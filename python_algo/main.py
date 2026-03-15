import json
import argparse
import os
import logging
import time

from src.infraProperties import InfraProperties
from src.networkGraph import NetworkGraph
from src.appProperties import AppProperties
from src.serviceGraph import ServiceGraph
from src.greedyFirstIteratePlacement import GreedyFirstIterate
from src.greedyFirstFitPlacement import GreedyFirstFit
from src.cspPlacement import CSP
from src.llmPlacement import LLMPlacement
from src.resultExporter import ResultExporter
from src.evaluation import Evaluator
from mappingUnitTest import MappingUnitTest


P_CPU_WATTS_DEFAULT = 45.0
SOLVER_POWER_UNCERTAINTY = 0.20
LLM_UNCERTAINTY_DEFAULT = 3.0

# Wh per 1k tokens (input, output)
LLM_TOKEN_ENERGY_WH_PER_1K = {
    'anthropic': (0.0015, 0.0025),
    'openai': (0.0015, 0.0025),
    'gemini': (0.0010, 0.0020),
}


def _compute_solver_energy_meta(strategy_name: str, result_meta: dict, resolution_time_s: float) -> dict:
    """Compute solver overhead energy and uncertainty band."""
    p_cpu_w = float(os.environ.get('P_CPU_WATTS', str(P_CPU_WATTS_DEFAULT)))
    provider = str(result_meta.get('provider', '') or '').lower()
    tokens_in = int(result_meta.get('tokens_in') or 0)
    tokens_out = int(result_meta.get('tokens_out') or 0)
    llm_time_s = float(result_meta.get('llm_time') or 0.0)

    # LLM cloud providers: token-energy model
    if strategy_name == 'LLM' and provider in LLM_TOKEN_ENERGY_WH_PER_1K and (tokens_in > 0 or tokens_out > 0):
        eps_in, eps_out = LLM_TOKEN_ENERGY_WH_PER_1K[provider]
        e_solver_wh = eps_in * (tokens_in / 1000.0) + eps_out * (tokens_out / 1000.0)
        unc = float(os.environ.get('LLM_SOLVER_UNCERTAINTY', str(LLM_UNCERTAINTY_DEFAULT)))
        unc = max(1.0, unc)
        e_solver_min_wh = e_solver_wh / unc
        e_solver_max_wh = e_solver_wh * unc
        model = 'llm_tokens'
        details = {
            'provider': provider,
            'epsilon_in_wh_per_1k': eps_in,
            'epsilon_out_wh_per_1k': eps_out,
            'uncertainty_factor': unc,
            'tokens_in': tokens_in,
            'tokens_out': tokens_out,
        }
    # Ollama local: power-time model when inference runtime is known
    elif strategy_name == 'LLM' and provider == 'ollama' and llm_time_s > 0:
        e_solver_wh = p_cpu_w * llm_time_s / 3600.0
        e_solver_min_wh = 0.8 * e_solver_wh
        e_solver_max_wh = 1.2 * e_solver_wh
        model = 'runtime_power'
        details = {
            'provider': provider,
            'p_cpu_watts': p_cpu_w,
            'runtime_s': llm_time_s,
            'uncertainty_ratio': SOLVER_POWER_UNCERTAINTY,
        }
    # Fallback for CSP/Greedy or missing LLM token/runtime data: power-time model
    else:
        e_solver_wh = p_cpu_w * resolution_time_s / 3600.0
        e_solver_min_wh = 0.8 * e_solver_wh
        e_solver_max_wh = 1.2 * e_solver_wh
        model = 'runtime_power'
        details = {
            'p_cpu_watts': p_cpu_w,
            'runtime_s': resolution_time_s,
            'uncertainty_ratio': SOLVER_POWER_UNCERTAINTY,
        }

    return {
        'e_solver_wh': e_solver_wh,
        'e_solver_min_wh': e_solver_min_wh,
        'e_solver_max_wh': e_solver_max_wh,
        'e_solver_j': e_solver_wh * 3600.0,
        'e_solver_min_j': e_solver_min_wh * 3600.0,
        'e_solver_max_j': e_solver_max_wh * 3600.0,
        'solver_energy_model': model,
        'solver_energy_details': details,
    }



if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)

    infra_properties_path = r'properties/Infra_16nodes_fog3tier.properties'
    app_properties_path = r'properties/Appli_8comps_smartbuilding.properties'

    parser = argparse.ArgumentParser(description='Demo placement runner')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the graphs')
    parser.add_argument('--verbose', action='store_true', help='Whether to print detailed graph info and results')
    parser.add_argument('--infra', type=str, default=infra_properties_path, help='Path to infrastructure properties file' \
    ' (default: properties/Infra_16nodes_fog3tier.properties)')
    parser.add_argument('--app', type=str, default=app_properties_path, help='Path to application properties file' \
    ' (default: properties/Appli_8comps_smartbuilding.properties)')
    parser.add_argument('--strategy', type=str, default='CSP', choices=['CSP', 'LLM', 'GreedyFirstFit', 'GreedyFirstIterate'], help='Placement strategy to use')
    parser.add_argument('--placement-csv', type=str, default='results/placement.csv', help='Optional path to export placement results as CSV')
    parser.add_argument('--metrics-csv', type=str, default='results/metrics.csv', help='Optional path to export evaluation metrics as CSV (rows are appended, useful for benchmarks)')
    args = parser.parse_args()


    # Load infrastructure and create network graph
    infra = InfraProperties.from_file(args.infra)

    G = NetworkGraph.from_infra_dict(infra.to_dict())
    logger.info("Nodes: %s", G.G.number_of_nodes())
    logger.info("Edges: %s", G.G.number_of_edges())
    # G.print_summary() # Skipping verbose prints
    
    if args.verbose:
        print("\nSummary:")
        G.print_summary()
        print("\nNodes:")
        G.print_nodes()
        print("\nEdges:")
        G.print_edges()
        print("\nDegree stats:")
        print(json.dumps(G.degree_stats(), indent=2))
        print("\nConnectivity:")
        print(json.dumps(G.connectivity_info(), indent=2))

    if args.plot:
        G.draw(block=False)

    # Load application and create service graph
    app = AppProperties.from_file(args.app)

    service_G = ServiceGraph.from_app_dict(app.to_dict())
    logger.info("Service Nodes: %s", service_G.G.number_of_nodes())
    logger.info("Service Edges: %s", service_G.G.number_of_edges())
    
    if args.verbose:
        print("\nService Graph Summary:")
        service_G.print_summary()
        print("\nService Graph Nodes:")
        service_G.print_nodes()
        print("\nService Graph Edges:")
        service_G.print_edges()
        print("\nService Graph Degree stats:")
        print(json.dumps(service_G.degree_stats(), indent=2))
        print("\nService Graph Connectivity:")
        print(json.dumps(service_G.connectivity_info(), indent=2))
        
    if args.plot:
        service_G.draw(block=False)

    # Create graph wrappers from properties
    net = NetworkGraph.from_infra_dict(infra.to_dict())
    svc = ServiceGraph.from_app_dict(app.to_dict())

    # wait for user input to start placement (to allow time to inspect graphs if plotted)
    if args.plot:
        input("\nPress Enter to start placement...")

    ## PLACEMENT
    logger.info("Running placement with %s strategy...", args.strategy)  
    if args.strategy == 'CSP':
        strategy = CSP()
    elif args.strategy == 'LLM':
        strategy = LLMPlacement()
    elif args.strategy == 'GreedyFirstFit':
        strategy = GreedyFirstFit()
    elif args.strategy == 'GreedyFirstIterate':
        strategy = GreedyFirstIterate()

    start_time = time.perf_counter()
    result = strategy.place(svc, net)
    resolution_time_s = time.perf_counter() - start_time
    result.meta['resolution_time_s'] = resolution_time_s
    result.meta.update(_compute_solver_energy_meta(args.strategy, result.meta, resolution_time_s))
    logger.info("Placement finished. Status: %s", result.meta.get('status'))
    logger.info("Placement resolution time: %.6f s", resolution_time_s)
    logger.info(
        "Solver overhead energy: %.6f Wh [%.6f, %.6f]",
        float(result.meta.get('e_solver_wh') or 0.0),
        float(result.meta.get('e_solver_min_wh') or 0.0),
        float(result.meta.get('e_solver_max_wh') or 0.0),
    )

    ## RESULTS
    if result.meta.get('status') == 'success':
        logger.info("Placement successful.")
    else:
        logger.warning("Placement failed. Reason: %s", result.meta.get('reason'))

    if args.verbose:
        print("\n========== Placement Result ==========")
        print('Path:')
        print(result.paths)
        print('Mapping (component -> host):')
        print(json.dumps(result.mapping, indent=2))
        
        if 'routing' in result.meta:
            print('Routing (service edges):')
            pretty = {f"{u}->{v}": info for (u, v), info in result.meta['routing'].items()}
            print(json.dumps(pretty, indent=2))
        else:
            print('Details:', json.dumps(result.meta, indent=2))
        print(f"Resolution time: {resolution_time_s:.6f} s")
        print(
            f"Solver overhead energy: {float(result.meta.get('e_solver_wh') or 0.0):.6f} Wh "
            f"[{float(result.meta.get('e_solver_min_wh') or 0.0):.6f}, {float(result.meta.get('e_solver_max_wh') or 0.0):.6f}]"
        )
        print(
            f"Solver overhead energy: {float(result.meta.get('e_solver_j') or 0.0):.2f} J "
            f"[{float(result.meta.get('e_solver_min_j') or 0.0):.2f}, {float(result.meta.get('e_solver_max_j') or 0.0):.2f}]"
        )
        
        if 'host_res' in result.meta:
            print('Final host resources:')
            print(json.dumps(result.meta['host_res'], indent=2))
        if 'edge_res' in result.meta:
            print('Final edge resources:')
            pretty = {f"{u}->{v}": info for (u, v), info in result.meta['edge_res'].items()}
            print(json.dumps(pretty, indent=2))
        print("\n")

    # Run unit tests
    is_valid = MappingUnitTest.run_tests(net, svc, result, logger=logger)

    # Evaluate placement
    logger.info("Evaluating placement...")
    metrics = Evaluator.evaluate(net, svc, result, verbose=args.verbose)  

    # Export results to CSV if requested
    if args.placement_csv:
        logger.info("Exporting placement results to CSV...")
        filename = args.placement_csv
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        ResultExporter.export_placement_to_csv(result, filename=filename)
        logger.info(f"Placement exported to {filename}")

    # Export evaluation metrics to CSV if requested
    if args.metrics_csv:
        logger.info("Exporting evaluation metrics to CSV...")
        metrics_filename = args.metrics_csv
        os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)
        path_taken = json.dumps(
            {
                f"{u}->{v}": path
                for (u, v), path in sorted(result.paths.items())
            },
            separators=(",", ":")
        )
        ResultExporter.export_metrics_to_csv(
            metrics,
            filename=metrics_filename,
            extra_fields={
                'strategy': args.strategy,
                'infra': os.path.basename(args.infra),
                'app': os.path.basename(args.app),
                'status': result.meta.get('status', ''),
                'resolution_time_s': resolution_time_s,
                'path_taken': path_taken,
            },
            append=True,
        )
        logger.info(f"Evaluation metrics exported to {metrics_filename}")

    
    # If graphs were plotted, inform the user to close them to exit
    if args.plot:
        input("\nPlacement complete. Press Enter to close graphs and exit...")
    
    logger.info("Done.")