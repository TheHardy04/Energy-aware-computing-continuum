import json
import argparse
import os
import logging

from src.cspPlacement import CSP
from src.infraProperties import InfraProperties
from src.networkGraph import NetworkGraph
from src.appProperties import AppProperties
from src.serviceGraph import ServiceGraph
from src.greedyFirstIteratePlacement import GreedyFirstIterate
from src.greedyFirstFitPlacement import GreedyFirstFit
from src.resultExporter import ResultExporter
from mappingUnitTest import MappingUnitTest



if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)

    infra_properties_path = r'properties/Infra_8nodes.properties'
    app_properties_path = r'properties/Appli_4comps.properties'

    parser = argparse.ArgumentParser(description='Demo placement runner')
    parser.add_argument('--start-host', type=int, default=None, help='Optional infra node id to start placement from')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the graphs')
    parser.add_argument('--verbose', action='store_true', help='Whether to print detailed graph info and results')
    parser.add_argument('--infra', type=str, default=infra_properties_path, help='Path to infrastructure properties file' \
    ' (default: properties/Infra_8nodes.properties)')
    parser.add_argument('--app', type=str, default=app_properties_path, help='Path to application properties file' \
    ' (default: properties/Appli_4comps.properties)')
    parser.add_argument('--placement-strategy', type=str, default='CSP', choices=['CSP', 'GreedyFirstFit', 'GreedyFirstIterate'], help='Placement strategy to use')
    parser.add_argument('--to-csv', type=str, default='results/placement.csv', help='Optional path to export results as CSV')
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
    logger.info("Running placement with strategy: %s", args.placement_strategy)
    if args.placement_strategy == 'CSP':
        strategy = CSP()
    elif args.placement_strategy == 'GreedyFirstFit':
        strategy = GreedyFirstFit()
    elif args.placement_strategy == 'GreedyFirstIterate':
        strategy = GreedyFirstIterate()

    result = strategy.place(svc, net)
    logger.info("Placement finished. Status: %s", result.meta.get('status'))

    ## RESULTS
    if result.meta.get('status') == 'infeasible':
        logger.error("Placement is infeasible!")
    else:
        logger.info("Placement successful!")

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
        
        if 'host_res' in result.meta:
            print('Final host resources:')
            print(json.dumps(result.meta['host_res'], indent=2))
        if 'edge_res' in result.meta:
            print('Final edge resources:')
            pretty = {f"{u}->{v}": info for (u, v), info in result.meta['edge_res'].items()}
            print(json.dumps(pretty, indent=2))
        print("\n")

    # Run unit tests
    MappingUnitTest.run_tests(net, svc, result)

    if args.to_csv:
        filename = args.to_csv
        # check if directory exists, if not create it
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        ResultExporter.export_placement_to_csv(result, filename=filename)
        logger.info(f"Placement exported to {filename}")

    # If graphs were plotted, inform the user to close them to exit
    if args.plot:
        input("\nPlacement complete. Press Enter to close graphs and exit...")