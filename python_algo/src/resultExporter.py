from src.placementAlgo import PlacementResult

import csv


class ResultExporter:
    """
    Utility class to export placement results.
    """
    @staticmethod
    def export_placement_to_csv(placement_result: PlacementResult, filename: str):
        """
        Exports the placement mapping to a CSV file with columns: Component, Host.

        :param placement_result: The placement result containing the mapping to export
        :type placement_result: PlacementResult
        :param filename: The name of the CSV file to write to
        :type filename: str
        """
        # check if filename ends with .csv, if not add it
        if not filename.endswith('.csv'):
            filename += '.csv'
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Component', 'Host'])
            for comp, host in placement_result.mapping.items():
                writer.writerow([comp, host])
        print(f"Placement mapping exported to {filename}")