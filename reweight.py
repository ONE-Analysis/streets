"""
1. Loads the all_segments_CoolCorridors.geojson (exported by main_citywide.py).
2. Defines new weights (if needed) for Cool Corridors.
3. Recalculates 'priority' for the scenario.
4. Selects the top N roads by priority (where N = config.number_of_top_roads).
5. Exports one file named topN_CoolCorridors.geojson.
6. Generates a separate webmap labeled with '_topN'.
"""

import os
import logging
import warnings
import geopandas as gpd

from config import CoolCorridorsConfig
from analysis_modules import FinalAnalysis
from webmap import generate_webmap

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    warnings.filterwarnings('ignore')

    # Load config
    config = CoolCorridorsConfig(analysis_type='citywide')

    scenario_file = os.path.join(config.output_dir, "all_segments_CoolCorridors.geojson")
    if not os.path.exists(scenario_file):
        logging.warning(f"File not found: {scenario_file}. Skipping CoolCorridors reweighting.")
        return

    logging.info(f"\nLoading ALL segments for CoolCorridors from {scenario_file}")
    roads_gdf = gpd.read_file(scenario_file)

    # Check for required columns
    required_cols = [
        'pave_indx', 'heat_indx', 'tree_indx', 'vuln_indx',
        'BusDensInx', 'BikeLnIndx', 'PedIndex', 'pop_dens_indx'
    ]
    missing_cols = [c for c in required_cols if c not in roads_gdf.columns]
    if missing_cols:
        logging.error(f"Missing columns {missing_cols} in {scenario_file}. Cannot re-run scenario.")
        return

    # Define (or reuse) weights for Cool Corridors
    new_weights = {
            # 'CoolPavement': {
            #     'PavementIndex': 0.25,          # uses 'v' shaped suitability
            #     'HeatHazIndex': 0,            # raw heat values from summers 2021-2024
            #     'TreeCanopyIndex': 0.1,        # assessment of tree canopy percentage
            #     'HeatVulnerabilityIndex': 0.1, # heat vulnerability index values
            #     'BusDensInx': 0,             # bus stops per mile
            #     'BikeLnIndx': 0.1,             # bike lane length per mile
            #     'PedIndex': 0.2,                  # road prioritization in NYC Pedestrian Mobility Plan
            #     'pop_dens_indx': 0.25,          # people per square mile within 1000' buffer
            #     'ComIndex': 0                   # commercial space within 1000' buffer
            # },
            'CoolCorridors': {
                'PavementIndex': 0,
                'HeatHazIndex': 0,
                'HeatVulnerabilityIndex': 0,
                'BusDensInxDensInx': 0,
                'BikeLnIndx': 0,
                'PedIndex': 0.1,
                'pop_dens_indx': 0.2,
                'TreeCanopyIndex': 0.3,
                'ComIndex': 0.2,
                'SidewalkIndex': 0.1,
                'RoadWidthIndex': 0.1
            }
        }
        
    analyzer = FinalAnalysis({'CoolCorridors': new_weights['CoolCorridors']})
    analysis_params = {"priority_threshold": 999999}
    results_dict = analyzer.run_all_scenarios(roads_gdf, analysis_params)
    scenario_result = results_dict.get('CoolCorridors', None)
    if scenario_result is None:
        logging.warning("No results for CoolCorridors after run_all_scenarios(). Skipping.")
        return

    # Get the number_of_top_roads from config
    number_of_top_roads = config.analysis_params.get('number_of_top_roads', 100)
    # Alternatively, if you set it directly on config: number_of_top_roads = config.number_of_top_roads

    # Select the top N roads by priority (where N = number_of_top_roads)
    topN_gdf = scenario_result.nlargest(number_of_top_roads, "priority").copy()
    topN_filename = f"top{number_of_top_roads}_CoolCorridors.geojson"
    topN_path = os.path.join(config.output_dir, topN_filename)
    topN_gdf.to_file(topN_path, driver="GeoJSON")
    logging.info(f"Exported top-{number_of_top_roads} for CoolCorridors -> {topN_filename} ({len(topN_gdf)} features)")

    # Generate a separate webmap with a "_topN" suffix
    scenario_name_topN = f"CoolCorridors_top{number_of_top_roads}"
    try:
        webmap_paths = generate_webmap(
            results_dict={scenario_name_topN: topN_gdf},
            exported_paths={scenario_name_topN: topN_path},
            config=config
        )
        if webmap_paths:
            for wmap in webmap_paths:
                logging.info(f"Generated webmap: {wmap}")
        else:
            logging.warning("No webmap generated (generate_webmap returned None).")
    except Exception as e:
        logging.error(f"Error generating webmap for {scenario_name_topN}: {e}")

    logging.info(
        f"\nDone! Created {topN_filename} and a webmap with '_top{number_of_top_roads}' in the name for CoolCorridors."
    )

if __name__ == "__main__":
    main()