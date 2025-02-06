"""
1. Loads each scenario's all_segments_{ScenarioName}.geojson (exported by main_citywide.py).
2. Defines new weights (or reuses original) for CoolPavement & CoolCorridors.
3. Recalculates 'priority' for each scenario.
4. Selects the top 200 roads by priority.
5. Exports exactly one file per scenario, named top200_{ScenarioName}.geojson.
6. Generates separate webmaps, each labeled with '_top200' in the name:
   - CoolPavement_top200_webmap.html
   - CoolCorridors_top200_webmap.html
"""

import os
import logging
import warnings
import geopandas as gpd

from config import CoolPavementConfig
from analysis_modules import FinalAnalysis, generate_webmap

def main():
    # --------------------------------------------------------------------------
    # 1) Basic setup
    # --------------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    warnings.filterwarnings('ignore')

    config = CoolPavementConfig(analysis_type='citywide')

    # --------------------------------------------------------------------------
    # 2) Indicate the "all_segments" files for our two scenarios
    # --------------------------------------------------------------------------
    scenario_files = {
        "CoolPavement": os.path.join(config.output_dir, "all_segments_CoolPavement.geojson"),
        "CoolCorridors": os.path.join(config.output_dir, "all_segments_CoolCorridors.geojson")
    }

    # --------------------------------------------------------------------------
    # 3) Define new (or possibly the same) weights for each scenario
    #    They must sum to 1.0 individually.
    # --------------------------------------------------------------------------
    # Weight scenarios
    new_weights = {
        'CoolPavement': {
            'PavementIndex': 0.25,          # uses 'v' shaped suitability
            'HeatHazIndex': 0,            # raw heat values from summers 2021-2024
            'TreeCanopyIndex': 0.1,        # assessment of tree canopy percentage
            'HeatVulnerabilityIndex': 0.1, # heat vulnerability index values
            'BusDensInx': 0,             # bus stops per mile
            'BikeLnIndx': 0.1,             # bike lane length per mile
            'PedIndex': 0.2,                  # road prioritization in NYC Pedestrian Mobility Plan
            'pop_dens_indx': 0.25,          # people per square mile within 1000' buffer
            'ComIndex': 0                   # commercial space within 1000' buffer
        },

        'CoolCorridors': {
            'PavementIndex': 0,             # uses 'v' shaped suitability
            'HeatHazIndex': 0,              # raw heat values from summers 2021-2024
            'TreeCanopyIndex': 0.2,         # assessment of tree canopy percentage
            'HeatVulnerabilityIndex': 0.1,  # heat vulnerability index values
            'BusDensInx': 0.05,             # bus stops per mile
            'BikeLnIndx': 0.05,             # bike lane length per mile
            'PedIndex': 0.3,                # road prioritization in NYC Pedestrian Mobility Plan
            'pop_dens_indx': 0.15,          # people per square mile within 1000' buffer
            'ComIndex': 0.15                # commercial space within 1000' buffer
        }

        #     'CoolCorridors': {
        #     'PavementIndex': 0,             # uses 'v' shaped suitability
        #     'HeatHazIndex': 0,              # raw heat values from summers 2021-2024
        #     'TreeCanopyIndex': 0.25,        # assessment of tree canopy percentage
        #     'HeatVulnerabilityIndex': 0,    # heat vulnerability index values
        #     'BusDensInx': 0.15,             # bus stops per mile
        #     'BikeLnIndx': 0.15,             # bike lane length per mile
        #     'PedIndex': 0.2,                # road prioritization in NYC Pedestrian Mobility Plan
        #     'pop_dens_indx': 0.25,          # people per square mile within 1000' buffer
        #     'ComIndex': 0                   # commercial space within 1000' buffer
        # }
    }
    # We'll store our "top 200" GeoDataFrames and paths, 
    # but generate separate webmaps so each has `_top200` in the filename.
    for scenario_name, file_path in scenario_files.items():
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}. Skipping {scenario_name}.")
            continue

        logging.info(f"\nLoading ALL segments for {scenario_name} from {file_path}")
        roads_gdf = gpd.read_file(file_path)

        # ----------------------------------------------------------------------
        # 4) Check required columns
        # ----------------------------------------------------------------------
        required_cols = [
            'pave_indx', 'heat_indx', 'tree_indx', 'vuln_indx',
            'BusDensInx', 'BikeLnIndx', 'PedIndex', 'pop_dens_indx'
        ]
        missing_cols = [c for c in required_cols if c not in roads_gdf.columns]
        if missing_cols:
            logging.error(
                f"Missing columns {missing_cols} in {file_path}. Cannot re-run scenario."
            )
            continue

        # ----------------------------------------------------------------------
        # 5) Build a FinalAnalysis for *this* scenario alone
        #    so we only produce one new 'priority' column
        # ----------------------------------------------------------------------
        single_scenario_weights = {scenario_name: new_weights[scenario_name]}
        analyzer = FinalAnalysis(weight_scenarios=single_scenario_weights)

        # Large threshold so we won't filter within the analysis
        analysis_params = {"priority_threshold": 999999}

        # Run analysis -> { scenario_name: df, "ALL": combined_df }
        results_dict = analyzer.run_all_scenarios(roads_gdf, analysis_params)
        scenario_result = results_dict.get(scenario_name, None)
        if scenario_result is None:
            logging.warning(f"No results for {scenario_name} after run_all_scenarios(). Skipping.")
            continue

        # ----------------------------------------------------------------------
        # 6) Pick the top 200
        # ----------------------------------------------------------------------
        top200_gdf = scenario_result.nlargest(200, "priority").copy()
        top200_filename = f"top200_{scenario_name}.geojson"
        top200_path = os.path.join(config.output_dir, top200_filename)
        top200_gdf.to_file(top200_path, driver="GeoJSON")
        logging.info(f"Exported top-200 for {scenario_name} -> {top200_filename} ({len(top200_gdf)} features)")

        # ----------------------------------------------------------------------
        # 7) Generate a separate webmap with a "_top200" suffix
        # ----------------------------------------------------------------------
        # We'll rename the scenario key to e.g. "CoolPavement_top200"
        # so the final HTML name has that suffix in it.
        scenario_name_top200 = f"{scenario_name}_top200"

        try:
            webmap_paths = generate_webmap(
                results_dict={scenario_name_top200: top200_gdf},
                exported_paths={scenario_name_top200: top200_path},
                config=config
            )

            if webmap_paths:
                for wmap in webmap_paths:
                    logging.info(f"Generated webmap: {wmap}")
            else:
                logging.warning("No webmap generated (generate_webmap returned None).")

        except Exception as e:
            logging.error(f"Error generating webmap for {scenario_name}_top200: {e}")

    logging.info("\nDone! Created top200_{ScenarioName}.geojson and a webmap with '_top200' in the name for each scenario.")

if __name__ == "__main__":
    main()