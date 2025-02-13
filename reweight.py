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
    config = CoolCorridorsConfig(analysis_type='citywide')
    scenario_file = os.path.join(config.output_dir, "all_segments_CoolCorridors.geojson")
    if not os.path.exists(scenario_file):
        logging.warning(f"File not found: {scenario_file}. Skipping reweighting.")
        return
    logging.info(f"\nLoading ALL segments for analysis from {scenario_file}")
    roads_gdf = gpd.read_file(scenario_file)
    
    # Use the full weight_scenarios from config
    new_weights = config.weight_scenarios
    analyzer = FinalAnalysis(new_weights)
    analysis_params = {"priority_threshold": 999999}
    results_dict = analyzer.run_all_scenarios(roads_gdf, analysis_params)
    
    for scenario, scenario_result in results_dict.items():
        if scenario_result is None:
            logging.warning(f"No results for {scenario} scenario. Skipping.")
            continue
        number_of_top_roads = config.analysis_params.get('number_of_top_roads', 100)
        topN_gdf = scenario_result.nlargest(number_of_top_roads, "priority").copy()
        topN_filename = f"top{number_of_top_roads}_{scenario}.geojson"
        topN_path = os.path.join(config.output_dir, topN_filename)
        topN_gdf.to_file(topN_path, driver="GeoJSON")
        logging.info(f"Exported top-{number_of_top_roads} for {scenario} -> {topN_filename} ({len(topN_gdf)} features)")
        
        scenario_name_topN = f"{scenario}_top{number_of_top_roads}"
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
    
    logging.info(f"\nDone! Created top road files and webmaps for scenarios: {', '.join(results_dict.keys())}.")

if __name__ == "__main__":
    main()