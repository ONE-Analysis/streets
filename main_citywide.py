import logging
import os
import warnings

from config import CoolCorridorsConfig
from utils import log_memory_usage
from data_preprocessing import load_and_preprocess_roads
from analysis_modules import (
    DataProcessors,
    OptimizedRasterProcessing,
    process_bus_stops,
    merge_street_segments,
    FinalAnalysis,
    export_results,
    incorporate_population_density,
    process_commercial_area
)

from webmap import build_webmap
from webmap import generate_webmap
def main():
    """
    Main entry point for the NYC Cool Corridors Analysis (Citywide).
    Usage: python main_citywide.py
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    warnings.filterwarnings('ignore')
    
    # 1) Initialize configuration and processors
    logging.info("Initializing configuration and processors...")
    config = CoolCorridorsConfig(analysis_type='citywide')
    raster_processor = OptimizedRasterProcessing()

    # 2) Load and preprocess roads
    logging.info("\n=== STEP 1: Loading and preprocessing road network ===")
    roads = load_and_preprocess_roads(config)
    logging.info(f"Preprocessed road network: {len(roads)} segments")

    # 3) Process basic attributes (pavement, bike lanes, vulnerability, etc.)
    logging.info("\n=== STEP 2: Processing basic attributes ===")
    roads = DataProcessors.batch_process_all(roads, config.input_dir)
    logging.info("Basic attributes processing complete")

    # 4) Process temperature data
    print("\nProcessing temperature data...")
    temp_raster_path = os.path.join(config.input_dir, 'Landsat9_ThermalComposite_ST_B10_2020-2023.tif')
    if not os.path.exists(temp_raster_path):
        raise FileNotFoundError(f"Temperature raster not found at: {temp_raster_path}")
    roads = raster_processor.optimize_temperature_processing(roads, temp_raster_path)
    log_memory_usage()
    roads['heat_indx'] = DataProcessors.normalize_to_index(roads['heat_mean'], 'temperature')
    print("\nTemperature Processing Results:")
    print(f"heat_mean range: {roads['heat_mean'].min():.1f}°F - {roads['heat_mean'].max():.1f}°F")
    print(f"heat_indx range: {roads['heat_indx'].min():.3f} - {roads['heat_indx'].max():.3f}")
    print(f"Missing values: {roads[['heat_mean', 'heat_indx']].isna().sum().to_dict()}")

    # 5) Process tree canopy data
    print("\nProcessing tree canopy data...")
    canopy_raster_path = os.path.join(config.input_dir, 'NYC_TreeCanopy.tif')
    if not os.path.exists(canopy_raster_path):
        raise FileNotFoundError(f"Tree canopy raster not found at: {canopy_raster_path}")
    roads = raster_processor.optimize_tree_canopy_processing(roads, canopy_raster_path)
    log_memory_usage()
    print("\nTree Canopy Processing Results:")
    if 'tree_pct' in roads.columns and 'tree_indx' in roads.columns:
        print(f"tree_pct range: {roads['tree_pct'].min():.1f}% - {roads['tree_pct'].max():.1f}%")
        print(f"tree_indx range: {roads['tree_indx'].min():.3f} - {roads['tree_indx'].max():.3f}")
        print(f"Missing values: {roads[['tree_pct', 'tree_indx']].isna().sum().to_dict()}")
    else:
        print("Warning: tree_pct or tree_indx columns not found in the results")

    # 6) Process bus stops
    print("\nProcessing bus stop proximity...")
    roads = process_bus_stops(roads, config.input_dir, config.crs)
    log_memory_usage()
    print("\nBus Stop Processing Results:")
    print(f"BusStpDens range: {roads['BusStpDens'].min():.1f} - {roads['BusStpDens'].max():.1f}")
    print(f"BusDensInx range: {roads['BusDensInx'].min():.3f} - {roads['BusDensInx'].max():.3f}")
    print(f"Missing values: {roads[['BusStpDens', 'BusDensInx']].isna().sum().to_dict()}")

    # 7) Incorporate population density
    logging.info("\nIncorporating population density...")
    roads = incorporate_population_density(roads, config.input_dir, buffer_ft=config.analysis_params["pop_buffer_ft"])
    log_memory_usage()

    # 8) Incorporate commercial activity areas
    print("\nProcessing commercial activity areas...")
    com_buffer_ft = config.analysis_params['pop_buffer_ft']
    roads = process_commercial_area(roads, config.input_dir, com_buffer_ft)
    data_processor = DataProcessors()
    roads['ComIndex'] = data_processor.normalize_to_index(roads['ComArea'], attribute_type='standard')
    print(f"ComIndex stats: min={roads['ComIndex'].min():.3f}, max={roads['ComIndex'].max():.3f}, mean={roads['ComIndex'].mean():.3f}")

    # 8) Merge road segments
    print("\nMerging street segments...")
    merged_roads = merge_street_segments(roads, config.analysis_params["min_segment_length"])
    log_memory_usage()

    # 9) Calculate final priority for the Cool Corridors scenario
    print("\nCalculating final priorities for Cool Corridors scenario...")
    analyzer = FinalAnalysis({'CoolCorridors': config.weight_scenarios['CoolCorridors']})
    results = analyzer.run_all_scenarios(merged_roads, config.analysis_params)
    scenario_data = results.get('CoolCorridors')
    if scenario_data is not None:
        print(f"\n=== Summary for CoolCorridors Scenario ===")
        priority_count = scenario_data['is_priority'].sum()
        total_count = len(scenario_data)
        print(f"Total segments analyzed: {total_count}")
        print(f"High priority segments: {priority_count} ({(priority_count/total_count)*100:.1f}%)")
        print(f"Mean priority score: {scenario_data['priority'].mean():.3f}")
    else:
        print("No results found for CoolCorridors scenario.")

    print("\nColumns in merged_roads:", list(merged_roads.columns))

    # 10) Export final results and generate a webmap
    print("\nExporting results and generating webmap...")
    try:
        exported_paths = export_results(results_dict=results, config=config)
        if exported_paths:
            webmap_paths = generate_webmap(results_dict=results, exported_paths=exported_paths, config=config)
            if webmap_paths:
                print("\nSuccessfully generated webmaps at:")
                for path in webmap_paths:
                    print(f"- {path}")
            else:
                print("Warning: Webmap generation failed")
        else:
            print("Warning: No results were exported, skipping webmap generation")
    except Exception as e:
        logging.error(f"Error in export and webmap process: {str(e)}", exc_info=True)

    print("\nAll done!")

if __name__ == "__main__":
    main()