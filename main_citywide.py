import logging
import os
import warnings

# Local imports (make sure these match your actual file/module names)
from config import CoolPavementConfig
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
    build_webmap,
    generate_webmap
)

def main():
    """
    Main entry point for the NYC Cool Pavement Prioritization Analysis.
    Run this script in a terminal: `python main.py`
    """
    # Configure logging and suppress warnings if desired
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    warnings.filterwarnings('ignore')
    
    # 1) Setup config and classes
    logging.info("Initializing configuration and processors...")
    config = CoolPavementConfig(analysis_type='citywide')
    raster_processor = OptimizedRasterProcessing()

    # 2) Load and preprocess roads
    logging.info("\n=== STEP 1: Loading and preprocessing road network ===")
    roads = load_and_preprocess_roads(config)
    logging.info(f"Preprocessed road network: {len(roads)} segments")

    # 3) Process basic attributes (pavement, bike lanes, vulnerability)
    logging.info("\n=== STEP 2: Processing basic attributes (pavement, bike lanes, vulnerability) ===")
    roads = DataProcessors.batch_process_all(roads, config.input_dir)
    logging.info("Basic attributes processing complete")

    # 4) Process temperature data
    print("\nProcessing temperature data...")
    temp_raster_path = os.path.join(config.input_dir, 'Landsat9_ThermalComposite_ST_B10_2020-2023.tif')
    if not os.path.exists(temp_raster_path):
        raise FileNotFoundError(f"Temperature raster not found at: {temp_raster_path}")

    roads = raster_processor.optimize_temperature_processing(roads, temp_raster_path)
    log_memory_usage()

    # Normalize heat_mean -> heat_indx
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

    # 7) Incorporate population density (downloads TIGER data & Census, saves GeoJSON, etc.)
    logging.info("\nIncorporating population density...")
    roads = incorporate_population_density(
        roads,
        config.input_dir,
        buffer_ft=config.analysis_params["pop_buffer_ft"]  # e.g. 1000
        # If needed, add census_api_key=...
    )
    log_memory_usage()

    # 8) Merge road segments
    print("\nMerging street segments...")
    merged_roads = merge_street_segments(roads, config.analysis_params["min_segment_length"])
    log_memory_usage()

    # 9) Calculate final priority
    print("\nCalculating final priorities for all scenarios...")

    # Initialize analyzer with scenarios from the config
    analyzer = FinalAnalysis(config.weight_scenarios)

    # Run analysis for all scenarios
    results = analyzer.run_all_scenarios(merged_roads, config.analysis_params)

    # Print summary stats
    for scenario_name, scenario_data in results.items():
        if scenario_name == 'ALL':
            print(f"\n=== Summary for ALL Scenario ===")
            print(f"Total segments: {len(scenario_data)}")
            for base_scenario in analyzer.weight_scenarios.keys():
                priority_count = scenario_data[f'is_priority_{base_scenario}'].sum()
                total_count = len(scenario_data)
                mean_priority = scenario_data[f'priority_{base_scenario}'].mean()
                print(f"\n{base_scenario} priorities within ALL scenario:")
                print(f"High priority segments: {priority_count} ({(priority_count/total_count)*100:.1f}%)")
                print(f"Mean priority score: {mean_priority:.3f}")
        else:
            print(f"\n=== Summary for {scenario_name} Scenario ===")
            priority_count = scenario_data['is_priority'].sum()
            total_count = len(scenario_data)
            print(f"Total segments analyzed: {total_count}")
            print(f"High priority segments: {priority_count} ({(priority_count/total_count)*100:.1f}%)")
            print(f"Mean priority score: {scenario_data['priority'].mean():.3f}")

    # Print columns in merged_roads
    print("\nColumns in merged_roads:", list(merged_roads.columns))


    # 10) Export final results and generate webmap
    print("\nExporting results and generating webmap...")
    try:
        # Export results to GeoJSON
        exported_paths = export_results(results_dict=results, config=config)

        if exported_paths:
            # Generate webmap using exported GeoJSON files
            webmap_path = generate_webmap(
                results_dict=results,
                exported_paths=exported_paths,
                config=config
            )

            if webmap_path:
                print(f"Successfully generated webmap at: {webmap_path}")
            else:
                print("Warning: Webmap generation failed")
        else:
            print("Warning: No results were exported, skipping webmap generation")

    except Exception as e:
        logging.error(f"Error in export and webmap process: {str(e)}", exc_info=True)

    print("\nAll done!")

# Standard entry point
if __name__ == "__main__":
    main()