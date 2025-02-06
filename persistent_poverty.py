import os
import time
import pandas as pd
import geopandas as gpd

def create_persistent_poverty_blocks(config):
    """
    Create a dataset of NYC census blocks in persistent poverty tracts.

    Args:
        config: Configuration object containing input_dir

    Returns:
        GeoDataFrame containing census blocks in persistent poverty tracts
    """
    try:
        print("Loading input datasets...")

        # Load the persistent poverty tracts CSV with error handling for encoding
        poverty_tracts_path = os.path.join(config.input_dir, 'census-tracts-in-persistent-poverty.csv')
        try:
            # First try UTF-8
            poverty_tracts = pd.read_csv(poverty_tracts_path)
        except UnicodeDecodeError:
            try:
                # Try with 'latin-1' encoding
                poverty_tracts = pd.read_csv(poverty_tracts_path, encoding='latin-1')
            except UnicodeDecodeError:
                # If that fails, try with 'cp1252' (Windows encoding)
                poverty_tracts = pd.read_csv(poverty_tracts_path, encoding='cp1252')

        # Load the NYC blocks GeoJSON
        blocks_path = os.path.join(config.input_dir, 'nyc_blocks_with_pop.geojson')
        blocks_gdf = gpd.read_file(blocks_path)

        print(f"Loaded {len(poverty_tracts)} poverty tracts and {len(blocks_gdf)} census blocks")

        # Create concatenated tract ID in blocks dataset
        blocks_gdf['tract_id'] = (
            blocks_gdf['STATEFP20'].astype(str).str.zfill(2) +
            blocks_gdf['COUNTYFP20'].astype(str).str.zfill(3) +
            blocks_gdf['TRACTCE20'].astype(str).str.zfill(6)
        )

        # Ensure the Tract column in poverty_tracts is string type and clean it
        poverty_tracts['Tract'] = poverty_tracts['Tract'].astype(str).str.strip()

        # Create list of poverty tract IDs
        poverty_tract_ids = set(poverty_tracts['Tract'])

        # Filter blocks to only those in poverty tracts
        poverty_blocks = blocks_gdf[blocks_gdf['tract_id'].isin(poverty_tract_ids)].copy()

        print(f"Found {len(poverty_blocks)} blocks in persistent poverty tracts")

        # Print some diagnostic information
        print("\nDiagnostic Information:")
        print(f"Sample of tract_id values: {list(blocks_gdf['tract_id'].head())}")
        print(f"Sample of Tract values from CSV: {list(poverty_tracts['Tract'].head())}")

        return poverty_blocks

    except Exception as e:
        print(f"Error in creating persistent poverty blocks dataset: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Input directory contents: {os.listdir(config.input_dir)}")
        raise

def main():
    """Main execution function for persistent poverty analysis."""
    class Config:
        def __init__(self):
            # Set paths relative to script location
            self.input_dir = "input"
            self.output_dir = "output"

    try:
        print("\n=== Starting Persistent Poverty Analysis ===")
        start_time = time.time()

        # Initialize config
        config = Config()

        # Verify input directory exists and contains required files
        if not os.path.exists(config.input_dir):
            raise FileNotFoundError(f"Input directory '{config.input_dir}' not found")

        required_files = [
            'census-tracts-in-persistent-poverty.csv',
            'nyc_blocks_with_pop.geojson'
        ]
        for file in required_files:
            if not os.path.exists(os.path.join(config.input_dir, file)):
                raise FileNotFoundError(f"Required file '{file}' not found in input directory")

        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)

        # Process the data
        print("\nProcessing persistent poverty blocks...")
        poverty_blocks = create_persistent_poverty_blocks(config)

        # Save results
        output_path = os.path.join(config.output_dir, 'nyc_persistent_poverty.geojson')
        print(f"\nSaving results to {output_path}")
        poverty_blocks.to_file(output_path, driver='GeoJSON')

        # Print summary
        print(f"\nAnalysis Summary:")
        print(f"Total census blocks in poverty tracts: {len(poverty_blocks)}")
        print(f"Total unique tracts: {len(poverty_blocks['tract_id'].unique())}")

        execution_time = time.time() - start_time
        print(f"\nAnalysis completed in {execution_time:.2f} seconds")
        print("=== Persistent Poverty Analysis Complete ===\n")

    except Exception as e:
        print(f"\nError in execution: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        raise

if __name__ == "__main__":
    main()
