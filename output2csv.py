import os
import geopandas as gpd
import pandas as pd
from analysis_modules import build_webmap  # Assumes analysis_modules.py is in the same folder

# Define a simple configuration class for directory paths.
class Config:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir  # Directory where supplementary files (e.g., neighborhoods, FOZ, poverty) are stored.
        self.output_dir = output_dir  # Directory where webmap output will be saved.

def main():
    # ---------------------------
    # 1. Define file paths.
    # You can switch between different files by commenting/uncommenting the desired options.
    input_geojson = ("/Users/oliveratwood/One Architecture Dropbox/_ONE LABS/[ Side Projects ]/ONE-Labs-Github/streets/output/all_segments_CoolPavement.geojson")
    output_csv = ("/Users/oliveratwood/One Architecture Dropbox/_NYC PROJECTS/P2415_CSC Year Two/09 Grant Applications/250224_DOT_PROTECT/2_Research/CoolPavementAnalysisOutput/citywide.csv")
    # input_geojson = ('/Users/oliveratwood/One Architecture Dropbox/_ONE LABS/[ Side Projects ]/ONE-Labs-Github/streets/output/East_New_York/all_segments_CoolPavement.geojson')
    # output_csv = ('/Users/oliveratwood/One Architecture Dropbox/_NYC PROJECTS/P2415_CSC Year Two/09 Grant Applications/250224_DOT_PROTECT/2_Research/CoolPavementAnalysisOutput/EastNewYork.csv')
    # input_geojson = ('/Users/oliveratwood/One Architecture Dropbox/_ONE LABS/[ Side Projects ]/ONE-Labs-Github/streets/output/Concourse/all_segments_CoolPavement.geojson')
    # output_csv = ('/Users/oliveratwood/One Architecture Dropbox/_NYC PROJECTS/P2415_CSC Year Two/09 Grant Applications/250224_DOT_PROTECT/2_Research/CoolPavementAnalysisOutput/Concourse.csv')
    # input_geojson = ('/Users/oliveratwood/One Architecture Dropbox/_ONE LABS/[ Side Projects ]/ONE-Labs-Github/streets/output/East_Harlem/all_segments_CoolPavement.geojson')
    # output_csv = ('/Users/oliveratwood/One Architecture Dropbox/_NYC PROJECTS/P2415_CSC Year Two/09 Grant Applications/250224_DOT_PROTECT/2_Research/CoolPavementAnalysisOutput/EastHarlem.csv')

    
    # ---------------------------
    # 2. Load the GeoJSON file into a GeoDataFrame.
    print("Loading GeoJSON data...")
    gdf = gpd.read_file(input_geojson)
    
    # ---------------------------
    # 3. Ensure the CRS is projected for accurate length calculations.
    if gdf.crs is None:
        print("Warning: CRS is not defined. Assuming geographic coordinates.")
    elif gdf.crs.is_geographic:
        print("GeoJSON is in a geographic CRS. Reprojecting to EPSG:3857 for accurate length calculations.")
        gdf = gdf.to_crs(epsg=3857)
    
    # ---------------------------
    # 4. Filter features by the 'priority' column.
    #    Modify top_x to change how many rows you want.
    top_x = 20
    gdf_top = gdf.sort_values(by="priority", ascending=False).head(top_x).copy()
    
    # ---------------------------
    # 5. Calculate the length of each geometry.
    # The .length attribute returns values in the CRS’s units.
    lengths = gdf_top.geometry.length
    if gdf_top.crs and gdf_top.crs.is_projected:
        # Convert from meters to feet (1 m = 3.28084 ft)
        gdf_top["length_ft"] = lengths * 3.28084
    else:
        gdf_top["length_ft"] = lengths
    
    # ---------------------------
    # 6. Save the filtered data as CSV.
    # Convert geometry to WKT so that it can be saved as text.
    gdf_csv = gdf_top.copy()
    gdf_csv["geometry"] = gdf_csv.geometry.apply(lambda geom: geom.wkt if geom is not None else None)
    gdf_csv.to_csv(output_csv, index=False)
    print(f"Filtered CSV saved to '{output_csv}'.")
    
    # ---------------------------
    # 7. Save the filtered data as a GeoJSON file for web mapping.
    # This file will be used by the build_webmap function.
    output_geojson = output_csv.replace(".csv", ".geojson")
    gdf_top.to_file(output_geojson, driver="GeoJSON")
    print(f"Filtered GeoJSON saved to '{output_geojson}'.")
    
    # ---------------------------
    # 8. Set up the configuration for the webmap.
    # config.input_dir should point to the directory containing supplementary files
    # (e.g., "CSC_Neighborhoods.geojson", "FOZ_NYC_Merged.geojson", and "nyc_persistent_poverty.geojson").
    # config.output_dir is where the webmap HTML will be saved.
    config_input_dir = (
        "/Users/oliveratwood/One Architecture Dropbox/_NYC PROJECTS/P2415_CSC Year Two/09 Grant Applications/250224_DOT_PROTECT/2_Research"
    )
    config_output_dir = os.path.dirname(output_csv)
    
    # Ensure the output directory exists.
    if not os.path.exists(config_output_dir):
        os.makedirs(config_output_dir)
    
    config = Config(input_dir=config_input_dir, output_dir=config_output_dir)
    
    # ---------------------------
    # 9. Create a dictionary for scenario GeoJSONs.
    # The build_webmap function expects a dictionary with keys as scenario names and values as GeoJSON file paths.
    scenario_geojsons = {"Citywide": output_geojson}
    
    # (Optional) Specify a neighborhood name if you want to zoom in on a specific area.
    neighborhood_name = None  # e.g., "East New York"
    
    # ---------------------------
    # 10. Build the webmap.
    print("Building webmap...")
    html_map_path = build_webmap(scenario_geojsons, config, neighborhood_name)
    print(f"Webmap created at: {html_map_path}")

if __name__ == "__main__":
    main()