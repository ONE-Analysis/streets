import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

import rasterio
from rasterio import features, mask
from rasterio.transform import from_origin
from rasterio.io import MemoryFile

def load_and_preprocess_roads(config):
    """Load and preprocess road network data."""
    try:
        print("Loading road network data...")

        # -------------------------------
        # Load road network data from GeoJSON
        # -------------------------------
        roads = gpd.read_file(os.path.join(config.input_dir, 'lion_data.geojson'))
        print(f"Initial load - total features: {len(roads)}")
        print(f"Columns available: {list(roads.columns)}")

        # Ensure correct CRS
        roads = roads.to_crs(config.crs)

        # Standardize SegmentID and create a padded segmentid for merging later
        roads['SegmentID'] = roads['SegmentID'].astype(str).str.strip()
        roads['segmentid'] = roads['SegmentID'].str.zfill(7)

        # Clean string fields
        string_columns = ['FeatureTyp', 'RW_TYPE', 'Status', 'NonPed']
        for col in string_columns:
            roads[col] = roads[col].str.strip()

        # =====================================
        # Filter by FeatureTyp
        # =====================================
        roads = roads[roads['FeatureTyp'].isin(['0', 'C'])]
        print(f"\nAfter FeatureTyp filter: {len(roads)} roads remaining")

        # =====================================
        # Filter by RW_TYPE
        # =====================================
        roads = roads[roads['RW_TYPE'] == '1']
        print(f"After RW_TYPE filter: {len(roads)} roads remaining")
        
        # =====================================
        # Filter by Status
        # =====================================
        roads = roads[roads['Status'] != '4']
        print(f"After Status filter: {len(roads)} roads remaining")

        # =====================================
        # Filter by StreetWidth_Min
        # =====================================
        min_street_width = config.analysis_params['min_street_width']
        roads = roads[roads['StreetWidth_Min'] >= min_street_width]
        print(f"After StreetWidth_Min filter: {len(roads)} roads remaining")

        # =====================================
        # Filter by Elevated Railways
        # =====================================
        print("Filtering roads that intersect with elevated railways buffer...")
        # Load the elevated railways GeoJSON
        elevated_rails = gpd.read_file(os.path.join(config.input_dir, 'ElevatedRailways.geojson'))
        elevated_rails = elevated_rails.to_crs(config.crs)

        # Get the buffer distance from the configuration
        rail_buffer = config.analysis_params['rail_buffer']

        # Buffer the elevated railways geometries by the rail_buffer distance
        elevated_rails['buffered_geom'] = elevated_rails.geometry.buffer(rail_buffer)
        # Combine all buffered geometries into a single geometry
        buffered_union = elevated_rails['buffered_geom'].unary_union

        # Remove roads that intersect with the buffered elevated railways
        roads = roads[~roads.geometry.intersects(buffered_union)]
        print(f"After Elevated Railways filter: {len(roads)} roads remaining")

        # =====================================
        # Filter by HVI intersection
        # =====================================
        vuln_path = os.path.join(config.input_dir, 'HeatVulnerabilityIndex.geojson')
        if os.path.exists(vuln_path):
            print("Loading vulnerability data for filtering...")
            vulnerability = gpd.read_file(vuln_path)

            # Convert HVI to numeric, handling string format
            vulnerability['hvi_numeric'] = (
                vulnerability['HVI']
                .astype(str)
                .str.strip()
                .replace({'': None, 'null': None, 'nan': None})
                .pipe(pd.to_numeric, errors='coerce')
            )

            # Filter vulnerability polygons using config parameter
            min_vuln = config.analysis_params['min_vulnerability']
            high_vuln = vulnerability[vulnerability['hvi_numeric'] >= min_vuln]

            if len(high_vuln) > 0:
                # Ensure same CRS
                if high_vuln.crs != roads.crs:
                    high_vuln = high_vuln.to_crs(roads.crs)

                print(f"Filtering for roads intersecting with HVI >= {min_vuln} areas...")
                print(f"Number of high vulnerability polygons: {len(high_vuln)}")

                # Create spatial index for efficiency
                spatial_index = high_vuln.sindex

                # Function to check if a road intersects with any high vulnerability polygon
                def intersects_high_vuln(road_geom):
                    possible_matches_idx = list(spatial_index.intersection(road_geom.bounds))
                    if not possible_matches_idx:
                        return False
                    possible_matches = high_vuln.iloc[possible_matches_idx]
                    return any(possible_matches.intersects(road_geom))

                # Apply the HVI filter
                roads = roads[roads.geometry.apply(intersects_high_vuln)]
                print(f"After HVI intersection filter: {len(roads)} roads remaining")
            else:
                print(f"Warning: No areas with HVI >= {min_vuln} found. Skipping HVI filter.")
        else:
            print(f"Warning: Vulnerability file not found at {vuln_path}. Skipping HVI filter.")

        # =====================================
        # Load and merge pedestrian demand data
        # =====================================
        print("\nLoading pedestrian demand data...")
        ped_demand = pd.read_csv(os.path.join(config.input_dir, 'Pedestrian_Mobility_Plan_Pedestrian_Demand_20250117.csv'))
        ped_demand['segmentid'] = ped_demand['segmentid'].astype(str).str.strip().str.zfill(7)

        # Remove duplicates so that each segmentid appears only once
        ped_demand = ped_demand.drop_duplicates(subset='segmentid')

        # Populate PedRank and invert the values
        ped_demand['PedRank'] = ped_demand['Rank']
        ped_demand['PedRankInverted'] = 6 - ped_demand['Rank']

        # Then merge using an inner join
        roads = roads.merge(
            ped_demand[['segmentid', 'PedRank', 'PedRankInverted']], 
            on='segmentid', 
            how='inner'
        )

        # Create PedIndex via min–max normalization on PedRankInverted
        min_inverted = roads['PedRankInverted'].min()
        max_inverted = roads['PedRankInverted'].max()
        if max_inverted - min_inverted != 0:
            roads['PedIndex'] = (roads['PedRankInverted'] - min_inverted) / (max_inverted - min_inverted)
        else:
            roads['PedIndex'] = 0

        print(f"Segments with PedRank data: {roads['PedRankInverted'].notna().sum()}")
        print(f"PedIndex range: {roads['PedIndex'].min():.3f} to {roads['PedIndex'].max():.3f}")

        # =====================================
        # Process NYC Sidewalk data (raster-based approach)
        # =====================================
        roads = process_sidewalk_data(roads, config)

        # =====================================
        # Process Road metrics
        # =====================================
        roads = process_road_width(roads)

        # =====================================
        # Return the final dataframe with all features.
        # =====================================
        return roads

    except Exception as e:
        print(f"Error in road network preprocessing: {str(e)}")
        raise


def process_sidewalk_data(df, config):
    """
    Process NYC Sidewalk data using a raster-based approach:
      - Rasterize the sidewalk GeoJSON to a binary raster
      - For each road segment, buffer by 0.75 * StreetWidth_Min
      - Use raster masking to sum sidewalk pixels within the buffer
    
    Parameters:
      df (GeoDataFrame): Roads data with StreetWidth_Min field
      config: Configuration object (provides input directory, CRS, etc.)
      
    Returns:
      df (GeoDataFrame): Updated with sidewalk_area and SidewalkIndex fields
    """
    import os
    import numpy as np
    import geopandas as gpd
    from rasterio import features, mask
    from rasterio.transform import from_origin
    from rasterio.io import MemoryFile

    print("\nProcessing NYC Sidewalk data (raster-based approach)...")
    sidewalk_path = os.path.join(config.input_dir, 'NYC_Sidewalk.geojson')
    sidewalk = gpd.read_file(sidewalk_path)
    if sidewalk.crs != df.crs:
        sidewalk = sidewalk.to_crs(df.crs)
    
    # Determine the bounds for rasterization based on the roads dataset, adding a margin
    xmin, ymin, xmax, ymax = df.total_bounds
    margin = 100  # feet margin to cover buffers extending beyond road bounds
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin

    # Define a suitable resolution (e.g., 5 feet per pixel)
    resolution = 5.0
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    transform = from_origin(xmin, ymax, resolution, resolution)
    pixel_area = resolution * resolution

    # Rasterize sidewalk polygons (1 = sidewalk, 0 = no sidewalk)
    shapes = ((geom, 1) for geom in sidewalk.geometry)
    sidewalk_raster = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # Open an in-memory raster dataset for masking
    with MemoryFile() as memfile:
        with memfile.open(driver='GTiff',
                          height=height,
                          width=width,
                          count=1,
                          dtype=sidewalk_raster.dtype,
                          transform=transform) as dataset:
            dataset.write(sidewalk_raster, 1)
            
            # Initialize list for results
            sidewalk_areas = []
            
            # Process each road segment
            for idx, row in df.iterrows():
                road_geom = row.geometry
                street_width = row.get('StreetWidth_Min', 0)
                buffer_dist = 0.75 * street_width
                buffered_geom = road_geom.buffer(buffer_dist)
                try:
                    # Mask the raster with the road buffer geometry
                    out_image, out_transform = mask.mask(dataset, [buffered_geom], crop=True)
                    # out_image has shape (1, h, w); count sidewalk pixels (value==1)
                    sidewalk_pixels = (out_image[0] == 1).sum()
                    area = sidewalk_pixels * pixel_area
                except Exception as e:
                    print(f"Error masking for road idx {idx}: {e}")
                    area = 0
                sidewalk_areas.append(area)
    
    # Add the sidewalk area field to the dataframe
    df['sidewalk_area'] = sidewalk_areas

    print(f"Calculated sidewalk_area for {len(df)} segments.")
    
    # Normalize sidewalk_area to [0,1] and then convert to a V-shaped index.
    min_sidewalk = df['sidewalk_area'].min()
    max_sidewalk = df['sidewalk_area'].max()
    if max_sidewalk - min_sidewalk != 0:
        normalized = (df['sidewalk_area'] - min_sidewalk) / (max_sidewalk - min_sidewalk)
        df['SidewalkIndex'] = 2 * abs(normalized - 0.5)
    else:
        df['SidewalkIndex'] = 0

    # Print summary statistics
    non_zero = df['sidewalk_area'][df['sidewalk_area'] > 0]
    print("\nSidewalk Area Summary Statistics:")
    print(f"  Total segments with sidewalks: {len(non_zero)}")
    print(f"  Mean area: {df['sidewalk_area'].mean():.2f} sq ft")
    print(f"  Range: {df['sidewalk_area'].min():.2f} to {df['sidewalk_area'].max():.2f} sq ft")
    print(f"  Median: {df['sidewalk_area'].median():.2f} sq ft")
    print(f"  Segments with zero area: {len(df) - len(non_zero)}")
    
    return df

def process_road_width(df):
    """
    Normalize StreetWidth_Min to create a RoadWidthIndex.
    
    Parameters:
      df (DataFrame): Road network data containing StreetWidth_Min
      
    Returns:
      df (DataFrame): Updated with RoadWidthIndex column.
    """
    import numpy as np
    df = df.copy()
    
    # Create a mask for valid rows
    mask = (df['StreetWidth_Min'] > 0)
    valid_data = df[mask].copy()
    
    if len(valid_data) == 0:
        print("No valid segments for normalized road metrics.")
        df['RoadWidthIndex'] = np.nan
        return df

    # Normalize StreetWidth_Min to [0,1] to create the index
    min_width = valid_data['StreetWidth_Min'].min()
    max_width = valid_data['StreetWidth_Min'].max()
    valid_data['RoadWidthIndex'] = (valid_data['StreetWidth_Min'] - min_width) / (max_width - min_width)

    # Merge the computed index back into the original dataframe
    df.loc[mask, 'RoadWidthIndex'] = valid_data['RoadWidthIndex']
    
    # Print summary statistics
    print("\nRoadWidthIndex Summary Statistics (Valid Segments):")
    valid_rpi = df['RoadWidthIndex'].dropna() 
    print(f"  Mean: {valid_rpi.mean():.4f}")
    print(f"  Median: {valid_rpi.median():.4f}")
    print(f"  Range: {valid_rpi.min():.4f} to {valid_rpi.max():.4f}")
    
    return df


        # # =====================================
        # # Filter by Capital Project Exclusion Buffer
        # # =====================================
        # # Buffer the DOT Capital Projects datasets (lines and points) and exclude any road that intersects the buffered areas.
        # cap_proj_buffer = config.analysis_params.get('CapitalProjectExclusionBuffer', 10)
        # cap_proj_lines_path = os.path.join(config.input_dir, 'DOTCapitalProjects_Lines.geojson')
        # cap_proj_pts_path = os.path.join(config.input_dir, 'DOTCapitalProjects_Pts.geojson')

        # if os.path.exists(cap_proj_lines_path) or os.path.exists(cap_proj_pts_path):
        #     buffers = []  # List to hold buffered geometries

        #     if os.path.exists(cap_proj_lines_path):
        #         print("Loading DOT Capital Projects Lines for exclusion...")
        #         cap_proj_lines = gpd.read_file(cap_proj_lines_path)
        #         cap_proj_lines = cap_proj_lines.to_crs(roads.crs)
        #         # Buffer the lines dataset by the specified distance (in feet)
        #         cap_proj_lines['geometry'] = cap_proj_lines.geometry.buffer(cap_proj_buffer)
        #         buffers.append(cap_proj_lines)

        #     if os.path.exists(cap_proj_pts_path):
        #         print("Loading DOT Capital Projects Points for exclusion...")
        #         cap_proj_pts = gpd.read_file(cap_proj_pts_path)
        #         cap_proj_pts = cap_proj_pts.to_crs(roads.crs)
        #         # Buffer the points dataset by the specified distance (in feet)
        #         cap_proj_pts['geometry'] = cap_proj_pts.geometry.buffer(cap_proj_buffer)
        #         buffers.append(cap_proj_pts)

        #     if buffers:
        #         # Combine the buffered geometries into one GeoDataFrame
        #         cap_proj_buffers = gpd.GeoDataFrame(
        #             pd.concat(buffers, ignore_index=True),
        #             crs=roads.crs
        #         )
        #         spatial_index = cap_proj_buffers.sindex

        #         # Function to check if a road intersects any buffered capital project feature
        #         def intersects_cap_proj(road_geom):
        #             possible_matches_idx = list(spatial_index.intersection(road_geom.bounds))
        #             if not possible_matches_idx:
        #                 return False
        #             possible_matches = cap_proj_buffers.iloc[possible_matches_idx]
        #             return any(possible_matches.intersects(road_geom))

        #         # Exclude roads that intersect with any buffered capital project feature
        #         roads = roads[~roads.geometry.apply(intersects_cap_proj)]
        #         print(f"After Capital Project Exclusion filter: {len(roads)} roads remaining")
        # else:
        #     print("Capital Project files not found. Skipping Capital Project Exclusion filter.")

        # =====================================
        # FOZ Intersection Filter (commented out)
        # =====================================
        # foz_path = os.path.join(config.input_dir, 'FOZ_NYC_Merged.geojson')
        # if os.path.exists(foz_path):
        #     print("Loading FOZ data for filtering...")
        #     foz = gpd.read_file(foz_path)
        #
        #     if len(foz) > 0:
        #         if foz.crs != roads.crs:
        #             foz = foz.to_crs(roads.crs)
        #
        #         print(f"Filtering for roads intersecting with FOZ areas...")
        #         print(f"Number of FOZ polygons: {len(foz)}")
        #
        #         foz_spatial_index = foz.sindex
        #
        #         def intersects_foz(road_geom):
        #             possible_matches_idx = list(foz_spatial_index.intersection(road_geom.bounds))
        #             if not possible_matches_idx:
        #                 return False
        #             possible_matches = foz.iloc[possible_matches_idx]
        #             return any(possible_matches.intersects(road_geom))
        #
        #         roads = roads[roads.geometry.apply(intersects_foz)]
        #         print(f"After FOZ intersection filter: {len(roads)} roads remaining")
        #     else:
        #         print("Warning: No FOZ areas found. Skipping FOZ filter.")
        # else:
        #     print("Warning: FOZ file not found. Skipping FOZ filter.")

        # =====================================
        # Persistent Poverty Intersection Filter (commented out)
        # =====================================
        # poverty_path = os.path.join(config.input_dir, 'nyc_persistent_poverty.geojson')
        # if os.path.exists(poverty_path):
        #     print("Loading persistent poverty data for filtering...")
        #     poverty = gpd.read_file(poverty_path)
        #
        #     if len(poverty) > 0:
        #         if poverty.crs != roads.crs:
        #             poverty = poverty.to_crs(roads.crs)
        #
        #         print(f"Filtering for roads intersecting with persistent poverty areas...")
        #         print(f"Number of poverty area polygons: {len(poverty)}")
        #
        #         poverty_spatial_index = poverty.sindex
        #
        #         def intersects_poverty(road_geom):
        #             possible_matches_idx = list(poverty_spatial_index.intersection(road_geom.bounds))
        #             if not possible_matches_idx:
        #                 return False
        #             possible_matches = poverty.iloc[possible_matches_idx]
        #             return any(possible_matches.intersects(road_geom))
        #
        #         roads = roads[roads.geometry.apply(intersects_poverty)]
        #         print(f"After persistent poverty intersection filter: {len(roads)} roads remaining")
        #     else:
        #         print("Warning: No persistent poverty areas found. Skipping poverty filter.")
        # else:
        #     print("Warning: Persistent poverty file not found. Skipping poverty filter.")