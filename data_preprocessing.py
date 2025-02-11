import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

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

        # =====================================
        # Final assignment for compatibility with following steps
        # =====================================
        roads_result = roads

        # =====================================
        # Load and merge pedestrian demand data
        # =====================================
        print("\nLoading pedestrian demand data...")
        ped_demand = pd.read_csv(
            os.path.join(config.input_dir, 'Pedestrian_Mobility_Plan_Pedestrian_Demand_20250117.csv')
        )

        # Ensure segmentid is in the same format for joining
        ped_demand['segmentid'] = ped_demand['segmentid'].astype(str).str.strip().str.zfill(7)

        # Merge pedestrian demand data
        result_df = roads_result.merge(
            ped_demand[['segmentid', 'Rank']], 
            on='segmentid', 
            how='left'
        )

        # Rename Rank to PedRank
        result_df = result_df.rename(columns={'Rank': 'PedRank'})

        # Create PedIndex (min-max normalization)
        min_rank = result_df['PedRank'].min()
        max_rank = result_df['PedRank'].max()
        result_df['PedIndex'] = (result_df['PedRank'] - min_rank) / (max_rank - min_rank)

        print(f"Segments with PedRank data: {result_df['PedRank'].notna().sum()}")
        print(f"PedIndex range: {result_df['PedIndex'].min():.3f} to {result_df['PedIndex'].max():.3f}")

        return result_df

    except Exception as e:
        print(f"Error in road network preprocessing: {str(e)}")
        raise


def process_bike_lanes(roads_gdf, buffer_distance=50):
    """
    Process bike lanes by calculating the length of nearby bike lanes for each road segment,
    then derive 'bike_ln_per_mile' based on road length.

    Args:
        roads_gdf (GeoDataFrame): GeoDataFrame of road segments, *must* have a 'segment_length' column in feet.
        buffer_distance (float): Distance in feet to buffer road segments (default: 10)

    Returns:
        roads_gdf (GeoDataFrame): Updated GeoDataFrame with 'bike_length' and 'bike_ln_per_mile' columns.
    """
    import geopandas as gpd
    import numpy as np
    from shapely.ops import unary_union

    try:
        print("\nProcessing bike lane data...")

        # Read the bike lanes GeoJSON
        bike_lane_gdf = gpd.read_file('input/bikelanes.geojson')
        print(f"Loaded {len(bike_lane_gdf)} bike lane features")

        print("\nBike lanes CRS:", bike_lane_gdf.crs)
        print("Roads CRS:", roads_gdf.crs)
        print("\nBike lanes geometry types:", bike_lane_gdf.geometry.geom_type.value_counts())

        # 1) Ensure same CRS
        if roads_gdf.crs != bike_lane_gdf.crs:
            print("Reprojecting bike lanes to match roads CRS...")
            bike_lane_gdf = bike_lane_gdf.to_crs(roads_gdf.crs)

        # Validate geometries
        bike_lane_gdf.geometry = bike_lane_gdf.geometry.make_valid()
        roads_gdf.geometry = roads_gdf.geometry.make_valid()

        # 2) Create union of bike lanes for intersection
        print("Creating bike lanes union...")
        bike_union = unary_union(bike_lane_gdf.geometry)
        print(f"Union geometry type: {bike_union.geom_type}")

        # 3) Calculate intersection lengths with buffering
        print(f"Calculating intersection lengths (using {buffer_distance} ft buffer)...")
        bike_lengths = []
        total_intersections = 0

        for idx, row in roads_gdf.iterrows():
            road_geom = row.geometry
            if road_geom is None or road_geom.is_empty:
                bike_lengths.append(0.0)
                continue

            try:
                buffered_road = road_geom.buffer(buffer_distance)
                bike_lanes_in_buffer = bike_union.intersection(buffered_road)

                if not bike_lanes_in_buffer.is_empty:
                    total_length = 0.0
                    # If multiple line segments
                    if bike_lanes_in_buffer.geom_type == 'MultiLineString':
                        total_length = sum(line.length for line in bike_lanes_in_buffer.geoms)
                    elif bike_lanes_in_buffer.geom_type == 'LineString':
                        total_length = bike_lanes_in_buffer.length

                    if total_length > 0:
                        total_intersections += 1
                        bike_lengths.append(total_length)
                    else:
                        bike_lengths.append(0.0)
                else:
                    bike_lengths.append(0.0)
            except Exception as e:
                print(f"Error calculating intersection for segment {idx}: {str(e)}")
                bike_lengths.append(0.0)

        # 4) Add raw bike lane length to dataframe
        roads_gdf['bike_length'] = bike_lengths

        # 5) Calculate the length of bike lanes per mile of road
        if 'segment_length' not in roads_gdf.columns:
            roads_gdf['segment_length'] = roads_gdf.geometry.length

        # Avoid division by zero
        roads_gdf['bike_ln_per_mile'] = 0.0
        non_zero_mask = roads_gdf['segment_length'] > 0
        roads_gdf.loc[non_zero_mask, 'bike_ln_per_mile'] = (
            roads_gdf.loc[non_zero_mask, 'bike_length'] / roads_gdf.loc[non_zero_mask, 'segment_length']
        ) * 5280

        # Debug information
        print("\nBike lane statistics:")
        print(f"Total bike length: {sum(bike_lengths):.1f}")
        print(f"Max bike length: {max(bike_lengths):.1f}")
        print(f"Number of non-zero bike lengths: {sum(1 for x in bike_lengths if x > 0)}")
        print(f"\nTotal bike lane intersections found: {total_intersections}")
        print(f"Average nearby bike lane length: {np.mean(bike_lengths):.1f}")
        print(f"Segments with any nearby bike lanes: {sum(1 for x in bike_lengths if x > 0)}")
        print(f"Maximum nearby bike lane length: {max(bike_lengths):.1f}")

        print("\nBike lane per mile statistics:")
        print(f"Mean bike lanes per mile: {roads_gdf['bike_ln_per_mile'].mean():.2f}")
        print(f"Max bike lanes per mile: {roads_gdf['bike_ln_per_mile'].max():.2f}")

        return roads_gdf

    except Exception as e:
        print(f"Error in bike lane processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
