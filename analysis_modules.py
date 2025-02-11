import os
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from tqdm.auto import tqdm

import rasterio
import rasterio.features
from rasterio.windows import Window
from shapely.ops import linemerge, unary_union
from shapely.validation import make_valid
from shapely import make_valid as shapely_make_valid  # rename if needed

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import folium

import zipfile
import io
import warnings
from pathlib import Path
from data_preprocessing import process_bike_lanes

# ------------------------------------
# Data Processing Classes
# ------------------------------------
import os
import pandas as pd
import geopandas as gpd
import numpy as np

from shapely.ops import unary_union
from shapely.validation import make_valid
from tqdm.auto import tqdm


class DataProcessors:
    def __init__(self):
        pass

    @staticmethod
    def normalize_to_index(series, attribute_type):
        """
        Normalize a series to an index between 0 and 1 based on attribute type.
        Uses 5th and 95th percentiles to handle outliers.
        - 'heat', 'canopy' => reversed normalization
        - 'pavement' => V-shaped normalization where extremes are good
        """
        try:
            if series.isna().all():
                return pd.Series(0.5, index=series.index)

            numeric_series = pd.to_numeric(series, errors='coerce')
            valid_data = numeric_series.dropna()
            if valid_data.nunique() <= 1:
                return pd.Series(0.5, index=series.index)

            # Calculate percentiles for outlier handling
            p05 = valid_data.quantile(0.05)
            p95 = valid_data.quantile(0.95)
            
            if attribute_type == 'pavement':
                # V-shaped normalization for pavement
                # Assuming rating scale is 0-10
                middle_point = 5.0
                distances = abs(valid_data - middle_point)
                
                # Handle outliers in distances
                dist_p05 = distances.quantile(0.05)
                dist_p95 = distances.quantile(0.95)
                
                if dist_p95 == dist_p05:
                    normalized = pd.Series(0.5, index=valid_data.index)
                else:
                    # Clip distances to remove outliers
                    clipped_distances = distances.clip(dist_p05, dist_p95)
                    # Normalize clipped distances
                    normalized = (clipped_distances - dist_p05) / (dist_p95 - dist_p05)
                
            else:
                # Standard normalization with outlier handling
                if p95 == p05:
                    normalized = pd.Series(0.5, index=valid_data.index)
                else:
                    # Clip values to remove outliers
                    clipped_data = valid_data.clip(p05, p95)
                    # Normalize clipped values
                    normalized = (clipped_data - p05) / (p95 - p05)

                # Reverse for 'heat', 'canopy'
                if attribute_type in ['heat', 'canopy']:
                    normalized = 1 - normalized

            # Ensure values are within [0, 1]
            normalized = normalized.clip(0, 1)
            
            result = pd.Series(normalized, index=valid_data.index)
            # Fill any missing or NaN with 0.5
            result = result.reindex(series.index).fillna(0.5)
            return result

        except Exception as e:
            print(f"Error in index normalization: {str(e)}")
            return pd.Series(0.5, index=series.index)

    @staticmethod
    def _calculate_vulnerability_detailed(road, vulnerability, spatial_index):
        """Helper function for vulnerability calculation."""
        try:
            possible_matches_idx = list(spatial_index.intersection(road.geometry.bounds))
            if not possible_matches_idx:
                return {'hvi': 0.0, 'intersections': 0, 'length': 0}

            possible_matches = vulnerability.iloc[possible_matches_idx]
            precise_matches = possible_matches[possible_matches.intersects(road.geometry)]
            if len(precise_matches) == 0:
                return {'hvi': 0.0, 'intersections': 0, 'length': 0}

            weighted_hvi = 0
            total_length = 0
            intersection_count = 0

            for _, vuln in precise_matches.iterrows():
                intersection = road.geometry.intersection(vuln.geometry)
                if not intersection.is_empty:
                    length = intersection.length
                    total_length += length
                    weighted_hvi += length * vuln['hvi_raw']
                    intersection_count += 1

            final_hvi = weighted_hvi / total_length if total_length > 0 else 0.0
            return {'hvi': final_hvi, 'intersections': intersection_count, 'length': total_length}

        except Exception as e:
            print(f"Error in vulnerability calculation: {str(e)}")
            return {'hvi': 0.0, 'intersections': 0, 'length': 0}

    @staticmethod
    def process_bike_lanes(roads_gdf, buffer_distance=50):
        """
        Process bike lanes by calculating the length of nearby bike lanes for each road segment,
        then derive 'bike_ln_per_mile' based on road length.

        Args:
            roads_gdf (GeoDataFrame): GeoDataFrame of road segments (in a feet-based CRS).
            buffer_distance (float): Buffer distance in feet (default: 10).

        Returns:
            roads_gdf (GeoDataFrame): Updated with 'bike_length' & 'bike_ln_per_mile' columns.
        """
        try:
            print("\nProcessing bike lane data...")

            # Read the bike lanes GeoJSON
            bike_lane_gdf = gpd.read_file(os.path.join('input', 'bikelanes.geojson'))
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

            # Ensure segment_length is present (in feet)
            if 'segment_length' not in roads_gdf.columns:
                roads_gdf['segment_length'] = roads_gdf.geometry.length

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

            roads_gdf['bike_length'] = bike_lengths

            # 4) Calculate bike lane length per mile of road
            roads_gdf['bike_ln_per_mile'] = 0.0
            non_zero_mask = roads_gdf['segment_length'] > 0
            roads_gdf.loc[non_zero_mask, 'bike_ln_per_mile'] = (
                roads_gdf.loc[non_zero_mask, 'bike_length'] / roads_gdf.loc[non_zero_mask, 'segment_length']
            ) * 5280

            # Debug info
            print("\nBike lane statistics:")
            print(f"  Total bike length (feet): {sum(bike_lengths):.1f}")
            print(f"  Max bike length (feet): {max(bike_lengths):.1f}")
            print(f"  Number of road segments with nearby bike lanes: "
                  f"{sum(1 for x in bike_lengths if x > 0)}")
            print(f"  Total intersections found: {total_intersections}")
            print(f"  Average bike length (feet): {np.mean(bike_lengths):.1f}")

            print("\nBike lanes per mile statistics:")
            print(f"  Mean (bike_ln_per_mile): {roads_gdf['bike_ln_per_mile'].mean():.2f}")
            print(f"  Max  (bike_ln_per_mile): {roads_gdf['bike_ln_per_mile'].max():.2f}")

            return roads_gdf

        except Exception as e:
            print(f"Error in bike lane processing: {str(e)}")
            raise

    @classmethod
    def batch_process_all(cls, roads, input_dir):
        """
        Process pavement, vulnerability, bike lanes, etc.
        """
        try:
            print("Starting batch processing...")

            # ------------------------------------------------------
            # 1) Load + Merge Pavement Data
            # ------------------------------------------------------
            pavement_path = os.path.join(input_dir, 'Street_Pavement_Rating.geojson')
            if os.path.exists(pavement_path):
                pavement = gpd.read_file(pavement_path)

                # Convert rating to numeric; rename if needed
                pavement['pav_rate'] = pd.to_numeric(pavement['manualrati'], errors='coerce')

                # Ensure 'segmentid' is present and zero-padded
                pavement['segmentid'] = (
                    pavement['segmentid']
                    .astype(str)
                    .str.strip()
                    .str.zfill(7)
                )

                # Remove duplicates so we don't merge duplicates on 'segmentid'
                pavement = pavement.drop_duplicates('segmentid', keep='first')

                # Merge into roads
                if 'segmentid' in roads.columns:
                    roads = roads.merge(
                        pavement[['segmentid', 'pav_rate']],
                        on='segmentid',
                        how='left'
                    )
                else:
                    print("Warning: 'segmentid' not found in roads, cannot merge pavement data.")
            else:
                print(f"Warning: Pavement file not found at: {pavement_path} - skipping merge.")

            # ------------------------------------------------------
            # 2) Pavement Index
            # ------------------------------------------------------
            if 'pav_rate' in roads.columns:
                roads['pave_indx'] = cls.normalize_to_index(roads['pav_rate'], 'pavement')
            else:
                print("Warning: 'pav_rate' column not found in roads. Using 0.5 as default.")
                roads['pave_indx'] = 0.5

            # ------------------------------------------------------
            # 3) Process Bike Lanes
            # ------------------------------------------------------
            roads = cls.process_bike_lanes(roads, buffer_distance=50)

            # Create a 0-1 BikeLnIndx by min-max normalizing 'bike_ln_per_mile'
            if 'bike_ln_per_mile' in roads.columns:
                min_val = roads['bike_ln_per_mile'].min()
                max_val = roads['bike_ln_per_mile'].max()
                if max_val > min_val:
                    roads['BikeLnIndx'] = (
                        (roads['bike_ln_per_mile'] - min_val) / (max_val - min_val)
                    )
                else:
                    # All segments have same bike_ln_per_mile
                    roads['BikeLnIndx'] = 0.0
            else:
                roads['BikeLnIndx'] = 0.0

            # ------------------------------------------------------
            # 4) Vulnerability Overlay
            # ------------------------------------------------------
            vuln_path = os.path.join(input_dir, 'HeatVulnerabilityIndex.geojson')
            if os.path.exists(vuln_path):
                vulnerability = gpd.read_file(vuln_path)

                # Handle string format HVI values
                vulnerability['hvi_raw'] = (
                    vulnerability['HVI']
                    .astype(str)
                    .str.strip()
                    .replace({'': None, 'null': None, 'nan': None})
                    .pipe(pd.to_numeric, errors='coerce')
                )

                # Drop rows where HVI couldn't be converted
                vulnerability = vulnerability.dropna(subset=['hvi_raw'])
                if len(vulnerability) == 0:
                    print("Warning: No valid HVI values found. Using 0.5 for vuln_indx.")
                    roads['vuln_indx'] = 0.5
                    return roads

                # Print raw HVI statistics before processing
                print("\nRaw HVI statistics:")
                print(f"Mean: {vulnerability['hvi_raw'].mean():.3f}")
                print(f"Std: {vulnerability['hvi_raw'].std():.3f}")
                print(f"Min: {vulnerability['hvi_raw'].min():.3f}")
                print(f"Max: {vulnerability['hvi_raw'].max():.3f}")

                # Optional: Remove extreme outliers using IQR method
                Q1 = vulnerability['hvi_raw'].quantile(0.25)
                Q3 = vulnerability['hvi_raw'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                vulnerability = vulnerability[
                    (vulnerability['hvi_raw'] >= lower_bound) & 
                    (vulnerability['hvi_raw'] <= upper_bound)
                ]

                if roads.crs and vulnerability.crs and roads.crs != vulnerability.crs:
                    vulnerability = vulnerability.to_crs(roads.crs)

                print("Calculating vulnerability data...")
                print(f"Number of valid vulnerability polygons: {len(vulnerability)}")
                print(f"HVI value range: {vulnerability['hvi_raw'].min():.2f} - {vulnerability['hvi_raw'].max():.2f}")

                spatial_index = vulnerability.sindex
                vuln_results = []
                for idx, road in tqdm(roads.iterrows(), total=len(roads), desc="Calculating vulnerability"):
                    result = cls._calculate_vulnerability_detailed(road, vulnerability, spatial_index)
                    vuln_results.append(result)

                roads['hvi_raw'] = [res['hvi'] for res in vuln_results]
                roads['intersections'] = [res['intersections'] for res in vuln_results]
                roads['vuln_length'] = [res['length'] for res in vuln_results]

                # Robust normalization for vulnerability index
                valid_hvi = pd.Series(roads['hvi_raw'])
                valid_hvi = valid_hvi.replace([np.inf, -np.inf], np.nan)

                if valid_hvi.isna().all() or valid_hvi.nunique() <= 1:
                    print("Warning: No valid HVI values for normalization. Using 0.5 for vuln_indx.")
                    roads['vuln_indx'] = 0.5
                else:
                    min_val = valid_hvi.min()
                    max_val = valid_hvi.max()

                    # Perform normalization
                    normalized = (valid_hvi - min_val) / (max_val - min_val)
                    # For heat vulnerability, higher values mean higher priority
                    roads['vuln_indx'] = normalized.clip(0, 1)

                # Validation and debug information
                print("\nVulnerability processing complete:")
                print(f"  Roads with valid HVI values: {(roads['hvi_raw'] > 0).sum()}")
                print(f"  Average HVI raw: {roads['hvi_raw'].mean():.3f}")
                print(f"  HVI raw range: {roads['hvi_raw'].min():.3f} - {roads['hvi_raw'].max():.3f}")
                print(f"  Final vulnerability index range: {roads['vuln_indx'].min():.3f} - {roads['vuln_indx'].max():.3f}")

                # Additional validation
                if not roads['vuln_indx'].between(0, 1).all():
                    print("Warning: Vulnerability index contains values outside 0-1 range!")
                    roads['vuln_indx'] = roads['vuln_indx'].clip(0, 1)
            else:
                print(f"Warning: No vulnerability file at {vuln_path}; using 0.5 for vuln_indx.")
                roads['vuln_indx'] = 0.5


            # ------------------------------------------------------
            # 5) Debug Stats
            # ------------------------------------------------------
            if 'pave_indx' in roads.columns:
                print(f"\nPavement index statistics:")
                print(f"  Mean pavement index: {roads['pave_indx'].mean():.3f}")
                print(f"  Range: {roads['pave_indx'].min():.3f} - {roads['pave_indx'].max():.3f}")

            if 'vuln_indx' in roads.columns:
                print(f"\nVulnerability index statistics:")
                print(f"  Mean vulnerability index: {roads['vuln_indx'].mean():.3f}")
                print(f"  Range: {roads['vuln_indx'].min():.3f} - {roads['vuln_indx'].max():.3f}")

            print(f"\nBike lane statistics:")
            if 'bike_length' in roads.columns:
                print(f"  Segments with nearby bike lanes: {(roads['bike_length'] > 0).sum()}")
                print(f"  Total bike lane length (feet): {roads['bike_length'].sum():.1f}")
                print(f"  Mean raw bike length (feet): {roads['bike_length'].mean():.1f}")
                print(f"  Max raw bike length (feet): {roads['bike_length'].max():.1f}")

            if 'bike_ln_per_mile' in roads.columns:
                print(f"  Mean bike_ln_per_mile: {roads['bike_ln_per_mile'].mean():.2f}")
                print(f"  Max  bike_ln_per_mile: {roads['bike_ln_per_mile'].max():.2f}")

            print(f"  Mean BikeLnIndx: {roads['BikeLnIndx'].mean():.3f}")
            print(f"  Max  BikeLnIndx: {roads['BikeLnIndx'].max():.3f}")

            return roads

        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            raise

# ------------------------------------
# Raster Processing Classes
# ------------------------------------
class OptimizedRasterProcessing:
    """CPU-only raster processing."""

    def __init__(self):
        pass

    def optimize_temperature_processing(self, roads, raster_path):
        """Process temperature data for each segment geometry."""
        import logging
        import numpy as np
        import pandas as pd
        import rasterio
        from tqdm.auto import tqdm

        try:
            with rasterio.open(raster_path) as src:
                logging.info("\nRaster properties:")
                logging.info(f"CRS: {src.crs}")
                logging.info(f"Transform: {src.transform}")

                processing_roads = roads.copy()
                if processing_roads.crs != src.crs:
                    processing_roads = processing_roads.to_crs(src.crs)

                raster_data = src.read(1)
                results = []
                temp_stats = []
                pad_size = 3
                batch_size = 100

                for i in tqdm(range(0, len(processing_roads), batch_size)):
                    batch = processing_roads.iloc[i : i + batch_size]
                    batch_results = []

                    for idx, row in batch.iterrows():
                        try:
                            bounds = row.geometry.bounds
                            minx, miny, maxx, maxy = bounds
                            col_start = int((minx - src.transform.c) / src.transform.a)
                            row_start = int((maxy - src.transform.f) / src.transform.e)
                            col_end = int((maxx - src.transform.c) / src.transform.a) + 1
                            row_end = int((miny - src.transform.f) / src.transform.e) + 1

                            col_start = max(0, col_start - pad_size)
                            row_start = max(0, row_start - pad_size)
                            col_end = min(src.width, col_end + pad_size)
                            row_end = min(src.height, row_end + pad_size)

                            width = col_end - col_start
                            height = row_end - row_start

                            if width > 0 and height > 0:
                                from rasterio.features import rasterize
                                window_transform = rasterio.transform.from_origin(
                                    src.bounds.left + col_start * src.transform.a,
                                    src.bounds.top + row_start * src.transform.e,
                                    src.transform.a,
                                    src.transform.e
                                )
                                data = raster_data[row_start:row_end, col_start:col_end]

                                if data.size > 0:
                                    try:
                                        mask = rasterize(
                                            [(row.geometry, 1)],
                                            out_shape=(height, width),
                                            transform=window_transform,
                                            all_touched=True,
                                            dtype=rasterio.uint8
                                        )
                                        if np.sum(mask) > 0:
                                            masked_data = data[mask == 1]
                                            if masked_data.size > 0:
                                                valid_data = masked_data[
                                                    (masked_data >= 200) &
                                                    (masked_data <= 400) &
                                                    ~np.isnan(masked_data)
                                                ]
                                                if valid_data.size > 0:
                                                    temp_value = float(np.mean(valid_data))
                                                    if 200 <= temp_value <= 400:
                                                        temp_stats.append(temp_value)
                                                        batch_results.append(temp_value)
                                                        continue
                                    except Exception as mask_error:
                                        logging.warning(f"Mask creation failed for segment {idx}: {str(mask_error)}")

                            # fallback approach
                            if data.size > 0:
                                valid_data = data[
                                    (data >= 200) &
                                    (data <= 400) &
                                    ~np.isnan(data)
                                ]
                                if valid_data.size > 0:
                                    temp_value = float(np.mean(valid_data))
                                    batch_results.append(temp_value)
                                    temp_stats.append(temp_value)
                                    continue

                            batch_results.append(None)

                        except Exception as e:
                            logging.warning(f"Error processing segment {idx}: {str(e)}")
                            batch_results.append(None)

                    results.extend(batch_results)

                if len(temp_stats) == 0:
                    raise ValueError("No valid temperature values extracted")

                roads['heat_mean'] = pd.Series(results)
                missing_count = roads['heat_mean'].isna().sum()
                if missing_count > 0:
                    median_temp = np.median(temp_stats)
                    roads['heat_mean'] = roads['heat_mean'].fillna(median_temp)
                    logging.info(f"Filled {missing_count} missing values with median: {median_temp:.2f}K")

                # Convert K -> F
                roads['heat_mean'] = (roads['heat_mean'] - 273.15) * 9 / 5 + 32
                logging.info("\nTemperature processing results:")
                logging.info(f"Total segments: {len(roads)}")
                logging.info(f"Segments with direct readings: {len(temp_stats)}")
                logging.info(f"Temperature range (F): [{min(roads['heat_mean']):.1f}, {max(roads['heat_mean']):.1f}]")

                return roads

        except Exception as e:
            logging.error(f"Temperature processing error: {str(e)}")
            raise

    def optimize_tree_canopy_processing(self, roads, raster_path):
        """
        Process tree canopy with improved raster alignment.
        Assumes both street widths and raster data are in feet.
        """
        import logging
        import rasterio
        from rasterio.windows import Window
        from rasterio.features import rasterize
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm

        try:
            if not os.path.exists(raster_path):
                raise FileNotFoundError(f"Tree canopy raster not found at: {raster_path}")

            with rasterio.open(raster_path) as src:
                print("\nTree canopy raster properties:")
                print(f"CRS: {src.crs}")
                print(f"Resolution: {src.res}")

                processing_roads = roads.copy()
                if processing_roads.crs != src.crs:
                    processing_roads = processing_roads.to_crs(src.crs)

                results = []
                canopy_stats = []
                batch_size = 100

                for i in tqdm(range(0, len(processing_roads), batch_size), desc="Processing tree canopy"):
                    batch = processing_roads.iloc[i : i + batch_size]
                    batch_results = []

                    for idx, road in batch.iterrows():
                        try:
                            # Use half the street width directly in feet for buffer
                            buffer_width = float(road['StreetWidth_Min']) / 2 if pd.notna(road['StreetWidth_Min']) else 12.5  # default to 25/2 feet
                            buffer = road.geometry.buffer(buffer_width)
                            bounds = buffer.bounds

                            window_start_col = int((bounds[0] - src.transform.c) / src.transform.a)
                            window_start_row = int((bounds[3] - src.transform.f) / src.transform.e)
                            window_end_col = int((bounds[2] - src.transform.c) / src.transform.a) + 1
                            window_end_row = int((bounds[1] - src.transform.f) / src.transform.e) + 1

                            window_start_col = max(0, window_start_col)
                            window_start_row = max(0, window_start_row)
                            window_end_col = min(src.width, window_end_col)
                            window_end_row = min(src.height, window_end_row)

                            window_width = window_end_col - window_start_col
                            window_height = window_end_row - window_start_row

                            if window_width > 0 and window_height > 0:
                                window = Window(
                                    window_start_col,
                                    window_start_row,
                                    window_width,
                                    window_height
                                )

                                window_transform = rasterio.windows.transform(window, src.transform)
                                data = src.read(1, window=window)

                                if data is not None and data.size > 0:
                                    mask = rasterize(
                                        [(buffer, 1)],
                                        out_shape=(window_height, window_width),
                                        transform=window_transform,
                                        all_touched=True,
                                        dtype=rasterio.uint8
                                    )
                                    roadbed_pixels = np.sum(mask)
                                    if roadbed_pixels > 0:
                                        tree_pixels = np.sum((data == 1) & (mask == 1))
                                        percentage = (tree_pixels / roadbed_pixels) * 100
                                        if 0 <= percentage <= 100:
                                            batch_results.append(percentage)
                                            canopy_stats.append(percentage)
                                            continue

                            batch_results.append(0)

                        except Exception as e:
                            print(f"Error processing segment {idx}: {str(e)}")
                            batch_results.append(0)

                    results.extend(batch_results)

                roads['tree_pct'] = results
                from analysis_modules import DataProcessors
                roads['tree_indx'] = DataProcessors.normalize_to_index(roads['tree_pct'], 'percentage')
                # Reverse => less canopy => higher priority
                roads['tree_indx'] = 1 - roads['tree_indx']

                print(f"\nTree canopy statistics:")
                print(f"Mean coverage: {roads['tree_pct'].mean():.1f}%")
                print(f"Coverage range: {roads['tree_pct'].min():.1f}% - {roads['tree_pct'].max():.1f}%")
                print(f"Mean tree index: {roads['tree_indx'].mean():.3f}")
                print(f"Tree index range: {roads['tree_indx'].min():.3f} - {roads['tree_indx'].max():.3f}")

                return roads

        except Exception as e:
            print(f"Tree canopy processing error: {str(e)}")
            raise


# ------------------------------------
# Bus stops
# ------------------------------------
def process_bus_stops(roads, input_dir, crs):
    """
    Calculate bus stop density within 50ft buffer of each road segment,
    then create normalized BusDensInx based on stops per mile of segment length.
    Uses DataProcessors class for normalization with outlier handling.
    """
    import geopandas as gpd
    import pandas as pd
    import os
    
    try:
        print("Processing bus stop density...")
        bus_stops = gpd.read_file(os.path.join(input_dir, 'BusStops.geojson'))
        bus_stops = bus_stops.to_crs(crs)
        print(f"Loaded {len(bus_stops)} bus stops")
        
        # Create 50ft buffer around each road segment
        roads['buffer'] = roads.geometry.buffer(50)
        
        def count_bus_stops_in_buffer(buffer_geom):
            return sum(bus_stops.geometry.intersects(buffer_geom))
        
        print("Calculating bus stop counts within buffers...")
        roads['bus_stop_count'] = roads['buffer'].apply(count_bus_stops_in_buffer)
        
        # Calculate segment lengths in feet and bus stop density per mile
        roads['segment_length'] = roads.geometry.length
        roads['BusStpDens'] = (roads['bus_stop_count'] / roads['segment_length']) * 5280
        
        # Create normalized index using DataProcessors
        data_processor = DataProcessors()
        roads['BusDensInx'] = data_processor.normalize_to_index(roads['BusStpDens'], attribute_type='standard')
        
        # Clean up intermediate columns
        roads = roads.drop(columns=['buffer'])
        
        print(f"Bus stop density processing complete")
        print(f"Density range: {roads['BusStpDens'].min():.4f} to {roads['BusStpDens'].max():.4f} stops per mile")
        print(f"Normalized index range: {roads['BusDensInx'].min():.4f} to {roads['BusDensInx'].max():.4f}")
        print(f"Mean density: {roads['BusStpDens'].mean():.4f} stops per mile")
        print(f"Total bus stops counted: {roads['bus_stop_count'].sum()}")
        
        return roads
        
    except Exception as e:
        print(f"Error processing bus stops: {str(e)}")
        raise


# ------------------------------------
# Population Assessment
# ------------------------------------
def incorporate_population_density(roads, input_dir, buffer_ft=1000):
    """
    Load pre-generated NYC blocks with population data,
    then compute pop_density for each road and pop_dens_indx from it.
    """
    import os
    import geopandas as gpd
    from analysis_modules import DataProcessors

    # Load the pre-generated blocks with population data
    blocks_file = os.path.join(input_dir, "nyc_blocks_with_pop.geojson")
    if not os.path.exists(blocks_file):
        raise FileNotFoundError(f"Blocks file with population data not found at: {blocks_file}")

    print("Loading pre-generated blocks with population data...")
    blocks_gdf = gpd.read_file(blocks_file)

    # Ensure correct CRS
    if str(blocks_gdf.crs) != "EPSG:2263":
        blocks_gdf = blocks_gdf.to_crs("EPSG:2263")
    roads = roads.to_crs("EPSG:2263")

    # Estimate population density around each road
    roads = estimate_population_density(roads, blocks_gdf, buffer_ft=buffer_ft)

    # Create a normalized index from pop_density
    roads["pop_dens_indx"] = DataProcessors.normalize_to_index(
        roads["pop_density"],
        "popdensity"
    )

    return roads

def estimate_population_density(roads, blocks_gdf, buffer_ft=1000):
    """
    For each road segment, create a buffer around its geometry (the entire linestring),
    then estimate population from intersecting Census blocks. Then compute population
    density (people per square mile).
    """
    import geopandas as gpd

    # Constants for conversion (1 sq ft = 3.587e-8 sq miles)
    SQFT_TO_SQMILE = 3.587e-8

    if roads.crs is None or str(roads.crs) != "EPSG:2263":
        raise ValueError("Roads must be in EPSG:2263 for population density.")
    if blocks_gdf.crs is None or str(blocks_gdf.crs) != "EPSG:2263":
        raise ValueError("Blocks must be in EPSG:2263 for population density.")

    block_sindex = blocks_gdf.sindex
    pop_estimates = []

    for idx, row in roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            pop_estimates.append(0.0)
            continue

        # Buffer the entire linestring
        buffer_geom = geom.buffer(buffer_ft)

        # Create a GeoDataFrame from the buffer geometry
        buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_geom], crs=roads.crs)

        candidate_idxs = list(block_sindex.intersection(buffer_geom.bounds))
        candidate_blocks = blocks_gdf.iloc[candidate_idxs].copy()
        if candidate_blocks.empty:
            pop_estimates.append(0.0)
            continue

        intersection = gpd.overlay(
            candidate_blocks,
            buffer_gdf,
            how='intersection'
        )

        if intersection.empty:
            pop_estimates.append(0.0)
            continue

        intersection["intersect_area"] = intersection.geometry.area
        candidate_blocks["block_area"] = candidate_blocks.geometry.area

        intersection = intersection.merge(
            candidate_blocks[["GEOID20", "block_area", "P1_001N"]],
            on="GEOID20", how="left", suffixes=("", "_block")
        )

        intersection["proportion"] = intersection["intersect_area"] / intersection["block_area"]
        intersection["weighted_pop"] = intersection["proportion"] * intersection["P1_001N_block"]
        total_pop = intersection["weighted_pop"].sum()

        buffer_area = buffer_geom.area
        if buffer_area <= 0:
            pop_estimates.append(0.0)
            continue

        # Convert area to square miles and calculate density
        buffer_area_sqmile = buffer_area * SQFT_TO_SQMILE
        pop_density = total_pop / buffer_area_sqmile
        pop_estimates.append(pop_density)

    roads["pop_density"] = pop_estimates

    # Print some statistics about the population density
    print("\nPopulation density statistics (people/sq mile):")
    print(f"Mean: {roads['pop_density'].mean():,.1f}")
    print(f"Median: {roads['pop_density'].median():,.1f}")
    print(f"Range: {roads['pop_density'].min():,.1f} - {roads['pop_density'].max():,.1f}")

    return roads

def process_commercial_area(roads_gdf, input_dir, buffer_distance):
    """
    Summarize the commercial area density (ComArea) around each road segment within 'buffer_distance' (in feet).
    Updates roads_gdf with a new column 'ComArea', which is the SUM of ComArea from intersecting polygons
    divided by the length of the road segment to get commercial area per foot of road.
    
    Args:
        roads_gdf      (GeoDataFrame): Road segments (EPSG:2263).
        input_dir      (str): Directory path with PLUTO_ComMix.geojson.
        buffer_distance(float): Distance in feet to buffer roads.
    Returns:
        roads_gdf (GeoDataFrame): Updated with 'ComArea' column (commercial area per foot of road).
    """
    import os
    import geopandas as gpd
    import numpy as np
    
    # 1) Load the polygons
    com_path = os.path.join(input_dir, 'PLUTO_ComMix.geojson')
    if not os.path.exists(com_path):
        print(f"Warning: {com_path} not found. Setting 'ComArea' = 0.")
        roads_gdf['ComArea'] = 0.0
        return roads_gdf
    
    print(f"Loading Commercial Mix polygons from {com_path}...")
    com_gdf = gpd.read_file(com_path)
    
    # Ensure correct CRS
    if com_gdf.crs != roads_gdf.crs:
        com_gdf = com_gdf.to_crs(roads_gdf.crs)
    
    # Confirm 'ComArea' field is present
    if 'ComArea' not in com_gdf.columns:
        print("Warning: 'ComArea' field not found in PLUTO_ComMix.geojson. Setting to 0.")
        com_gdf['ComArea'] = 0.0
    
    # 2) Create a spatial index for faster bounding-box queries
    com_sindex = com_gdf.sindex
    
    # 3) Calculate segment lengths
    roads_gdf['segment_length'] = roads_gdf.geometry.length
    
    # 4) For each road, buffer by buffer_distance and sum the ComArea of intersecting polygons
    summed_areas = []
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            summed_areas.append(0.0)
            continue
            
        # Create the buffer around the road
        road_buffer = geom.buffer(buffer_distance)
        
        # Quickly get candidate polygons by bounding box intersection
        candidate_idxs = list(com_sindex.intersection(road_buffer.bounds))
        if not candidate_idxs:
            summed_areas.append(0.0)
            continue
            
        candidate_polys = com_gdf.iloc[candidate_idxs]
        
        # Precisely filter by intersection
        intersecting_polys = candidate_polys[candidate_polys.geometry.intersects(road_buffer)]
        
        # Sum the entire ComArea of those intersecting polygons
        total_com_area = intersecting_polys['ComArea'].sum()
        
        # Normalize by segment length
        segment_length = row['segment_length']
        if segment_length > 0:  # Prevent division by zero
            normalized_com_area = total_com_area / segment_length
        else:
            normalized_com_area = 0.0
            
        summed_areas.append(normalized_com_area)
    
    # 5) Attach results to roads_gdf
    roads_gdf['ComArea'] = summed_areas
    
    # 6) Debug info
    print(f"\nCalculated ComArea per foot for {len(roads_gdf)} roads (buffer={buffer_distance} ft).")
    print(f"ComArea density stats -> min={roads_gdf['ComArea'].min():.3f}, "
          f"max={roads_gdf['ComArea'].max():.3f}, "
          f"mean={roads_gdf['ComArea'].mean():.3f}")
    
    # 7) Clean up
    roads_gdf = roads_gdf.drop(columns=['segment_length'])
    
    return roads_gdf

def merge_street_segments(roads, min_segment_length):
    """
    Merge road segments with common street names, applying appropriate aggregation methods
    for different metrics including bike lane calculations.

    Args:
        roads (GeoDataFrame): GeoDataFrame of road segments.
        min_segment_length (float): Minimum segment length (in feet) to keep in the final output.

    Returns:
        merged_gdf (GeoDataFrame): GeoDataFrame of merged segments with aggregated metrics.
    """
    import numpy as np
    import geopandas as gpd
    from shapely.ops import linemerge, unary_union
    from shapely.validation import make_valid
    from tqdm.auto import tqdm

    try:
        print("Merging road segments with common names...")

        # If not already present, ensure each segment has a length in feet
        if 'segment_length' not in roads.columns:
            roads['segment_length'] = roads.geometry.length

        grouped = roads.groupby('StreetCode')
        merged_segments = []
        skipped_streets = []

        # Define aggregation methods for different column types
        #  - 'bike_ln_per_mile' must be aggregated in a length-weighted manner.
        length_weighted_cols = [
            'pav_rate', 'heat_mean', 'tree_pct', 'hvi_raw', 'pop_density',
            'pave_indx', 'heat_indx', 'tree_indx', 'vuln_indx', 'pop_dens_indx',
            'bike_ln_per_mile'
        ]

        sum_cols = [
            'vuln_length', 'intersections', 'bus_stop_count', 'bike_length', 'ComArea'
        ]

        # We'll skip geometry, StreetCode, etc. from these dicts since we handle them separately
        for street_name, street_group in tqdm(grouped, desc="Merging segments"):
            if len(street_group) > 1:
                try:
                    weights = street_group['segment_length']
                    total_length = weights.sum()
                    if total_length == 0:
                        print(f"Street {street_name} has total length of zero - skipping merge")
                        skipped_streets.append(street_name)
                        continue

                    aggregated_values = {}

                    for col in street_group.columns:
                        if col in sum_cols:
                            aggregated_values[col] = street_group[col].sum(skipna=True)
                        elif col in length_weighted_cols:
                            valid_mask = ~street_group[col].isna()
                            if valid_mask.any() and weights[valid_mask].sum() > 0:
                                aggregated_values[col] = np.average(
                                    street_group[col][valid_mask],
                                    weights=weights[valid_mask]
                                )
                            else:
                                aggregated_values[col] = np.nan
                        elif col in ['StreetCode', 'geometry', 'SegmentCount', 'segment_length']:
                            # We'll handle geometry and segment_length outside the loop
                            continue
                        else:
                            # Default to taking the first value (e.g., name fields)
                            aggregated_values[col] = street_group[col].iloc[0]

                    # Union and linemerge the geometry
                    merged_unary = unary_union(street_group.geometry)
                    merged_unary = make_valid(merged_unary)

                    final_geometries = []
                    if merged_unary.is_empty:
                        print(f"Street {street_name} is empty after union - skipping.")
                        skipped_streets.append(street_name)
                        continue

                    if merged_unary.geom_type == 'LineString':
                        final_geometries.append(merged_unary)
                    elif merged_unary.geom_type == 'MultiLineString':
                        try:
                            merged_geom = linemerge(merged_unary)
                            if merged_geom.geom_type == 'MultiLineString':
                                for part in merged_geom.geoms:
                                    final_geometries.append(part)
                            elif merged_geom.geom_type == 'LineString':
                                final_geometries.append(merged_geom)
                            else:
                                print(f"Street {street_name} produced {merged_geom.geom_type}, skipping.")
                                skipped_streets.append(street_name)
                                continue
                        except Exception as merge_error:
                            print(f"Could not linemerge street {street_name}: {merge_error}")
                            skipped_streets.append(street_name)
                            continue
                    else:
                        print(f"Street {street_name} is {merged_unary.geom_type} - skipping.")
                        skipped_streets.append(street_name)
                        continue

                    # Build final merged segment(s)
                    for geom in final_geometries:
                        segment_data = {
                            'StreetCode': street_name,
                            'Street': street_group['Street'].iloc[0],
                            'geometry': geom,
                            'SegmentCount': len(street_group),
                            'segment_length': geom.length  # Recompute for the merged segment
                        }
                        segment_data.update(aggregated_values)

                        # Calculate bus stop density if applicable
                        if 'bus_stop_count' in segment_data and segment_data['segment_length'] > 0:
                            segment_data['BusStpDens'] = (
                                segment_data['bus_stop_count'] / segment_data['segment_length']
                            ) * 5280

                        merged_segments.append(segment_data)

                except Exception as e:
                    print(f"Error merging segments for {street_name}: {str(e)}")
                    skipped_streets.append(street_name)
            else:
                # Single segment; keep as is but ensure metrics are calculated
                segment_data = street_group.iloc[0].to_dict()
                if 'segment_length' not in segment_data:
                    segment_data['segment_length'] = segment_data['geometry'].length

                if 'bus_stop_count' in segment_data and segment_data['segment_length'] > 0:
                    segment_data['BusStpDens'] = (
                        segment_data['bus_stop_count'] / segment_data['segment_length']
                    ) * 5280

                merged_segments.append(segment_data)

        merged_gdf = gpd.GeoDataFrame(merged_segments, crs=roads.crs)

        # 3) Apply length filter
        total_segments = len(merged_gdf)
        merged_gdf = merged_gdf[merged_gdf['segment_length'] >= min_segment_length]
        filtered_segments = total_segments - len(merged_gdf)

        print(f"\nLength filtering results:")
        print(f"Minimum length threshold: {min_segment_length} ft")
        print(f"Total segments before filter: {total_segments}")
        print(f"Segments removed: {filtered_segments}")
        print(f"Segments remaining: {len(merged_gdf)} ({(len(merged_gdf)/total_segments)*100:.1f}%)")

        # 4) Calculate final BikeLnIndx using min-max normalization of 'bike_ln_per_mile'
        if 'bike_ln_per_mile' in merged_gdf.columns:
            max_val = merged_gdf['bike_ln_per_mile'].max()
            min_val = merged_gdf['bike_ln_per_mile'].min()
            if max_val > min_val:
                merged_gdf['BikeLnIndx'] = (
                    (merged_gdf['bike_ln_per_mile'] - min_val) / (max_val - min_val)
                )
            else:
                # If all segments have the same bike_ln_per_mile
                merged_gdf['BikeLnIndx'] = 0.0

        # Recompute ComIndex by min-max in the merged_gdf
        if 'ComArea' in merged_gdf.columns:
            min_val = merged_gdf['ComArea'].min()
            max_val = merged_gdf['ComArea'].max()
            if max_val > min_val:
                merged_gdf['ComIndex'] = (
                    (merged_gdf['ComArea'] - min_val) / (max_val - min_val)
                )
            else:
                merged_gdf['ComIndex'] = 0.0
        else:
            merged_gdf['ComIndex'] = 0.0

        # Print helpful stats
        if 'BusStpDens' in merged_gdf.columns:
            print(f"\nBus stop density statistics:")
            print(f"Mean density: {merged_gdf['BusStpDens'].mean():.4f} stops per mile")
            print(f"Range: {merged_gdf['BusStpDens'].min():.4f} - {merged_gdf['BusStpDens'].max():.4f} stops per mile")

        if 'bike_ln_per_mile' in merged_gdf.columns:
            print(f"\nBike lane statistics:")
            print("Bike lanes per mile:")
            print(f"  Mean: {merged_gdf['bike_ln_per_mile'].mean():.2f}")
            print(f"  Range: {merged_gdf['bike_ln_per_mile'].min():.2f} - {merged_gdf['bike_ln_per_mile'].max():.2f}")
            print(f"  Segments with any nearby bike lanes: "
                  f"{(merged_gdf['bike_ln_per_mile'] > 0).sum()} "
                  f"({(merged_gdf['bike_ln_per_mile'] > 0).mean()*100:.1f}%)")
            print("Bike Lane Index (normalized 0-1):")
            print(f"  Mean: {merged_gdf['BikeLnIndx'].mean():.3f}")
            print(f"  Range: {merged_gdf['BikeLnIndx'].min():.3f} - {merged_gdf['BikeLnIndx'].max():.3f}")

        return merged_gdf

    except Exception as e:
        print(f"Error in street segment merging: {str(e)}")
        return roads

# ------------------------------------
# Final Analysis Classes
# ------------------------------------
class FinalAnalysis:
    def __init__(self, weight_scenarios):
        self.weight_scenarios = weight_scenarios

    def run_all_scenarios(self, roads, analysis_params):
        """
        If you only have one scenario, you can keep the structure or simplify.
        We'll keep it for flexibility.
        """
        results = {}
        all_scenario_roads = roads.copy()

        for scenario_name, weights in self.weight_scenarios.items():
            print(f"\nRunning analysis for {scenario_name}")
            scenario_results = self.calculate_priority(roads, weights, analysis_params)
            all_scenario_roads[f'priority_{scenario_name}'] = scenario_results['priority']
            all_scenario_roads[f'is_priority_{scenario_name}'] = scenario_results['is_priority']
            results[scenario_name] = scenario_results

        # Also store an 'ALL' scenario (aggregated) if needed
        results['ALL'] = all_scenario_roads
        return results

    @staticmethod
    def calculate_priority(roads, weights, analysis_params):
        import numpy as np
        import pandas as pd

        print("Calculating priority index...")

        working_df = roads.copy()
        priority_count = analysis_params.get('priority_threshold', 100)
        print(f"Using priority threshold of {priority_count} segments")

        # Weighted index approach
        working_df['priority'] = 0.0
        index_weights = {
            # 'traf_indx': weights.get('TrafficIndex', 0),
            'pave_indx': weights.get('PavementIndex', 0),
            'heat_indx': weights.get('HeatHazIndex', 0),
            'tree_indx': weights.get('TreeCanopyIndex', 0),
            'BusDensInx': weights.get('BusDensInx', 0),
            'vuln_indx': weights.get('HeatVulnerabilityIndex', 0),
            'BikeLnIndx': weights.get('BikeLnIndx', 0),
            'PedIndex': weights.get('PedIndex', 0),
            'pop_dens_indx': weights.get('pop_dens_indx', 0),
            'ComIndex': weights.get('ComIndex', 0)

        }

        # Check sum
        weight_sum = sum(index_weights.values())
        if not np.isclose(weight_sum, 1.0, rtol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        # Weighted sum for each factor
        for index, w in index_weights.items():
            if w > 0:
                if index not in working_df.columns:
                    raise ValueError(f"Missing required index: {index}")
                working_df['priority'] += working_df[index].fillna(0) * w
                print(f"{index} -> weight={w:.2f}, mean={(working_df[index]*w).mean():.3f}")

        # Mark top X
        working_df['is_priority'] = False
        if len(working_df) > 0:
            if priority_count >= len(working_df):
                print(f"Warning: Requested priority count ({priority_count}) >= total segments ({len(working_df)})")
                working_df['is_priority'] = True
            else:
                threshold_value = working_df.nlargest(priority_count, 'priority')['priority'].min()
                working_df['is_priority'] = working_df['priority'] >= threshold_value
                print(f"Priority threshold value for top {priority_count}: {threshold_value:.3f}")

        # Stats
        actual_priority_count = working_df['is_priority'].sum()
        print(f"\nFinal prioritization results:")
        print(f"High priority segments: {actual_priority_count}")
        if actual_priority_count > priority_count:
            print(f"Due to ties, {actual_priority_count - priority_count} extra segments included.")

        # Correlations
        factor_cols = [k for k in index_weights if k in working_df.columns]
        correlation_matrix = working_df[['priority'] + factor_cols].corr()['priority'].sort_values(ascending=False)
        print("\nCorrelations with final priority:")
        print(correlation_matrix)

        return working_df


# ------------------------------------
# Export Results + Webmap
# ------------------------------------

def export_results(results_dict, config):
    exported_paths = {}

    for scenario_name, final_roads in results_dict.items():
        if scenario_name == 'ALL':
            continue

        try:
            print(f"\nProcessing exports for scenario: {scenario_name}")

            export_fields = [
                # Basic identifiers
                'Street',
                'StreetCode',
                # Raw values
                # 'vol_adj',         # Traffic volume
                'pav_rate',        # Pavement rating
                'heat_mean',       # Heat hazard
                'tree_pct',        # Tree canopy percentage
                'hvi_raw',         # Heat vulnerability
                'BusStpDens',      # Number of Bus Stops per mile
                'bike_length',     # Nearby bike lane length
                'PedRank',         # Pedestrian Mobility Priority
                'pop_density',     # Population density
                'ComArea',         # Commercial Area

                # Index values
                # 'traf_indx',       # Traffic index
                'pave_indx',       # Pavement index
                'heat_indx',       # Heat hazard index
                'tree_indx',       # Tree canopy index
                'vuln_indx',       # Heat Vulnerability index
                'BusDensInx',      # Bus stop proximity index
                'BikeLnIndx',      # Bike lane index (0-1)
                'PedIndex',        # Pedestian mobility index
                'pop_dens_indx',   # Population density index
                'ComIndex',        # Commercial Area index

                # Priority results
                'priority',
                'is_priority'
            ]

            # Filter to available columns
            available_fields = [f for f in export_fields if f in final_roads.columns]

            # Create export dataframe
            export_gdf = final_roads[export_fields + ['geometry']].copy()
            output_filename = f"all_segments_{scenario_name}.geojson"
            output_path = os.path.join(config.output_dir, output_filename)
            export_gdf.to_file(output_path, driver='GeoJSON')

            exported_paths[scenario_name] = output_path
            print(f"Exported {len(export_gdf)} segments to {output_filename}")

        except Exception as e:
            print(f"Error exporting {scenario_name}: {str(e)}")

    return exported_paths

# ------------------------------------
# Neighborhood Analysis
# ------------------------------------
def run_neighborhood_analysis(config):
    """
    Run analysis for each neighborhood separately, ensuring that each neighborhood
    has its indices (pave_indx, heat_indx, ComIndex, etc.) re-min-maxed based on 
    only that neighborhood's data.
    
    This version generates separate HTML webmaps for each scenario.
    """
    import os
    import geopandas as gpd

    # Local imports from your project
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
        process_commercial_area   # if you have a separate function for ComArea
    )

    # -------------------------------------------------------------------------
    # 1) Load neighborhood boundaries
    # -------------------------------------------------------------------------
    nbhd_path = os.path.join(config.input_dir, "CSC_Neighborhoods.geojson")
    if not os.path.exists(nbhd_path):
        raise FileNotFoundError(f"Neighborhood boundaries not found at: {nbhd_path}")

    neighborhoods = gpd.read_file(nbhd_path)
    if neighborhoods.crs is None:
        neighborhoods.set_crs("EPSG:2263", inplace=True)
    elif neighborhoods.crs.to_string() != "EPSG:2263":
        neighborhoods = neighborhoods.to_crs("EPSG:2263")

    # -------------------------------------------------------------------------
    # 2) Load and preprocess citywide roads (one time)
    # -------------------------------------------------------------------------
    roads = load_and_preprocess_roads(config)
    if roads.crs is None:
        roads.set_crs("EPSG:2263", inplace=True)
    elif roads.crs.to_string() != "EPSG:2263":
        roads = roads.to_crs("EPSG:2263")

    # 2A) Run your main attribute processors (pavement, bike, etc.)
    raster_processor = OptimizedRasterProcessing()
    roads = DataProcessors.batch_process_all(roads, config.input_dir)

    # 2B) Process commercial area (if applicable)
    if hasattr(config.analysis_params, "pop_buffer_ft"):
        buffer_ft = config.analysis_params["pop_buffer_ft"]
    else:
        buffer_ft = 1000  # fallback
    roads = process_commercial_area(roads, config.input_dir, buffer_ft)

    # 2C) (Optional) Citywide ComIndex, but we will re-min-max at the neighborhood level,
    # so this step is not strictly necessary:
    #   roads['ComIndex'] = ...
    #   roads['heat_indx'] = ...
    # etc. 
    # We'll do a citywide pass if you want to see global distributions, 
    # but final min-max will happen per neighborhood.

    # 2D) Temperature
    temp_raster_path = os.path.join(config.input_dir, 'Landsat9_ThermalComposite_ST_B10_2020-2023.tif')
    roads = raster_processor.optimize_temperature_processing(roads, temp_raster_path)
    # If you want citywide indices, do it here, but we *will* re-min-max later

    # 2E) Tree canopy
    canopy_raster_path = os.path.join(config.input_dir, 'NYC_TreeCanopy.tif')
    roads = raster_processor.optimize_tree_canopy_processing(roads, canopy_raster_path)

    # 2F) Bus stops
    roads = process_bus_stops(roads, config.input_dir, config.crs)

    # 2G) Population density
    roads = incorporate_population_density(
        roads,
        config.input_dir,
        buffer_ft=config.analysis_params["pop_buffer_ft"]
    )

    # -------------------------------------------------------------------------
    # 3) Initialize final analyzer
    # -------------------------------------------------------------------------
    analyzer = FinalAnalysis(config.weight_scenarios)

    # -------------------------------------------------------------------------
    # 4) Process each neighborhood
    # -------------------------------------------------------------------------
    for idx, nbhd_row in neighborhoods.iterrows():
        nbhd_name = nbhd_row['Name']
        print(f"\nProcessing neighborhood: {nbhd_name}")

        # 4A) Clip roads to this neighborhood
        roads_sindex = roads.sindex
        nbhd_bounds = nbhd_row.geometry.bounds
        candidate_indices = list(roads_sindex.intersection(nbhd_bounds))
        candidate_roads = roads.iloc[candidate_indices]

        nbhd_roads = gpd.clip(candidate_roads, nbhd_row.geometry)
        if len(nbhd_roads) == 0:
            print(f"No roads found in neighborhood {nbhd_name}, skipping...")
            continue
        print(f"Found {len(nbhd_roads)} road segments in {nbhd_name}")

        # 4B) Merge segments within this neighborhood
        nbhd_roads = merge_street_segments(nbhd_roads, config.analysis_params["min_segment_length"])
        print(f"After merging: {len(nbhd_roads)} segments for {nbhd_name}")

        # ---------------------------------------------------------------------
        # 4C) Re-min-max each relevant index *within* the neighborhood
        #     This ensures each index is 0-1 based on neighborhood's data only.
        # ---------------------------------------------------------------------
        from numpy import nanmin, nanmax

        # Example: heat_mean -> heat_indx
        if 'heat_mean' in nbhd_roads.columns:
            nbhd_roads['heat_indx'] = DataProcessors.normalize_to_index(nbhd_roads['heat_mean'], 'temperature')

        # Pavement rating -> pave_indx
        if 'pav_rate' in nbhd_roads.columns:
            nbhd_roads['pave_indx'] = DataProcessors.normalize_to_index(nbhd_roads['pav_rate'], 'pavement')

        # Tree canopy -> tree_indx
        if 'tree_pct' in nbhd_roads.columns:
            nbhd_roads['tree_indx'] = DataProcessors.normalize_to_index(nbhd_roads['tree_pct'], 'canopy')

        # Heat vulnerability -> vuln_indx (if you store raw in 'hvi_raw')
        if 'hvi_raw' in nbhd_roads.columns:
            nbhd_roads['vuln_indx'] = DataProcessors.normalize_to_index(nbhd_roads['hvi_raw'], 'basic')

        # Bus stops -> BusDensInx
        if 'BusStpDens' in nbhd_roads.columns:
            nbhd_roads['BusDensInx'] = DataProcessors.normalize_to_index(nbhd_roads['BusStpDens'], 'basic')

        # Bike lanes -> BikeLnIndx
        if 'bike_ln_per_mile' in nbhd_roads.columns:
            nbhd_roads['BikeLnIndx'] = DataProcessors.normalize_to_index(nbhd_roads['bike_ln_per_mile'], 'basic')

        # Population density -> pop_dens_indx
        if 'pop_density' in nbhd_roads.columns:
            nbhd_roads['pop_dens_indx'] = DataProcessors.normalize_to_index(nbhd_roads['pop_density'], 'popdensity')

        # Commercial Area -> ComIndex
        if 'ComArea' in nbhd_roads.columns:
            nbhd_roads['ComIndex'] = DataProcessors.normalize_to_index(nbhd_roads['ComArea'], 'basic')

        # (Add any other fields you want to re-min-max, e.g., PedRank -> PedIndex, etc.)

        # ---------------------------------------------------------------------
        # 4D) Now run final priority analysis
        # ---------------------------------------------------------------------
        nbhd_results = analyzer.run_all_scenarios(nbhd_roads, config.analysis_params)

        # 4E) Create neighborhood-specific output directory
        nbhd_output_dir = os.path.join(config.output_dir, nbhd_name.replace(" ", "_"))
        os.makedirs(nbhd_output_dir, exist_ok=True)

        # 4F) Export results (which by default might create "all_segments_{scenario}.geojson" etc.)
        config_copy = config.copy()
        config_copy.output_dir = nbhd_output_dir
        exported_gdfs = export_results(nbhd_results, config_copy)

        # ---------------------------------------------------------------------
        # 4G) Generate separate HTML webmaps for each scenario
        # ---------------------------------------------------------------------
        try:
            scenario_geojsons = {}
            for scenario_name in nbhd_results.keys():
                # Skip any keys you do not want to map (e.g., a combined "ALL" scenario)
                if scenario_name == 'ALL':
                    continue
                # Construct the expected filename (adjust this if your export naming differs)
                out_name = f"all_segments_{scenario_name}.geojson"
                geojson_path = os.path.join(nbhd_output_dir, out_name)
                if os.path.exists(geojson_path):
                    scenario_geojsons[scenario_name] = geojson_path

            if scenario_geojsons:
                nbhd_config = config_copy
                nbhd_config.input_dir = config.input_dir  # maintain consistent reference paths

                # Generate a separate HTML webmap for each scenario
                for scenario_name, geojson_path in scenario_geojsons.items():
                    scenario_webmap_path = build_webmap(
                        {scenario_name: geojson_path},  # pass a single scenario in a dict
                        nbhd_config,
                        neighborhood_name=f"{nbhd_name}_{scenario_name}"
                    )
                    print(f"Generated webmap for {nbhd_name} scenario {scenario_name}: {scenario_webmap_path}")
            else:
                print(f"No GeoJSON files found for {nbhd_name} webmap generation")

        except Exception as e:
            print(f"Warning: HTML map generation failed for {nbhd_name}: {str(e)}")

        print(f"Completed processing for {nbhd_name}")

    print("\nNeighborhood analysis complete!")

# ------------------------------------
# Webmap Builder
# ------------------------------------
def build_webmap(scenario_geojsons, config, neighborhood_name=None):
    """
    Create a single HTML folium map with heat-map style visualization 
    and detailed popups showing raw and index values.
    """
    import folium
    import geopandas as gpd
    import pandas as pd
    import os
    import numpy as np

    print(f"\nBuilding layered HTML webmap{' for ' + neighborhood_name if neighborhood_name else ''}...")

    def get_color(priority_score, feature_collection):
        """Returns color based on priority score normalization."""
        if pd.isna(priority_score):
            return '#808080'  # gray for missing values

        colors = ['#FFE066', '#FFB84D', '#FF9933', '#FF7A1A', '#FF5C00', '#E63D00', '#CC0000']

        priorities = [f['properties'].get('priority') for f in feature_collection['features'] 
                     if f['properties'].get('priority') is not None]
        min_priority = min(priorities)
        max_priority = max(priorities)

        normalized_score = (0 if max_priority == min_priority 
                          else (priority_score - min_priority) / (max_priority - min_priority))
        idx = int(np.floor(normalized_score * (len(colors) - 1)))
        return colors[max(0, min(idx, len(colors) - 1))]

    def style_function(feature, feature_collection):
        """Style function for GeoJSON features."""
        priority = feature['properties'].get('priority', None)
        return {
            'color': get_color(priority, feature_collection),
            'weight': 3,
            'opacity': 0.8
        }

    def create_popup_content(properties):
        """Creates detailed HTML popup content with raw and index values."""
        try:
            # Helper function to format values safely
            def format_value(value, format_spec, default='N/A'):
                if value is None:
                    return default
                try:
                    if isinstance(value, (int, float)):
                        return format_spec.format(value)
                    return str(value)
                except:
                    return default

            return f"""
            <div style="font-family: Arial; min-width: 200px; max-width: 300px;">
                <h4 style="margin-bottom: 10px; border-bottom: 1px solid #ccc;">
                    {properties.get('Street', 'N/A')}
                </h4>

                <div style="margin-bottom: 10px;">
                    <b>Raw Values</b><br>
                    • Average Pavement Rating: {format_value(properties.get('pav_rate'), '{:.1f}')}<br>
                    • Average Summer Heat: {format_value(properties.get('heat_mean'), '{:.1f}')} (°F)<br>
                    • Tree Canopy: {format_value(properties.get('tree_pct'), '{:.1f}')}%<br>
                    • Heat Vulnerability Index Average: {format_value(properties.get('hvi_raw'), '{:.2f}')}<br>
                    • Bus Stop Density: ~{format_value(properties.get('BusStpDens'), '{:,.0f}')} Stops per Mile<br>
                    • Bike Lane Density: {format_value(properties.get('bike_length', 0), '{:,.0f}')} ft Bike Lane per Mile<br>
                    • Pedestrian Mobility Priority: {format_value(properties.get('PedRank', 0), '{:,.0f}')}<br>
                    • Population Density: ~{format_value(properties.get('pop_density'), '{:,.0f}')} People per Square Mile<br>
                    • Commercial Area: ~{format_value(properties.get('ComArea'), '{:,.0f}')} nearby sq ft per 1 ft of road
                </div>

                <div style="margin-bottom: 10px;">
                    <b>Index Values</b><br>
                    • Pavement: {format_value(properties.get('pave_indx'), '{:.3f}')}<br>
                    • Heat: {format_value(properties.get('heat_indx'), '{:.3f}')}<br>
                    • Tree Canopy: {format_value(properties.get('tree_indx'), '{:.3f}')}<br>
                    • Heat Vulnerability: {format_value(properties.get('vuln_indx'), '{:.3f}')}<br>
                    • Bus Stops: {format_value(properties.get('BusDensInx'), '{:.3f}')}<br>
                    • Bike Lanes: {format_value(properties.get('BikeLnIndx'), '{:.3f}')}<br>
                    • Pedestrian Mobility: {format_value(properties.get('PedIndex'), '{:.3f}')}<br>
                    • Population Density: {format_value(properties.get('pop_dens_indx'), '{:.3f}')}<br>
                    • Commercial Area: {format_value(properties.get('ComIndex'), '{:.3f}')}

                </div>

                <div style="font-weight: bold; color: #CC0000;">
                    Priority Score: {format_value(properties.get('priority'), '{:.3f}')}
                </div>
            </div>
            """
        except Exception as e:
            print(f"Error creating popup content: {str(e)}")
            return "<div>Error creating popup content</div>"

    # Load and process neighborhoods
    neighborhoods_4326 = None
    neighborhoods_path = os.path.join(config.input_dir, "CSC_Neighborhoods.geojson")
    if os.path.exists(neighborhoods_path):
        try:
            neighborhoods_gdf = gpd.read_file(neighborhoods_path)
            if neighborhoods_gdf.crs is None:
                neighborhoods_gdf.set_crs("EPSG:2263", inplace=True)
            neighborhoods_4326 = neighborhoods_gdf.to_crs("EPSG:4326")
            if neighborhood_name:
                neighborhoods_4326 = neighborhoods_4326[
                    neighborhoods_4326['Name'] == neighborhood_name
                ]
        except Exception as e:
            print(f"Warning: Could not process neighborhoods file: {str(e)}")

    # Calculate bounds and process scenarios
    bounds_list = []
    if neighborhoods_4326 is not None and not neighborhoods_4326.empty:
        bounds_list.append(neighborhoods_4326.total_bounds)

    scenario_data = {}
    for scenario_name, reprojected_path in scenario_geojsons.items():
        if not os.path.exists(reprojected_path):
            continue
        try:
            gdf = gpd.read_file(reprojected_path)
            if gdf.crs is None:
                gdf.set_crs("EPSG:2263", inplace=True)
            gdf_4326 = gdf.to_crs("EPSG:4326")
            bounds_list.append(gdf_4326.total_bounds)
            scenario_data[scenario_name] = gdf_4326
        except Exception as e:
            print(f"Warning: Could not process {scenario_name}: {str(e)}")
            continue

    # Create base map
    if bounds_list:
        bounds_array = np.array(bounds_list)
        center_lat = (bounds_array[:, 1].min() + bounds_array[:, 3].max()) / 2
        center_lon = (bounds_array[:, 0].min() + bounds_array[:, 2].max()) / 2
    else:
        center_lat, center_lon = 40.7128, -74.0060

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14 if neighborhood_name else 12,
        tiles="CartoDB Positron"
    )

    # Add title
    title_html = f'''
    <div style="position: fixed; 
                top: 20px; 
                left: 12%; 
                transform: translateX(-50%);
                z-index: 1000;
                background-color: white;
                padding: 10px;
                border: 2px solid grey;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
                font-family: Arial;">
        {list(scenario_geojsons.keys())[0]} Roads
    </div>
    '''

    m.get_root().html.add_child(folium.Element(title_html))

    # Add FOZ layer
    foz_path = os.path.join(config.input_dir, 'FOZ_NYC_Merged.geojson')
    if os.path.exists(foz_path):
        try:
            foz_gdf = gpd.read_file(foz_path)
            if foz_gdf.crs is None or foz_gdf.crs != "EPSG:4326":
                foz_gdf = foz_gdf.to_crs("EPSG:4326")
            folium.GeoJson(
                foz_gdf,
                name="Federal Opportunity Zones",
                style_function=lambda x: {
                    'fillColor': 'blue',
                    'color': 'blue',
                    'weight': 0.5,
                    'fillOpacity': 0.05,
                    'opacity': 1.0
                }
            ).add_to(m)
        except Exception as e:
            print(f"Warning: Could not process FOZ file: {str(e)}")

    # Add Persistent Poverty layer
    poverty_path = os.path.join(config.input_dir, 'nyc_persistent_poverty.geojson')
    if os.path.exists(poverty_path):
        try:
            poverty_gdf = gpd.read_file(poverty_path)
            if poverty_gdf.crs is None or poverty_gdf.crs != "EPSG:4326":
                poverty_gdf = poverty_gdf.to_crs("EPSG:4326")
            folium.GeoJson(
                poverty_gdf,
                name="Persistent Poverty Areas",
                style_function=lambda x: {
                    'fillColor': 'green',
                    'color': 'green',
                    'weight': 0.5,
                    'fillOpacity': 0.05,
                    'opacity': 1.0
                }
            ).add_to(m)
        except Exception as e:
            print(f"Warning: Could not process persistent poverty file: {str(e)}")


    # Add neighborhoods layer
    if neighborhoods_4326 is not None and not neighborhoods_4326.empty:
        folium.GeoJson(
            neighborhoods_4326,
            name="CSC_Neighborhoods",
            style_function=lambda x: {
                "color": "gray",
                "weight": 1,
                "fillOpacity": 0.1
            }
        ).add_to(m)

    legend_html = """
    <div style="
        position: fixed; 
        bottom: 50px; 
        right: 50px; 
        z-index: 1000;
        background: white; 
        padding: 10px; 
        border: 2px solid grey;
        border-radius: 5px;
        ">
        <h4 style="margin-bottom: 10px;">Legend</h4>
        <div style="display: flex; flex-direction: column; gap: 10px;">
            <div>
                <h5 style="margin: 5px 0;">Road Priority Score</h5>
                <div style="display: flex; flex-direction: column; gap: 5px;">
    """

    colors = ['#CC0000', '#E63900', '#FF4D00', '#FF6B1A', 
              '#FF8533', '#FF9940', '#FFB366']
    labels = ['1.0', '0.83', '0.67', '0.5', '0.33', '0.17', '0.0']

    for color, label in zip(colors, labels):
        legend_html += f"""
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 20px; height: 3px; background: {color};"></div>
                <span>{label}</span>
            </div>
        """


    legend_html += """
                </div>
            </div>
            <div>
                <h5 style="margin: 5px 0;">Area Overlays</h5>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: gray; opacity: 0.2; border: 1px solid gray;"></div>
                        <span>CSC Neighborhoods</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: green; opacity: 0.1; border: 1px solid green;"></div>
                        <span>Persistent Poverty Areas</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: blue; opacity: 0.1; border: 1px solid blue;"></div>
                        <span>Federal Opportunity Zones</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """

    # Create a custom Legend element
    legend = folium.Element(legend_html)

    # Add the legend to the map
    m.get_root().html.add_child(legend)

    # Add scenario layers with popups
    for scenario_name, gdf_4326 in scenario_data.items():
        geojson_data = gdf_4326.__geo_interface__

        def style_callback(feature):
            return style_function(feature, geojson_data)

        gjson = folium.GeoJson(
            gdf_4326,
            name=scenario_name,
            style_function=style_callback,
            tooltip=folium.GeoJsonTooltip(
                fields=['priority'],
                aliases=['Priority Score:'],
                sticky=False,
                labels=True,
                style="""
                    background-color: white;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """
            )
        )

        # Add popups
        for feature in gjson.data['features']:
            if feature['properties'] is not None:
                popup_content = create_popup_content(feature['properties'])
                if popup_content:
                    folium.Popup(popup_content, max_width=300).add_to(
                        folium.GeoJson(
                            feature,
                            style_function=style_callback
                        ).add_to(gjson)
                    )

        gjson.add_to(m)


    folium.LayerControl(collapsed=False).add_to(m)

    # Save map
    html_map_path = os.path.join(
        config.output_dir,
        f"{scenario_name}_webmap{'_' + neighborhood_name.replace(' ', '_') if neighborhood_name else ''}.html"
    )
    m.save(html_map_path)
    print(f"Webmap saved to: {html_map_path}")

    return html_map_path



def generate_webmap(results_dict, exported_paths, config):
    """Generate the webmap using exported GeoJSON files."""
    try:
        print("\nGenerating interactive HTML maps...")

        # Check if we have any valid GeoJSON files
        valid_geojsons = {
            scenario: path for scenario, path in exported_paths.items()
            if os.path.exists(path)
        }

        if not valid_geojsons:
            print("Warning: No GeoJSON files found for HTML map generation")
            return None

        webmap_paths = []
        # Generate a separate webmap for each scenario
        for scenario_name, geojson_path in valid_geojsons.items():
            scenario_geojsons = {scenario_name: geojson_path}
            webmap_path = build_webmap(scenario_geojsons, config)
            if webmap_path:
                webmap_paths.append(webmap_path)
                print(f"Generated webmap for {scenario_name} at: {webmap_path}")

        return webmap_paths

    except Exception as e:
        print(f"Error generating webmap: {str(e)}")
        return None

# Usage example:
def run_exports_and_webmap(results_dict, config):
    """Run the full export and webmap generation process."""
    try:
        # Export results to GeoJSON
        exported_paths = export_results(results_dict, config)

        # Generate webmap
        if exported_paths:
            webmap_path = generate_webmap(results_dict, exported_paths, config)
            if webmap_path:
                print(f"Successfully generated webmap at: {webmap_path}")
        else:
            print("No results were exported, skipping webmap generation")

    except Exception as e:
        print(f"Error in export and webmap process: {str(e)}")

