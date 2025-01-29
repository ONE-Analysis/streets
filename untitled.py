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

# ------------------------------------
# Data Processing Classes
# ------------------------------------
class DataProcessors:
    @staticmethod
    def normalize_to_index(series, attribute_type):
        """
        Normalize a series to an index between 0 and 1 based on attribute type.
        - 'bike' => direct 0/1
        - 'pavement', 'heat', 'canopy' => reversed normalization
        """
        try:
            if series.isna().all():
                return pd.Series(0.5, index=series.index)

            if attribute_type == 'bike':
                # For bike lanes, we already have a 0/1 indicator
                return series.astype(float)

            numeric_series = pd.to_numeric(series, errors='coerce')
            valid_data = numeric_series.dropna()
            if valid_data.nunique() <= 1:
                return pd.Series(0.5, index=series.index)

            min_val = valid_data.min()
            max_val = valid_data.max()
            normalized = (valid_data - min_val) / (max_val - min_val)

            # Reverse for certain attributes
            if attribute_type in ['pavement', 'heat', 'canopy']:
                normalized = 1 - normalized

            result = pd.Series(normalized, index=valid_data.index)
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
    def batch_process_all(roads, input_dir):
        """
        Process traffic, pavement, vulnerability, bike lanes, etc.
        """
        try:
            print("Starting batch processing...")

            # Traffic data
            traffic = gpd.read_file(os.path.join(input_dir, 'Automated_Traffic_Volume_Counts.geojson'))
            traffic['segmentid'] = traffic['SegmentID'].astype(str).str.strip().str.zfill(7)
            traffic_agg = (traffic.groupby('segmentid')
                                   .agg({'Vol': ['mean', 'count', 'std']})
                                   .reset_index())
            traffic_agg.columns = ['segmentid', 'traf_vol', 'vol_count', 'vol_std']

            # Pavement data
            pavement = gpd.read_file(os.path.join(input_dir, 'Street_Pavement_Rating.geojson'))
            pavement['pav_rate'] = pd.to_numeric(pavement['manualrati'], errors='coerce')
            pavement = pavement.sort_values('pav_rate').drop_duplicates('segmentid', keep='first')

            # Vulnerability data
            vulnerability = gpd.read_file(os.path.join(input_dir, 'HeatVulnerabilityIndex.geojson'))
            vulnerability['hvi_raw'] = pd.to_numeric(vulnerability['HVI'], errors='coerce')
            vulnerability = vulnerability.to_crs(roads.crs)

            # Merge everything
            roads = (roads.merge(traffic_agg, on='segmentid', how='left')
                          .merge(pavement[['segmentid', 'pav_rate']], on='segmentid', how='left'))

            # Adjusted volume
            roads['vol_conf'] = roads['vol_count'] / roads['vol_count'].max()
            roads['vol_adj'] = roads['traf_vol'] * (1 + roads['vol_conf'] * 0.1)
            if roads['vol_adj'].isna().any():
                median_vol = roads['vol_adj'].median()
                roads['vol_adj'] = roads['vol_adj'].fillna(median_vol)

            # Normalize
            roads['traf_indx'] = DataProcessors.normalize_to_index(roads['vol_adj'], 'traffic')
            roads['pave_indx'] = DataProcessors.normalize_to_index(roads['pav_rate'], 'pavement')

            # Bike lanes
            from data_preprocessing import process_bike_lanes
            roads = process_bike_lanes(roads)

            # Vulnerability overlay
            spatial_index = vulnerability.sindex
            vulnerability_results = []
            for idx, road in tqdm(roads.iterrows(), total=len(roads), desc="Calculating vulnerability"):
                result = DataProcessors._calculate_vulnerability_detailed(road, vulnerability, spatial_index)
                vulnerability_results.append(result)

            roads['hvi_raw'] = [res['hvi'] for res in vulnerability_results]
            roads['intersections'] = [res['intersections'] for res in vulnerability_results]
            roads['vuln_length'] = [res['length'] for res in vulnerability_results]
            roads['vuln_indx'] = DataProcessors.normalize_to_index(roads['hvi_raw'], 'heat')

            # Some debug stats
            print(f"\nTraffic index statistics:")
            print(f"Mean traffic index: {roads['traf_indx'].mean():.3f}")
            print(f"Range: {roads['traf_indx'].min():.3f} - {roads['traf_indx'].max():.3f}")

            print(f"\nPavement index statistics:")
            print(f"Mean pavement index: {roads['pave_indx'].mean():.3f}")

            print(f"\nVulnerability index statistics:")
            print(f"Mean vulnerability index: {roads['vuln_indx'].mean():.3f}")

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
        """Process tree canopy with improved raster alignment."""
        import logging
        import rasterio
        from rasterio.windows import Window
        from rasterio.features import rasterize
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm

        try:
            FEET_TO_METERS = 0.3048
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
                            buffer_width = float(road['StreetWidth_Min']) * FEET_TO_METERS / 2 if pd.notna(road['StreetWidth_Min']) else 25 * FEET_TO_METERS / 2
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
    """Calculate distances to nearest bus stops, then create BusProxInx."""
    import geopandas as gpd
    import pandas as pd

    try:
        print("Processing bus stop proximity...")
        bus_stops = gpd.read_file(os.path.join(input_dir, 'BusStops.geojson'))
        bus_stops = bus_stops.to_crs(crs)
        print(f"Loaded {len(bus_stops)} bus stops")

        def nearest_bus_stop(geometry):
            distances = bus_stops.geometry.distance(geometry)
            return distances.min()

        print("Calculating distances to nearest bus stops...")
        roads['bus_distance'] = roads.geometry.apply(nearest_bus_stop)

        max_dist = roads['bus_distance'].max()
        min_dist = roads['bus_distance'].min()
        if max_dist == min_dist:
            roads['BusProxInx'] = 1.0
        else:
            roads['BusProxInx'] = 1 - ((roads['bus_distance'] - min_dist) / (max_dist - min_dist))

        print(f"Bus stop proximity processing complete")
        print(f"Distance range: {min_dist:.2f} to {max_dist:.2f} feet")
        print(f"Mean distance: {roads['bus_distance'].mean():.2f} feet")

        return roads

    except Exception as e:
        print(f"Error processing bus stops: {str(e)}")
        raise


# ------------------------------------
# Population Assessment
# ------------------------------------
def create_nyc_blocks_with_pop(input_dir, target_crs="EPSG:2263"):
    """
    Load Tiger blocks for NY state from local shapefile, 
    filter for NYC counties, and prepare population data.
    """
    from pathlib import Path
    import geopandas as gpd
    import warnings

    nyc_counties = ['005','047','061','081','085']  # Bronx, Kings, New York, Queens, Richmond

    # Load from local shapefile
    tiger_dir = Path(input_dir) / "tiger_blocks"
    shp_path = tiger_dir / "tl_2020_36_tabblock20.shp"

    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found at: {shp_path}")

    print("Loading blocks shapefile...")
    blocks = gpd.read_file(str(shp_path))
    print(f"Total blocks loaded: {len(blocks)}")

    # Filter to NYC counties
    blocks = blocks[blocks['GEOID20'].str[2:5].isin(nyc_counties)]
    print(f"Filtered to NYC counties: {len(blocks)} blocks")

    # Ensure correct CRS
    if blocks.crs is None:
        warnings.warn("Vector data has no CRS. Setting to target CRS.")
        blocks = blocks.set_crs(target_crs)
    elif blocks.crs.to_string() != target_crs:
        blocks = blocks.to_crs(target_crs)

    # Rename POP20 to P1_001N for consistency with rest of code
    blocks = blocks.rename(columns={'POP20': 'P1_001N'})
    blocks['P1_001N'] = blocks['P1_001N'].fillna(0).astype(int)
    blocks['P1_001N'] = pd.to_numeric(blocks['P1_001N'], errors='coerce').fillna(0)

    # Save final data as GeoJSON
    out_path = Path(input_dir) / "nyc_blocks_with_pop.geojson"
    print(f"Saving combined blocks+pop to {out_path}")
    blocks.to_file(out_path, driver='GeoJSON')

    return str(out_path)

def incorporate_population_density(roads, input_dir, buffer_ft=1000):
    """
    Load NYC blocks from local shapefile,
    then compute pop_density for each road and pop_dens_indx from it.
    """
    import os
    import geopandas as gpd
    from analysis_modules import DataProcessors

    # 1) Load and process blocks from local shapefile
    merged_file = create_nyc_blocks_with_pop(
        input_dir=input_dir,
        target_crs="EPSG:2263"
    )

    # 2) Confirm the GeoJSON file exists, then read
    if not os.path.exists(merged_file):
        raise FileNotFoundError(f"Blocks file with population data not found at: {merged_file}")

    blocks_gdf = gpd.read_file(merged_file)
    if str(blocks_gdf.crs) != "EPSG:2263":
        blocks_gdf = blocks_gdf.to_crs("EPSG:2263")

    blocks_gdf = blocks_gdf.to_crs("EPSG:2263")
    roads = roads.to_crs("EPSG:2263")

    # 3) Estimate population density around each road
    roads = estimate_population_density(roads, blocks_gdf, buffer_ft=buffer_ft)

    # 4) Create a normalized index from pop_density
    roads["pop_dens_indx"] = DataProcessors.normalize_to_index(
        roads["pop_density"],
        "popdensity"
    )

    return roads

# ------------------------------------
# Population Density Calculation
# ------------------------------------
def estimate_population_density(roads, blocks_gdf, buffer_ft=1000):
    """
    For each road segment, create a buffer around its geometry (the entire linestring),
    then estimate population from intersecting Census blocks. Then compute population
    density (pop / buffer area).
    """
    import geopandas as gpd

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
        
        # Create a GeoDataFrame from the buffer geometry - THIS IS THE KEY FIX
        buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_geom], crs=roads.crs)
        
        candidate_idxs = list(block_sindex.intersection(buffer_geom.bounds))
        candidate_blocks = blocks_gdf.iloc[candidate_idxs].copy()
        if candidate_blocks.empty:
            pop_estimates.append(0.0)
            continue

        # Use buffer_gdf instead of buffer_ft in the overlay
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

        pop_density = total_pop / buffer_area
        pop_estimates.append(pop_density)

    roads["pop_density"] = pop_estimates
    return roads

# ------------------------------------
# Merging Segments
# ------------------------------------ 
def merge_street_segments(roads, min_segment_length):
    """
    Merge road segments with common street names, applying length-weighted
    averages (including new pop_density, pop_dens_indx, heat_mean, tree_pct, etc.).
    Also sums columns like vuln_length or intersections.
    """
    import numpy as np
    import geopandas as gpd
    from shapely.ops import linemerge, unary_union
    from shapely.validation import make_valid
    from tqdm.auto import tqdm

    try:
        print("Merging road segments with common names...")
        roads['segment_length'] = roads.geometry.length
        grouped = roads.groupby('StreetCode')
        merged_segments = []
        skipped_streets = []

        # Define which columns to do length-weighted averages for,
        # and which columns to do sums for (e.g., intersections, vuln_length).
        length_weighted_cols = [
            'traf_vol', 'vol_adj', 'pav_rate', 'heat_mean', 'tree_pct', 'hvi_raw',
            'BikeLane', 'bus_distance', 'BusProxInx', 'BoroCode',
            'pop_density', 'pop_dens_indx', 'traf_indx', 'pave_indx',
            'heat_indx', 'tree_indx', 'vuln_indx'
        ]
        sum_cols = ['vuln_length', 'intersections']

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
                            # Sum columns like intersections or vuln_length
                            aggregated_values[col] = street_group[col].sum(skipna=True)
                        elif col in length_weighted_cols or col.endswith('_indx'):
                            # Length-weighted average for these columns
                            valid_mask = ~street_group[col].isna()
                            if valid_mask.any() and weights[valid_mask].sum() > 0:
                                aggregated_values[col] = np.average(
                                    street_group[col][valid_mask],
                                    weights=weights[valid_mask]
                                )
                            else:
                                aggregated_values[col] = np.nan
                        elif col in ['StreetCode', 'geometry', 'SegmentCount', 'segment_length']:
                            # We'll handle geometry below, plus add these after union
                            continue
                        else:
                            # Carry forward the first value (if it's consistent).
                            # Or you can choose some other logic (e.g. mode).
                            aggregated_values[col] = street_group[col].iloc[0]

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
                            'segment_length': geom.length
                        }
                        segment_data.update(aggregated_values)
                        merged_segments.append(segment_data)

                except Exception as e:
                    print(f"Error merging segments for {street_name}: {str(e)}")
                    skipped_streets.append(street_name)
            else:
                # Single segment; just keep as is
                segment_data = street_group.iloc[0].to_dict()
                merged_segments.append(segment_data)

        merged_gdf = gpd.GeoDataFrame(merged_segments, crs=roads.crs)
        total_segments = len(merged_gdf)
        merged_gdf = merged_gdf[merged_gdf['segment_length'] >= min_segment_length]
        filtered_segments = total_segments - len(merged_gdf)

        print(f"\nLength filtering results:")
        print(f"Minimum length threshold: {min_segment_length} feet")
        print(f"Total segments before filter: {total_segments}")
        print(f"Segments removed: {filtered_segments}")
        print(f"Segments remaining: {len(merged_gdf)} ({(len(merged_gdf)/total_segments)*100:.1f}%)")

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
            'traf_indx': weights.get('TrafficIndex', 0),
            'pave_indx': weights.get('PavementIndex', 0),
            'heat_indx': weights.get('HeatHazIndex', 0),
            'tree_indx': weights.get('TreeCanopyIndex', 0),
            'BusProxInx': weights.get('BusProxInx', 0),
            'BikeLane': weights.get('BikeLane', 0),
            # NEW factor: pop_dens_indx
            'pop_dens_indx': weights.get('PopDensity', 0)
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
# Visualization (Optional)
# ------------------------------------
class PriorityVisualization:
    def __init__(self):
        self.color_scheme = {
            'traffic': '#346df1',
            'heat': '#f8b00b',
            'tree': '#2d9b42',
            'bus': '#3bb1b9',
            'bike': '#6997f5'
        }

    def create_single_map(self, data, column, title, ax, color, is_binary=False):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        if is_binary:
            data.plot(
                column=column,
                ax=ax,
                colors=['white', color],
                legend=True,
                legend_kwds={
                    'label': title,
                    'orientation': 'vertical',
                    'shrink': 0.5
                }
            )
        else:
            cmap = LinearSegmentedColormap.from_list('custom', ['white', color])
            data.plot(
                column=column,
                cmap=cmap,
                ax=ax,
                legend=True,
                legend_kwds={
                    'label': title,
                    'orientation': 'vertical',
                    'shrink': 0.5
                }
            )
        ax.set_title(title)
        ax.axis('off')

    def create_visualization(self, roads, output_path):
        import matplotlib.pyplot as plt

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
            fig.suptitle('NYC Cool Pavement Priority Analysis', fontsize=16)

            self.create_single_map(roads, 'traf_indx', 'Traffic Index', ax1, self.color_scheme['traffic'])
            self.create_single_map(roads, 'heat_indx', 'Heat Hazard Index', ax2, self.color_scheme['heat'])
            self.create_single_map(roads, 'tree_indx', 'Tree Canopy Index', ax3, self.color_scheme['tree'])

            ax4.set_title('Priority Segments')
            roads.plot(color='lightgray', alpha=0.2, ax=ax4)
            priority_roads = roads[roads['is_priority']]
            if len(priority_roads) > 0:
                priority_roads.plot(
                    color='red',
                    ax=ax4,
                    legend=True,
                    legend_kwds={
                        'label': 'Priority Segments',
                        'orientation': 'vertical',
                        'shrink': 0.5
                    }
                )
            ax4.axis('off')

            stats_text = (
                f"Total segments analyzed: {len(roads)}\n"
                f"High priority segments: {roads['is_priority'].sum()} "
                f"({(roads['is_priority'].mean()*100):.1f}%)\n"
                f"Segments with bike lanes: {roads['BikeLane'].sum()} "
                f"({(roads['BikeLane'].mean()*100):.1f}%)\n"
                f"Mean bus proximity: {roads['BusProxInx'].mean():.3f}"
            )
            plt.figtext(0.02, 0.02, stats_text, fontsize=10, ha='left')

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()

        except Exception as e:
            print(f"Error in visualization creation: {str(e)}")
            raise


# ------------------------------------
# Export Results + Webmap
# ------------------------------------

def export_results(results_dict, config):
    """Export analysis results to GeoJSON files with raw and index values."""
    exported_paths = {}

    for scenario_name, final_roads in results_dict.items():
        if scenario_name == 'ALL':
            continue

        try:
            print(f"\nProcessing exports for scenario: {scenario_name}")

            # Define fields to export
            export_fields = [
                # Basic identifiers
                'Street', 'StreetCode',

                # Raw values
                'vol_adj',           # Traffic volume
                'pav_rate',          # Pavement rating
                'heat_mean',         # Heat hazard
                'tree_pct',          # Tree canopy percentage
                'hvi_raw',           # Heat vulnerability
                'bus_distance',      # Distance to bus stops
                'BikeLane',          # Bike lane presence
                'pop_density',       # Population density

                # Index values
                'traf_indx',         # Traffic index
                'pave_indx',         # Pavement index
                'heat_indx',         # Heat hazard index
                'tree_indx',         # Tree canopy index
                'vuln_indx',         # Vulnerability index
                'BusProxInx',        # Bus proximity index
                'pop_dens_indx',     # Population density index

                # Priority results
                'priority',
                'is_priority'
            ]

            # Filter to available columns
            available_fields = [f for f in export_fields if f in final_roads.columns]

            # Create export dataframe
            export_gdf = final_roads[final_roads['is_priority']][
                available_fields + ['geometry']].copy()

            # Export to GeoJSON
            output_filename = f"priority_segments_{scenario_name}.geojson"
            output_path = os.path.join(config.output_dir, output_filename)
            export_gdf.to_file(output_path, driver='GeoJSON')

            exported_paths[scenario_name] = output_path
            print(f"Exported {len(export_gdf)} priority segments to {output_filename}")

        except Exception as e:
            print(f"Error exporting {scenario_name}: {str(e)}")

    return exported_paths

# ------------------------------------
# Neighborhood Analysis
# ------------------------------------

def run_neighborhood_analysis(config):
    """Run analysis for each neighborhood separately."""
    import os
    import geopandas as gpd

    from data_preprocessing import load_and_preprocess_roads
    from analysis_modules import (
        DataProcessors,
        OptimizedRasterProcessing,
        process_bus_stops,
        merge_street_segments,
        FinalAnalysis,
        PriorityVisualization,
        export_results,
        incorporate_population_density,
        build_webmap
    )

    # Load neighborhood boundaries
    nbhd_path = os.path.join(config.input_dir, "CSC_Neighborhoods.geojson")
    if not os.path.exists(nbhd_path):
        raise FileNotFoundError(f"Neighborhood boundaries not found at: {nbhd_path}")

    neighborhoods = gpd.read_file(nbhd_path)
    if neighborhoods.crs is None:
        neighborhoods.set_crs("EPSG:2263", inplace=True)
    elif neighborhoods.crs.to_string() != "EPSG:2263":
        neighborhoods = neighborhoods.to_crs("EPSG:2263")

    # Load and preprocess citywide roads
    roads = load_and_preprocess_roads(config)

    # Ensure roads are in the correct CRS
    if roads.crs is None:
        roads.set_crs("EPSG:2263", inplace=True)
    elif roads.crs.to_string() != "EPSG:2263":
        roads = roads.to_crs("EPSG:2263")

    # Process all attributes for the full dataset
    raster_processor = OptimizedRasterProcessing()
    roads = DataProcessors.batch_process_all(roads, config.input_dir)

    # Process temperature
    temp_raster_path = os.path.join(config.input_dir, 'Landsat9_ThermalComposite_ST_B10_2020-2023.tif')
    roads = raster_processor.optimize_temperature_processing(roads, temp_raster_path)
    roads['heat_indx'] = DataProcessors.normalize_to_index(roads['heat_mean'], 'temperature')

    # Process tree canopy
    canopy_raster_path = os.path.join(config.input_dir, 'NYC_TreeCanopy.tif')
    roads = raster_processor.optimize_tree_canopy_processing(roads, canopy_raster_path)

    # Process bus stops
    roads = process_bus_stops(roads, config.input_dir, config.crs)

    # Process population density
    roads = incorporate_population_density(
        roads,
        config.input_dir,
        buffer_ft=config.analysis_params["pop_buffer_ft"]
    )

    # Initialize analyzer
    analyzer = FinalAnalysis(config.weight_scenarios)

    # Process each neighborhood
    for idx, nbhd_row in neighborhoods.iterrows():
        nbhd_name = nbhd_row['Name']
        print(f"\nProcessing neighborhood: {nbhd_name}")

        # Create spatial index for efficiency
        roads_sindex = roads.sindex

        # Get candidate roads that might intersect with neighborhood
        nbhd_bounds = nbhd_row.geometry.bounds
        candidate_indices = list(roads_sindex.intersection(nbhd_bounds))
        candidate_roads = roads.iloc[candidate_indices]

        # Perform actual intersection check
        nbhd_roads = gpd.clip(candidate_roads, nbhd_row.geometry)

        if len(nbhd_roads) == 0:
            print(f"No roads found in neighborhood {nbhd_name}, skipping...")
            continue

        print(f"Found {len(nbhd_roads)} road segments in {nbhd_name}")

        # Merge segments
        nbhd_roads = merge_street_segments(nbhd_roads, config.analysis_params["min_segment_length"])
        print(f"After merging: {len(nbhd_roads)} segments")

        # Run analysis
        nbhd_results = analyzer.run_all_scenarios(nbhd_roads, config.analysis_params)

        # Create neighborhood-specific output directory
        nbhd_output_dir = os.path.join(config.output_dir, nbhd_name.replace(" ", "_"))
        os.makedirs(nbhd_output_dir, exist_ok=True)

        # Export results
        config_copy = config.copy()
        config_copy.output_dir = nbhd_output_dir
        exported_gdfs = export_results(nbhd_results, config_copy)

        # Generate webmap for this neighborhood
        try:
            scenario_geojsons = {}
            for scenario_name in nbhd_results.keys():
                if scenario_name != 'ALL':
                    geojson_path = os.path.join(
                        nbhd_output_dir,
                        f"priority_segments_{scenario_name}.geojson"
                    )
                    if os.path.exists(geojson_path):
                        scenario_geojsons[scenario_name] = geojson_path

            if scenario_geojsons:
                # Create neighborhood-specific config for correct boundary file reference
                nbhd_config = config_copy
                nbhd_config.input_dir = config.input_dir  # Keep original input dir for boundary file

                # Generate the webmap
                webmap_path = build_webmap(
                    scenario_geojsons,
                    nbhd_config,
                    neighborhood_name=nbhd_name
                )
                print(f"Generated webmap for {nbhd_name}: {webmap_path}")
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

        colors = [
            '#FFB366',  # light orange (lowest priority)
            '#FF9940', '#FF8533', '#FF6B1A',
            '#FF4D00', '#E63900', '#CC0000'   # red (highest priority)
        ]

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
                    • Traffic Volume: {format_value(properties.get('vol_adj'), '{:,.0f}')}<br>
                    • Pavement Rating: {format_value(properties.get('pav_rate'), '{:.1f}')}<br>
                    • Heat (°F): {format_value(properties.get('heat_mean'), '{:.1f}')}<br>
                    • Tree Canopy: {format_value(properties.get('tree_pct'), '{:.1f}')}%<br>
                    • Heat Vulnerability: {format_value(properties.get('hvi_raw'), '{:.2f}')}<br>
                    • Bus Stop Distance (ft): {format_value(properties.get('bus_distance'), '{:,.0f}')}<br>
                    • Bike Lane: {'Yes' if properties.get('BikeLane', 0) == 1 else 'No'}<br>
                    • Pop. Density: {format_value(properties.get('pop_density'), '{:,.0f}')}
                </div>

                <div style="margin-bottom: 10px;">
                    <b>Index Values</b><br>
                    • Traffic: {format_value(properties.get('traf_indx'), '{:.3f}')}<br>
                    • Pavement: {format_value(properties.get('pave_indx'), '{:.3f}')}<br>
                    • Heat: {format_value(properties.get('heat_indx'), '{:.3f}')}<br>
                    • Tree Canopy: {format_value(properties.get('tree_indx'), '{:.3f}')}<br>
                    • Vulnerability: {format_value(properties.get('vuln_indx'), '{:.3f}')}<br>
                    • Bus Proximity: {format_value(properties.get('BusProxInx'), '{:.3f}')}<br>
                    • Pop. Density: {format_value(properties.get('pop_dens_indx'), '{:.3f}')}
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

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; right: 50px; 
                background: white; padding: 10px; border: 2px solid grey;">
        <h4>Priority Score</h4>
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
    legend_html += "</div></div>"
    m.get_root().html.add_child(folium.Element(legend_html))

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
        f"cool_pavement_webmap{'_' + neighborhood_name.replace(' ', '_') if neighborhood_name else ''}.html"
    )
    m.save(html_map_path)
    print(f"Webmap saved to: {html_map_path}")

    return html_map_path



def generate_webmap(results_dict, exported_paths, config):
    """Generate the webmap using exported GeoJSON files."""
    try:
        print("\nGenerating interactive HTML map...")

        # Check if we have any valid GeoJSON files
        valid_geojsons = {
            scenario: path for scenario, path in exported_paths.items()
            if os.path.exists(path)
        }

        if not valid_geojsons:
            print("Warning: No GeoJSON files found for HTML map generation")
            return None

        # Generate the webmap
        webmap_path = build_webmap(valid_geojsons, config)
        return webmap_path

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

