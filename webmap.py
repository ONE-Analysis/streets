import os
import requests
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def build_webmap(scenario_geojsons, config, neighborhood_name=None):
    """
    Create an HTML folium map with:
      - A styled GeoJSON layer (with popups bound via onEachFeature using config.dataset_info),
      - A donut chart overlay of analysis weights (using config.dataset_info for labels, prefixes, suffixes, and hex colors),
      - A floating list of top roads for each scenario.
    """
    # Determine scenario name from the passed-in dictionary
    scenario_name = list(scenario_geojsons.keys())[0]
    
    # Try to match the scenario name to one of the keys in config.weight_scenarios.
    # For example, if scenario_name is "all_segments_CoolCorridors", extract "CoolCorridors".
    matched_key = None
    for key in config.weight_scenarios.keys():
        if key.lower() in scenario_name.lower():
            matched_key = key
            break
    if matched_key is None:
        matched_key = scenario_name
    scenario_weights = config.weight_scenarios.get(matched_key, {})
    print(f"\nBuilding layered HTML webmap for {matched_key}{' for ' + neighborhood_name if neighborhood_name else ''}...")
    
    def get_color(priority_score, feature_collection):
        if pd.isna(priority_score):
            return '#808080'
        colors = ['#FFE066', '#FFB84D', '#FF9933', '#FF7A1A', '#FF5C00', '#E63D00', '#CC0000']
        priorities = [
            f['properties'].get('priority')
            for f in feature_collection['features']
            if f['properties'].get('priority') is not None
        ]
        min_priority = min(priorities)
        max_priority = max(priorities)
        normalized_score = 0 if max_priority == min_priority else (priority_score - min_priority) / (max_priority - min_priority)
        idx = int(np.floor(normalized_score * (len(colors) - 1)))
        return colors[max(0, min(idx, len(colors) - 1))]
    
    def style_function(feature, feature_collection):
        priority = feature['properties'].get('priority', None)
        return {
            'color': get_color(priority, feature_collection),
            'weight': 3,
            'opacity': 0.8
        }
    
    def create_popup_content(properties, dataset_info, scenario_weights):
        """
        Build popup (or tooltip) content using the data dictionary and the active weight scenario.
        Only include metrics with nonzero weights.
        
        Structure:
          [Street Name]
          Priority Index: [priority]
          
          Input Values:
            • [name]: [prefix][raw value (2dp)][suffix] ([index value (3dp)])
              - with [name] in the specified hex color.
        """
        street_name = properties.get('Street', 'Unknown')
        pr = properties.get('priority')
        try:
            priority = f"{float(pr):.3f}" if pr not in (None, '', 'N/A') else 'N/A'
        except Exception:
            priority = 'N/A'
        header = f"<h4 style='margin-bottom:5px;'>{street_name}</h4>"
        priority_line = f"<strong>Priority Index:</strong> {priority}"
        
        input_lines = ""
        for key, weight in scenario_weights.items():
            if weight == 0:
                continue
            info = dataset_info.get(key)
            if not info:
                continue
            raw_field = info.get('raw')
            raw_val = properties.get(raw_field)
            if raw_val is None:
                raw_val = properties.get(key)
            if raw_val in (None, '', 'N/A'):
                raw_val_disp = 'N/A'
            else:
                try:
                    raw_val_disp = f"{float(raw_val):.2f}"
                except Exception:
                    raw_val_disp = str(raw_val)
            index_val = properties.get(key)
            if index_val is None:
                index_val = raw_val
            if index_val in (None, '', 'N/A'):
                index_val_disp = 'N/A'
            else:
                try:
                    index_val_disp = f"{float(index_val):.3f}"
                except Exception:
                    index_val_disp = str(index_val)
            colored_name = f"<span style='color: {info.get('hex')}; font-weight:bold;'>{info.get('name')}</span>"
            line = f"{colored_name}: {info.get('prefix','')}{raw_val_disp}{info.get('suffix','')} ({index_val_disp})"
            input_lines += f"<li style='margin-bottom:2px;'>{line}</li>"
        
        input_values = f"<strong>Input Values:</strong><ul style='margin: 0; padding-left:15px;'>{input_lines}</ul>"
        popup_html = f"<div style='font-family: Helvetica;'>{header}<p>{priority_line}</p><p>{input_values}</p></div>"
        return popup_html

    def bind_popup_to_feature(feature, layer, dataset_info, scenario_weights):
        popup_content = create_popup_content(feature['properties'], dataset_info, scenario_weights)
        layer.bindPopup(popup_content)
    
    # Process neighborhoods (if available)
    bounds_list = []
    neighborhoods_path = os.path.join(config.input_dir, "CSC_Neighborhoods.geojson")
    neighborhoods_4326 = None
    if os.path.exists(neighborhoods_path):
        try:
            neighborhoods_gdf = gpd.read_file(neighborhoods_path)
            if neighborhoods_gdf.crs is None:
                neighborhoods_gdf.set_crs("EPSG:2263", inplace=True)
            neighborhoods_4326 = neighborhoods_gdf.to_crs("EPSG:4326")
            if neighborhood_name:
                neighborhoods_4326 = neighborhoods_4326[neighborhoods_4326['Name'] == neighborhood_name]
        except Exception as e:
            print(f"Warning: Could not process neighborhoods file: {str(e)}")
    if neighborhoods_4326 is not None and not neighborhoods_4326.empty:
        bounds_list.append(neighborhoods_4326.total_bounds)
    
    # Process the scenario GeoJSON files and add a tooltip column if not present.
    scenario_data = {}
    for s_name, reproj_path in scenario_geojsons.items():
        if not os.path.exists(reproj_path):
            continue
        try:
            gdf = gpd.read_file(reproj_path)
            if gdf.crs is None:
                gdf.set_crs("EPSG:2263", inplace=True)
            gdf_4326 = gdf.to_crs("EPSG:4326")
            if "tooltip" not in gdf_4326.columns:
                if "Street" in gdf_4326.columns and "priority" in gdf_4326.columns:
                    gdf_4326["tooltip"] = gdf_4326.apply(
                        lambda row: f"<b>{row['Street']}</b>: {row['priority']:.3f}", axis=1
                    )
                else:
                    gdf_4326["tooltip"] = "No tooltip info"
            bounds_list.append(gdf_4326.total_bounds)
            scenario_data[s_name] = gdf_4326
        except Exception as e:
            print(f"Warning: Could not process {s_name}: {str(e)}")
            continue

    if bounds_list:
        bounds_array = np.array(bounds_list)
        center_lat = (bounds_array[:, 1].min() + bounds_array[:, 3].max()) / 2
        center_lon = (bounds_array[:, 0].min() + bounds_array[:, 2].max()) / 2
    else:
        center_lat, center_lon = 40.7128, -74.0060

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14 if neighborhood_name else 12,
        tiles="CartoDB Positron",
        zoom_control=False
    )

    # ---------------------------
    # Add meta viewport for mobile responsiveness
    meta_viewport = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
    m.get_root().html.add_child(folium.Element(meta_viewport))

    # ---------------------------
    # Add responsive CSS for fixed overlays
    # The mobile media query now hides the donut chart and top roads lists, and makes the legend smaller.
    responsive_css = """
    <style>
    /* Base styles for overlays */
    .resilience-title, .analysis-text, .legend-box, .donut-overlay, .top-roads-container {
        position: fixed;
        background-color: white;
        border: 2px solid grey;
        border-radius: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        font-family: Helvetica, sans-serif;
        z-index: 1000;
        padding: 10px;
    }
    .resilience-title {
        top: 20px;
        left: 20px;
        font-size: 30px;
        font-weight: bold;
    }
    .analysis-text {
        bottom: 20px;
        left: 20px;
        font-size: 30px;
    }
    .legend-box {
        bottom: 20px;
        right: 20px;
        padding: 12px;
        font-size: 12px;
        width: 200px;
    }
    .donut-overlay {
        top: 100px;
        left: 20px;
        font-size: 25px;
        text-align: center;
    }
    .top-roads-container {
        top: 20px;
        right: 20px;
        width: 200px;
        max-height: 350px;
    }

    /* Mobile adjustments: hide donut chart and top roads, and make legend smaller */
    @media (max-width: 600px) {
        .resilience-title {
            top: 5px;
            left: 5px;
            font-size: 18px;
            padding: 10px;
            border-width: 1px;
            border-radius: 10px;
        }
        .analysis-text {
            bottom: 5px;
            left: 5px;
            font-size: 12px;
            padding: 10px;
            border-width: 1px;
            border-radius: 10px;
        }
        .legend-box {
            top: 60px;
            left: 5px;
            font-size: 9px;
            padding: 10px;
            border-width: 1px;
            border-radius: 10px;
            width: 150px;
            height: 270px;
        }
        .donut-overlay, .top-roads-container {
            display: none !important;
        }
    }

    /* Tooltip CSS for donut legend */
    .tooltip-container {
        position: relative;
        display: inline-block;
    }
    .tooltip {
        visibility: hidden;
        position: absolute;
        left: 25px;
        background-color: white;
        color: #333;
        padding: 10px;
        border-radius: 20px;
        font-size: 8pt;
        width: 200px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid #ddd;
        z-index: 1001;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip-container:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }
    .legend-item {
        position: relative;
        white-space: nowrap;
    }
    .info-icon {
        color: inherit;
        font-weight: bold;
    }
    </style>
    """
    m.get_root().html.add_child(folium.Element(responsive_css))

    # ---------------------------
    # Add Title overlay
    title_html = f'''
    <div class="resilience-title">
        {list(scenario_geojsons.keys())[0]} Roads
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # ---------------------------
    # Add ONE Analysis logo text box
    analysis_text_html = f'''
    <div class="analysis-text">
        <span style="font-family: 'Futura', sans-serif; font-weight: bold; color: #4c5da4;">one</span>
        <span style="font-family: 'Futura', sans-serif; font-weight: 300; color: #4c5da4;"> analysis</span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(analysis_text_html))

    # ---------------------------
    # Add data layers
    # FOZ layer
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
    
    # Persistent Poverty layer
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
    
    # Zoning Map Adjustments layer
    nyzma_path = os.path.join(config.input_dir, 'nyzma_since2020.geojson')
    if os.path.exists(nyzma_path):
        try:
            nyzma = gpd.read_file(nyzma_path)
            if nyzma.crs is None or nyzma.crs != "EPSG:4326":
                nyzma = nyzma.to_crs("EPSG:4326")
            folium.GeoJson(
                nyzma,
                name="Zoning Map Adjustments (2020-Present)",
                style_function=lambda x: {
                    'fillColor': 'cyan',
                    'color': 'cyan',
                    'weight': 0.5,
                    'fillOpacity': 0.2,
                    'opacity': 1.0
                }
            ).add_to(m)
        except Exception as e:
            print(f"Warning: Could not process nyzma file: {str(e)}")
    
    # Unplantable Roads Layer
    unplantable_path = os.path.join(config.input_dir, 'roads_unplantable_DPR.geojson')
    if os.path.exists(unplantable_path):
        try:
            unplantable = gpd.read_file(unplantable_path)
            if unplantable.crs is None or unplantable.crs != "EPSG:4326":
                unplantable = unplantable.to_crs("EPSG:4326")
            folium.GeoJson(
                unplantable,
                name="Zoning Map Adjustments (2020-Present)",
                style_function=lambda x: {
                    'color': 'green',
                    'weight': 5,
                    'opacity': 0.4
                }
            ).add_to(m)
        except Exception as e:
            print(f"Warning: Could not process unplantable file: {str(e)}")
    
    # Neighborhoods layer
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
    
    # FEMA Community Disaster Resilience Zones layer
    try:
        fema_url = "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/FEMA_Community_Disaster_Resilience_Zones/FeatureServer/0/query"
        fema_params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson"
        }
        fema_response = requests.get(fema_url, params=fema_params)
        fema_geojson = fema_response.json()
        folium.GeoJson(
            fema_geojson,
            name="FEMA Community Disaster Resilience Zones",
            style_function=lambda feature: {
                'fillColor': 'purple',
                'color': 'purple',
                'weight': 1,
                'fillOpacity': 0.1,
            }
        ).add_to(m)
    except Exception as e:
        print(f"Warning: Could not fetch FEMA layer: {str(e)}")
    
    # ---------------------------
    # Add the roads with custom tooltip using popup content as tooltip
    if scenario_data:
        gdf_4326 = scenario_data[scenario_name].copy()
        def compute_tooltip(row):
            props = row.to_dict()
            props.pop('geometry', None)
            return create_popup_content(props, config.dataset_info, scenario_weights)
        gdf_4326['tooltip'] = gdf_4326.apply(compute_tooltip, axis=1)
        geojson_data = gdf_4326.__geo_interface__
        def style_callback(feature):
            return style_function(feature, geojson_data)
        gjson = folium.GeoJson(
            gdf_4326,
            name=scenario_name,
            style_function=style_callback,
            tooltip=folium.GeoJsonTooltip(
                fields=['tooltip'],
                aliases=[''],
                sticky=False,
                labels=False,
                parse_html=True,
                style="""
                    background-color: white;
                    border: 2px solid black;
                    border-radius: 15px;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
                """
            )
        )
        gjson.add_to(m)
    
    # ---------------------------
    # Add Legend overlay
    legend_html = """
    <div class="legend-box overlay">
        <h4 style="margin-bottom: 5px; font-size: 17px; font-weight: bold;">Legend</h4>
        <div style="display: flex; flex-direction: column; gap: 10px;">
            <!-- Road Priority Score: Single Horizontal Gradient -->
            <div>
                <h5 style="margin: 5px 0; font-weight: bold;">Road Priority Score</h5>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <div style="width: 120px; height: 10px; border-radius: 5px; background: linear-gradient(to right, #FFB366, #CC0000);"></div>
                    <div style="display: flex; justify-content: space-between; width: 120px;">
                        <span>lower</span>
                        <span>higher</span>
                    </div>
                </div>
            </div>
            <!-- Area Overlays -->
            <div>
                <h5 style="margin: 5px 0; font-weight: bold;">Area Overlays</h5>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: rgba(128, 128, 128, 0.2); border: 2px solid gray; border-radius: 3px;"></div>
                        <span>CSC Neighborhoods</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 5px; background: green; opacity: 0.4; border: 1px solid green;"></div>
                        <span>Unplantable Roads (DPR)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px;background: rgba(0, 128, 0, 0.2); border: 1px solid green; border-radius: 3px;"></div>
                        <span>Persistent Poverty Areas</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: rgba(0, 0, 255, 0.2); border: 1px solid blue; border-radius: 3px;"></div>
                        <span>Federal Opportunity Zones</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: rgba(0, 255, 255, 0.2); border: 1px solid cyan; border-radius: 3px;"></div>
                        <span>NYZMA (2020-Present)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: rgba(128, 0, 128, 0.2); border: 1px solid purple; border-radius: 3px;"></div>
                        <span>FEMA CDRZ</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # ---------------------------
    # Donut Chart Overlay with Interactive Info Icons in Legend
    active_items = [(k, v) for k, v in scenario_weights.items() if v != 0 and k in config.dataset_info]
    if active_items:
        sizes = [v for k, v in active_items]
        colors = [config.dataset_info[k]['hex'] for k, v in active_items]
        fig, ax = plt.subplots(figsize=(3, 3), dpi=90)
        wedges, texts, autotexts = ax.pie(
             sizes,
             labels=None,
             autopct=lambda pct: f"{int(round(pct))}%",
             pctdistance=0.75,
             startangle=90,
             wedgeprops={'width': 0.5, 'edgecolor': 'white'},
             colors=colors
        )
        ax.set_aspect("equal")
        plt.setp(autotexts, size=12, fontfamily='Verdana', weight='bold', color='black', va='center', ha='center')
        svg_buf = io.BytesIO()
        plt.savefig(svg_buf, format='svg', transparent=True, bbox_inches='tight')
        svg_buf.seek(0)
        svg_data = svg_buf.read().decode('utf-8')
        svg_buf.close()
        plt.close(fig)
        
        legend_items_html = '<div style="margin-top:10px;">'
        for k, v in active_items:
            info = config.dataset_info[k]
            label = info['name']
            description = info.get('description', 'No description available.')
            tooltip_content = f"""
                <div class='tooltip-content'>
                    <strong>{label}</strong><br>
                    {description}
                </div>
            """
            legend_items_html += f"""
                <div class="legend-item" style="font-size:12pt; color:{info['hex']}; margin-bottom:3px; margin-left:30px; display:flex; align-items:center;">
                    <div class="tooltip-container">
                        <span class="info-icon" style="cursor:pointer; margin-right:5px; font-size:12pt;">&#9432;</span>
                        <div class="tooltip">{tooltip_content}</div>
                    </div>
                    <span>{label}</span>
                </div>
            """
        legend_items_html += '</div>'
        donut_combined = svg_data + legend_items_html
    else:
        donut_combined = "<svg></svg>"
    
    donut_html = f'''
    <div class="donut-overlay overlay">
        <h4 style="margin-top: 0; margin-bottom: 0; font-weight: bold; font-size: 20px;">Analysis Weights</h4>
        {donut_combined}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(donut_html))
    
    # ---------------------------
    # Add list of top roads for each scenario
    if scenario_data:
        offset = 20  # starting offset (in pixels) from the top for the first container
        for scenario_name, gdf_4326 in scenario_data.items():
            gdf_4326['Street'] = gdf_4326['Street'].fillna("Unknown").str.strip().str.title()
            if "priority" in gdf_4326.columns:
                topx_gdf = gdf_4326.nlargest(100, "priority")
            else:
                continue
            unique_streets = []
            seen = set()
            for _, row in topx_gdf.iterrows():
                street_name = row.get("Street", "Unknown")
                if street_name not in seen:
                    seen.add(street_name)
                    unique_streets.append(street_name)
            roads_html = "".join(f"<li>{st}</li>" for st in unique_streets)
            topx_html = f"""
            <div class="top-roads-container overlay" style="top: {offset}px;">
                <h4 style="margin-top: 0; margin-bottom: 5px; font-size: 14px; font-weight: bold;">
                    Top Roads (by Priority)
                </h4>
                <div style="max-height: 250px; overflow-y: auto; padding-right: 10px;">
                    <ul style="margin: 0; padding-left: 20px; font-size: 12px; list-style-type: disc;">
                        {roads_html}
                    </ul>
                </div>
            </div>
            """
            m.get_root().html.add_child(folium.Element(topx_html))
            offset += 400

    html_map_path = os.path.join(
        config.output_dir,
        f"{matched_key}_webmap{'_' + neighborhood_name.replace(' ', '_') if neighborhood_name else ''}.html"
    )
    m.save(html_map_path)
    print(f"Webmap saved to: {html_map_path}")
    return html_map_path

def generate_webmap(results_dict, exported_paths, config):
    """Generate the webmap using exported GeoJSON files."""
    try:
        print("\nGenerating interactive HTML maps...")
        valid_geojsons = {scenario: path for scenario, path in exported_paths.items() if os.path.exists(path)}
        if not valid_geojsons:
            print("Warning: No GeoJSON files found for HTML map generation")
            return None
        webmap_paths = []
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

def run_exports_and_webmap(results_dict, config):
    """Run the full export and webmap generation process."""
    try:
        exported_paths = export_results(results_dict, config)
        if exported_paths:
            webmap_path = generate_webmap(results_dict, exported_paths, config)
            if webmap_path:
                print(f"Successfully generated webmap at: {webmap_path}")
        else:
            print("No results were exported, skipping webmap generation")
    except Exception as e:
        print(f"Error in export and webmap process: {str(e)}")