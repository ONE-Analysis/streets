import os
import requests
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# ------------------------------------
# Webmap Builder
# ------------------------------------
def build_webmap(scenario_geojsons, config, neighborhood_name=None):
    """
    Create a single HTML folium map with heat-map style visualization 
    and detailed popups showing raw and index values.
    """

    print(f"\nBuilding layered HTML webmap{' for ' + neighborhood_name if neighborhood_name else ''}...")

    def get_color(priority_score, feature_collection):
        """Returns color based on priority score normalization."""
        if pd.isna(priority_score):
            return '#808080'  # gray for missing values

        colors = ['#FFE066', '#FFB84D', '#FF9933', '#FF7A1A', '#FF5C00', '#E63D00', '#CC0000']

        priorities = [
            f['properties'].get('priority')
            for f in feature_collection['features']
            if f['properties'].get('priority') is not None
        ]
        min_priority = min(priorities)
        max_priority = max(priorities)

        if max_priority == min_priority:
            normalized_score = 0
        else:
            normalized_score = (priority_score - min_priority) / (max_priority - min_priority)

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
            <div style="font-family: Helvetica; min-width: 200px; max-width: 300px;">
                <h4 style="margin-bottom: 10px; border-bottom: 1px solid #ccc;">
                    {properties.get('Street', 'N/A')}
                </h4>

                <div style="margin-bottom: 10px;">
                    <b>Raw Values</b><br>
                    • Average Pavement Rating: {format_value(properties.get('pav_rate'), '{:.1f}')}<br>
                    • Average Daytime Summer Heat: {format_value(properties.get('heat_mean'), '{:.1f}')} °F<br>
                    • Tree Canopy Roadway Coverage: {format_value(properties.get('tree_pct'), '{:.1f}')}%<br>
                    • Heat Vulnerability Index Average: {format_value(properties.get('hvi_raw'), '{:.2f}')}<br>
                    • Bus Stop Density: ~{format_value(properties.get('BusStpDens'), '{:,.0f}')} Stops per Mile<br>
                    • Bike Lane Density: {format_value(properties.get('bike_length', 0), '{:,.0f}')} ft Bike Lane per Mile<br>
                    • Pedestrian Demand Priority: {format_value(properties.get('PedRank', 0), '{:,.0f}')}<br>
                    • Population Density: ~{format_value(properties.get('pop_density'), '{:,.0f}')} People per Square Mile<br>
                    • Commercial Area: ~{format_value(properties.get('ComArea'), '{:,.0f}')} sq ft within 1000ft per ft of road
                    • Sidewalk Area: ~{format_value(properties.get('sidewalk_area'), '{:,.0f}')} sq ft
                    • Road Width: ~{format_value(properties.get('StreetWidth_Min'), '{:,.0f}')} ft
                </div>

                <div style="margin-bottom: 10px;">
                    <b>Index Values</b><br>
                    • Tree Canopy: {format_value(properties.get('tree_indx'), '{:.3f}')}<br>
                    • Bus Stops: {format_value(properties.get('BusDensInx'), '{:.3f}')}<br>
                    • Population Density: {format_value(properties.get('pop_dens_indx'), '{:.3f}')}<br>
                    • Commercial Area: {format_value(properties.get('ComIndex'), '{:.3f}')}<br>
                    • Sidewalk: {format_value(properties.get('SidewalkIndex'), '{:.3f}')}<br>
                    • Road Width: {format_value(properties.get('RoadWidthIndex'), '{:.3f}')}
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
                neighborhoods_4326 = neighborhoods_4326[neighborhoods_4326['Name'] == neighborhood_name]
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
        tiles="CartoDB Positron",
        zoom_control=False  # Disable default zoom control
    )

    # Add title
    title_html = f'''
    <div style="position: fixed; 
                top: 30px; 
                left: 30px; 
                z-index: 1000;
                background-color: white;
                padding: 10px;
                border: 2px solid grey;
                border-radius: 20px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
                font-size: 30px;
                font-weight: bold;
                font-family: Helvetica;">
        {list(scenario_geojsons.keys())[0]} Roads
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # -------------------------------
    # Add ONE Analysis logo text box
    # -------------------------------
    analysis_text_html = f'''
    <div style="
        position: fixed;
        top: 110px;  /* Adjust vertical position as needed */
        left: 30px;
        z-index: 1000;
        background-color: white;
        padding: 10px;
        border: 2px solid grey;
        border-radius: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        font-size: 30px;
    ">
        <span style="font-family: Futura Bold, sans-serif; color: #4c5da4;">one</span>
        <span style="font-family: Futura Light, sans-serif; color: #4c5da4;"> analysis</span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(analysis_text_html))

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

    # Add Zoning Map Adjustments layer
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

    # Add FEMA Community Disaster Resilience Zones layer
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

    # Add Legend
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 30px; 
        right: 30px; 
        z-index: 1000;
        background: white; 
        padding: 12px; 
        border: 2px solid grey;
        border-radius: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        ">
        <h4 style="margin-bottom: 10px; font-size: 20px; font-weight: bold;">Legend</h4>
        <div style="display: flex; flex-direction: column; gap: 10px;">
            <!-- Road Priority Score: Single Horizontal Gradient -->
            <div>
                <h5 style="margin: 5px 0;">Road Priority Score</h5>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <!-- Gradient bar from 0 to 1 -->
                    <div style="width: 120px; height: 10px; background: linear-gradient(to right, #FFB366, #CC0000);"></div>
                    <!-- Labels for 0 and 1 -->
                    <div style="display: flex; justify-content: space-between; width: 120px;">
                        <span>0</span>
                        <span>1</span>
                    </div>
                </div>
            </div>
            <!-- Area Overlays -->
            <div>
                <h5 style="margin: 5px 0;">Area Overlays</h5>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: gray; opacity: 0.2; border: 1px solid gray;"></div>
                        <span>CSC Neighborhoods</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: green; opacity: 0.2; border: 1px solid green;"></div>
                        <span>Persistent Poverty Areas</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: blue; opacity: 0.2; border: 1px solid blue;"></div>
                        <span>Federal Opportunity Zones</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: cyan; opacity: 0.2; border: 1px solid cyan;"></div>
                        <span>NYZMA (2020-Present)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 5px;">
                        <div style="width: 20px; height: 20px; background: purple; opacity: 0.2; border: 1px solid purple;"></div>
                        <span>FEMA CDRZ</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    legend = folium.Element(legend_html)
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
                    border-radius: 15px;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
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

    # -------------------------------
    # Insert the new donut chart overlay (SVG)
    # -------------------------------
    # Automatically pull non-zero values from the CoolCorridors weight scenario in config
    cc_weights = config.weight_scenarios['CoolCorridors']
    friendly_names = {
        'PavementIndex': 'Pavement',
        'HeatHazIndex': 'Heat Hazard',
        'TreeCanopyIndex': 'Tree Canopy Gap',
        'HeatVulnerabilityIndex': 'Heat Vulnerability',
        'BusDensInx': 'Bus Density',
        'BikeLnIndx': 'Bike Lane',
        'PedIndex': 'Pedestrian',
        'pop_dens_indx': 'Population Density',
        'ComIndex': 'Commercial Density',
        'SidewalkIndex': 'Sidewalk',
        'RoadWidthIndex': 'Road Width'
    }
    # Filter out keys with a zero value and map keys to friendly names
    cc_data = {friendly_names.get(k, k): v for k, v in cc_weights.items() if v != 0}

    cc_labels = list(cc_data.keys())
    cc_sizes = list(cc_data.values())

    # Create a donut chart in Matplotlib
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=90)  # Increase if you want it larger
    wedges, _, autotexts = ax.pie(
        cc_sizes,
        labels=None,  # We'll use a legend instead of wedge labels
        autopct=lambda pct: f"{int(round(pct))}%",  # Whole-number percentages
        pctdistance=0.8,
        startangle=90,
        wedgeprops={'width': 0.5, 'edgecolor': 'white'}
    )
    ax.set_aspect("equal")

    # Style the percentage labels
    plt.setp(
        autotexts,
        size=10,
        fontfamily='Helvetica',
        weight='bold',
        color='black',
        va='center'
    )

    # Add a legend below the donut
    ax.legend(
        wedges,
        cc_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=1,
        fontsize=9
    )

    plt.tight_layout()

    # Convert the figure to an SVG in memory
    svg_buf = io.BytesIO()
    plt.savefig(svg_buf, format='svg', transparent=True, bbox_inches='tight')
    svg_buf.seek(0)
    svg_data = svg_buf.read().decode('utf-8')
    svg_buf.close()
    plt.close(fig)

    # Wrap the raw SVG in a white container with rounded corners + bolded title
    donut_html = f'''
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 1000;
        background: white;
        padding: 10px;
        border: 2px solid grey;
        border-radius: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        font-family: 'Helvetica', sans-serif;
    ">
        <h4 style="margin-top: 0; margin-bottom: 8px; font-weight: bold; font-size: 20px; text-align: center;">Analysis Weights</h4>
        {svg_data}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(donut_html))
    # -------------------------------

    # ---------------------------------------
    # ADD LIST OF TOP x ROADS TO MAP
    # ---------------------------------------
    if scenario_data:
        # For example, pick the first scenario in the dictionary:
        scenario_name = list(scenario_data.keys())[0]
        gdf_4326 = scenario_data[scenario_name]

        # 1) Remove duplicates, convert Street to title case
        #    (Here we assume 'Street' may have duplicates or inconsistent casing.)
        gdf_4326['Street'] = gdf_4326['Street'].fillna("Unknown").str.strip().str.title()
        
        #    Optionally, if you want only one row per street, keep the highest priority:
        #    gdf_4326 = gdf_4326.sort_values('priority', ascending=False)
        #    gdf_4326 = gdf_4326.drop_duplicates(subset=['Street'], keep='first')

        # 2) Select the top x rows by 'priority'
        topx_gdf = gdf_4326.nlargest(100, "priority")

        # 3) Build a bullet list (no numbering) of unique street names from topx_gdf
        unique_streets = []
        seen = set()
        for _, row in topx_gdf.iterrows():
            street_name = row.get("Street", "Unknown")
            if street_name not in seen:
                seen.add(street_name)
                unique_streets.append(street_name)

        roads_html = ""
        for st in unique_streets:
            roads_html += f"<li>{st}</li>"

        # 4) Wrap the list in a styled container in the top-right corner
        # The inner div uses "max-height", "overflow-y: auto", and padding-right to create a fixed-height, scrollable list with extra space for the scrollbar.
        topx_html = f"""
        <div style="
            position: fixed;
            top: 30px;
            right: 30px;
            z-index: 1000;
            background: white;
            padding: 10px;
            border: 2px solid grey;
            border-radius: 20px;
            font-family: 'Helvetica', sans-serif;
            max-width: 250px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        ">
            <h4 style="margin-top: 0; margin-bottom: 8px; font-weight: bold;">
                Top Roads by Priority Index
            </h4>
            <div style="max-height: 350px; overflow-y: auto; padding-right: 10px;">
                <ul style="margin: 0; padding-left: 20px; font-size: 12px; list-style-type: disc;">
                    {roads_html}
                </ul>
            </div>
        </div>
        """

        # 5) Add the floating list to the map
        m.get_root().html.add_child(folium.Element(topx_html))

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

