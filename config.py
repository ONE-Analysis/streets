# config.py
import os

class CoolCorridorsConfig:
    def __init__(self, analysis_type='citywide'):
        try:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            self.script_dir = os.getcwd()

        # Directories and CRS
        self.input_dir = os.path.join(self.script_dir, 'input')
        self.output_dir = os.path.join(self.script_dir, 'output')
        self.temp_dir = os.path.join(self.output_dir, 'temp')
        self.crs = 'EPSG:2263'

        # Base parameters common to all analyses
        base_params = {
            'min_vulnerability': 4,
            'min_street_width': 15,
            'rail_buffer': 30,
            'pop_buffer_ft': 1000,
            'CapitalProjectExclusionBuffer': 5
        }

        # Specific parameters for each analysis type
        citywide_specific = {
            'number_of_top_roads': 100,
            'min_segment_length': 1320,  # 1/4 mile
        }

        neighborhood_specific = {
            'number_of_top_roads': 20,
            'min_segment_length': 500,
        }

        self.analysis_params = base_params.copy()
        if analysis_type.lower() == 'citywide':
            self.analysis_params.update(citywide_specific)
        elif analysis_type.lower() == 'neighborhood':
            self.analysis_params.update(neighborhood_specific)

        # Define both weight scenarios – listing all indices even if some are 0.
        # Weight scenarios for analysis:
        self.weight_scenarios = {
            'CoolCorridors': {
                'PavementIndex': 0,
                'HeatHazIndex': 0,
                'HeatVulnerabilityIndex': 0,
                'BusDensInx': 0.1,
                'BikeLnIndx': 0,
                'PedIndex': 0.1,
                'pop_dens_indx': 0.15,
                'TreeCanopyIndex': 0.3,
                'ComIndex': 0.15,
                'SidewalkIndex': 0.1,
                'RoadWidthIndex': 0.1
            },
            'CoolPavement': {
                'PavementIndex': 0.25,
                'HeatHazIndex': 0,
                'HeatVulnerabilityIndex': 0.1,
                'BusDensInx': 0,
                'BikeLnIndx': 0.1,
                'PedIndex': 0.2,
                'pop_dens_indx': 0.25,
                'TreeCanopyIndex': 0.1,
                'ComIndex': 0,
                'SidewalkIndex': 0,
                'RoadWidthIndex': 0
            }
        }

        # Data dictionary for input datasets / metrics.
        self.dataset_info = {
            "PavementIndex": {
                "alias": "PavementIndex",
                "raw": "pav_rate",
                "name": "Pavement Rating",
                "description": "Prioritizes very high or very low ratings (Street Pavement Rating, DOT)",
                "prefix": "",
                "suffix": "",
                "hex": "#F4D03F"
            },
            "HeatHazIndex": {
                "alias": "HeatHazIndex",
                "raw": "Heat_Mean",
                "name": "Heat Hazard",
                "description": "Priorizizes higher heat hazard areas (daytime summer temperature, Landsat via Google Earth Engine)",
                "prefix": "",
                "suffix": "",
                "hex": "#CD6155"
            },
            "HeatVulnerabilityIndex": {
                "alias": "HeatVulnerabilityIndex",
                "raw": "hvi_raw",
                "name": "Heat Vulnerability",
                "description": "Prioritizes higher HVI areas (Heat Vulnerability Index)",
                "prefix": "",
                "suffix": "",
                "hex": "#DC7633"
            },
            "BusDensInx": {
                "alias": "BusDensInx",
                "raw": "BusStpDens",
                "name": "Bus Stop Density",
                "description": "Prioritizes higher density of bus stops",
                "prefix": "~",
                "suffix": " Stops per Mile",
                "hex": "#5499C7"
            },
            "BikeLnIndx": {
                "alias": "BikeLnIndx",
                "raw": "bike_length",
                "name": "Bike Lane Density",
                "description": "Prioritizes higher bike lane presence (NYC Bike Lanes)",
                "prefix": "~",
                "suffix": " ft per Mile",
                "hex": "#48C9B0"
            },
            "PedIndex": {
                "alias": "PedIndex",
                "raw": "PedRank",
                "name": "Pedestrian Demand",
                "description": "Prioritizes higher pedestrian demand rating (Pedestrian Mobility Plan)",
                "prefix": "",
                "suffix": "",
                "hex": "#5DADE2"
            },
            "pop_dens_indx": {
                "alias": "pop_dens_indx",
                "raw": "pop_density",
                "name": "Population Density",
                "description": "Prioritizes higher population density within 1000 ft (US Census, 2020)",
                "prefix": "~",
                "suffix": " People per Sq Mile",
                "hex": "#A569BD"
            },
            "TreeCanopyIndex": {
                "alias": "TreeCanopyIndex",
                "raw": "tree_pct",
                "name": "Tree Canopy Gap",
                "description": "Prioritizes lower tree canopy coverage (LiDAR-based 6in Land Cover, 2021)",
                "prefix": "~",
                "suffix": " % roadway canopy coverage",
                "hex": "#52BE80"
            },
            "ComIndex": {
                "alias": "ComIndex",
                "raw": "ComArea",
                "name": "Commercial Activity",
                "description": "Prioritizes higher commercial or mixed-use building areas within 1000 ft",
                "prefix": "~",
                "suffix": " sq ft",
                "hex": "#F5B041"
            },
            "SidewalkIndex": {
                "alias": "SidewalkIndex",
                "raw": "sidewalk_area",
                "name": "Sidewalk Availability",
                "description": "Prioritizes very wide and very small sidewalks",
                "prefix": "~",
                "suffix": " sq ft",
                "hex": "#95A5A6"
            },
            "RoadWidthIndex": {
                "alias": "RoadWidthIndex",
                "raw": "StreetWidth_Min",
                "name": "Road Width",
                "description": "Prioritizes higher-width roads (LION Roads Data)",
                "prefix": "",
                "suffix": " ft",
                "hex": "#566573"
            } 
        }

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def copy(self):
        import copy
        return copy.deepcopy(self)