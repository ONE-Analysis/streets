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
            'number_of_top_roads': 300,
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
                'BusDensInx': 0,
                'BikeLnIndx': 0,
                'PedIndex': 0.1,
                'pop_dens_indx': 0.2,
                'TreeCanopyIndex': 0.3,
                'ComIndex': 0.2,
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
                "name": "Pavement",
                "description": "Prioritizes very high or very low ratings (Street Pavement Rating, DOT)",
                "prefix": "",
                "suffix": "",
                "hex": "#E2E8F0"
            },
            "HeatHazIndex": {
                "alias": "HeatHazIndex",
                "name": "Heat Hazard",
                "description": "Priorizizes higher heat hazard areas (daytime summer temperature, Landsat via Google Earth Engine)",
                "prefix": "",
                "suffix": "",
                "hex": "#FDB5B5"
            },
            "HeatVulnerabilityIndex": {
                "alias": "HeatVulnerabilityIndex",
                "name": "Heat Vulnerability",
                "description": "Prioritizes higher HVI areas (Heat Vulnerability Index)",
                "prefix": "",
                "suffix": "",
                "hex": "#ebc5a9"
            },
            "BusDensInx": {
                "alias": "BusDensInx",
                "name": "Bus Stop Density",
                "description": "Prioritizes higher density of bus stops",
                "prefix": "~",
                "suffix": " Stops per Mile",
                "hex": "#93C5FD"
            },
            "BikeLnIndx": {
                "alias": "BikeLnIndx",
                "name": "Bike Lane Density",
                "description": "Prioritizes higher bike lane presence (NYC Bike Lanes)",
                "prefix": "",
                "suffix": " ft per Mile",
                "hex": "#BBF7D0"
            },
            "PedIndex": {
                "alias": "PedIndex",
                "name": "Pedestrian Demand",
                "description": "Prioritizes higher pedestrian demand rating (Pedestrian Mobility Plan)",
                "prefix": "",
                "suffix": "",
                "hex": "#BFDBFE"
            },
            "pop_dens_indx": {
                "alias": "pop_dens_indx",
                "name": "Population Density",
                "description": "Prioritizes higher population density within 1000 ft (US Census, 2020)",
                "prefix": "~",
                "suffix": " People per Sq Mile",
                "hex": "#DDD6FE"
            },
            "TreeCanopyIndex": {
                "alias": "TreeCanopyIndex",
                "name": "Tree Canopy Gap",
                "description": "Prioritizes lower tree canopy coverage (LiDAR-based 6in Land Cover, 2021)",
                "prefix": "",
                "suffix": " %",
                "hex": "#bcebc1"
            },
            "ComIndex": {
                "alias": "ComIndex",
                "name": "Commercial Activity",
                "description": "Prioritizes higher commercial or mixed-use building areas within 1000 ft",
                "prefix": "~",
                "suffix": " sq ft",
                "hex": "#FDE68A"
            },
            "SidewalkIndex": {
                "alias": "SidewalkIndex",
                "name": "Sidewalk Availability",
                "description": "Prioritizes very wide and very small sidewalks",
                "prefix": "",
                "suffix": " sq ft",
                "hex": "#e7f0b6"
            },
            "RoadWidthIndex": {
                "alias": "RoadWidthIndex",
                "name": "Road Width",
                "description": "Prioritizes higher-width roads (LION Roads Data)",
                "prefix": "",
                "suffix": " ft",
                "hex": "#94A3B8"
            } 
        }

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def copy(self):
        import copy
        return copy.deepcopy(self)