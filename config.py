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
            'min_segment_length': 1320, # 1/4 mile
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

        # Cool Corridors weight scenario:
        self.weight_scenarios = {
            # 'CoolPavement': {
            #     'PavementIndex': 0.25,          # uses 'v' shaped suitability
            #     'HeatHazIndex': 0,            # raw heat values from summers 2021-2024
            #     'TreeCanopyIndex': 0.1,        # assessment of tree canopy percentage
            #     'HeatVulnerabilityIndex': 0.1, # heat vulnerability index values
            #     'BusDensInx': 0,             # bus stops per mile
            #     'BikeLnIndx': 0.1,             # bike lane length per mile
            #     'PedIndex': 0.2,                  # road prioritization in NYC Pedestrian Mobility Plan
            #     'pop_dens_indx': 0.25,          # people per square mile within 1000' buffer
            #     'ComIndex': 0                   # commercial space within 1000' buffer
            # },
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
            }
        }

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def copy(self):
        import copy
        return copy.deepcopy(self)