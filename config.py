# config.py
import os

class CoolPavementConfig:
    def __init__(self, analysis_type='citywide'):
        # Attempt to determine the script directory; fallback to the current working directory
        try:
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            self.script_dir = os.getcwd()

        # Directories, CRS, etc.
        self.input_dir = os.path.join(self.script_dir, 'input')
        self.output_dir = os.path.join(self.script_dir, 'output')
        self.temp_dir = os.path.join(self.output_dir, 'temp')
        self.crs = 'EPSG:2263'

        # Base parameters that are same for both analyses
        base_params = {
            'min_vulnerability': 4,
            'min_street_width': 15,
            'rail_buffer': 25,
            'pop_buffer_ft': 1000
        }

        # Specific parameters for each analysis type
        citywide_specific = {
            'number_of_top_roads': 200,     # This will select top 200 segments citywide
            'min_segment_length': 1500,     # Minimum segment length in feet
            'priority_threshold': 200        # Explicitly set the priority threshold
        }

        neighborhood_specific = {
            'number_of_top_roads': 50,      # This will select top 10 segments per neighborhood
            'min_segment_length': 1000,     # Minimum segment length in feet
            'priority_threshold': 50        # Explicitly set the priority threshold
        }

        # Set analysis parameters based on analysis type
        self.analysis_params = base_params.copy()
        if analysis_type.lower() == 'citywide':
            self.analysis_params.update(citywide_specific)
        elif analysis_type.lower() == 'neighborhood':
            self.analysis_params.update(neighborhood_specific)

        # Weight scenarios
        self.weight_scenarios = {
            'CoolPavement': {
                'PavementIndex': 0.25,           # uses 'v' shaped suitability
                'HeatHazIndex': 0.1,             # raw heat values from summers 2021-2024
                'TreeCanopyIndex': 0.15,         # assessment of tree canopy percentage
                'HeatVulnerabilityIndex': 0.15,  # heat vulnerability index values
                'BusDensInx': 0.05,              # bus stops per mile
                'BikeLnIndx': 0.05,              # bike lane length per mile
                'PedIndex': 0,                   # road prioritization in NYC Pedestrian Mobility Plan
                'pop_dens_indx': 0.1,            # people per square mile within 1000' buffer
                'ComIndex': 0.15                 # commercial space within 1000' buffer
            },
            'CoolCorridors': {
                'PavementIndex': 0,             # uses 'v' shaped suitability
                'HeatHazIndex': 0,              # raw heat values from summers 2021-2024
                'TreeCanopyIndex': 0.25,        # assessment of tree canopy percentage
                'HeatVulnerabilityIndex': 0,    # heat vulnerability index values
                'BusDensInx': 0.15,             # bus stops per mile
                'BikeLnIndx': 0.15,             # bike lane length per mile
                'PedIndex': 0.2,                # road prioritization in NYC Pedestrian Mobility Plan
                'pop_dens_indx': 0.25,          # people per square mile within 1000' buffer
                'ComIndex': 0                   # commercial space within 1000' buffer
            }
        }

        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def copy(self):
        """Create a copy of the configuration object."""
        import copy
        return copy.deepcopy(self)