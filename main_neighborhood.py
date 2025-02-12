import logging
import warnings

from config import CoolCorridorsConfig
from analysis_modules import run_neighborhood_analysis

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    warnings.filterwarnings('ignore')

    # Setup configuration for neighborhood analysis using Cool Corridors
    config = CoolCorridorsConfig(analysis_type='neighborhood')
    run_neighborhood_analysis(config)

if __name__ == "__main__":
    main()