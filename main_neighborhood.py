import logging
import warnings

from config import CoolPavementConfig
from analysis_modules import run_neighborhood_analysis

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    warnings.filterwarnings('ignore')

    # 1) Setup config for 'neighborhood' analysis
    config = CoolPavementConfig(analysis_type='neighborhood')

    # 2) Run neighborhood analysis
    run_neighborhood_analysis(config)

if __name__ == "__main__":
    main()