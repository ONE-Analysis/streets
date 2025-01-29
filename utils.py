import os
import psutil

def standardize_street_names(df, street_col='Street'):
    """Standardize street names to handle common variations."""
    if street_col not in df.columns:
        print(f"Street column '{street_col}' not found in dataset")
        return df
    
    df = df.copy()
    df[street_col] = df[street_col].str.upper().str.strip()
    
    suffixes = {
        r'\bSTREET\b': 'ST',
        r'\bAVENUE\b': 'AVE',
        r'\bBOULEVARD\b': 'BLVD',
        r'\bROAD\b': 'RD',
        r'\bPLACE\b': 'PL',
        r'\bLANE\b': 'LN',
        r'\bDRIVE\b': 'DR',
        r'\bCOURT\b': 'CT',
        r'\bTERRACE\b': 'TER',
    }
    
    for pattern, replacement in suffixes.items():
        df[street_col] = df[street_col].str.replace(pattern, replacement, regex=True)
    
    df[street_col] = df[street_col].str.replace(r'[^\w\s]', '', regex=True)
    df[street_col] = df[street_col].str.replace(r'\s+', ' ', regex=True)
    df[street_col] = df[street_col].str.strip()
    
    return df


def log_memory_usage():
    """Log current CPU memory usage."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    print(f"CPU Memory Usage: {mem:.2f} MB")