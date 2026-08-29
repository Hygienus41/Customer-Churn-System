from pathlib import Path
import pandas as pd

def load_data(filepath):
    """
    Load a CSV dataset.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    return pd.read_csv(filepath)