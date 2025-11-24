import yaml
from src.exception.exception import Exception
import sys





def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    
    Parameters:
        file_path (str): Path to the YAML file.
    
    Returns:
        dict: Parsed YAML content as a dictionary.
    """
    try:
        with open(file_path, "rb") as yaml_file:  # Open the YAML file in binary read mode
            return yaml.safe_load(yaml_file)  # Parse the YAML content
    except Exception as e:
        raise Exception(e, sys) from e  
    
    



