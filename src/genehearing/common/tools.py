import yaml
import pandas as pd

def load_config(path="config.yaml"):
    # Read in the configuration file
    with open(path) as p:
        config = yaml.safe_load(p)
    return config

def parse_list(arg_str):
    """Konwertuje string 'a,b,c' na listę ['a','b','c']"""
    return [x.strip() for x in arg_str.split(',')] if arg_str else []


def parse_map(map_str):
    """
    Konwertuje string 'Arkusz1=klucz1,Arkusz2=klucz2' na słownik
    """
    mapping = {}
    for pair in map_str.split(','):
        sheet, key = pair.split('=')
        mapping[sheet.strip()] = key.strip()
    return mapping

def convert_to_datetime(df, column, suffix):
    col = f"{column}_{suffix}"
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df