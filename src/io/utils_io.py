import os
import pickle
from src.config import maxar_items_path, town_path, date_event, maxar_items_columns_path
import geopandas as gpd 



def make_path(file_name, *path):
    return os.path.join(*path, file_name)

def save_pickle(data, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
        
def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_maxar_items(type_item: str):
    columns = load_pickle(maxar_items_columns_path.format(type_item))
    gdf = gpd.read_file(maxar_items_path.format(type_item)).rename(columns, axis=1)
    gdf = gdf.assign(
        after_event=(gdf.datetime > date_event)*1
    )
    return gdf


def load_roi_town():
    return gpd.read_file(town_path)


def check_dir(*path):
    os.makedirs(os.path.join(*path), exist_ok=True)
    return os.path.join(*path)