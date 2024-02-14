import os
import pickle
from src.config import IMG_TEST, maxar_items_path, town_path, date_event, maxar_items_columns_path, interim_dir_path, VALID_MODE
from src.utils import create_bbox
import xarray as xr
from typing import Dict, List, Any, Union
import glob 
import logging
import geopandas as gpd


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def make_path(file_name, *path):
    return os.path.join(*path, file_name)


def save_pickle(data, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def load_maxar_items(type_item: str):

    columns = load_pickle(maxar_items_columns_path.format(type_item))
    gdf = gpd.read_file(maxar_items_path.format(type_item)).rename(columns, axis=1)
    gdf = gdf.assign(after_event=(gdf.datetime > date_event) * 1)
    return gdf


def load_roi_town(buffer_size: int=None):
    gdf = gpd.read_file(town_path)
    gdf = gdf.assign(town_name=gdf.town_name.str.replace(" ", "_"))
    if buffer_size:
        gdf.geometry = create_bbox(gdf.geometry, buffer_size)
    
    return gdf 


def check_dir(*path):
    os.makedirs(os.path.join(*path), exist_ok=True)
    return os.path.join(*path)


def load_rs_path_toy(name: str, 
                 root_path:str=interim_dir_path, 
                 type_item: str="visual",
                 temp: str="both", 
                 n:int=1) -> Dict[str, xr.DataArray]:
    
    
    if temp not in ["after", "before", "both"]:
        raise ValueError("Please provide valid temp : after, before, both")
    if type_item not in VALID_MODE:
        raise ValueError(f"Please provide valid temp : {VALID_MODE}")

    if temp in ["both", "before"]:
        #path_before = glob.glob(os.path.join(root_path, type_item, name, "before", "*.tif"))[:n]
        path_before = [IMG_TEST[name]["before"]]

    if temp in ["both", "after"]:
        #path_after = glob.glob(os.path.join(root_path, type_item, name, "after", "*.tif"))[:n]
        path_after = [IMG_TEST[name]["after"]]

    return dict(before=path_before, after=path_after)

def load_rs_toy(name: str, 
                 root_path:str=interim_dir_path, 
                 type_item: str="visual",
                 temp: str="both", 
                 n:int=1) -> Dict[str, xr.DataArray]:
    
    import rioxarray as rioxr

    
    res = dict(zip(["after", "before"], [None, None]))

    path_dict = load_rs_path_toy(name, root_path, type_item, temp, n)

    res["before"] = [rioxr.open_rasterio(_) for _ in path_dict["before"]]
    res["after"] = [rioxr.open_rasterio(_) for _ in path_dict["after"]]

    return res
    
    
    
