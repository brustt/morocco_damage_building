import rasterio as rio
import geopandas as gpd
import fiona
import os
import pickle
# enable kml file reading for geopandas
from shapely.geometry import box, Polygon
import pandas as pd
import numpy as np
import leafmap.foliumap as leafmap # use folium backend
import rioxarray as rioxr
import matplotlib.pyplot as plt
from typing import List, Union, Dict
from pathlib import Path
import logging
from shutil import copyfile as shtl_copyfile

from arosics import COREG_LOCAL, COREG

from src.config import *
from src.io.utils_io import load_maxar_items, load_rs_path_toy, load_roi_town, check_dir, make_path, save_pickle
from src.rs_processing.registration import compute_cc
from src.stac.utils_stac import get_nearest_match_geometry_view
from src.utils import create_bbox, get_maxar_items_on_roi
from src.rs_processing.processing import crop_rs_on_shape

pd.options.display.max_columns = 50

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


type_item = 'visual'
s_buffer = 500
prefix = f"after_{s_buffer}_shifted__local"
name = "Talat_Nyaaqoub"
download = False

maxar_items = load_maxar_items(type_item=type_item)
town = load_roi_town()

failed = {}
field_geometry = "view_incidence_angle"

# default params for local shift - grid_res:50 - window_size: (256, 256)
local_params = {
    'grid_res'     : 50,
    'window_size'  : (128, 128),
    'q'            : False,
    'max_shift': 50
}

town = load_roi_town()

nearest = get_nearest_match_geometry_view(town=town, 
                                       field=field_geometry,
                                       maxar_items=maxar_items, 
                                       type_item=type_item, 
                                       s_bbox=s_buffer)


for name in town.town_name.unique():
    logger.info(f"=== {name} ===")

    town_dir = os.path.join(interim_dir_path, type_item, name)

    b_path = nearest[name]["before"]["href"].item()
    a_path = nearest[name]["after"]["href"].item()

    in_dir_b = check_dir(town_dir, "nearest", "before")
    in_dir_a = check_dir(town_dir, "nearest", "after")

    b_name = Path(b_path).stem
    a_name = Path(a_path).stem

    if download and not os.path.exists(make_path("full_tiles", town_dir, "nearest")):
        full_rs_path = check_dir(town_dir, "nearest", "full_tiles")
        leafmap.download_file(b_path, output=make_path(f"full_{b_name}.tif", full_rs_path), overwrite=False)
        leafmap.download_file(a_path, output=make_path(f"full_{a_name}.tif", full_rs_path), overwrite=False)


    if not os.path.exists(make_path(f"{s_buffer}_{b_name}.tif", in_dir_b)):
        shape = create_bbox(town[town.town_name == name], s_buffer=s_buffer)

        # load from s3 - return local crop path
        b_path = crop_rs_on_shape(b_path, 
                                shape, 
                                out_path=os.path.join(in_dir_b, 
                                                    f"{s_buffer}_{b_name}.tif")
                                )

        logger.info(f"before image : {b_path}")

        a_path = crop_rs_on_shape(a_path, 
                                shape, 
                                out_path=os.path.join(in_dir_a, 
                                                    f"{s_buffer}_{a_name}.tif")
                                )    

        logger.info(f"after image : {a_path}")
    else:
        logger.info(f"Already local files")
        b_path = make_path(f"{s_buffer}_{b_name}.tif", in_dir_b)
        a_path = make_path(f"{s_buffer}_{a_name}.tif", in_dir_a)


    out_shift_dir = os.path.join(processed_dir_path, 
                                type_item, 
                                name, 
                                "registration")

    _=check_dir(out_shift_dir)

    local_params["path_out"] = os.path.join(out_shift_dir, f"{prefix}_{a_name}.tif")
    # before : ref - after : target
    if not os.path.exists(local_params["path_out"]):
        CRL = COREG_LOCAL(
            b_path, 
            a_path, 
            **local_params
            )
        CRL.correct_shifts() 
    
    # check on smaller image
    min_buffer = 200
    shape = create_bbox(town[town.town_name == name], s_buffer=min_buffer)

    rs_a_shift = crop_rs_on_shape(local_params["path_out"], 
                        shape, 
                        out_path=None
                        ) 

    rs_b = crop_rs_on_shape(b_path, 
                        shape, 
                        out_path=None
                        )    
    rs_a_shift = rs_a_shift.transpose(1, 2, 0)
    rs_b = rs_b.transpose(1, 2, 0)

    logger.info(f"Corr with shift : {compute_cc(rs_a_shift, rs_b)}")
    