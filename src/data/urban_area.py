from rasterio.warp import reproject, Resampling, calculate_default_transform
import rasterio as rio
import geopandas as gpd
import pandas as pd
import os
from shapely import wkt
from shapely.geometry import box
import leafmap.foliumap as leafmap
import folium
import rioxarray as rioxr
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from skimage import io
import xarray as xr
import shapely
from typing import Any, Union, Dict, List

from shapely.geometry import shape as geo_shape
from shapely import Polygon, MultiPolygon
from rasterio.features import shapes as rio_shapes
from rasterio import features
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import  watershed
from pathlib import Path
from tqdm import tqdm

from src.config import *
from src.io.utils_io import load_maxar_items, load_rs_path_toy, load_roi_town, check_dir, make_path, save_pickle
from src.stac.utils_stac import get_nearest_match_geometry_view
from src.utils import create_bbox, get_maxar_items_on_roi
from src.rs_processing.processing import crop_rs_on_shape, rasterize_shp
from src.io.utils_io import load_pair_raster,make_path,load_pickle

import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_building_from_location(loc: Union[shapely.Point, shapely.Polygon], s_buffer: int, crs_ox: int=4326, crs_src: int=CRS): 

    if isinstance(loc, shapely.Point):

        loc = (
            gpd.GeoSeries([loc], 
                          crs=crs_src)
            .buffer(s_buffer)
            .to_crs(crs_ox)
        )

    shape = box(*loc.bounds.values.ravel())

    try:
    
        buildings = ox.features_from_polygon(shape, tags={"building":True})
    except:
        print("no buildings found")
        return None
        
    buildings = buildings.reset_index(drop=True)[["geometry"]]
    buildings = buildings.to_crs(crs_src)
    
    return buildings


def vectorize_rs(rs: np.ndarray, rs_src: xr.DataArray, out_shp_path:str, col_id:str): 
    rs = rs.astype(np.int16)
    shape_gen = [(geo_shape(s), v) for s, v in rio_shapes(rs, transform=rs_src.rio.transform())]
    gdf = gpd.GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=CRS) #rs_src.rio.crs
    if isinstance(col_id, str):
        gdf[col_id] = np.arange(len(gdf))
    if isinstance(out_shp_path, str):
        gdf.to_file(out_shp_path)

    return gdf


def extract_urban_area(name:str,
                    before_rs_path: str, 
                    loc: Union[shapely.Point,shapely.Polygon], 
                    method_seg: str,
                    fill_holes:bool,
                    resolution: int,
                    building_buffer: int,
                    s_buffer: int=157,): 
    
    out_dir = check_dir(processed_dir_path, "dataset", "buildings")

    fname = f"{name}_{method_seg}_hex{resolution}_b_buffer{building_buffer}_fh{fill_holes}"
    urb_path = make_path(f"{fname}.shp", out_dir)
    
    if not os.path.exists(urb_path): 
        logger.info(f"build urban zone {urb_path}")
    
        rs_b = rioxr.open_rasterio(before_rs_path)

        # if no building is found
        urban_zone=gpd.GeoDataFrame(geometry=[loc], crs=CRS)
        
        geom_building = get_building_from_location(loc, s_buffer)
        #print(f"found {geom_building.shape[0]} buildings")

        if geom_building is not None:

            urban_zone = segmentation_factory(method_seg, **{"rs_b":rs_b, "buffer":building_buffer, "geom_building":geom_building})

            urban_zone = process_urban_area(urban_zone, resolution, method_seg, fill_holes)
        else:
            # check how to manage missing columns
            print(f"No building found : {Path(before_rs_path).stem}")
            
        if isinstance(urb_path, str):
            urban_zone.to_file(urb_path)
            return urb_path
        
    return urb_path

def segmentation_factory(method_name, **kwargs): 
    factory_func = {
        "watershed": watershed_segmentation, 
        "osm_buffer": osm_buffer_segmentation, 
    }
    return factory_func[method_name](**kwargs)

def watershed_segmentation(**kwargs) -> gpd.GeoDataFrame: 
    rs_b = kwargs["rs_b"]
    data = rs_b.data.transpose(1, 2, 0)
    geom_building = kwargs["geom_building"]
    
    gradient = sobel(rgb2gray(data))
    lines = watershed(gradient, markers=500, compactness=0.0001,watershed_line=True)
    lines = np.where(lines == 0, 1, 0)
    lines = lines.astype(np.int16)
    
    gdf = vectorize_rs(lines, rs_b, out_shp_path=None, col_id="id_seg")    
    
    urban_area = gdf.sjoin(geom_building, how="inner", predicate="intersects").drop("index_right", axis=1)
    urban_area.geometry = urban_area.simplify(5)
    urban_area.geometry = urban_area.buffer(5)
    
    return urban_area

def osm_buffer_segmentation(**kwargs) ->  gpd.GeoDataFrame:
    urban_area = kwargs["geom_building"]
    logger.info(f"buffer : {kwargs['buffer']}")
    urban_area.geometry = urban_area.buffer(kwargs["buffer"])
    return urban_area


import h3pandas as h3
def process_urban_area(urban_area: gpd.GeoDataFrame, resolution:int, method:str, fill_holes: bool) -> gpd.GeoDataFrame: 
    """
    - simplify, dissolve and fill holes of urban area
    - create hex grid
    """
    if fill_holes:
        urban_area = urban_area.dissolve()
        
        if isinstance(urban_area.geometry.item(), MultiPolygon): 
            geoms = list(urban_area.geometry.item().geoms)
        else:
            geoms = list(urban_area.geometry)
            
        urban_area["geometry"] =  MultiPolygon(Polygon(geom.exterior) for geom in geoms)

    urban_area = create_hex_grid(urban_area, resolution)
    #urban_area = urban_area[["id_hex", "geometry"]] 

    return urban_area


def create_hex_grid(gdf, resolution: int): 
    """
    reolution table : https://h3geo.org/docs/core-library/restable/
    """
    gdf = gdf.to_crs(4326)
    hexagons = gdf.h3.polyfill_resample(resolution)
    hexagons = hexagons.h3.h3_get_resolution()
    hexagons = hexagons.h3.cell_area("m^2")
    hexagons = hexagons.to_crs(CRS).reset_index().rename({"h3_polyfill":"id_hex", "h3_cell_area":"cell_area", "h3_resolution":"grid_resolution"}, axis=1)
    return hexagons


def rasterize_shp(gdf: gpd.GeoDataFrame, src_rs: str, out_path: str) -> str:
    
    with rio.open(os.path.join(src_rs)) as rs_src:
        meta = rs_src.meta.copy()

        meta = {
            **meta, 
            **{
                "count":1, 
                "driver": "GTiff",
            }
        }
        
        with rio.open(out_path, 'w+', **meta) as out:
            out_arr = out.read(1)
        
            shapes = [geom  for geom  in gdf.geometry]
        
            burned = rio.features.rasterize(shapes=shapes, 
                                            out_shape = rs_src.shape,
                                            transform = rs_src.transform,
                                            all_touched = True,
                                            fill = 0,  
                                            dtype = None)
            out.write_band(1, burned)
            
    return out_path