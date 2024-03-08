from pathlib import PosixPath
import geopandas as gpd 
from shapely import box, Point
import logging
import itertools
import numpy as np
import xarray as xr
from src.config import *
from typing import Union
import rioxarray as rioxr

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_href_nearest(nearest, roi_name):
    return {
        "before": nearest[roi_name]["before"]["href"].item(),
        "after": nearest[roi_name]["after"]["href"].item()
    }



def convert_to_raster(img: np.ndarray, rs_ref: Union[PosixPath, str, xr.DataArray], crs=CRS, out_path: Union[str, None]=None):
    
    if isinstance(rs_ref, str) or isinstance(rs_ref, PosixPath):
        rs_ref = rioxr.open_rasterio(rs_ref)
    print(rs_ref.shape)
    # put band dim at first dim
    img = img.transpose(2, 0, 1)    
    
    da = xr.DataArray(
        img,
        dims=("band", "y", "x"), 
        coords={
            "y": rs_ref.coords["y"].values,
            "x":rs_ref.coords["x"].values
        },
        attrs=rs_ref.attrs 
    ).rio.write_crs(crs)
    if out_path:
        da.rio.to_raster(out_path)
        logger.info(f"raster registered : {out_path}")
   

def create_bbox(s_geom: gpd.GeoSeries, s_buffer: int):
    """create polygon bbox buffer from points geom

    Args:
        s_geom (gpd.GeoSeries): geoseries point geometry
        s_buffer (int): buffer size in meter
    """
    return [box(*geom.bounds) for geom in s_geom.buffer(s_buffer).values]

def compute_intersection(shp_rs, shp_roi):
    assert shp_rs.crs == shp_roi.crs
    shp_rs = gpd.overlay(shp_rs, shp_roi, how="intersection")
    shp_rs["intersec_roi_perc"] = shp_rs.area / shp_roi.area.item()
    return shp_rs


def get_maxar_items_on_roi(
    maxar_items: gpd.GeoDataFrame, roi: gpd.GeoDataFrame, th: float = 0.9, s_bbox:int=None
):

    if s_bbox or isinstance(maxar_items.geometry.iloc[0], Point):
        roi.geometry = [
                box(*_.bounds) for _ in roi.buffer(s_bbox)
            ]
    items_roi = compute_intersection(maxar_items, roi)
    if th is not None:
        items_roi = items_roi[items_roi.intersec_roi_perc > th]
    # order by date to be sur to get both before and after - workaround
    items_roi = items_roi.sort_values("after_event", ascending=False)
    logger.info(f"found {items_roi.shape[0]} maxar items")
    return items_roi



    
    