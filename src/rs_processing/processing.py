import rioxarray as rioxr
import rasterio as rio
import geopandas as gpd
from typing import List, Union, Dict
from shapely import Polygon
from src.io.utils_io import check_dir, load_roi_town, load_maxar_items, make_path, save_pickle
from src.config import *
from src.stac.utils_stac import get_nearest_match_geometry_view
from src.utils import create_bbox, get_href_nearest
from pathlib import Path
from shapely import Polygon
import pandas as pd
from skimage.exposure import match_histograms
import numpy as np
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def pipeline_processing(bbox_size):
    town = load_roi_town(buffer_size=bbox_size)
    pass

def histogram_alignment(target, ref):
    matched = match_histograms(target, ref, channel_axis=-1)
    return matched

def pipeline_download_nearest_items(type_item: str, 
                           town: Union[gpd.GeoDataFrame, None]=None,
                           field_geometry: str="view_incidence_angle", 
                           bbox_size: int=BBOX_SIZE):
    
    local_path = []
    meta_b = []
    meta_a = []

    if town is None:
        town = load_roi_town(buffer_size=bbox_size)
        
    maxar_items = load_maxar_items(type_item=type_item)

    nearest = get_nearest_match_geometry_view(town=town, 
                                       field=field_geometry,
                                       maxar_items=maxar_items, 
                                       type_item=type_item, 
                                       s_bbox=bbox_size)
    for name in nearest:
        hrefs = get_href_nearest(nearest, name)
        out_dir = make_path(name, interim_dir_path, type_item)
        shape = town[town.town_name == name].geometry.item()
        local_path.append(crop_nearest(hrefs, shape, out_dir, s_buffer=bbox_size))
        meta_b.append(get_meta_img(hrefs["before"], maxar_items, COL_META_IMG))
        meta_a.append(get_meta_img(hrefs["after"], maxar_items, COL_META_IMG))

    local_path = gpd.GeoDataFrame(local_path, crs=CRS)
    local_path = local_path.assign(name=list(nearest))
    meta_b = pd.DataFrame(meta_b, columns=[f"before_{_}" for _ in COL_META_IMG])    
    meta_a = pd.DataFrame(meta_a, columns=[f"after_{_}" for _ in COL_META_IMG])

    local_path = pd.concat([local_path, meta_b, meta_a], axis=1)
    
    save_pickle(local_path, make_path(f"nearest_{type_item}.pkl", interim_dir_path, type_item)) 

def get_meta_img(href: str, maxar_items: gpd.GeoDataFrame, columns_meta: List[str]):
    return maxar_items.loc[maxar_items.href == href, columns_meta].values.squeeze()

def crop_nearest(hrefs: Dict[str, str], shape: Polygon, out_dir: str, s_buffer: int):
    
    out_dir_b = check_dir(out_dir, "nearest")
    out_dir_a = check_dir(out_dir, "nearest")

    b_name = Path(hrefs["before"]).stem
    a_name = Path(hrefs["after"]).stem
    
    # load from s3 - return local crop path
    b_path = crop_rs_on_shape(hrefs["before"], 
                             shape, 
                             out_path=os.path.join(out_dir_b, 
                                                   f"{s_buffer}_{B_TAG}_{b_name}.tif")
                            )

    a_path = crop_rs_on_shape(hrefs["after"], 
                             shape, 
                             out_path=os.path.join(out_dir_a, 
                                                   f"{s_buffer}_{A_TAG}_{a_name}.tif")
                            )

    return dict(before_path=b_path, after_pat=a_path, geometry=shape)


def crop_rs_on_shape(rs_path: str, shape: Union[Polygon, List], out_path: str=None) -> Union[str, np.ndarray]:
    """
    Crop raster from shapely geometry shape

    - add check crs equality
    - clip_box with rioxarray => pyproj CRS error

    """
    if not isinstance(shape, list):
        shape = [shape]

    with rio.open(rs_path) as src:
        out_image, out_transform = rio.mask.mask(src, shape, crop=True)
        out_meta = src.meta

        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        if out_path:
            with rio.open(out_path, "w", **out_meta) as dest:
                dest.write(out_image)
            return out_path
        else:
            return out_image
        
        
        
        
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

if __name__ == "__main__":
    path_df = pipeline_download_nearest_items(type_item="visual", bbox_size=BBOX_SIZE)