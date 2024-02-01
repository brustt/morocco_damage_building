import rioxarray as rioxr
import rasterio as rio
import geopandas as gpd
from typing import List, Union, Dict
from shapely import Polygon


def crop_rs_on_shape(rs_path: str, shape: Union[Polygon, List], out_path: str):
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
    
        out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    
        with rio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)
    return out_path