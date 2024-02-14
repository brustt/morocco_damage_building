import os
from shapely.geometry import box, Polygon
import leafmap.foliumap as leafmap  # use folium backend for download now to change to thread
from typing import List, Union, Dict
from src.io.utils_io import check_dir, load_maxar_items, load_roi_town, make_path
from src.config import date_event, interim_dir_path
from src.rs_processing.processing import crop_rs_on_shape
from src.stac.utils_stac import get_maxar_items_on_roi
import logging
from shapely import Point
import geopandas as gpd

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
need to be refacto in a cleaner way
- find specific items before donwload
- multi threading download
"""


def download_maxar_on_roi(
    type_item: str,
    roi_gdf: gpd.GeoDataFrame,
    maxar_items: gpd.GeoDataFrame = None,
    th_intersect: float = 0.90,
    buffer_size: int = 200,
    save_items_shp: bool = True,
    limit_download: Union[int, None] = 5,
    remove_full_rs: bool = True,
) -> List[str]:
    

    tif_local_path = []

    if not maxar_items:
        maxar_items = load_maxar_items(type_item=type_item)
        
  

    for name in roi_gdf.town_name.unique():
        logger.info(f"==== TOWN : {name} =====")
        
        name = "_".join(name.split(" "))

        roi_buffer = roi_gdf[roi_gdf.town_name == name]
        
        if isinstance(roi_buffer.geometry.iloc[0], Point) and not buffer_size:
            raise ValueError("Please provide buffer size for point geometry")
        
        s_bbox = buffer_size if buffer_size else None
            
        bbox_buffer = box(*roi_buffer.bounds.values.squeeze())

        items_town = get_maxar_items_on_roi(maxar_items, roi_buffer, th_intersect, s_bbox)

        path_town_dir = check_dir(interim_dir_path, type_item, name)

        if save_items_shp:
            items_town.to_file(
                os.path.join(path_town_dir, f"maxar_items_{name}_{type_item}.shp")
            )

        cnt = 0
        for i, row in items_town.iterrows():
            if limit_download and cnt >= limit_download:
                # stop download after limit_download tiles per town
                break

            delta_event = "after" if row["after_event"] == 1 else "before"

            path_rs = check_dir(path_town_dir, delta_event)

            """
            select "last before" tile
            """

            path_tmp_rs = make_path("full_rs_tmp.tif", path_rs)

            # change to multi thread
            leafmap.download_file(row["href"], output=path_tmp_rs, overwrite=False)

            path_rs_clip = os.path.join(path_rs, row["href"].split("/")[-1])

            try:
                rs_path = crop_rs_on_shape(path_tmp_rs, bbox_buffer, path_rs_clip)
            except:
                raise ValueError(f"erreur : {i} : {path_tmp_rs}")

            logger.info(
                f"Save {path_rs_clip} for {name} : {delta_event} event {row['datetime']}"
            )
            tif_local_path.append(rs_path)
            if remove_full_rs:
                # delete full rs
                os.remove(path_tmp_rs)
            cnt += 1
        logger.info(f"Downloaded {cnt} images for {name}")
    logger.info(f"=== End downloads ====")

    return tif_local_path
