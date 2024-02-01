from pystac_client import Client
import pystac
from shapely import Polygon
import geopandas as gpd
from pydantic import ValidationError, BaseModel

from typing import Dict, List, Optional, Union, Any
import concurrent.futures

from tqdm import tqdm

import logging
from src.config import raw_dir_path
from src.io.utils_io import *
from src.stac.utils_stac import *
import argparse

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_items(type_item: str) -> None:
    """get maxar items metadata

    local items = {
         "hrefs": make_path("maxar_href_stac_items.pkl", interim_dir_path, type_item),
         "items":make_path("maxar_stac_items.pkl", interim_dir_path, type_item)
     } | None

    Args:
        type_item (str) :  'data-mask', 'ms_analytic', 'pan_analytic', 'visual'

    """
    if type_item not in ["data-mask", "ms_analytic", "pan_analytic", "visual"]:
        raise ValueError

    reader = StaticStacReader(
        model_validation=MaxarOpenData,
        url_catalog="https://maxar-opendata.s3.amazonaws.com/events/catalog.json",
        collection="Morocco-Earthquake-Sept-2023",
        local_items=None,
        max_connections=100,
        timeout=30,
    )

    gdf = reader.read(type_item=type_item)

    gdf = gdf.assign(
        gsd=gdf.gsd.astype("float32"),
        utm_zone=gdf.utm_zone.astype("int16"),
        view_off_nadir=gdf.view_off_nadir.astype("float32"),
        view_azimuth=gdf.view_azimuth.astype("float32"),
        view_sun_elevation=gdf.view_sun_elevation.astype("float"),
        proj_epsg=gdf.proj_epsg.astype("int16"),
        data_area=gdf.tile_data_area.astype("float16"),
        clouds_area=gdf.tile_clouds_area.astype("float16"),
        clouds_percent=gdf.tile_clouds_percent.astype("float16"),
        geometry_bbox=[str(x) for x in gdf.proj_bbox],
    ).drop(
        {
            "proj_geometry",
            "proj_bbox",
            "tile_data_area",
            "tile_clouds_area",
            "tile_clouds_percent",
        },
        axis=1,
    )

    mapping_columns = {n[:10]: n for n in gdf.columns}
    save_pickle(
        mapping_columns,
        make_path("mapping_columns_stac_collection.pkl", raw_dir_path, type_item),
    )
    gdf.to_file(make_path("maxar_stac_items.shp", raw_dir_path, type_item))


class StaticStacReader:

    def __init__(
        self,
        model_validation: BaseModel,
        url_catalog: str,
        collection: str,
        local_items: Dict[str, str] = None,
        max_connections: int = 20,
        timeout: int = 20,
    ):

        self.model_validation = model_validation
        self.timeout = timeout
        self.max_connections = max_connections
        self.url_catalog = url_catalog
        self.collection = collection
        self.catalog = None
        self.local_items = local_items

    def open_catalog(self, url_catalog: str, collection: str) -> pystac.Catalog:
        root_catalog = Client.open(url_catalog)
        return root_catalog.get_child(collection)

    def read(self, type_item: str, validate: bool = False) -> gpd.GeoDataFrame:
        """
        add validation json retun with pydantic (currently implemented but not tested)
        """
        items = self.get_items(type_item)
        if validate:
            self.validate(items)

        geom = [Polygon(_["proj_geometry"]) for _ in items]
        gdf = gpd.GeoDataFrame(items, crs=int(items[0]["proj_epsg"]), geometry=geom)

        return gdf

    def validate(self, items: List[Dict]) -> None:
        # to check
        for item in items:
            try:
                self.model_validation(**item)
            except ValidationError as e:
                print(e.errors())

    def get_items(self, type_item: str) -> List[Dict[Any, Any]]:

        items: List[Dict[Any, Any]] = []
        logger.info("Build absolute path for stac items")

        hrefs = self.get_hrefs()
        items = self._get_items(hrefs)
        items = self.process_items(items, hrefs, type_item)

        return items

    def get_hrefs(self) -> List[str]:
        hrefs = []
        if self.local_items is None:
            self.catalog = self.open_catalog(self.url_catalog, self.collection)
            # build absolute path for file stac items
            for parent_catalog, _, _ in self.catalog.walk():
                hrefs += [
                    link.get_absolute_href() for link in parent_catalog.get_item_links()
                ]
        else:
            hrefs = load_pickle(self.local_items["hrefs"])
        return hrefs

    def process_items(
        self, items: List[Dict[Any, Any]], hrefs: List[str], type_item: str
    ) -> List[Dict[Any, Any]]:
        """items: Generator"""
        items_clean = []
        hrefs_mapping = id_to_href(hrefs, self.collection)

        for item in items:
            n_item = extract_assets(item, type_item=type_item)
            n_item = build_link_product(n_item, hrefs_mapping)
            items_clean.append(n_item)

        return items_clean

    def _get_items(self, hrefs: List[str]) -> Any:  # add Generator type
        if self.local_items is not None:
            # return generator
            logger.info("Load stac items from local file..")

            return (x for x in load_pickle(self.local_items["items"]))
        else:
            logger.info("Start download stac items..")
            return self.download_items_threaded(hrefs)

    def download_items_threaded(
        self, hrefs: List[str]
    ) -> Any:  # add Generator type - extract from class

        with tqdm(total=len(hrefs)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_connections
            ) as executor:
                future_to_href = [
                    executor.submit(read_http_stac_file, href, self.timeout)
                    for href in hrefs
                ]
                for future in concurrent.futures.as_completed(future_to_href):
                    pbar.update(1)
                    yield future.result()


if __name__ == "__main__":
    # et cli args
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_item", help="type of items to match", required=True)
    args = parser.parse_args()
    get_items(type_item=args.type_item)
