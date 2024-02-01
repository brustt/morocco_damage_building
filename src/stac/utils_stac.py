from datetime import datetime
import logging
from requests import Session
from threading import local
from urllib.error import URLError
import requests
import numpy as np
import socket
from pydantic import BaseModel
import geopandas as gpd

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_root_url_id_ = "ard/29/"  # change for other collection maxar - see STAC catalog

thread_local = local()


def compute_intersection(shp_rs, shp_roi):
    assert shp_rs.crs == shp_roi.crs
    shp_rs = gpd.overlay(shp_rs, shp_roi, how="intersection")
    shp_rs["intersec_roi_perc"] = shp_rs.area / shp_roi.area.item()
    return shp_rs


def get_maxar_items_on_roi(
    maxar_items: gpd.GeoDataFrame, roi: gpd.GeoDataFrame, th: float = 0.9
):

    items_roi = compute_intersection(maxar_items, roi)
    items_roi = items_roi[items_roi.intersec_roi_perc > th]
    # order by date to be sur to get both before and after - workaround
    items_roi = items_roi.sort_values("after_event", ascending=False)
    logger.info(f"found {items_roi.shape[0]} maxar items")
    return items_roi


class TypeAssetError(BaseException):
    """Error on img asset selection"""


class STACOpenerError(Exception):
    """An error indicating that a STAC file could not be opened"""


def id_to_href(hrefs, collection):
    """
    build dictionnary to map id product - threads with .submits() doesn't keep order
    hope it's consistent with stac standards
    id_item : quadkey/date/catalog_id
    """

    hrefs_mapping = {}
    for h in hrefs:
        id_item = h.split("/".join([collection, _root_url_id_]))[1:][0][:-5]
        hrefs_mapping[id_item] = "/".join(h.split("/")[:-1])
    return hrefs_mapping


def isotime(date_string):
    return date_string.replace("T", " ").replace("Z", "").rstrip()


def build_link_product(item, hrefs_map):
    """faster than use STAC Item
    not very readable
    """

    tmp_id = "/".join(
        [
            item["quadkey"],
            datetime.strftime(
                datetime.strptime(isotime(item["datetime"]), "%Y-%m-%d %H:%M:%S"),
                "%Y-%m-%d",
            ),
            item["catalog_id"],
        ]
    )
    item["href"] = "/".join([hrefs_map[tmp_id], item["href"].split("/")[-1]])
    return item


def extract_assets(content, type_item="visual"):
    """
    Parse json - selection of interesting properties
    type_item :  'data-mask', 'ms_analytic', 'pan_analytic', 'visual'
    """
    assets_selection = {
        "id": "id",
        "assets": {
            type_item: [
                "title",
                "type",
                "href",
            ]
        },
        "properties": [
            "datetime",
            "platform",
            "gsd",
            "catalog_id",
            "utm_zone",
            "quadkey",
            "grid:code",
            "proj:epsg",
            "proj:bbox",
            "proj:shape",
        ]
        + [_ for _ in content["properties"] if _.startswith("view")]
        + [_ for _ in content["properties"] if _.startswith("tile")],
    }

    def process_key(k):
        return k.replace(":", "_")

    new_content = {}
    for group, asset_keys in assets_selection.items():
        if isinstance(asset_keys, dict):
            ((sub_group, sub_asset_keys),) = asset_keys.items()
            try:
                new_content = new_content | {
                    k: v
                    for k, v in content[group][sub_group].items()
                    if k in sub_asset_keys
                }
            except KeyError as e:
                logger.error(f"Authorized types : {content[group].keys()}")
                raise TypeAssetError

        elif isinstance(asset_keys, str):
            new_content = new_content | {asset_keys: content[group]}
        else:
            # list
            # asset_keys = [process_key(_) for _ in asset_keys]
            new_content = new_content | {
                k: v for k, v in content[group].items() if k in asset_keys
            }
    # np array to prevent error of index out of bounds
    geom = (
        np.array(content["properties"]["proj:geometry"]["coordinates"])
        .squeeze()
        .tolist()
    )
    new_content = new_content | {"proj_geometry": geom}
    new_content = {process_key(k): v for k, v in new_content.items()}

    return new_content


def get_session() -> Session:
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()  # Create a new Session if not exists
    return thread_local.session


def read_http_stac_file(url, timeout=None):
    """requests with session object"""
    try:
        session = get_session()
        # cannot set timeout to request session
        with session.get(url) as response:
            return response.json()

    except URLError as e:
        if isinstance(e.reason, socket.timeout):
            logger.error("%s: %s", url, e)
            raise socket.timeout(f"{url} with a timeout of {timeout} seconds") from None
        else:
            logger.debug("read_http_remote_json is not the right STAC opener")
            raise STACOpenerError


# implemented but not tested
class MaxarOpenData(BaseModel):
    id: str
    type: str
    title: str
    href: str
    proj_bbox: list[float]
    shape: list[int]
    platform: str
    gsd: float
    catalog_id: str
    utm_zone: int
    quadkey: str
    view_off_nadir: float
    view_azimuth: float
    view_incidence_angle: float
    view_sun_azimuth: float
    view_sun_elevation: float
    proj_epsg: int
    proj_geometry: list
    grid_code: str
    tile_data_area: float
    tile_clouds_area: float
    tile_clouds_percent: float
