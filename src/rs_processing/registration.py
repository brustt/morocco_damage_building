from src.io.utils_io import check_dir, load_pickle, load_roi_town, make_path, save_pickle
from src.rs_processing.processing import histogram_alignment
from src.utils import convert_to_raster
import torch
import cv2
from src.config import *
from pathlib import Path
from PIL import Image
import kornia as K
import kornia.feature as KF
import logging
import rioxarray as rioxr
import numpy as np
import xarray as xr
from src.rs_processing.LoFTR.src.loftr import LoFTR, default_cfg
from typing import Dict, Any, List
import geopandas as gpd
from shapely import box
from skimage.exposure import match_histograms

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.manual_seed(12)

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def load_model():
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(Path(models_dir_path, "loftr", "outdoor_ds.ckpt"))['state_dict'])
    matcher = matcher.eval().cpu()
    return matcher

def predict(matcher, batch):
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    return mkpts0, mkpts1, mconf

def convert_to_png(in_path, out_path):
    im0 = Image.open(in_path)
    im0.save(out_path)
    
def process_img_cv2(img_path, ref_img=None, resize=(640, 480)):
    img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_raw = cv2.resize(img_raw, resize)
    if ref_img is not None:
        img_raw = match_histograms(img_raw, ref_img, channel_axis=-1)
    
    return img_raw

def make_input_batch(img0_path, img1_path):
    img0 = process_img_cv2(img0_path)
    # best coor without histogram match - wtf
    img1 = process_img_cv2(img1_path, ref_img=None)

    img0 = torch.from_numpy(img0.astype(np.float32))[None][None].cpu() / 255.
    img1 = torch.from_numpy(img1.astype(np.float32))[None][None].cpu() / 255.
    
    batch = {
            "image0": img1,  # inverse imgs
            "image1": img0,
        }
    return batch

def register_pair_img(img0_path: str, img1_path: str, out_path: str, dir_tmp_png:str, hist_alignment:bool=True, to_int:bool=True, verbose=True):
    
    img0_path_png, img1_path_png = make_path(Path(img0_path).stem+".png", dir_tmp_png), make_path(Path(img1_path).stem+".png", dir_tmp_png), 

    rs_a = rioxr.open_rasterio(img1_path)
    arr_a = rs_a.data.transpose(1, 2, 0).astype(np.float32)

    rs_b = rioxr.open_rasterio(img0_path)
    arr_b = rs_b.data.transpose(1, 2, 0).astype(np.float32)
    
    convert_to_png(img0_path, img0_path_png)
    convert_to_png(img1_path, img1_path_png)

    batch = make_input_batch(img0_path_png, img1_path_png)

    matcher = load_model()
    # Inference with LoFTR and get prediction
    mkpts0, mkpts1, mconf = predict(matcher, batch)
        
    print(mkpts0.shape)
    (H, mask) = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=10000)
    
    (h, w) = rs_b.shape[-2:]
    aligned = cv2.warpPerspective(arr_a, H, (w, h))
    aligned = aligned.astype(np.float32)
    
    if hist_alignment:
        aligned = histogram_alignment(aligned, arr_b)
        
    if to_int:
        aligned = aligned.astype(np.uint8)
        arr_b = arr_b.astype(np.uint8)
        arr_a = arr_a.astype(np.uint8)
    
    convert_to_raster(aligned, rs_b, out_path=out_path)
    
    coor_map = compute_cc(aligned, arr_b)
    coor_map_ref = compute_cc(arr_a, arr_b)
    
    if verbose:
        logger.info(f"coor aligned: {coor_map}")
        logger.info(f"coor ref: {coor_map_ref}")
        
    clean_png_files(dir_tmp_png)
        
    return dict(
        before_path=img0_path, 
        after_path=out_path, 
        corr=coor_map, 
        corr_ref=coor_map_ref,
        geometry=box(*rs_b.rio.bounds()))
    
def compute_cc(rs_a: np.ndarray, rs_b: np.ndarray) -> float:
    """compute cross correlation metric with opencv
    
    https://stackoverflow.com/questions/58158129/understanding-and-evaluating-template-matching-methods

    Args:
        rs_a (np.ndarray): arr image a
        rs_b (np.ndarray): arr image b
        
    Try TM_CCOEFF_NORMED | TM_CCORR_NORMED

    Returns:
        float: cross correlation
    """
    coor_map = cv2.matchTemplate(rs_a, rs_b, cv2.TM_CCOEFF_NORMED)
    return coor_map[0][0]

def clean_png_files(dir):
    _ = [os.remove(make_path(f, dir)) for f in os.listdir(dir) if Path(f).suffix ==".png"]

    

    
    
if __name__ == "__main__":
    town_test = "Talat_Nyaaqoub"
    type_item = "visual"
    zoom = 200
    process_tag = "rgsted"
    in_path = make_path(f"nearest_{type_item}.pkl", interim_dir_path, type_item)
    nearest_crop = load_pickle(in_path)
    process_items = []
    for name in nearest_crop.name.unique():
        logger.info(f"=== register for {name} ===")
        
        b_path = nearest_crop[nearest_crop.name == name]["before"].item()
        a_path = nearest_crop[nearest_crop.name == name]["after"].item()
        
        b_name = Path(b_path).stem
        a_name = Path(a_path).stem

        out_dir = check_dir(processed_dir_path, type_item)
        dir_tmp_png = check_dir(processed_dir_path, type_item, name)

        # check dir
        res = register_pair_img(img0_path=b_path, 
                                img1_path=a_path, 
                                out_path=make_path(f"{a_name}_{process_tag}.tif", out_dir, name), 
                                hist_alignment=False,
                                to_int=True,
                                dir_tmp_png=dir_tmp_png)
        
        process_items.append(res)
    # add metadata sensor and acquisition
    process_items = gpd.GeoDataFrame(process_items, crs=CRS)
    process_items = process_items.assign(name=nearest_crop.name.unique())

    save_pickle(process_items, make_path(f"{process_tag}_{type_item}.pkl", out_dir))
    logger.info(f"""register done for {len(process_items)} - paths saved {make_path(f"{process_tag}_{type_item}.pkl", out_dir)}""")

   
   
   
    
"""
abandon registration with kornia pretrained - wtf not the same model as git repo

def process_imgs(img0, img1, direction="ab", resize=(640, 480)) -> Dict[str, torch.tensor]:


    # why this resize w,h ??
    # output size : (b, c, h, w)
    img0 = K.geometry.resize(img0, resize, antialias=True)
    img1 = K.geometry.resize(img1, resize, antialias=True)
    
    if direction == "ab":
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1) / 255,  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(img0) / 255,
        }
    elif direction == "ba":
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1) / 255,  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(img0) / 255,
        }
    else:
        raise ValueError("please provide valid direction stitching : ab == after towards before images")
    
    return input_dict

def load_model():
    matcher = KF.LoFTR(pretrained=None, config=default_cfg)
    matcher.load_state_dict(torch.load(Path(models_dir_path, "loftr", "outdoor_ds_v2.ckpt"))['state_dict'])
    matcher = matcher.eval().cpu()
    return matcher


def register_imgs(img0_path: str, img1_path: str, out_path: str, direction: str="ab", verbose=True):
    
    img0 = K.io.load_image(img0_path, K.io.ImageLoadType.RGB32)[None, ...]
    img1 = K.io.load_image(img1_path, K.io.ImageLoadType.RGB32)[None, ...]
    
    input_dict = process_imgs(img0, img1)
    matcher = load_model()
    
    with torch.inference_mode():
        correspondences = matcher(input_dict)
        
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    logger.info(mkpts0.shape)
    
    (H, mask) = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=10000)
    
    # cautious need to change if direction ba
    (h, w) = img0.shape[-2:]
    # squeeze batch dim and transpose to cv2 channel order
    img1 = img1.squeeze().numpy().transpose(1, 2, 0)
    #img0 = img0.squeeze().numpy().transpose(1, 2, 0)
    img0 = rioxr.open_rasterio(img0_path).data.astype(np.float32).transpose(1, 2, 0)

    aligned = cv2.warpPerspective(img1, H, (w, h))
    aligned = aligned.astype(np.float32)
    
    print(aligned.shape)

    convert_to_raster(aligned, img0_path, out_path=out_path)
    if verbose:
        corr = compute_cc(aligned, img0)
    # cautious need to change if direction ba
    
    #rs_a_shift = rioxr.open_rasterio(out_path)
    #arr_a_shift = rs_a_shift.data.transpose(1, 2, 0)
    #_ = compute_cc(img0, img1)
    return dict(rs_b_path=img0_path, rs_a_path=out_path, corr=corr)
    
def compute_cc(rs_a, rs_b):
    print(rs_a.dtype)
    print(rs_b.dtype)
    coor_map = cv2.matchTemplate(rs_a, rs_b, cv2.TM_CCOEFF_NORMED)
    logger.info(f"Cross Correlation : {coor_map}")
    return coor_map
    
# img read / grey conversion by kornia decline performanace alignement - wtf
img0_path_png = str(Path(interim_dir_path, type_item, town_test, "nearest", "test_eolearn", "200_1040010077691A00-visual.png"))
img1_path_png = str(Path(interim_dir_path, type_item, town_test, "nearest", "test_eolearn", "200_10300500E4F92300-visual.png"))

img0 = K.io.load_image(img0_path_png, K.io.ImageLoadType.GRAY8)[None, ...] / 255
img1 = K.io.load_image(img1_path_png, K.io.ImageLoadType.GRAY8)[None, ...] / 255
#img0 = K.color.rgb_to_grayscale(img0) / 255
#img1 = K.color.rgb_to_grayscale(img1) / 255
img0 = K.geometry.resize(img0, (640, 480), antialias=True)
img1 = K.geometry.resize(img1, (640, 480), antialias=True)

"""















"""
cannot import rasterio | rioxarray module with otbApplication
seems to be a pb with gdal
=> segmentation fault (import rasterio before otb)
=> shapely/lib.cpython-310-x86_64-linux-gnu.so: undefined symbol: GEOSGeom_getExtent_r (import rasterio after)
try to install --no-bynary :all: but maybe conda load rasterio from local install and do not link gdal with otb

same with geopandas - works import before otb




from src.config import *
#import rasterio as rio
from src.io.utils_io import check_dir, load_rs_path_toy
from pathlib import Path


import logging
from typing import Dict
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


type_item = 'pan_analytic'
s_buffer=100
town_test = "Talat Nyaaqoub"


path_ref = "/home/rustt/Documents/Projects/building_damage/morocco_damage_building/data/interim/visual/Tafeghaghte/before/200_1040010045AE4B00-visual.tif"
path_slave = "/home/rustt/Documents/Projects/building_damage/morocco_damage_building/data/interim/visual/Tafeghaghte/after/200_10300500E4F92300-visual.tif"
out_dir = os.path.join(processed_dir_path, type_item, town_test, "test", "registration")
check_dir(out_dir)
dem_dir = os.path.join(external_dir_path, "dem", "MoroccoDEM", "DEM")
geoid_dir = os.path.join(external_dir_path, "geoid")

geom_path = os.path.join(out_dir, f"{Path(path_slave).stem}.geom")
new_geom_path = os.path.join(out_dir, f"new_{Path(path_slave).stem}.geom")

s1 = pyotb.ReadImageInfo({
    "in": path_slave,
    "outgeom":geom_path,
})


pyotb.HomologousPointsExtraction({
    "in1":path_slave,
    "in2":path_ref,
    "algorithm":"surf",
    "mode":"full",
    #"2wgs84":0,
    "out": os.path.join(out_dir, "homologous_points.txt"),
    "outvector": os.path.join(out_dir, "points.shp"),
    "elev.dem": dem_dir,
    "elev.geoid": geoid_dir
})


#doesn't exist
pyotb.RefineSensorModel({
    "elev.dem":dem_dir,
    "elev.geoid": geoid_dir,
    "ingeom": geom_path,
    "outgeom": new_geom_path,
    "inpoints": os.path.join(out_dir, "homologous_points.txt"),
    "outstat": os.path.join(out_dir, "stats.txt"),
    "outvector": os.path.join(out_dir, "refined_slave_image.shp")

})


otbcli_RefineSensorModel   -elev.dem dem_path/SRTM4-HGT/
                           -elev.geoid OTB-Data/Input/DEM/egm96.grd
                           -ingeom slave_image.geom
                           -outgeom refined_slave_image.geom
                           -inpoints homologous_points.txt
                           -outstat stats.txt
                           -outvector refined_slave_image.shp

otbcli_HomologousPointsExtraction   -in1 slave_image
                                    -in2 reference_image
                                    -algorithm surf
                                    -mode geobins
                                    -mode.geobins.binstep 512
                                    -mode.geobins.binsize 512
                                    -mfilter 1
                                    -precision 20
                                    -2wgs84 1
                                    -out homologous_points.txt
                                    -outvector points.shp
                                    -elev.dem dem_path/SRTM4-HGT/
                                    -elev.geoid OTB-Data/Input/DEM/egm96.grd
                                    

"""