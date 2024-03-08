import cv2
import numpy as np
import imageio
import time
from src.io.utils_io import check_dir
from src.modelling.utils import otsu
from src.modelling.utils import stad_img
from pathlib import Path 
from src.config import *
import rioxarray as rioxr
from src.rs_processing.processing import histogram_alignment

def CVA(img_X, img_Y, stad=False):
    # CVA has not affinity transformation consistency, so it is necessary to normalize multi-temporal images to
    # eliminate the radiometric inconsistency between them
    if stad:
        img_X = stad_img(img_X)
        img_Y = stad_img(img_Y)
    img_diff = img_X - img_Y
    L2_norm = np.sqrt(np.sum(np.square(img_diff), axis=0))
    return L2_norm


def main():
    type_item = "visual"
    town_test = "Talat_Nyaaqoub"
    s_buffer = 200
    out_dir = check_dir(results_dir_path, "cva", town_test)
    
    registration_path = Path(processed_dir_path, type_item, town_test)
    #post_path = Path(registration_path, f"after_{s_buffer}_shifted__global_10300500E4F91500-visual.tif")
    #pre_path = Path(registration_path, f"before_{s_buffer}_104001004F028F00-visual.tif")
    pre_path = Path(interim_dir_path, type_item, town_test, "nearest","before", f"{s_buffer}_1040010077691A00-visual.tif")
    #post_path = Path(registration_path, "registration", f"after_{s_buffer}_shifted__global_10300500E4F92300-visual.tif")
    post_path = Path(registration_path, f"{s_buffer}_after_10300500E4F92300-visual_rgsted.tif")



    pre_img = rioxr.open_rasterio(pre_path).data
    post_img = rioxr.open_rasterio(post_path).data

    pre_img = pre_img.transpose(1, 2, 0)
    post_img = post_img.transpose(1, 2, 0)

    post_img = histogram_alignment(post_img, pre_img)
    
    pre_img = cv2.GaussianBlur(pre_img, (15,15),0)
    post_img = cv2.GaussianBlur(post_img, (15,15),0)
    
    pre_img = pre_img.transpose(2, 0, 1)
    post_img = post_img.transpose(2, 0, 1)

    channel, img_height, img_width = pre_img.shape
    tic = time.time()
    L2_norm = CVA(pre_img, post_img)

    bcm = np.ones((img_height, img_width))
    thre = otsu(L2_norm.reshape(1, -1))
    bcm[L2_norm > thre] = 255
    bcm = np.reshape(bcm, (img_height, img_width))
    imageio.imwrite(os.path.join(out_dir, f'{2}_{s_buffer}_{town_test}.png'), bcm.astype(np.uint8))
    toc = time.time()
    print(toc - tic)


if __name__ == '__main__':
    main()