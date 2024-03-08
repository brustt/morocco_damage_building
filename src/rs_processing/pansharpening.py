
import itertools
#from src.io.utils_io import *
from src.config import *
#from src.rs_processing.processing import crop_rs_on_shape
#from src.stac.utils_stac import get_nearest_match_geometry_view
#from src.utils import get_maxar_items_on_roi
import numpy as np
from pathlib import Path
import os
import otbApplication
import pyotb

"""
DRAFT NOT USED
To CLEAN
OTB and Rasterio issue
"""


type_item = 'visual'
s_buffer=200
town_test = "Talat_Nyaaqoub"

#maxar_items = load_maxar_items(type_item=type_item)
#town = load_roi_town(buffer_size=s_buffer)

#out_path_psh_b = check_dir(processed_dir_path, "pan_sharp", town_test, "before")
#out_dir_pan_a = check_dir(interim_dir_path, type_item, town_test, "test_ortho")
out_dir_pan_b = os.path.join(project_path, "data/tmp")
#out_dir_psh_a = check_dir(processed_dir_path, type_item, town_test, "test_ortho")
field = "view_incidence_angle"

#nearest = get_nearest_match_geometry_view(town, field, maxar_items, type_item=type_item, s_bbox=None)
#in_path = make_path(f"nearest_{type_item}.pkl", interim_dir_path, type_item)
#nearest = load_pickle(in_path)

#b_rs_path = nearest[nearest.name == town_test]["before"].item()
#a_rs_path = nearest[nearest.name == town_test]["after"].item()

#a_rs_name = Path(a_rs_path).stem

b_rs_path = os.path.join(out_dir_pan_b, '3_300_103001008244DA00-visual.tif') 
b_rs_name = Path(b_rs_path).stem

dem_dir = os.path.join(external_dir_path, "dem", "MoroccoDEM", "DEM")
geoid_dir = os.path.join(external_dir_path, "geoid")

#out_path_pan = os.path.join(out_dir_pan_a, f"{a_rs_name}.tif")
#out_path_psh = os.path.join(out_dir_psh_a, f"{a_rs_name}.tif")

#shape = create_bbox(town[town.town_name == town_test], s_buffer=s_buffer)
#pan_img = crop_rs_on_shape(a_rs_path, shape, out_path=out_path_pan)


b_ortho = pyotb.OrthoRectification({
    'io.in': b_rs_path, 
    'io.out':os.path.join(out_dir_pan_b, f"ortho_{b_rs_name}.tif"),
    'elev.dem': dem_dir, 
    'elev.geoid': geoid_dir
})

""""

a_ortho = pyotb.OrthoRectification({
    'io.in': a_rs_path, 
    'io.out' : os.path.join(out_dir_pan_a, f"{b_rs_name}.tif"),
    'elev.dem': dem_dir, 
    'elev.geoid': geoid_dir
})

"""
"""
get multispectral
"""

# default pansharpening

"""
psh = pyotb.Pansharpening({
    "inp":"",
    "inxs":"",
    "out":"",
    
})
"""

"""

which does not use sensor models as Pan and XS products are already coregistered but only estimate an affine transformation to superimpose XS over the Pan

pan_ortho = pyotb.OrthoRectification({
    'io.in': pan, 
    'elev.dem': srtm, 
    'elev.geoid': geoid
})
ms_ortho = pyotb.OrthoRectification({
    'io.in': ms, 
    'elev.dem': srtm, 
    'elev.geoid': geoid
})

pxs = pyotb.BundleToPerfectSensor(
    inp=pan_ortho, 
    inxs=ms_ortho, 
    method='bayes', 
    mode='default'
)

exfn = '?gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:BIGTIFF=YES'
# Here we trigger every app in the pipeline and the process is blocked until 
# result is written to disk
pxs.write('pxs_image.tif', pixel_type='uint16', ext_fname=exfn)

"""