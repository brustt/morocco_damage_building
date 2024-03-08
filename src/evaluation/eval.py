

from src.io.utils_io import load_pickle, make_path
from src.config import *
import rasterio as rio 
import rioxarray as rioxr
from rasterstats import zonal_stats, point_query
import geopandas as gpd

img_id ="49a91dea-ed40-4772-b6b9-f9683509b8ac"


urban_area = load_pickle(make_path("urban_area.pkl", processed_dir_path, "dataset"))
urban_test = gpd.read_file(urban_area.loc[urban_area["img_id"] == img_id, "urban_area_path"].item())

rs_path_test = make_path("out_test_pred.tif", project_path, "notebooks", "evaluation")

rs = rioxr.open_rasterio(rs_path_test)
arr = rs.sel(band=1).data

print(rs.shape)
zs = zonal_stats(urban_test, arr, affine=rs.rio.transform(), stats=['count', 'mean', "max",'median', 'majority', 'sum'])
print(zs)