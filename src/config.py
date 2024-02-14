import os
from dotenv import find_dotenv

CRS = 32629
project_path = os.path.dirname(find_dotenv())
raw_data_path = os.path.join(project_path, "data/raw")
interim_dir_path = os.path.join(project_path, "data/interim")
processed_dir_path = os.path.join(project_path, "data/processed")
external_dir_path = os.path.join(project_path, "data/external")
models_dir_path = os.path.join(project_path, "models")

town_path = os.path.join(external_dir_path, "villages_location", "marocco_town.shp")

maxar_items_path = os.path.join(raw_data_path, "maxar", "{}", "maxar_stac_items.shp")
maxar_items_columns_path = os.path.join(
    raw_data_path, "maxar", "{}", "mapping_columns_stac_collection.pkl"
)


VALID_MODE = ["ms_analytic", "visual", "data_mask", "pan_analytic"]
date_event = "2023-09-09"


IMG_TEST = {
    "Tafeghaghte":{
        "before":os.path.join(interim_dir_path, "visual","Tafeghaghte","before", "before_10400100797DAC00-visual.png"), 
        "after":os.path.join(interim_dir_path, "visual","Tafeghaghte","after","after_10300500E4F92300-visual.png")
        },
}