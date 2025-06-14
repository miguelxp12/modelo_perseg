import os
from typing import Dict, Any
import argparse

FORECAST_DSS_PER_SEG_FOLDER = os.getenv("FORECAST_DSS_PER_SEG_FOLDER")
TRAINING_FOLDER_SEG = os.getenv("TRAINING_FOLDER_SEG")
OUTPUT_FOLDER_SEG = os.getenv("OUTPUT_FOLDER_SEG")
TRAINING_FOLDER_PER = os.getenv("TRAINING_FOLDER_PER")
OUTPUT_FOLDER_PER = os.getenv("OUTPUT_FOLDER_PER")

RANGE_CAMPAIGNS = int(os.getenv("RANGE_CAMPAIGNS", "18"))

MODEL_PER = 'PER'
MODEL_SEG = 'SEG'
MODEL_ALL = 'ALL'

TAG_MODELS = [
    MODEL_PER,
    MODEL_SEG
]

MAP_PATH = {
    MODEL_PER: {
        "PATH_GLOBAL_PREFIX_TRAINING": (
            f"{FORECAST_DSS_PER_SEG_FOLDER}/"
            f"{TRAINING_FOLDER_PER}"
        ),
        "PATH_OUTPUT_MODEL_METRIC": (
            f"{FORECAST_DSS_PER_SEG_FOLDER}/"
            f"{OUTPUT_FOLDER_PER}"
        )
    },
    MODEL_SEG: {
        "PATH_GLOBAL_PREFIX_TRAINING": (
            f"{FORECAST_DSS_PER_SEG_FOLDER}/"
            f"{TRAINING_FOLDER_SEG}"
        ),
        "PATH_OUTPUT_MODEL_METRIC": (
            f"{FORECAST_DSS_PER_SEG_FOLDER}/"
            f"{OUTPUT_FOLDER_SEG}"
        )
    }
}

COD_COUNTRIES = [
    'CO',
    'PE',
    'MX',
    'EC',
    'CL',
    'BO',
    'CR',
    'GT',
    'SV',
    'PA',
    'DO',
    'PR'
]

OFFER_DATA = 'Offer.parquet'
CONSULTANT_DATA = 'Consultant.parquet'
EVENT_DATA = 'Event.parquet'
BUCKET_NAME_PREDICT_DATA = os.getenv("BUCKET_NAME_PREDICT")
PATH_TO_DATA_PREDICT = os.getenv("PATH_TO_DATA_PREDICT")
KEY_PATH_OFFER_DATA = os.path.join(
    BUCKET_NAME_PREDICT_DATA,
    PATH_TO_DATA_PREDICT
)


class PREDICT_DATA:
    OFFER = f"{KEY_PATH_OFFER_DATA}/{OFFER_DATA}"
    CONSULTANT = f"{KEY_PATH_OFFER_DATA}/{CONSULTANT_DATA}"
    EVENT = f"{KEY_PATH_OFFER_DATA}/{EVENT_DATA}"


BUCKET_PRODUCTS = os.getenv("BUCKET_NAME_PRODUCTS")
PATH_PRODUCTS = os.getenv("PATH_PRODUCTS")
PATH_PRODUCT_FILES = os.path.join(
    BUCKET_PRODUCTS,
    PATH_PRODUCTS
)

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

def parser_arguments(conf_params: Dict) -> Dict[str, Any]:

    parser = argparse.ArgumentParser()
    for item in conf_params:
        name_param = item[0]
        type_param = item[1]
        help_param = item[2]
        default_param = item[3]

        parser.add_argument(
            name_param,
            type_param,
            help_param,
            default_param
        )
    known_args, _ = parser.parse_known_args()
    args = vars(known_args)
    return args
