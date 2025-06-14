
import os

FORECAST_DSS_PER_SEG_FOLDER = os.getenv("FORECAST_DSS_PER_SEG_FOLDER")
TRAINING_FOLDER_SEG = os.getenv("TRAINING_FOLDER_SEG")
OUTPUT_FOLDER_SEG = os.getenv("OUTPUT_FOLDER_SEG")
TRAINING_FOLDER_PER = os.getenv("TRAINING_FOLDER_PER")
OUTPUT_FOLDER_PER = os.getenv("OUTPUT_FOLDER_PER")

RANGE_CAMPAIGNS = int(os.getenv("RANGE_CAMPAIGNS", "18"))

MODEL_PER = 'PER'
MODEL_SEG = 'SEG'

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