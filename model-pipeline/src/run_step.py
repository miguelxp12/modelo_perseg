# for local tests
from dotenv import load_dotenv
load_dotenv('forecastperseg.env')

import argparse
import traceback
import http.client
import json
import os

from typing import Dict, Any
from model_base import ModelInter
from model_per import ModelPer
from model_seg import ModelSeg
from training.preprocess_base import PreprocessInter
from training.preprocess_per import PreProcessDataPer
from training.preprocess_seg import PreProcessDataSeg
from training.train_data_service import TrainDataService
from util.generate_data_training import APIData
from util.enums import (
    MODEL_PER,
    MODEL_SEG
)
from predict.predict_service import PredictService


def parser_arguments() -> Dict[str, Any]:

    def str2bool(v):
        return v.lower() in ('yes', 'true', 't', '1')

    parser = argparse.ArgumentParser()
    parser.add_argument('--type_operation', type=str, help='type_operation', default="TRAIN")
    parser.add_argument('--pais', type=str, help="pais", default="")
    parser.add_argument('--periodo', type=str, help="periodo", default="")
    parser.add_argument('--save_features', type=str2bool, help='save_features', default=False)
    parser.add_argument('--gen_data', type=str2bool, help='gen_data', default=False)
    parser.add_argument('--type_model', type=str, help="type_model", default="all")
    known_args, _ = parser.parse_known_args()
    args = vars(known_args)
    return args

def run_training_process(
    country: str,
    campaign: str,
    save_features: bool,
    type_model: str,
    ModelServ: ModelInter,
    PreProcessData: PreprocessInter
) -> None:
    try:
        df_data_raw = None
        train_data_serv = TrainDataService(type_model)
        if country:
            df_data_raw = train_data_serv.generated_df_by_codpais_aniocampana(
                country,
                campaign,
            )
            df_data_raw.to_csv('df_train.csv', sep='	')
        else:
            df_data_raw = train_data_serv.generated_corportivo_df(campaign)

        print(df_data_raw.info())
        print(f"shape: {df_data_raw.shape}")
        preproc_data_serv = PreProcessData(df_data_raw)
        data_res = preproc_data_serv.prepare_data(is_production=True)

        if save_features:
            preproc_data_serv.generate_data_metrics(
                campaign,
                country
            )

        x_train = data_res[0]
        y_train = data_res[1]
        x_test = data_res[2]
        y_test = data_res[3]

        model_serv = ModelServ(
            x_train,
            y_train,
            x_test,
            y_test,
            preproc_data_serv.weights_train,
            preproc_data_serv.weights_test,
            type_model,
            PreProcessData.VAR_CATEGORICAS_2
        )
        model_serv.train()
        model_serv.predict()
        model_serv.save(campaign, country)

        if save_features:
            model_serv.save_features_metrics_as_plot(
                preproc_data_serv.get_data_train_raw(),
                preproc_data_serv.FINAL_COLUMNS,
                campaign,
                country
            )
            model_serv.save_metrics_for_ds(
                preproc_data_serv.get_data_train_raw(),
                preproc_data_serv.FINAL_COLUMNS,
                campaign,
                country,
                name_dataset='training'
            )
        if x_test is not None and y_test is not None:
            model_serv.save_metrics_for_ds(
                preproc_data_serv.get_data_test(),
                preproc_data_serv.FINAL_COLUMNS,
                campaign,
                country,
                name_dataset='testing'
            )
            model_serv.save_metrics_test_without_outliers(
                preproc_data_serv.get_data_test(),
                preproc_data_serv.FINAL_COLUMNS,
                preproc_data_serv.VARS_TARGET,
                campaign,
                country,
                name_dataset='testing_without_outliers'
            )
            model_serv.save_shap_metrics(
                preproc_data_serv.get_data_test(),
                preproc_data_serv.FINAL_COLUMNS,
                campaign,
                country
             )

    except Exception as e:
        print(f"error: {e}")
        print(traceback.print_exc())


def run_predicting_process(
    type_model: str,
    predict_serv: PredictService()
) -> None:
    predict_serv.predict(type_model)

# python run_step.py --gen_data true
# python run_step.py --periodo 202501 --save_features true
# python run_step.py --pais PE --periodo 202504 --save_features true
# python run_step.py --pais PE --periodo 202504 --save_features true --tyoe_model ALL

# local
if __name__ == "__main__":
    args = parser_arguments()
    print(f"args: {args}")
    type_operation = args['type_operation']
    gen_data = args['gen_data']

    if type_operation.upper() in ['TRAIN', 'PREDICT']:

        if type_operation.upper() == 'TRAIN':
            if gen_data:
                APIData.generate_data()
            else:
                type_model = args['type_model']
                type_model = type_model.replace('', '')
                if type_model.upper() in ('PER', 'ALL'):
                    run_training_process(
                        args['pais'],
                        args['periodo'],
                        args['save_features'],
                        MODEL_PER,
                        ModelServ=ModelPer,
                        PreProcessData=PreProcessDataPer
                    )

                if type_model.upper() in ('SEG', 'ALL'):
                    run_training_process(
                        args['pais'],
                        args['periodo'],
                        args['save_features'],
                        MODEL_SEG,
                        ModelServ=ModelSeg,
                        PreProcessData=PreProcessDataSeg
                    )

        if type_operation.upper() == 'PREDICT':
            predict_serv = PredictService()
            predict_serv.generate_live_features()

            if type_model.upper() in ('PER', 'ALL'): # This line has an issue: type_model is not defined in this scope
                run_predicting_process(
                    MODEL_PER,
                    predict_serv
                )

            if type_model.upper() in ('SEG', 'ALL'): # This line has an issue: type_model is not defined in this scope
                run_predicting_process(
                    MODEL_SEG,
                    predict_serv
                )
