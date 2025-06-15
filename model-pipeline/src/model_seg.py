from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from util.s3_manager import S3Buckets
from util.enums import MAP_PATH
from util.metrics import Metrics
import lightgbm as lgb
import optuna
import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Any, List
import shap
import matplotlib.pyplot as plt
import io
from model_base import ModelInter
import random
import numpy as np


class ModelSeg(ModelInter):

    HYPERPARAMS = {
        'n_estimators': 3200,
        'lambda_l1': 0.01183819335630482,
        'lambda_l2': 0.49314301297520013,
        'num_leaves': 10,
        'max_depth': 12,
        'subsample': 0.7101439953475182,
        'colsample_bytree': 0.7758295336726955,
        'min_child_samples': 26
    }

    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame
    weights_train: pd.DataFrame
    weights_test: pd.DataFrame
    best_model: Any
    path_output_model_metric: str
    type_model: str
    vars_categoricas2: List[str]
    SEED = 42

    def __init__(
        self,
        _x_train: pd.DataFrame,
        _y_train: pd.DataFrame,
        _weights_train: pd.DataFrame,
        _type_model: str,
        _var_categoricas2: List[str]
    ) -> None:
        self.x_train = _x_train
        self.y_train = _y_train
        self.weights_train = _weights_train
        self.path_output_model_metric = MAP_PATH[_type_model]['PATH_OUTPUT_MODEL_METRIC']
        self.type_model = _type_model
        self.vars_categoricas2 = _var_categoricas2

    def _weighted_smape(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        demand: np.ndarray
    ) -> np.ndarray:
        denominator = demand.sum()
        numerator   = np.abs(y_pred - y_true) / ((np.abs(y_pred) + np.abs(y_true)) / 2)
        return np.sum(numerator * demand) / denominator

    def train(
        self,
    ):
        random.seed(self.SEED)
        np.random.seed(self.SEED)

        # Train final model with best params
        self.best_model = lgb.LGBMRegressor(**self.HYPERPARAMS, random_state=self.SEED)
        print(f"columns: {self.x_train.columns.tolist()}")
        print(f"self.vars_categoricas2: {self.vars_categoricas2}")
        self.best_model.fit(
            self.x_train,
            self.y_train,
            categorical_feature=self.vars_categoricas2,
            eval_set=[(self.x_train, self.y_train)]
        )
        # Test set prediction and metrics removed from here

    def predict(
        self
    ) -> Any:
        # Predict on training set, if needed for other methods
        self.y_pred_train = self.best_model.predict(self.x_train)
        # Test set prediction and metrics removed
        # Optionally, print training MAE or other metrics if desired
        print(
            "Model MAE on Training Data: "
            f"{mean_absolute_error(self.y_train, self.y_pred_train):.4f}"
        )

    def save(
        self,
        anio_campana: str,
        country_code: str
    ) -> None:
        country_path = f'codpais={country_code}/' if country_code else ''
        s3_serv = S3Buckets()
        key_model = (
            f'{self.path_output_model_metric}/{anio_campana}/'
            f'{country_path}'
            f'model/model_{self.type_model}.pkl'
        )
        s3_serv.save_file(
            key=key_model,
            obj=self.best_model,
            format= S3Buckets.PICKLE_FORMAT
        )
        print(
            "saving model in "
            f"{key_model}"
        )

    def save_features_metrics_as_plot(
        self,
        df_train: pd.DataFrame,
        columns: List[str],
        name_model: str,
        country_model: str
    ) -> None:
        feat_imp = pd.Series(
            self.best_model.feature_importances_,
            index=df_train[columns].columns
        )
        feat_imp = feat_imp.nlargest(20).sort_values()
        feat_imp.plot(kind='barh', figsize=(6,8), color='#CA0D0A')
        plt.title("Importance Feature")

        country_path = f'codpais={country_model}/' if country_model else ''
        s3_serv = S3Buckets()
        s3_serv.save_file(
            key=(
                f'{self.path_output_model_metric}/{name_model}/'
                f'{country_path}'
                'metrics/feature_importance.png'
            ),
            obj=plt,
            format= S3Buckets.PLOT_FORMAT
        )

    def save_metrics_for_ds(
        self,
        df_train_raw: pd.DataFrame,
        final_columns: list,
        anio_campana: str,
        country_code: str
    ) -> None:
        # Create a copy to avoid modifying the original DataFrame passed to the function
        df_processed = df_train_raw.copy()
        df_processed['PUP_PREDICT'] = self.best_model.predict(df_processed[final_columns])

        llave = ['COD_PERIODO','COD_PAIS','COD_CUC','DESMARCA','DESCATEGORIA'] # Note: Original comment about DESMARCA/DESCATEGORIA vs DESMARCA3/DESCATEGORIA3 still applies
        primera_agrupacion = df_processed.groupby(llave).agg({
            'PUP_CALCULADO': 'sum',
            'PUP_PREDICT': 'sum',
            'REAL_UNIDADES_VENDIDAS': 'sum'
        })
        metric_1 = primera_agrupacion.groupby(['COD_PAIS', 'COD_PERIODO']).apply(Metrics.compute_aggregates).reset_index()
        metric_2 = primera_agrupacion.groupby(['COD_PAIS', 'COD_PERIODO','DESCATEGORIA']).apply(Metrics.compute_aggregates).reset_index() # Note: Original comment about DESCATEGORIA vs DESCATEGORIA3 still applies

        country_path = f'codpais={country_code}/' if country_code else ''
        s3_serv = S3Buckets()
        s3_serv.save_file(
            key=(
                f'{self.path_output_model_metric}/{anio_campana}/'
                f'{country_path}'
                f'metrics/metrica_training_cuc_pais-campana.csv'  # name_dataset is now fixed to 'training'
            ),
            obj=metric_1,
            format=S3Buckets.DATA_FRAME_FORMAT
        )
        s3_serv.save_file(
            key=(
                f'{self.path_output_model_metric}/{anio_campana}/'
                f'{country_path}'
                f'metrics/metrica_training_cuc_pais-campana-categoria.csv'  # name_dataset is now fixed to 'training'
            ),
            obj=metric_2,
            format=S3Buckets.DATA_FRAME_FORMAT
        )

        cod_periods = sorted(df_processed['COD_PERIODO'].unique().tolist())
        print(f"cod_periods: {cod_periods}")
        for cod_period in cod_periods:
            _df_period = df_processed[df_processed['COD_PERIODO'] == cod_period]
            plt_ = Metrics.plot_bullseye(_df_period)
            title = (
                f'bullseye-for-model-{self.type_model}_{anio_campana}_'
                f'name-ds_training_'  # name_dataset is now fixed to 'training'
                f'aniocampana_{cod_period}'
            )
            plt_.title(title)

            s3_serv.save_file(
                key=(
                    f'{self.path_output_model_metric}/{anio_campana}/'
                    f'{country_path}'
                    f'metrics/{title}.png'
                ),
                obj=plt_,
                format= S3Buckets.PLOT_FORMAT
            )
