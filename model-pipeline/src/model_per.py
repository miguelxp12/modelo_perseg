```python
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


class ModelPer(ModelInter):

    HYPERPARAMETERS = {
        'n_estimators': 558,
        'lambda_l1': 0.0017906758087515785,
        'lambda_l2': 0.03895805691818066,
        'num_leaves': 26,
        'min_child_samples': 9,
        'max_depth': 19
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
        _x_test: pd.DataFrame,
        _y_test: pd.DataFrame,
        _weights_train: pd.DataFrame,
        _weights_test: pd.DataFrame,
        _type_model: str,
        _var_categoricas2:  List[str]
    ) -> None:
        self.x_train = _x_train
        self.y_train = _y_train
        self.x_test = _x_test
        self.y_test = _y_test
        self.weights_train = _weights_train
        self.weights_test = _weights_test
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
        self
    ) -> None:
        random.seed(self.SEED)
        np.random.seed(self.SEED)

        self.best_model = lgb.LGBMRegressor(**self.HYPERPARAMETERS, random_state=self.SEED)
        self.best_model.fit(
            self.x_train,
            self.y_train,
            categorical_feature=self.vars_categoricas2,
            eval_set=[(self.x_train, self.y_train)]
        )

        test_pred = self.best_model.predict(self.x_test)
        test_wsmape = self._weighted_smape(
            self.y_test.values,
            test_pred,
            self.weights_test.values
        )

        print(
            "Optuna Optimized Model MAE: "
            f"{mean_absolute_error(self.y_test, test_pred):.7f}"
        )
        print(
            "Optuna Optimized W-SMAPE: "
            f"{test_wsmape:.7f}"
        )

    def predict(
        self
    ) -> None:
        test_pred = self.best_model.predict(self.x_test)
        print(
            "Optuna Optimized Model MAE: "
            f"{mean_absolute_error(self.y_test, test_pred):.4f}"
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
        df_: pd.DataFrame,
        columns: List[str],
        name_model: str,
        country_model: str,
        name_dataset: str
    ) -> None:
        df_['PUP_PREDICT'] = self.best_model.predict(df_[columns])
        llave = ['COD_PERIODO','COD_PAIS','COD_CUC','DES_MARCA','DES_CATEGORIA']
        primera_agrupacion = df_.groupby(llave).agg({
            'PUP_CALCULADO': 'sum',
            'PUP_PREDICT': 'sum',
            'REAL_UNIDADES_VENDIDAS': 'sum'
        })
        metric_1 = primera_agrupacion.groupby(['COD_PAIS', 'COD_PERIODO']).apply(Metrics.compute_aggregates).reset_index()
        metric_2 = primera_agrupacion.groupby(['COD_PAIS', 'COD_PERIODO','DES_CATEGORIA']).apply(Metrics.compute_aggregates).reset_index()

        country_path = f'codpais={country_model}/' if country_model else ''
        s3_serv = S3Buckets()
        s3_serv.save_file(
            key=(
                f'{self.path_output_model_metric}/{name_model}/'
                f'{country_path}'
                f'metrics/metrica_{name_dataset}_cuc_pais-campana.csv'
            ),
            obj=metric_1,
            format=S3Buckets.DATA_FRAME_FORMAT
        )
        s3_serv.save_file(
            key=(
                f'{self.path_output_model_metric}/{name_model}/'
                f'{country_path}'
                f'metrics/metrica_{name_dataset}_cuc_pais-campana-categoria.csv'
            ),
            obj=metric_2,
            format=S3Buckets.DATA_FRAME_FORMAT
        )

        cod_periods = sorted(df_['COD_PERIODO'].unique().tolist())
        print(f"cod_periods: {cod_periods}")
        for cod_period in cod_periods:
            _df = df_[df_['COD_PERIODO'] == cod_period]
            plt_ = Metrics.plot_bullseye(_df)
            title = (
                f'bullseye-for-model-{self.type_model}_{name_model}_'
                f'name-ds_{name_dataset}_'
                f'aniocampana_{cod_period}'
            )
            plt_.title(title)

            s3_serv.save_file(
                key=(
                    f'{self.path_output_model_metric}/{name_model}/'
                    f'{country_path}'
                    f'metrics/{title}.png'
                ),
                obj=plt_,
                format= S3Buckets.PLOT_FORMAT
            )

    def save_metrics_test_without_outliers(
        self,
        df_test: pd.DataFrame,
        columns: List[str],
        columns_target: List[str],
        name_model: str,
        country_model: str,
        name_dataset: str
    ) -> None:

        print(f"data_shape test: {df_test.shape}")

        upper_bound = 0.01
        outliers_v2_test = df_test[(df_test['PUP_CALCULADO'] > upper_bound)].copy()
        df_test_cleaned1 = df_test[(df_test['PUP_CALCULADO'] <= upper_bound)].copy()

        print('% DE PUP QUE VAMOS A ELIMINAR')
        print(outliers_v2_test.groupby(['COD_PAIS','COD_PERIODO'])['PUP_CALCULADO'].sum() / df_test.groupby(['COD_PAIS','COD_PERIODO'])['PUP_CALCULADO'].sum())
        print(outliers_v2_test.groupby(['COD_PAIS'])['PUP_CALCULADO'].sum() / df_test.groupby(['COD_PAIS'])['PUP_CALCULADO'].sum())

        df_test_cleaned2 = df_test_cleaned1[df_test_cleaned1['DES_CLASE'].isin(['CUIDADO PERSONAL','FRAGANCIAS','MAQUILLAJE'])].copy()
        #===
        print('% DE PUP CON EL QUE NOS QUEDAMOS')
        print(df_test_cleaned2.groupby(['COD_PAIS','COD_PERIODO'])['PUP_CALCULADO'].sum() / df_test_cleaned1.groupby(['COD_PAIS','COD_PERIODO'])['PUP_CALCULADO'].sum())
        print(df_test_cleaned2.groupby(['COD_PAIS'])['PUP_CALCULADO'].sum() / df_test_cleaned1.groupby(['COD_PAIS'])['PUP_CALCULADO'].sum())

        df_test_cleaned2['PUP_PREDICT'] = self.best_model.predict(df_test_cleaned2[columns])
        #===
        y_test_cleaned2 = df_test_cleaned2[columns_target].copy()
        print(f"Optuna Optimized Model MAE: {mean_absolute_error(y_test_cleaned2, df_test_cleaned2['PUP_PREDICT'] ):.6f}")

        llave = ['COD_PERIODO','COD_PAIS','COD_CUC','DES_MARCA','DES_CATEGORIA']
        primera_agrupacion_test = df_test_cleaned1.groupby(llave).agg({
            'PUP_CALCULADO': 'sum',
            'PUP_PREDICT': 'sum',
            'REAL_UNIDADES_VENDIDAS': 'sum'
        })

        metric_1 = primera_agrupacion_test.groupby(['COD_PAIS', 'COD_PERIODO']).apply(Metrics.compute_aggregates).reset_index()
        metric_2 = primera_agrupacion_test.groupby(['COD_PAIS', 'COD_PERIODO','DES_CATEGORIA']).apply(Metrics.compute_aggregates).reset_index()

        country_path = f'codpais={country_model}/' if country_model else ''
        s3_serv = S3Buckets()
        s3_serv.save_file(
            key=(
                f'{self.path_output_model_metric}/{name_model}/'
                f'{country_path}'
                f'metrics/metrica_{name_dataset}_cuc_pais-campana.csv'
            ),
            obj=metric_1,
            format=S3Buckets.DATA_FRAME_FORMAT
        )
        s3_serv.save_file(
            key=(
                f'{self.path_output_model_metric}/{name_model}/'
                f'{country_path}'
                f'metrics/metrica_{name_dataset}_cuc_pais-campana-categoria.csv'
            ),
            obj=metric_2,
            format=S3Buckets.DATA_FRAME_FORMAT
        )

        plt_ = Metrics.plot_bullseye(df_test_cleaned1)
        title = (
            f'bullseye-for-model-{self.type_model}_{name_model}_'
            'test_without_outliers_'
            f'name-ds_{name_dataset}'
        )
        plt_.title(title)

        s3_serv.save_file(
            key=(
                f'{self.path_output_model_metric}/{name_model}/'
                f'{country_path}'
                f'metrics/{title}.png'
            ),
            obj=plt_,
            format= S3Buckets.PLOT_FORMAT
        )

    def save_shap_metrics(
        self,
        df_: pd.DataFrame,
        columns: List[str],
        name_model: str,
        country_model: str
    ) -> None:

        country_path = f'codpais={country_model}/' if country_model else ''
        def save_shap_fig(
            _shap_values: Any,
            _df: pd.DataFrame,
            _name_imgs: str,
            _bar: bool = False
        ):
            if _bar:
                shap.summary_plot(
                    _shap_values,
                    _df,
                    plot_size=(15,8),
                    show=False,
                    plot_type='bar'
                )
            else:
                shap.summary_plot(
                    _shap_values,
                    _df,
                    plot_size=(15,8),
                    show=False
                )

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close()

            key=(
                    f'{self.path_output_model_metric}/{name_model}/'
                    f'{country_path}'
                    f'metrics/{_name_imgs}.png'
            )
            print(f"key file to save: {key}")

            s3_serv = S3Buckets()
            s3_serv.save_file(
                key=key,
                obj=buf,
                format= S3Buckets.SHAP_FORMAT
            )

        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(df_[columns])
        save_shap_fig(shap_values, df_[columns], 'shap_metrics')

        # Error = actual - predicted
        errors = df_['PUP_CALCULADO'].values - df_['PUP_PREDICT'].values
        over_mask = errors < 0    # Overestimated instances
        under_mask = errors > 0

        # Split SHAP values into error groups
        shap_over = shap_values[over_mask, :]
        shap_under = shap_values[under_mask, :]

        # Mean SHAP values per feature in each group
        mean_shap_over = np.mean(shap_over, axis=0)
        mean_shap_under = np.mean(shap_under, axis=0)

        print(mean_shap_over)
        print(mean_shap_under)

        save_shap_fig(
            shap_over,
            df_[columns][over_mask],
            'shap_metrics_over',
            _bar=True
        )
        save_shap_fig(
            shap_under,
            df_[columns][over_mask], # Corrected: This should be df_[columns][under_mask]
            'shap_metrics_under',
            _bar=True
        )
```
