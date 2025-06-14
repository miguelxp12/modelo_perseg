import pandas as pd
import os
from typing import Tuple
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from util.s3_manager import S3Buckets
from util.enums import MAP_PATH
from training.preprocess_base import PreprocessInter


class PreProcessDataPer(PreprocessInter):

    VARS_TARGET = 'PUP_CALCULADO'
    VARS_TO_PROCESS_AND_TRAIN = [
        'FACTOR_CUADRE',
        'FACTOR_REPETICION',
        'COUNT_COD_SAP',
        'PRIORIDADEVENTO',
        'CTD_CONSULTORAS_EXPUESTAS',
        'DSCT_CUC_COMPONENTE',
        'PUP_LAG_1',
        'PUP_LAG_2',
        'PUP_LAG_3',
        'PUP_LAG_4',
        'PUP_LAG_5',
        'PUP_LAG_6',
        'PREV_PUP_AVG_ALL_6',
        'PREV_PUP_STD_ALL_6',
        'PREV_PUP_AVG_1_3',
        'PREV_PUP_AVG_4_6',
        'PREV_PUP_WEIGHTED_AVG',
        'PREV_PUP_DIF_1_2',
        'PREV_PUP_DIF_1_3',
        'PREV_PUP_DIF_1_6',
        'PREV_PUP_DIF_2_3',
        'PREV_PUP_SHIFT',
        'MEDIAN_ORDEN_Q1',
        'MEDIAN_ORDEN_Q2',
        'MEDIAN_ORDEN_Q3',
        'MEDIAN_ORDEN_Q4',
        'RATIO_CONSTANTES_1',
        'RATIO_CONSTANTES_2',
        'RATIO_CONSTANTES_3',
        'RATIO_INCONSTANTES',
        'RATIO_TOPS',
        'RATIO_NUEVAS',
        'RATIO_BRILLA',
        'RATIO_CONVERSION_ACUMULADO_6M',
        'AVG_RATIO_CONVERSION_X_CAMP_6M',
        'STD_RATIO_CONVERSION_X_CAMP_6M',
        'AVG_CATEGORIAOFERTA_PROPORTION_FRAGANCIAS',
        'AVG_CATEGORIAOFERTA_PROPORTION_MAQUILLAJE',
        'AVG_CATEGORIAOFERTA_PROPORTION_RELOJES',
        'AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_FACIAL',
        'AVG_CATEGORIAOFERTA_PROPORTION_VARIOS',
        'AVG_CATEGORIAOFERTA_PROPORTION_CUIDADO_PERSONAL',
        'AVG_CATEGORIAOFERTA_PROPORTION_COMPLEMENTOS',
        'AVG_CATEGORIAOFERTA_PROPORTION_FINART_BIJOUTERIE',
        'AVG_CATEGORIAOFERTA_PROPORTION_DAMAS',
        'AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_CORPORAL',
        'AVG_CATEGORIAOFERTA_PROPORTION_LENTES',
        'AVG_CATEGORIAOFERTA_PROPORTION_HOGAR',
        # que se usa para transformaciones
        'PUP',
        'ES_PADRE',
        'COD_PERIODO',
        'COD_PAIS',
        'DES_MARCA',
        'DES_CATEGORIA',
        'DES_CLASE',
        'REAL_UNIDADES_VENDIDAS'
    ]
    DATA_MAP_DTYPE = {
        "FACTOR_CUADRE": "float64",
        "FACTOR_REPETICION": "float64",
        "COUNT_COD_SAP": "int64",
        "PRIORIDADEVENTO": "int64",
        "CTD_CONSULTORAS_EXPUESTAS": "int64",
        "DSCT_CUC_COMPONENTE": "float64",
        "PUP_LAG_1": "float64",
        "PUP_LAG_2": "float64",
        "PUP_LAG_3": "float64",
        "PUP_LAG_4": "float64",
        "PUP_LAG_5": "float64",
        "PUP_LAG_6": "float64",
        "PREV_PUP_AVG_ALL_6": "float64",
        "PREV_PUP_STD_ALL_6": "float64",
        "PREV_PUP_AVG_1_3": "float64",
        "PREV_PUP_AVG_4_6": "float64",
        "PREV_PUP_WEIGHTED_AVG": "float64",
        "PREV_PUP_DIF_1_2": "float64",
        "PREV_PUP_DIF_1_3": "float64",
        "PREV_PUP_DIF_1_6": "float64",
        "PREV_PUP_DIF_2_3": "float64",
        "PREV_PUP_SHIFT": "float64",
        "MEDIAN_ORDEN_Q1": "float64",
        "MEDIAN_ORDEN_Q2": "float64",
        "MEDIAN_ORDEN_Q3": "float64",
        "MEDIAN_ORDEN_Q4": "float64",
        "RATIO_CONSTANTES_1": "float64",
        "RATIO_CONSTANTES_2": "float64",
        "RATIO_CONSTANTES_3": "float64",
        "RATIO_INCONSTANTES": "float64",
        "RATIO_TOPS": "float64",
        "RATIO_NUEVAS": "float64",
        "RATIO_BRILLA": "float64",
        "RATIO_CONVERSION_ACUMULADO_6M": "float64",
        "AVG_RATIO_CONVERSION_X_CAMP_6M": "float64",
        "STD_RATIO_CONVERSION_X_CAMP_6M": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_FRAGANCIAS": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_MAQUILLAJE": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_RELOJES": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_FACIAL": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_VARIOS": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_CUIDADO_PERSONAL": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_COMPLEMENTOS": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_FINART_BIJOUTERIE": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_DAMAS": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_CORPORAL": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_LENTES": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_HOGAR": "float64",
        "PUP": "float64",
        "ES_PADRE": "int64",
        "COD_PERIODO": "int64",
        "COD_PAIS": "object",
        "DES_MARCA": "object",
        "DES_CATEGORIA": "object",
        "DES_CLASE": "object",
        "REAL_UNIDADES_VENDIDAS": "float64"
    }
    VARS_NUMERICAS = [
        'FACTOR_CUADRE',
        'FACTOR_REPETICION',
        'COUNT_COD_SAP',
        'PRIORIDADEVENTO',
        'CTD_CONSULTORAS_EXPUESTAS',
        'DSCT_CUC_COMPONENTE',
        'PUP_LAG_1',
        'PUP_LAG_2',
        'PUP_LAG_3',
        'PUP_LAG_4',
        'PUP_LAG_5',
        'PUP_LAG_6',
        'PREV_PUP_AVG_ALL_6',
        'PREV_PUP_STD_ALL_6',
        'PREV_PUP_AVG_1_3',
        'PREV_PUP_AVG_4_6',
        'PREV_PUP_WEIGHTED_AVG',
        'PREV_PUP_DIF_1_2',
        'PREV_PUP_DIF_1_3',
        'PREV_PUP_DIF_1_6',
        'PREV_PUP_DIF_2_3',
        'PREV_PUP_SHIFT',
        'MEDIAN_ORDEN_Q1',
        'MEDIAN_ORDEN_Q2',
        'MEDIAN_ORDEN_Q3',
        'MEDIAN_ORDEN_Q4',
        'RATIO_CONSTANTES_1',
        'RATIO_CONSTANTES_2',
        'RATIO_CONSTANTES_3',
        'RATIO_INCONSTANTES',
        'RATIO_TOPS',
        'RATIO_NUEVAS',
        'RATIO_BRILLA',
        'RATIO_CONVERSION_ACUMULADO_6M',
        'AVG_RATIO_CONVERSION_X_CAMP_6M',
        'STD_RATIO_CONVERSION_X_CAMP_6M',
        'AVG_CATEGORIAOFERTA_PROPORTION_FRAGANCIAS',
        'AVG_CATEGORIAOFERTA_PROPORTION_MAQUILLAJE',
        'AVG_CATEGORIAOFERTA_PROPORTION_RELOJES',
        'AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_FACIAL',
        'AVG_CATEGORIAOFERTA_PROPORTION_VARIOS',
        'AVG_CATEGORIAOFERTA_PROPORTION_CUIDADO_PERSONAL',
        'AVG_CATEGORIAOFERTA_PROPORTION_COMPLEMENTOS',
        'AVG_CATEGORIAOFERTA_PROPORTION_FINART_BIJOUTERIE',
        'AVG_CATEGORIAOFERTA_PROPORTION_DAMAS',
        'AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_CORPORAL',
        'AVG_CATEGORIAOFERTA_PROPORTION_LENTES',
        'AVG_CATEGORIAOFERTA_PROPORTION_HOGAR',
    ]

    VAR_CATEGORICAS_2 = [
        'DESMARCA3',
        'DESCATEGORIA3'
    ]

    FINAL_COLUMNS = VARS_NUMERICAS + VAR_CATEGORICAS_2

    DICT_MARCA = {
        'ESIKA': 1,
        'CYZONE': 2,
        'LBEL': 3,
    }

    DICT_CATEGORIA = {
        'MAQUILLAJE': 1,
        'CUIDADO PERSONAL': 2,
        'FRAGANCIAS': 3,
        'TRATAMIENTO FACIAL': 4,
        'TRATAMIENTO CORPORAL': 5,
    }

    UPPER_BOUND = 0.02
    PERCENTAGE_SPLIT = 0.7

    df: pd.DataFrame
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    df_train_orig: pd.DataFrame
    weights_train: pd.DataFrame
    weights_test: pd.DataFrame

    def __init__(
        self,
        _df: pd.DataFrame
    ) -> None:
        self.df = _df

    def prepare_data(
        self
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        # preprocess for data from s3 buckets.
        self.df = self.df.dropna(subset=['REACTIONFLAG'])
        self.df = self.df[self.VARS_TO_PROCESS_AND_TRAIN]
        self.df = self.df.astype(self.DATA_MAP_DTYPE)

        self.df['PUP_CALCULADO'] = self.df['PUP']
        self.df = self.df[self.df['ES_PADRE'] == 1]
        self.df['DESMARCA3'] = self.df['DES_MARCA'].map(self.DICT_MARCA).fillna(1)
        self.df['DESCATEGORIA3'] = self.df['DES_CATEGORIA'].map(self.DICT_CATEGORIA).fillna(1)
        self.df = self.df[self.df[self.VARS_TARGET].notnull()]
        self.df, self.df_train_orig, self.df_test = self.split_by_percentage(self.df, self.PERCENTAGE_SPLIT)
        self.df_train = self.df_train_orig.copy()
        self.df_train = self.df_train_orig[(self.df_train['PUP_CALCULADO'] <= self.UPPER_BOUND)]
        # self.df_train = self.df_train[self.df_train['DESCLASE'].isin(['CUIDADO PERSONAL','FRAGANCIAS','MAQUILLAJE'])]
        self.df_train = self.df_train[self.df_train['DES_CLASE'].isin(['CUIDADO PERSONAL','FRAGANCIAS','MAQUILLAJE'])]

        self.weights_train = self.df_train['REAL_UNIDADES_VENDIDAS']
        self.weights_test = self.df_test['REAL_UNIDADES_VENDIDAS']

        # print information
        #self.print_information_datasets_from_percentage_split()
        self.print_outliers_v1(self.df_train)
        self.print_outliers_v2(self.df_train)

        x_train = self.df_train[self.FINAL_COLUMNS].copy()
        y_train = self.df_train[self.VARS_TARGET].copy()

        x_test = self.df_test[self.FINAL_COLUMNS].copy()
        y_test = self.df_test[self.VARS_TARGET].copy()

        return (
            x_train,
            y_train,
            x_test,
            y_test
        )

    def split_by_periodo(
        self,
        df: pd.DataFrame,
        periodo_split: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df.loc[(df['COD_PERIODO'] <= periodo_split), 'type_subset'] = "1. Train/Valid"
        df.loc[(df['COD_PERIODO'] > periodo_split), 'type_subset'] = "2. Test"
        df_train = df[df['type_subset'] == '1. Train/Valid'].copy()
        df_test  = df[df['type_subset'] == '2. Test'].copy()
        return df, df_train, df_test

    def split_by_percentage(
        self,
        df: pd.DataFrame,
        percentage: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_train, df_test = train_test_split(df, train_size=percentage)
        return df, df_train, df_test

    def print_information_datasets_from_period_split(
        self
    ) -> None:
        print(f'================================================')
        print(self.df['type_subset'].value_counts().sort_index())
        print(f'================================================')
        print('TOTAL BASE DESPUES SPLIT')
        print(self.df[self.VARS_TARGET].mean())
        print('TRAINING SET')
        print(self.df_train['COD_PERIODO'].value_counts(dropna=False))
        print(self.df_train[self.VARS_TARGET].mean())
        print('TEST SET')
        print(self.df_test['COD_PERIODO'].value_counts(dropna=False))
        print(self.df_test[self.VARS_TARGET].mean())

    def print_information_datasets_from_percentage_split(
        self
    ) -> None:
        print(f'================================================')
        print(f"total de rows para data train raw: {self.df_train_orig.shape[0]}")
        print(f"total de rows para data train procesada: {self.df_train.shape[0]}")
        print(f"total de rows para data test: {self.df_test.shape[0]}")
        print(f'================================================')
        print('TOTAL BASE DESPUES SPLIT')
        print(self.df[self.VARS_TARGET].mean())
        print('TRAINING SET')
        print(self.df_train['COD_PERIODO'].value_counts(dropna=False))
        print(self.df_train[self.VARS_TARGET].mean())
        print('TEST SET')
        print(self.df_test['COD_PERIODO'].value_counts(dropna=False))
        print(self.df_test[self.VARS_TARGET].mean())

    def print_outliers_v1(
        self,
        df_train: pd.DataFrame
    ) -> None:
        # Compute outliers using the 1.5*IQR rule for the entire dataset
        Q1 = df_train['PUP_CALCULADO'].quantile(0.25)
        Q3 = df_train['PUP_CALCULADO'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out the outlierss
        outliers = df_train[(df_train['PUP_CALCULADO'] < lower_bound)].copy()

        print(f'lower bound: {lower_bound}')
        print(f'upper bound: {upper_bound}')
        print(f"Total number of outliers: {len(outliers)}")
        print(outliers.groupby(['COD_PAIS','COD_PERIODO'])['PUP_CALCULADO'].sum())

    def print_outliers_v2(
        self,
        df_train: pd.DataFrame
    ) -> None:
        outliers_v2 = df_train[(df_train['PUP_CALCULADO'] > self.UPPER_BOUND)].copy()
        print('% DE OUTLIERS SIENDO ELIMINADOS EN PUP')
        print(outliers_v2.groupby(['COD_PAIS','COD_PERIODO'])['PUP_CALCULADO'].sum() / df_train.groupby(['COD_PAIS','COD_PERIODO'])['PUP_CALCULADO'].sum())
        print(outliers_v2.groupby(['COD_PAIS'])['PUP_CALCULADO'].sum() / df_train.groupby(['COD_PAIS'])['PUP_CALCULADO'].sum())

    def get_data_train_raw(
        self
    ) -> pd.DataFrame:
        return self.df_train_orig

    def get_data_test(
        self
    ) -> pd.DataFrame:
        return self.df_test

    def generate_data_metrics(
        self,
        anio_campana: str,
        country_code: str
    ) -> None:

        country_path = f'codpais={country_code}/' if country_code else ''
        path_metric = os.path.join(
            MAP_PATH['PER']['PATH_OUTPUT_MODEL_METRIC'],
            anio_campana,
            country_path,
            "metric_training"
        )

        # data to plot
        data0 = self.df[self.df['ES_PADRE'] == 1].copy()
        data1 = data0[data0['REACTIONFLAG'] == 0].copy()

        # plot 01
        counts = data1['COD_PERIODO'].value_counts().sort_index()
        counts.index = counts.index.astype(str)
        ax = counts.plot(kind='line', marker='o')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45)
        plt.ylim(0, 2500)

        s3_serv = S3Buckets()
        s3_serv.save_file(
            key=os.path.join(path_metric, "histogram_codperiod.png"),
            obj=plt,
            format=S3Buckets.PLOT_FORMAT
        )
        plt.close()

        # plot 02
        plt.figure(figsize=(12, 6))
        data1.boxplot(column='PUP_CALCULADO', by='COD_PERIODO', grid=False)
        plt.xlabel('Month')
        plt.ylabel('Demand')
        plt.title('Monthly Demand Distribution with Outliers')
        plt.suptitle('')  # Removes the default pandas subtitle
        plt.xticks(rotation=45)
        plt.tight_layout()

        s3_serv.save_file(
            key=os.path.join(path_metric, "monthly_demand_distribution.png"),
            obj=plt,
            format=S3Buckets.PLOT_FORMAT
        )
        plt.close()

        # plot 03
        data2 = data1[~data1['COD_PERIODO'].isin([202416,202417])].copy()
        counts = data2['COD_PERIODO'].value_counts().sort_index()
        counts.index = counts.index.astype(str)
        ax = counts.plot(kind='line', marker='o')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45)
        plt.ylim(0, 2500)

        s3_serv.save_file(
            key=os.path.join(path_metric, "history_cod_period.png"),
            obj=plt,
            format=S3Buckets.PLOT_FORMAT
        )
        plt.close()
