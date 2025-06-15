import pandas as pd
import os
from typing import Tuple
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from util.s3_manager import S3Buckets
from util.enums import MAP_PATH
from training.preprocess_base import PreprocessInter
import numpy as np


class PreProcessDataSeg(PreprocessInter):

    VARS_TARGET = 'PUP_CALCULADO'
    VARS_TO_PROCESS_AND_TRAIN = [
        'RATIO_CONVERSION_ACUMULADO_6M',
        'AVG_RATIO_CONVERSION_X_CAMP_6M',
        'PUP_CUC_MVA_LAG_1',
        'PUP_CUC_MVA_LAG_2',
        'PREV_PUP_CUC_MVA_DIF_1_2',
        'PUP_CUC_MV_LAG_1',
        'PUP_CUC_MV_LAG_2',
        'PREV_PUP_CUC_MV_DIF_1_2',
        'MEDIAN_MV_PROPORTION_WEB',
        'MEDIAN_MV_PROPORTION_CATALOGO',
        'MEDIAN_MV_PROPORTION_REVISTA',
        'MEDIAN_MV_PROPORTION_OTROS',
        'AVG_ORDEN_Q1',
        'AVG_ORDEN_Q2',
        'AVG_ORDEN_Q3',
        'AVG_ORDEN_Q4',
        'RATIO_CONSTANTES_1',
        'RATIO_CONSTANTES_2',
        'RATIO_CONSTANTES_3',
        'RATIO_TOPS',
        'RATIO_NUEVAS',
        'RATIO_BRILLA',
        'AVG_CATEGORIAOFERTA_PROPORTION_FRAGANCIAS',
        'AVG_CATEGORIAOFERTA_PROPORTION_MAQUILLAJE',
        'AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_FACIAL',
        'AVG_CATEGORIAOFERTA_PROPORTION_CUIDADO_PERSONAL',
        'AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_CORPORAL',
        # que se usa para transformaciones
        'PUP',
        'ES_PADRE',
        'REACTIONFLAG',
        'COD_PERIODO',
        'FACTOR_REPETICION',
        'FACTOR_CUADRE',
        'COUNT_COD_SAP',
        'PRIORIDADEVENTO',
        'DISCOUNT_RANGE',
        'CTD_CONSULTORAS_EXPUESTAS',
        'RATIO_INCONSTANTES',
        'PUP_CUC_MVA_LAG_3',
        'PUP_CUC_MV_LAG_3',
        'DES_MARCA',
        'DES_CATEGORIA',
        'DES_CLASE',
        'REAL_UNIDADES_VENDIDAS',
        'COD_PAIS',
        'DSCT_CUC_COMPONENTE'
    ]
    DATA_MAP_DTYPE = {
        "RATIO_CONVERSION_ACUMULADO_6M": "float64",
        "AVG_RATIO_CONVERSION_X_CAMP_6M": "float64",
        "PUP_CUC_MVA_LAG_1": "float64",
        "PUP_CUC_MVA_LAG_2": "float64",
        "PREV_PUP_CUC_MVA_DIF_1_2": "float64",
        "PUP_CUC_MV_LAG_1": "float64",
        "PUP_CUC_MV_LAG_2": "float64",
        "PREV_PUP_CUC_MV_DIF_1_2": "float64",
        "MEDIAN_MV_PROPORTION_WEB": "float64",
        "MEDIAN_MV_PROPORTION_CATALOGO": "float64",
        "MEDIAN_MV_PROPORTION_REVISTA": "float64",
        "MEDIAN_MV_PROPORTION_OTROS": "float64",
        "AVG_ORDEN_Q1": "float64",
        "AVG_ORDEN_Q2": "float64",
        "AVG_ORDEN_Q3": "float64",
        "AVG_ORDEN_Q4": "float64",
        "RATIO_CONSTANTES_1": "float64",
        "RATIO_CONSTANTES_2": "float64",
        "RATIO_CONSTANTES_3": "float64",
        "RATIO_TOPS": "float64",
        "RATIO_NUEVAS": "float64",
        "RATIO_BRILLA": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_FRAGANCIAS": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_MAQUILLAJE": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_FACIAL": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_CUIDADO_PERSONAL": "float64",
        "AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_CORPORAL": "float64",
        "PUP": "float64",
        "ES_PADRE": "int64",
        "REACTIONFLAG": "int64",
        "COD_PERIODO": "int64",
        "FACTOR_REPETICION": "float64",
        "FACTOR_CUADRE": "float64",
        "COUNT_COD_SAP": "int64",
        "PRIORIDADEVENTO": "int64",
        "DISCOUNT_RANGE": "object",
        "CTD_CONSULTORAS_EXPUESTAS": "int64",
        "RATIO_INCONSTANTES": "float64",
        "PUP_CUC_MVA_LAG_3": "float64",
        "PUP_CUC_MV_LAG_3": "float64",
        "DES_MARCA": "object",
        "DES_CATEGORIA": "object",
        "DES_CLASE": "object",
        "REAL_UNIDADES_VENDIDAS": "float64",
        "COD_PAIS": "object",
        "DSCT_CUC_COMPONENTE": "float64"
    }

    VARS_NUMERICAS = [
        'FACTOR_REPETICION2',
        'COUNT_COD_SAP2',
        'PRIORIDADEVENTO2',
        'DISCOUNT_RANGE_MID2',
        'CTD_CONSULTORAS_EXPUESTAS2',
        'RATIO_CONVERSION_ACUMULADO_6M',
        'AVG_RATIO_CONVERSION_X_CAMP_6M',
        'PUP_CUC_MVA_LAG_1',
        'PUP_CUC_MVA_LAG_2',
        'PREV_PUP_CUC_MVA_DIF_1_2',
        'PUP_CUC_MV_LAG_1',
        'PUP_CUC_MV_LAG_2',
        'MEDIAN_MV_PROPORTION_WEB',
        'MEDIAN_MV_PROPORTION_CATALOGO',
        'MEDIAN_MV_PROPORTION_REVISTA',
        'MEDIAN_MV_PROPORTION_OTROS',
        'AVG_ORDEN_Q1',
        'AVG_ORDEN_Q2',
        'AVG_ORDEN_Q3',
        'AVG_ORDEN_Q4',
        'RATIO_CONSTANTES_1',
        'RATIO_CONSTANTES_2',
        'RATIO_CONSTANTES_3',
        'RATIO_TOPS',
        'RATIO_NUEVAS',
        'RATIO_BRILLA',
        'AVG_CATEGORIAOFERTA_PROPORTION_FRAGANCIAS',
        'AVG_CATEGORIAOFERTA_PROPORTION_MAQUILLAJE',
        'AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_FACIAL',
        'AVG_CATEGORIAOFERTA_PROPORTION_CUIDADO_PERSONAL',
        'AVG_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_CORPORAL'
    ]

    COLUMNS_TO_REMOVE = [
        'AVG_CATEGORIAOFERTA_COUNT_EMPTYSTRING',
        'AVG_CATEGORIAOFERTA_PROPORTION_EMPTYSTRING',
        'AVG_CATEGORIAOFERTA_COUNT_NINAS',
        'AVG_CATEGORIAOFERTA_PROPORTION_NINAS',
        'STD_CATEGORIAOFERTA_COUNT_EMPTYSTRING',
        'STD_CATEGORIAOFERTA_PROPORTION_EMPTYSTRING',
        'STD_CATEGORIAOFERTA_COUNT_NINAS',
        'STD_CATEGORIAOFERTA_PROPORTION_NINAS',
        'AVG_CATEGORIAOFERTA_COUNT_NINOS',
        'AVG_CATEGORIAOFERTA_PROPORTION_NINOS',
        'STD_CATEGORIAOFERTA_COUNT_NINOS',
        'STD_CATEGORIAOFERTA_PROPORTION_NINOS',
        'RATIO_EMPTYSTRING',
        'CNT_EMPTYSTRING',
        'CNT_SIN_SEGMENTO',
        'RATIO_SIN_SEGMENTO',
        'AVG_CATEGORIAOFERTA_PROPORTION_DAMAS',
        'AVG_CATEGORIAOFERTA_COUNT_DAMAS',
        'STD_CATEGORIAOFERTA_COUNT_DAMAS',
        'STD_CATEGORIAOFERTA_PROPORTION_DAMAS',
        'AVG_CATEGORIAOFERTA_PROPORTION_VARIOS',
        'AVG_CATEGORIAOFERTA_COUNT_VARIOS',
        'PUP_ND_LAG_5',
        'PUP_ND_LAG_3',
        'PUP_LAG_3',
        'PUP_LAG_5',
        'PUP_LAG_1',
        'PUP_ND_LAG_6',
        'STD_CATEGORIAOFERTA_COUNT_VARIOS',
        'STD_CATEGORIAOFERTA_PROPORTION_VARIOS',
        'AVG_CATEGORIAOFERTA_PROPORTION_MUESTRAS_COSMETICOS',
        'AVG_CATEGORIAOFERTA_COUNT_MUESTRAS_COSMETICOS',
        'STD_CATEGORIAOFERTA_PROPORTION_MUESTRAS_COSMETICOS',
        'STD_CATEGORIAOFERTA_COUNT_MUESTRAS_COSMETICOS',
        'AVG_CATEGORIAOFERTA_PROPORTION_LENTES',
        'AVG_CATEGORIAOFERTA_COUNT_LENTES',
        'AVG_CATEGORIAOFERTA_COUNT_PROMOCION_USUARIOS',
        'AVG_CATEGORIAOFERTA_PROPORTION_PROMOCION_USUARIOS',
        'STD_CATEGORIAOFERTA_PROPORTION_LENTES',
        'STD_CATEGORIAOFERTA_COUNT_LENTES',
        'STD_CATEGORIAOFERTA_PROPORTION_PROMOCION_USUARIOS',
        'STD_CATEGORIAOFERTA_COUNT_PROMOCION_USUARIOS',
        'AVG_CATEGORIAOFERTA_COUNT_HOGAR',
        'AVG_CATEGORIAOFERTA_PROPORTION_HOGAR',
        'STD_CATEGORIAOFERTA_COUNT_HOGAR',
        'STD_CATEGORIAOFERTA_PROPORTION_HOGAR',
        'AVG_CATEGORIAOFERTA_COUNT_RELOJES',
        'AVG_CATEGORIAOFERTA_PROPORTION_RELOJES',
        'STD_CATEGORIAOFERTA_PROPORTION_RELOJES',
        'STD_CATEGORIAOFERTA_COUNT_RELOJES',
        'AVG_CATEGORIAOFERTA_PROPORTION_FINART_BIJOUTERIE',
        'AVG_CATEGORIAOFERTA_COUNT_FINART_BIJOUTERIE',
        'STD_CATEGORIAOFERTA_COUNT_TRATAMIENTO_CORPORAL',
        'STD_CATEGORIAOFERTA_PROPORTION_TRATAMIENTO_CORPORAL',
        'AVG_CATEGORIAOFERTA_PROPORTION_COMPLEMENTOS',
        'AVG_CATEGORIAOFERTA_COUNT_COMPLEMENTOS',
        'STD_CATEGORIAOFERTA_COUNT_FINART_BIJOUTERIE',
        'STD_CATEGORIAOFERTA_PROPORTION_FINART_BIJOUTERIE',
        'STD_CATEGORIAOFERTA_COUNT_COMPLEMENTOS',
        'STD_CATEGORIAOFERTA_PROPORTION_COMPLEMENTOS'
    ]

    VAR_CATEGORICAS_2 = [
        'DESMARCA3',
        'DESCATEGORIA3'
    ]

    FINAL_COLUMNS = VARS_NUMERICAS + [
        'DESMARCA3',
        'DESCATEGORIA3'
    ]
    FINAL_COLUMNS: list = None

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

    new_colnames_multiplicacion = []

    UPPER_BOUND = 0.04
    PERCENTAGE_SPLIT = 0.7

    df: pd.DataFrame
    df_train: pd.DataFrame
    weights_train: pd.DataFrame

    def __init__(
        self,
        _df: pd.DataFrame
    ) -> None:
        self.df = _df
        self.new_colnames_multiplicacion = []
        self.FINAL_COLUMNS = []

    def prepare_data(
        self
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # preprocess for data from s3 buckets.
        self.df = self.df.dropna(subset=['REACTIONFLAG'])
        self.df = self.df[self.VARS_TO_PROCESS_AND_TRAIN]
        self.df = self.df.astype(self.DATA_MAP_DTYPE)
        self.df = self.df[self.df['DISCOUNT_RANGE'] != 'Out of Range']

        self.df['PUP_CALCULADO'] = self.df['PUP'] # data01['PUP_CALCULADO'] = data01['PUP']
        self.df = self.df[self.df['ES_PADRE'] == 1].copy()
        self.df = self.df[self.df['REACTIONFLAG'] == 0].copy()

        # FACTOR REPETICION
        self.df['FACTOR_REPETICION2'] = self.df['FACTOR_REPETICION']
        self.df.loc[(self.df['FACTOR_REPETICION'] >= 2), 'FACTOR_REPETICION2'] = 2

        # FACTOR CUADRE
        self.df['FACTOR_CUADRE2'] = self.df['FACTOR_CUADRE']
        self.df.loc[(self.df['FACTOR_CUADRE'] >= 3), 'FACTOR_CUADRE2'] = 3

        # COUNT_COD_SAP
        self.df['COUNT_COD_SAP2'] = self.df['COUNT_COD_SAP']
        self.df.loc[(self.df['COUNT_COD_SAP'] >= 2), 'COUNT_COD_SAP2'] = 2

        # PRIORIDADEVENTO
        self.df['PRIORIDADEVENTO2'] = self.df['PRIORIDADEVENTO']
        self.df.loc[(self.df['PRIORIDADEVENTO'] >= 11), 'PRIORIDADEVENTO2'] = 11

        # DISCOUNT_RANGE_MID
        parts = self.df['DISCOUNT_RANGE'].str.split('-', expand=True).astype(float)
        self.df['disc_low'], self.df['disc_high'] = parts[0], parts[1]
        self.df['DISCOUNT_RANGE_MID'] = (self.df['disc_low'] + self.df['disc_high']) / 2

        #===
        self.df['DISCOUNT_RANGE_MID2'] = self.df['DISCOUNT_RANGE_MID']
        self.df.loc[(self.df['DISCOUNT_RANGE_MID'] <= 18), 'DISCOUNT_RANGE_MID2'] = 10.5
        self.df.loc[(self.df['DISCOUNT_RANGE_MID'] >= 50), 'DISCOUNT_RANGE_MID2'] = 55

        # CTD_CONSULTORAS_EXPUESTAS
        self.df['CTD_CONSULTORAS_EXPUESTAS2'] = np.log1p(self.df['CTD_CONSULTORAS_EXPUESTAS'])

        lag_cols  = [
            'PUP_CUC_MVA_LAG_1',
            'PUP_CUC_MVA_LAG_2',
            'PUP_CUC_MVA_LAG_3',
            'PUP_CUC_MV_LAG_1',
            'PUP_CUC_MV_LAG_2',
            'PUP_CUC_MV_LAG_3'
        ]

        ratio_cols = [
            'RATIO_CONSTANTES_1',
            'RATIO_CONSTANTES_2',
            'RATIO_CONSTANTES_3',
            'RATIO_INCONSTANTES',
            'RATIO_TOPS',
            'RATIO_NUEVAS',
            'RATIO_BRILLA'
        ]

        for lag in lag_cols:
            for ratio in ratio_cols:
                new_col = f"{lag}_x_{ratio}"
                self.df[new_col] = self.df[lag] * self.df[ratio]
                self.new_colnames_multiplicacion.append(new_col)

        final_columns = self.VARS_NUMERICAS +                ['DESMARCA3', 'DESCATEGORIA3'] +                self.new_colnames_multiplicacion

        self.FINAL_COLUMNS = [i for i in final_columns if i not in self.COLUMNS_TO_REMOVE]
        print(f"self.FINAL_COLUMNS: {self.FINAL_COLUMNS}")
        final_col = self.FINAL_COLUMNS
        final_col.sort()
        print(final_col)

        self.df = self.df[self.df['ES_PADRE'] == 1].copy()

        self.df['DESMARCA3'] = self.df['DES_MARCA'].map(self.DICT_MARCA).fillna(1)
        self.df['DESCATEGORIA3'] = self.df['DES_CATEGORIA'].map(self.DICT_CATEGORIA).fillna(1)

        self.df = self.df[self.df[self.VARS_TARGET].notnull()]

        self.df_train = self.df.copy() # Assign self.df and then filter
        self.df_train = self.df_train[(self.df_train['PUP_CALCULADO'] <= self.UPPER_BOUND)]
        self.df_train = self.df_train[self.df_train['DES_CLASE'].isin([
            'CUIDADO PERSONAL',
            'FRAGANCIAS',
            'MAQUILLAJE',
            'TRATAMIENTO FACIAL',
            'TRATAMIENTO CORPORAL'
        ])]

        self.weights_train = self.df_train['REAL_UNIDADES_VENDIDAS']

        # print information
        self.print_outliers_v1(self.df_train)
        self.print_outliers_v2(self.df_train)

        x_train = self.df_train[self.FINAL_COLUMNS].copy()
        y_train = self.df_train[self.VARS_TARGET].copy()

        return (
            x_train,
            y_train
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

    def generate_data_metrics(
        self,
        anio_campana: str,
        country_code: str
    ) -> None:

        country_path = f'codpais={country_code}/' if country_code else ''
        path_metric = os.path.join(
            MAP_PATH['SEG']['PATH_OUTPUT_MODEL_METRIC'],
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
        ax.set_ylim([0,1500])

        s3_serv = S3Buckets()
        s3_serv.save_file(
            key=os.path.join(path_metric, "histogram_codperiod.png"),
            obj=ax,
            format=S3Buckets.PLOT_FORMAT
        )
        plt.close(ax.figure)

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
        counts = data1['COD_PERIODO'].value_counts().sort_index()
        counts.index = counts.index.astype(str)
        ax = counts.plot(kind='line', marker='o')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45)
        ax.set_ylim([0, 600])

        s3_serv.save_file(
            key=os.path.join(path_metric, "history_cod_period.png"),
            obj=ax,
            format=S3Buckets.PLOT_FORMAT
        )
        plt.close()
