import pandas as pd
import dask.dataframe as dd
import numpy as np
from util.s3_manager import S3Buckets
from predict.feature_base import FeatureInter
from predict.sql_queries import (
    SQL_QUERY_FEATURE_FACTOR_REPETICION
)
from util.enums import (
    PREDICT_DATA,
    BUCKET_PRODUCTS,
    PATH_PRODUCTS
)
from predict.operation_service import OperationDask as OpD


class FeatureCuadre(FeatureInter):

    """
        Esta clase calcula los features:            
            - DSCT_CUC_COMPONENTE
            - FACTOR_CUADRE
            - FACTOR_REPETICION
        basado en la agrupación de:
            - COD_PERIODO
            - COD_PAIS
            - TARGET
            - ALCANCE
            - COD_MEDIOVENTA
            - COD_SUBESTRATEGIA
            - ID_OFERTA
            - ID_SUBESTRATEGIA
            - COD_GRUPO
            - ES_GRUPO_PADRE
            - COD_SUBGRUPO
            - COD_TIPO_GRUPO
            - COD_TIPO_SUBGRUPO
            - COD_CUC
            - ES_PADRE
    """

    data_file: str = ""
    b_products: str = "" 
    data_product_files: str = ""
    df: dd.DataFrame = None
    df_offer: dd.DataFrame = None
    df_products: dd.DataFrame = None
    feature_df: dd.DataFrame = None
    meta_dict = {
        "CODPAIS": "object",
        "ANIOCAMPANA": "int64",
        "OFFERID": "int64",
        "INICIOEVENTO": "object",
        "FINEVENTO": "object",
        "DESCRIPCIONEVENTO": "object",
        "PRIORIDADEVENTO": "float64"
    }
    cols_offer = [
        "COD_PERIODO",
        "COD_PAIS",
        "COD_ALCANCE",
        "COD_MEDIO_VENTA",
        "COD_TIPO_SUBESTRATEGIA",
        "ID_OFERTA",
        "ID_SUBESTRATEGIA",
        "ID_SUBGRUPO",
        "ES_GRUPO_PADRE",
        "ID_GRUPO",
        "COD_TIPO_GRUPO",
        "COD_TIPO_SUBGRUPO",
        "ES_PADRE",
        "COD_SAP",
        "FACTOR_CUADRE",
        "FACTOR_REPETICION",
        'PRECIO_NORMAL_MN',
        'PRECIO_UNITARIO_FINAL_MN'
    ]
    columns_group_by = [
        "COD_PERIODO",
        "COD_PAIS",
        "TARGET",
        "ALCANCE",
        "COD_MEDIOVENTA",
        "COD_SUBESTRATEGIA",
        "ID_OFERTA",
        "ID_SUBESTRATEGIA",
        "COD_GRUPO",
        "ES_GRUPO_PADRE",
        "COD_SUBGRUPO",
        "COD_TIPO_GRUPO",
        "COD_TIPO_SUBGRUPO",
        "COD_CUC",
        "ES_PADRE"    
    ]

    def __init__(self):
        self.data_file = PREDICT_DATA.OFFER
        self.b_products = BUCKET_PRODUCTS
        self.data_product_files = PATH_PRODUCTS

    def _get_data_offer(self) -> dd.DataFrame:
        df_offer = OpD.instance_df_from_path_parquet(
            self.data_file
        )
        df_offer = df_offer[self.cols_offer]
        return df_offer

    def _get_data_product(self) -> dd.DataFrame:
        s3_serv = S3Buckets(self.b_products)
        product_parquets = s3_serv.get_list_files_by_prefix(
            self.data_product_files
        )
        df_products = OpD.instance_df_from_list_parquet(
            product_parquets
        )
        return df_products

    def get_data(self) -> None:
        self.df_offer = self._get_data_offer()
        self.df_products = self._get_data_product()

    def generate_features(self) -> None:
        self.df = self.df_offer.merge(
            self.df_products,
            left_on=['COD_PAIS', 'COD_SAP'],
            right_on=['CODPAIS', 'CODPRODUCTOSAP'],
            how='inner'
        )

        df_offer_product = self.df.rename(
            columns={
                'PRECIO_NORMAL_MN': 'PRECIO_NORMAL',
                'PRECIO_UNITARIO_FINAL_MN': 'PRECIO_OFERTA',
                'COD_MEDIO_VENTA': 'COD_MEDIOVENTA',
                'COD_TIPO_SUBESTRATEGIA': 'COD_SUBESTRATEGIA',
                'ID_SUBGRUPO': 'COD_SUBGRUPO',
                'ID_GRUPO': 'COD_GRUPO',
                "COD_ALCANCE": "ALCANCE",
                "CODCUC": "COD_CUC"
            }
        )
        df_offer_product_trans_01 = OpD.apply_sql_sentence_over_daskframe(
            dask_frame=df_offer_product,
            query=SQL_QUERY_FEATURE_FACTOR_REPETICION
        )
        self.feature_df = df_offer_product_trans_01.groupby(
                self.columns_group_by
            ).apply(
                FeatureCuadre.calc_ratio,
                    meta={
                        'DSCT_CUC_COMPONENTE': 'float64',
                        'FACTOR_CUADRE': 'float64',
                        'FACTOR_REPETICION': 'float64'
                    }
        )

    @property
    def export(self) -> dd.DataFrame:
        return self.feature_df

    def _get_data_products(self) -> None:
        self.df_products = None

    @staticmethod
    def calc_ratio(group: dd.DataFrame) -> pd.DataFrame:
        avg_factor_cuadre = group['FACTOR_CUADRE'].mean()

        avg_col1 = group['PRECIO_OFERTA'].mean()
        avg_col2 = group['PRECIO_NORMAL'].mean()
        dsc_cuc_componente = avg_col1 / avg_col2 if avg_col2 != 0 else pd.NA    

        conditions = [
            group['COD_TIPO_SUBGRUPO'].iloc[0] == 'F'
        ]
        sum_factor_repeticion = group['FACTOR_REPETICION'].sum()
        choices = [sum_factor_repeticion]
        default = group['FACTOR_REPETICION'].mean()
        case_result = np.select(conditions, choices, default)

        return pd.DataFrame({
            'DSCT_CUC_COMPONENTE': [dsc_cuc_componente],
            'FACTOR_CUADRE': [avg_factor_cuadre],
            'FACTOR_REPETICION': [case_result]
        })


if __name__ == "__main__":
    features_serv = FeatureCuadre()
    features_serv.get_data()
    features_serv.generate_features()
    df_ = features_serv.export

    print(f"{'='*10}FEATURES{'='*10}")
    print(df_.compute().head(10))