import pandas as pd
import dask.dataframe as dd
from predict.feature_base import FeatureInter
from predict.sql_queries import (
    SQL_QUERY_FEATURE_ROLLING
)
from util.enums import PREDICT_DATA
from predict.operation_service import OperationDask as OpD


class FeatureRolling(FeatureInter):

    """
        Esta clase calcula los features:            
            - CNT_CONSTANTES_1
            - CNT_CONSTANTES_2
            - CNT_CONSTANTES_3
            - CNT_INCONSTANTES
            - CNT_SIN_SEGMENTO
            - CNT_TOPS
            - CNT_NUEVAS
            - CNT_BRILLA
            - CNT_EMPTYSTRING
            - RATIO_CONSTANTES_1
            - RATIO_CONSTANTES_2
            - RATIO_CONSTANTES_3
            - RATIO_INCONSTANTES
            - RATIO_SIN_SEGMENTO
            - RATIO_TOPS
            - RATIO_NUEVAS
            - RATIO_BRILLA
            - RATIO_EMPTYSTRING
        basado en la agrupación de:
            - CODPAIS
            - ANIOCAMPANA
            - OFFERID
    """

    data_file = ""
    df: dd.DataFrame = None
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

    def __init__(self):
        self.data_file = PREDICT_DATA.CONSULTANT

    def get_data(self) -> None:
        self.df = OpD.instance_df_from_path_parquet(
            self.data_file
        )

    def generate_features(self) -> None:
        self.feature_df = OpD.apply_sql_sentence_over_daskframe(
            dask_frame=self.df,
            query=SQL_QUERY_FEATURE_ROLLING
        )

    @property
    def export(self) -> dd.DataFrame:
        return self.feature_df


if __name__ == "__main__":
    features_serv = FeatureRolling()
    features_serv.get_data()
    features_serv.generate_features()
    df_ = features_serv.export

    print(f"{'='*10}FEATURES{'='*10}")
    print(df_.compute().head(10))
