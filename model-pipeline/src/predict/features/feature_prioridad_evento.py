import pandas as pd
import dask.dataframe as dd
from predict.feature_base import FeatureInter
from predict.sql_queries import (
    SQL_QUERY_FEATURE_PRIORIDAD_EVENTO
)
from util.enums import PREDICT_DATA
from predict.operation_service import OperationDask as OpD


class FeaturePrioridadEvento(FeatureInter):

    """
        Esta clase calcula los features:            
            - CTD_CONSULTORAS_EXPUESTAS
        basado en la agrupación de:
            - CODPAIS
            - ANIOCAMPANA
            - OFFERID
    """


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
            query=SQL_QUERY_FEATURE_PRIORIDAD_EVENTO
        )

    @property
    def export(self) -> dd.DataFrame:
        return self.feature_df


if __name__ == "__main__":
    features_serv = FeaturePrioridadEvento()
    features_serv.get_data()
    features_serv.generate_features()
    df_ = features_serv.export

    print(f"{'='*10}FEATURES{'='*10}")
    print(df_.compute().head(10))
