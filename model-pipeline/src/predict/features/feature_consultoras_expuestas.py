import pandas as pd
import dask.dataframe as dd
from predict.feature_base import FeatureInter
from predict.sql_queries import (
    SQL_QUERY_FEATURE_CONSULTORAS_EXPS
)
from util.enums import PREDICT_DATA
from predict.operation_service import OperationDask as OpD

class FeatureConsExp(FeatureInter):

    """
        Esta clase calcula los features:            
            - PRIORIDADEVENTO
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
        "CODEBELISTA": "int64",
        "ORDEN": "float64",
        "DESCRIPCIONROLLING": "object"
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
            query=SQL_QUERY_FEATURE_CONSULTORAS_EXPS
        )

    @property
    def export(self) -> dd.DataFrame:
        return self.feature_df


if __name__ == "__main__":
    print("obteniendo features de cons expuestas")
    features_serv = FeatureConsExp()
    features_serv.get_data()
    features_serv.generate_features()
    df_ = features_serv.export

    print(f"{'='*10}FEATURES{'='*10}")
    print(df_.compute().head(10))
