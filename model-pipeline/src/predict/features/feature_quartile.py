import pandas as pd
import dask.dataframe as dd
import numpy as np
from predict.feature_base import FeatureInter
from predict.sql_queries import (
    SQL_QUERY_FEATURE_ORDEN,
    SQL_QUERY_FEATURE_QUARTILE
)
from util.enums import PREDICT_DATA
from predict.operation_service import OperationDask as OpD


class FeatureQuartile(FeatureInter):

    """
        Esta clase calcula los features:            
            - MIN_ORDEN_Q1
            - MIN_ORDEN_Q2
            - MIN_ORDEN_Q3
            - MIN_ORDEN_Q4
            - MAX_ORDEN_Q1
            - MAX_ORDEN_Q2
            - MAX_ORDEN_Q3
            - MAX_ORDEN_Q4
            - AVG_ORDEN_Q1
            - AVG_ORDEN_Q2
            - AVG_ORDEN_Q3
            - AVG_ORDEN_Q4
            - MEDIAN_ORDEN_Q1
            - MEDIAN_ORDEN_Q2
            - MEDIAN_ORDEN_Q3
            - MEDIAN_ORDEN_Q4
            - STD_ORDEN_Q1
            - STD_ORDEN_Q2
            - STD_ORDEN_Q3
            - STD_ORDEN_Q4
            - CTD_ORDEN_Q1
            - CTD_ORDEN_Q2
            - CTD_ORDEN_Q3
            - CTD_ORDEN_Q4
        basado en la agrupación de:
            - CODPAIS
            - ANIOCAMPANA
            - OFFERID
    """

    df: dd.DataFrame = None
    feature_dd: dd.DataFrame = None
    meta_dict = {
        'CODPAIS': 'object', 
        'ANIOCAMPANA': 'int', 
        'OFFERID': 'int', 
        'ORDEN': 'float64',    
        'DESCRIPCIONROLLING': 'object',
        'QUARTILE': 'int'
    }

    def __init__(self):
        self.data_file = PREDICT_DATA.CONSULTANT

    def get_data(self) -> None:
        self.df = OpD.instance_df_from_path_parquet(
            self.data_file
        )

    def generate_features(self) -> None:
        # prepare dataframe raw
        col_des = [k for k in self.meta_dict]
        df = self.df.groupby('OFFERID').apply(
            FeatureQuartile._ntile,
            n=4,
            over_part_col='ORDEN',
            cols_des=col_des,
            ntile_cole='QUARTILE',
            meta=self.meta_dict
        )
        df = df.reset_index(drop=True)

        # Dataframe 01 for inner join
        cols_group_by = [
            "ANIOCAMPANA",
            "CODPAIS",
            "OFFERID",
            "QUARTILE"
        ]
        df_1 = df.groupby(cols_group_by).apply(
            FeatureQuartile._median
        )
        df_1 = df_1.reset_index(drop=True)

        if isinstance(df_1, pd.DataFrame):
            df_1 = dd.from_pandas(df_1, npartitions=2)

        # Dataframe 02 for inner join
        df_2 = OpD.apply_sql_sentence_over_daskframe(
            dask_frame=df,
            query=SQL_QUERY_FEATURE_ORDEN
        )

        # join previous dataframes
        df_3 = dd.merge(
            df_1.drop(columns=["ORDEN", "DESCRIPCIONROLLING"]),
            df_2,
            how='inner',
            on=["CODPAIS", "ANIOCAMPANA", "OFFERID", "QUARTILE"]
        )

        self.feature_dd = OpD.apply_sql_sentence_over_daskframe(
            dask_frame=df_3,
            query=SQL_QUERY_FEATURE_QUARTILE
        )

    @property
    def export(self) -> dd.DataFrame:
        return self.feature_dd

    @staticmethod
    def _ntile(
        subdf: dd.DataFrame,
        n: int,
        over_part_col: str,
        ntile_col: str
    ) -> dd.DataFrame:
        jitter = np.random.uniform(0, 1e-6, size=len(subdf))
        subdf = subdf.assign(
            rank=(subdf[over_part_col] + jitter).rank(method='dense')
        )
        try:
            subdf[ntile_col] = pd.qcut(subdf['rank'], n, labels=False) + 1
        except ValueError:
            subdf[ntile_col] = 1
        subdf = subdf.drop(columns=['rank'])
        return subdf.reset_index(drop=True)

    @staticmethod
    def _median(subdf: dd.DataFrame) -> dd.DataFrame:
        medians = subdf[['ORDEN']].median()
        subdf['MEDIAN_ORDEN'] = medians['ORDEN']
        return subdf


if __name__ == "__main__":
    features_serv = FeatureQuartile()
    features_serv.get_data()
    features_serv.generate_features()
    df_ = features_serv.export

    print(f"{'='*10}FEATURES{'='*10}")
    print(df_.compute().head(10))