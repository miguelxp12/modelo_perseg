import pandas as pd
import dask.dataframe as dd
from predict.features.feature_consultoras_expuestas import FeatureConsExp
from predict.features.feature_factor_cuadre import FeatureCuadre
from predict.features.feature_prioridad_evento import FeaturePrioridadEvento
from predict.features.feature_quartile import FeatureQuartile
from predict.features.feature_rolling import FeatureRolling
from predict.feature_base import FeatureInter


class PredictService:

    df_features: dd.DataFrame = None

    def _get_feature(
        self,
        FeatureServ: FeatureInter
    ) -> dd.DataFrame:
        feature_serv = FeatureServ()
        feature_serv.get_data()
        feature_serv.generate_features()
        return feature_serv.export

    def generate_live_features(
        self
    ):
        feat_cons_exp = self._get_feature(FeatureConsExp)
        feat_prioridad = self._get_feature(FeaturePrioridadEvento)
        feat_quartile = self._get_feature(FeatureQuartile)
        feat_rolling = self._get_feature(FeatureRolling)
        feat_cuadre = self._get_feature(FeatureCuadre)

        join_columns = [
            "ANIOCAMPANA",
            "CODPAIS",
            "OFFERID"
        ]

        df_merged = feat_rolling
        df_merged = dd.merge(df_merged, feat_quartile, on=join_columns, how='inner')
        df_merged = dd.merge(df_merged, feat_prioridad, on=join_columns, how='inner')
        df_merged = dd.merge(df_merged, feat_cons_exp, on=join_columns, how='inner')
        # join factor_cuadre_
        df_merged = dd.merge(df_merged, feat_cuadre, on=join_columns, how='inner')

        self.df_features = df_merged

    def predict(
        self,
        type_model: str
    ) -> None:
        pass

    @property
    def export(
        self
    ) -> dd.DataFrame:
        return self.df_features
