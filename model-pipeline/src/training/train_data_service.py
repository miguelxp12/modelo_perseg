import io
from typing import List
import pandas as pd
from util.s3_manager import S3Buckets
from util.enums import (
    MAP_PATH,
    COD_COUNTRIES,
    RANGE_CAMPAIGNS
)

class TrainDataService:

    bucket_name: str
    s3_service: S3Buckets
    path_global_prefix_training: str

    def __init__(
        self,
        type_model: str
    ) -> None:
        self.s3_service = S3Buckets()
        self.path_global_prefix_training = MAP_PATH[type_model]['PATH_GLOBAL_PREFIX_TRAINING']

    def _gen_prefix_train_path_for_codpais(
        self,
        codpais
    ) -> str:
        codpais = codpais.upper()
        p_path = f"{self.path_global_prefix_training}/codpais={codpais}"
        return p_path

    def _get_files_for_codpais(
        self,
        codpais: str
    ) -> List[str]:
        p_path = self._gen_prefix_train_path_for_codpais(codpais)
        ll_in_pp = self.s3_service.get_list_files_by_prefix(p_path)
        return ll_in_pp

    def _get_files_by_codpais_from_aniocampana(
        self,
        codpais: str,
        aniocampana: str
    ) -> List[str]:
        ll_codpais = self._get_files_for_codpais(codpais)
        p_path_begin = (
            f"{self._gen_prefix_train_path_for_codpais(codpais)}"
            f"/aniocampana={aniocampana}"
        )
        ll_tmp = [ii for ii in ll_codpais if ii <= p_path_begin or aniocampana in ii]
        ll_tmp.sort(reverse=True)
        ll_codpais_campana, ii = [], 0
        while ii < len(ll_tmp):
            ll_codpais_campana.append(ll_tmp[ii])
            ii += 1
            if ii >= RANGE_CAMPAIGNS:
                break

        return ll_codpais_campana

    def _instance_df(
        self,
        key_file: str
    ) -> pd.DataFrame:
        response = self.s3_service.get_file_by_key(key_file)
        df = pd.DataFrame()
        if response:
            content = io.BytesIO(response['Body'].read())
            df = pd.read_parquet(content)
        return df

    def generated_df_by_codpais_aniocampana(
        self,
        codpais: str,
        aniocampana: str
    ) -> pd.DataFrame:
        df_train = pd.DataFrame()
        ll_parquets = self._get_files_by_codpais_from_aniocampana(
            codpais,
            aniocampana
        )
        
        ll_parquets = [item for item in ll_parquets if '.parquet' in item]
        if ll_parquets:
            df_train = self._instance_df(ll_parquets[0])
            for file_parquet in ll_parquets[1:]:
                df_tmp = self._instance_df(file_parquet)
                df_train = pd.concat([df_train, df_tmp], ignore_index=True)
            self.delete_columns(df_train)
            self.rename_important_columns(df_train)
            return df_train

        df_empty = pd.DataFrame()
        return df_empty

    def generated_corportivo_df(
        self,
        aniocampana: str
    ) -> pd.DataFrame:
        df_train = pd.DataFrame()
        df_lst = []
        for cod_pais in COD_COUNTRIES:
            _df = self.generated_df_by_codpais_aniocampana(
                cod_pais,
                aniocampana
            )
            df_lst.append(_df)
        df_train = pd.concat(df_lst, ignore_index=True)
        return df_train

    def delete_columns(
        self,
        df: pd.DataFrame
    ) -> None:
        if 'codpais' in df:
            df.drop(columns=['codpais'], inplace=True)
        if 'aniocampana' in df:
            df.drop(columns=['aniocampana'], inplace=True)

    def rename_important_columns(
        self,
        df: pd.DataFrame
    ) -> None:
        if not df.empty:
            df.rename(
                columns={
                    'CODPAIS': 'COD_PAIS',
                    'OFFERID': 'ID_OFERTA',
                    'ANIOCAMPANA': 'COD_PERIODO'
                },
                inplace=True
            )

    def cast_data_type_for_columns(
        self,
        df: pd.DataFrame
    ) -> None:
        object_columns = [
            'ALCANCE',
            'COD_CUC',
            'COD_MARCA',
            'COD_PAIS',
            'COD_TIPO_GRUPO',
            'COD_TIPO_SUBGRUPO',
            'DES_CATEGORIA',
            'DES_CLASE',
            'DES_CUC',
            'DES_MARCA',
            'DES_TIPOSOLO',
            'DES_UNIDAD_NEGOCIO',
            'DISCOUNT_RANGE',
            'TARGET',
            'AGRESIVIDAD_TACTICA',
            'CODIGO_NIVEL_PRECIO',
            'DESCRIPCION_NIVEL_PRECIO',
            'INDICADORNUEVOTONO'
        ]
        for col in df.columns:
            if col not in object_columns:
                df[col] = df[col].astype('float64')
