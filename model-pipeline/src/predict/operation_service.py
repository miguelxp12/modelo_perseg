import os
import traceback
import pandas as pd
import dask
import dask.dataframe as dd
from dask_sql import Context
from typing import List


class OperationDask:

    @staticmethod
    def apply_sql_sentence_over_daskframe(
        dask_frame: dd.DataFrame,
        query: str
    ) -> dd.DataFrame:
        print(f"applying query: {query}")
        c = Context()
        c.create_table("table_dd", dask_frame)
        result = c.sql(query)
        result.compute()
        return result

    @staticmethod
    def _empty_dask_dataframe(
        columns: List[str] = ['col1', 'col2']
    ) -> dd.DataFrame:
        meta = pd.DataFrame(columns=columns)
        df = dd.from_pandas(meta, npartitions=1)
        return df

    @staticmethod
    def instance_df_from_path_parquet(
        path_key: str
    ) -> dd.DataFrame:
        s3_key = f"s3://{path_key}"
        try:
            df = dd.read_parquet(
                s3_key,
                storage_options={
                    'key': os.getenv("USER_S3_MODEL"),
                    'secret': os.getenv("USER_SECRET_MODEL")
                }
            )
        except (FileNotFoundError, OSError) as e:
            print(f"Erro al leer el archivo ({s3_key}): {e}")
            print(traceback.print_exc())
            df = OperationDask._empty_dask_dataframe()

        return df

    @staticmethod
    def instance_df_from_list_parquet(
        path_keys: List[str]
    ) -> dd.DataFrame:
        if not path_keys:
            df = OperationDask._empty_dask_dataframe()
            return df
        try:
            dfs = [
                OperationDask.instance_df_from_path_parquet(path_key)
                for path_key in path_keys
            ]
            df = dd.concat(dfs)
        except (FileNotFoundError, OSError) as e:
            print(f"Erro al leer el archivo: {e}")
            df = OperationDask._empty_dask_dataframe()
        return df
