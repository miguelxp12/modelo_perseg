from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple


class PreprocessInter(ABC):

    @abstractmethod
    def prepare_data(
        self
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pass

    @abstractmethod
    def get_data_train_raw(
        self
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_data_test(
        self
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def generate_data_metrics(
        self,
        anio_campana: str,
        country_code: str
    ) -> None:
        pass