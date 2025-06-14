```python
from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class ModelInter(ABC):

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self) -> None:
        pass

    @abstractmethod
    def save(self, anio_campana: str, country_code: str) -> None:
        pass

    @abstractmethod
    def save_features_metrics_as_plot(
        self,
        df_train: pd.DataFrame,
        columns: List[str],
        name_model: str,
        country_model: str
    ) -> None:
        pass

    @abstractmethod
    def save_metrics_for_ds(
        self,
        df_: pd.DataFrame,
        columns: List[str],
        name_model: str,
        country_model: str,
        name_dataset: str
    ) -> None:
        pass

    @abstractmethod
    def save_metrics_test_without_outliers(
        self,
        df_test: pd.DataFrame,
        columns: List[str],
        columns_target: List[str],
        name_model: str,
        country_model: str,
        name_dataset: str
    ) -> None:
        pass

    @abstractmethod
    def save_shap_metrics(
        self,
        df_: pd.DataFrame,
        columns: List[str],
        name_model: str,
        country_model: str
    ) -> None:
        pass
```
