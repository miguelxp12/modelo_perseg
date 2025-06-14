from abc import ABC, abstractmethod


class FeatureInter(ABC):

    @abstractmethod
    def get_data(self) -> None:
        pass

    @abstractmethod
    def generate_features(self) -> None:
        pass

    @property
    @abstractmethod
    def export(self) -> None:
        pass
