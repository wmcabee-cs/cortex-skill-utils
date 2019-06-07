from abc import ABC, abstractmethod
import pandas as pd
from ..data_types.api_types import FSContent


class CortexFS(ABC):

    @abstractmethod
    def write_df(self, df: pd.DataFrame, to_path: str, domain_type: str):
        ...

    @abstractmethod
    def read(self, from_path: str) -> FSContent:
        ...

    def write(self, content: FSContent, to_path) -> None:
        if content.native_format.value == 'pandas_df':
            self.write_df(df=content.value, to_path=to_path, domain_type=content.domain_type)
        else:
            raise ValueError(f"unexpected native format type {content.native_format.value}")
