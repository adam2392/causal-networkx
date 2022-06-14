from abc import ABCMeta, abstractmethod
from typing import Any, Tuple

import pandas as pd


class BaseConditionalIndependenceTest(metaclass=ABCMeta):
    @abstractmethod
    def test(self, df: pd.DataFrame, x: Any, y: Any, z: Any = None) -> Tuple[float, float]:
        """Abstract method for all conditional independence tests.

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        x : Any
            _description_
        y : Any
            _description_
        z : Any, optional
            _description_, by default None

        Returns
        -------
        Tuple[float, float]
            _description_
        """
        pass
