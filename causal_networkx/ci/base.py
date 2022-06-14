from abc import ABCMeta, abstractmethod
from typing import Any, Tuple

import pandas as pd


class BaseConditionalIndependenceTest(metaclass=ABCMeta):
    """Abstract class for any conditional independence test.

    All CI tests are used in constraint-based causal discovery algorithms. This
    class interface is expected to be very lightweight to enable anyone to convert
    a function for CI testing into a class, which has a specific API.
    """

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
