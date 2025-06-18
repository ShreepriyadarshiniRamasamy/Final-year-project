"""PySpark XGBoost integration interface"""
"""yspark is a module or opensource Distributed computing framework or python API used to train large dataset or million of records.
faster and better than pandas and SQL and also can be runned in multiple machines and can be deployed in different cloud platforms"""
"""WE are using try and expect because to avoid the import error """

try:
    import pyspark
except ImportError as e:
    raise ImportError("pyspark package needs to be installed to use this module") from e

from .estimator import (
    SparkXGBClassifier,
    SparkXGBClassifierModel,
    SparkXGBRanker,
    SparkXGBRankerModel,
    SparkXGBRegressor,
    SparkXGBRegressorModel,
)

__all__ = [
    "SparkXGBClassifier",
    "SparkXGBClassifierModel",
    "SparkXGBRegressor",
    "SparkXGBRegressorModel",
    "SparkXGBRanker",
    "SparkXGBRankerModel",
]
