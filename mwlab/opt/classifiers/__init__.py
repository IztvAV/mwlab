#mwlab/opt/classifiers/__init__.py
"""
mwlab.opt.classifiers
=====================
Содержит surrogate-классификаторы (pass/fail).

* `SpecClassifier` – обучается на synthetic-датасете,
  сгенерированном базовым surrogate (NN, GP, …) + Specification.
"""

from .spec_classifier import SpecClassifier
import warnings, re

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=re.escape("X does not have valid feature names, but LGBMClassifier was fitted with feature names"),
)

__all__ = ["SpecClassifier"]


