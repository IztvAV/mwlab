#mwlab/opt/classifiers/spec_classifier.py
"""
spec_classifier.py
==================
`SpecClassifier` – обёртка над быстрым бинарным
классификатором (*HistGradientBoosting* по-умолчанию).

Класс полностью совместим с `BaseSurrogate`:
    • predict(x)        → bool
    • batch_predict(X)  → list[bool]

Основной вход – метод `from_training`, создающий synthetic-датасет
через существующий surrogate-регрессор и Specification.
"""

from __future__ import annotations
from typing import Mapping, Sequence, Dict, Any, Tuple

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,  balanced_accuracy_score

from mwlab.opt.surrogates.base import BaseSurrogate
from mwlab.opt.design.space import DesignSpace
from mwlab.opt.design.samplers import get_sampler
from mwlab.opt.objectives.specification import Specification


# ────────────────────────────────────────────────────────────────────────────
class SpecClassifier(BaseSurrogate):
    """
    Быстрый surrogate-классификатор pass/fail.

    *clf* – любой `predict_proba`-совместимый классификатор sklearn.
    """

    supports_uncertainty = False

    # ------------------------------------------------------------------ init
    def __init__(self, clf, design_space: DesignSpace):
        self.clf = clf
        self.space = design_space

    # ------ helper --------------------------------------------------------
    def _vec_df(self, x: Mapping[str, float]):
        """dict → DataFrame(1 × d)  с именами признаков"""
        return pd.DataFrame(
            [self.space.vector(x)],  # одна строка-вектор
            columns=self.space.names(),  # feature names!
        )

    # ---------------------------------------------------------------- predict
    def predict(self, x, *, return_std=False):
        return bool(self.clf.predict(self._vec_df(x))[0])

    def batch_predict(self, xs, *, return_std=False):
        X = pd.DataFrame(
            [self.space.vector(p) for p in xs],
            columns=self.space.names(),
        )
        return list(self.clf.predict(X).astype(bool))

    # ---------------------------------------------------------------- save/load
    def save(self, path: str | Path):
        """
        Сохраняет два файла:
        • .pkl – модель joblib
        • .json – сериализация DesignSpace
        """
        path = Path(path)
        joblib.dump(self.clf, path.with_suffix(".pkl"))
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(self.space.to_json(), fh, indent=2)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        clf = joblib.load(path.with_suffix(".pkl"))
        with open(path.with_suffix(".json"), encoding="utf-8") as fh:
            space_json = json.load(fh)
        space = DesignSpace.from_dict(space_json)
        return cls(clf, space)
    # ---------------------------------------------------------------- eval —
    def evaluate(
        self,
        points: Sequence[Mapping[str, float]],
        specification: Specification,
        *,
        teacher: BaseSurrogate,  # ← НОВОЕ: суррогат-«учитель» для y_true
        name: str = "eval",
        verbose: bool = True,
    ):
        """
        Считает y_true через переданный teacher-суррогат и Specification,
        затем сравнивает с предсказаниями классификатора.
        """
        y_true = teacher.passes_spec(points, specification).astype(int)
        y_pred = np.asarray(self.batch_predict(points), dtype=int)

        if verbose:
            print(f"\nClassification report ({name}):")
            print(classification_report(y_true, y_pred, digits=3, zero_division=0))
            ba = balanced_accuracy_score(y_true, y_pred)
            print(f"Balanced accuracy: {ba:.4f}")

        return balanced_accuracy_score(y_true, y_pred)
    # ---------------------------------------------------------------- factory
    @classmethod
    def from_training(
        cls,
        *,
        surrogate: BaseSurrogate,
        design_space: DesignSpace,
        specification: Specification,
        delta_limits: Dict[str, float] | Dict[str, Tuple[float, float]],
        n_samples: int = 524288,
        sampler: str = "sobol",
        sampler_kwargs: Mapping[str, Any] | None = None,
        algo: str = "hgb",           # "hgb" | "lightgbm" | "catboost"
        seed: int = 0,
        verbose: bool = True,
        val_frac: float = 0.20,
        early_stopping_rounds: int | None = 500,  # для LGBM / CatBoost
    ):
        """
        Обучает классификатор на synthetic-датасете.

        Parameters
        ----------
        delta_limits : dict
            k → δ   или  k → (−δ⁻, +δ⁺).  Все абсолютные.
        """
        # 1) формируем DesignSpace ограниченной области
        lows, highs = design_space.bounds()
        centers = {n: 0.5 * (lo + hi)
                   for n, lo, hi in zip(design_space.names(), lows, highs)}

        sub_space = DesignSpace.from_center_delta(
            centers,
            delta=delta_limits,
            mode="abs",
        )

        # 2) генерируем Monte-Carlo
        smp = get_sampler(sampler, **(sampler_kwargs or {}))
        pts = sub_space.sample(n_samples, sampler=smp, reject_invalid=False)

        # 3) метки pass/fail
        y_bool = surrogate.passes_spec(pts, specification)  # ndarray bool
        cols = sub_space.names()  # список имён признаков
        X = pd.DataFrame([sub_space.vector(p) for p in pts], columns=cols)
        y = y_bool.astype(int)

        if verbose:
            pos = y.mean()*100
            print(f"Generated dataset: {n_samples} pts, pass={pos:5.2f}%")

        # 3.5) train / val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_frac, random_state=seed, stratify=y
        )

        # 4) train classifier
        if algo == "hgb":
            clf = HistGradientBoostingClassifier(
                learning_rate=0.025,
                max_iter=10000,
                max_depth=None,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=seed,
            ).fit(X_train, y_train)
        elif algo == "lightgbm":
            try:
                import lightgbm as lgb
            except ModuleNotFoundError as err:
                raise ImportError("LightGBM не установлен") from err

            clf = lgb.LGBMClassifier(
                n_estimators=20000,
                learning_rate=0.02,
                subsample=0.65,  # bagging_fraction
                subsample_freq=1,  # каждый boosting-раунд
                colsample_bytree=0.65,  # feature_fraction
                lambda_l2=10.0, # L2-регуляризация
                class_weight="balanced",
                random_state=seed,
            )

            # --- колбэк ранней остановки: работает во всех версиях LightGBM -----
            callbacks = []
            if early_stopping_rounds is not None:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds,
                        first_metric_only=False,  # нам важен logloss
                        verbose=False,
                    )
                )

            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss",
                feature_name=cols,
                callbacks=callbacks
            )
        elif algo == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ModuleNotFoundError as err:
                raise ImportError("CatBoost не установлен") from err

            clf = CatBoostClassifier(
                iterations=100000,
                learning_rate=0.01,
                depth=8,
                l2_leaf_reg=10.0,  # сильная L2
                bagging_temperature=2.0,  # стохастический семплинг строк
                random_strength=2.0,  # шум в сплиты (anti overfit)

                loss_function="Logloss",
                eval_metric="AUC",  # устойчивый критерий
                auto_class_weights="Balanced",
                random_seed=seed,
                verbose=False,
            )
            clf.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True if early_stopping_rounds else False,
                early_stopping_rounds=early_stopping_rounds,
            )

        else:
            raise ValueError("algo должен быть 'hgb', 'lightgbm' или 'catboost'")

        # 5) отчёты
        if verbose:
            print("\nClassification report (train):")
            print(classification_report(
                y_train, clf.predict(X_train),
                digits=3, zero_division=0,
            ))
            ba_tr = balanced_accuracy_score(y_train, clf.predict(X_train))
            print(f"Balanced accuracy (train): {ba_tr:.4f}")

            print("\nClassification report (val):")
            print(classification_report(
                y_val, clf.predict(X_val),
                digits=3, zero_division=0,
            ))
            ba_val = balanced_accuracy_score(y_val, clf.predict(X_val))
            print(f"Balanced accuracy (val):   {ba_val:.4f}")

        return cls(clf, sub_space)

