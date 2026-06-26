import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
import sys
sys.path.append('..')
from config import (
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, RANDOM_FOREST_PARAMS,
    LOGISTIC_PARAMS, FINEGRAY_PARAMS, RSF_PARAMS, RANDOM_SEED
)


class LogisticRegressionModel:
    def __init__(self, params: dict = None):
        self.params = params or LOGISTIC_PARAMS
        self.model = LogisticRegression(**self.params)
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_coefficients(self, feature_names: list) -> pd.DataFrame:
        coefs = self.model.coef_[0]
        return pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
        }).sort_values("coefficient", ascending=False)


class XGBoostModel:
    def __init__(self, params: dict = None):
        self.params = params or XGBOOST_PARAMS
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: list = None) -> None:
        fit_params = {}
        if eval_set:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False
        self.model.fit(X, y, **fit_params)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)


class LightGBMModel:
    def __init__(self, params: dict = None):
        self.params = params or LIGHTGBM_PARAMS
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: list = None) -> None:
        fit_params = {}
        if eval_set:
            fit_params["eval_set"] = eval_set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y, **fit_params)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)


class RandomForestModel:
    def __init__(self, params: dict = None):
        self.params = params or RANDOM_FOREST_PARAMS
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)


class CoxCompetingRisksModel:
    def __init__(self, penalizer: float = None):
        self.penalizer = penalizer or FINEGRAY_PARAMS["penalizer"]
        self.model = CoxPHFitter(penalizer=self.penalizer)
        self.feature_names = None

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray, feature_names: list) -> None:
        self.feature_names = feature_names
        df = pd.DataFrame(X, columns=feature_names)
        df["duration"] = time
        df["event"] = event
        self.model.fit(df, duration_col="duration", event_col="event")

    def predict_cumulative_hazard(self, X: np.ndarray, times: np.ndarray = None) -> np.ndarray:
        df = pd.DataFrame(X, columns=self.feature_names)
        if times is not None:
            return self.model.predict_cumulative_hazard(df, times=times).values
        return self.model.predict_cumulative_hazard(df).values

    def predict_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict_partial_hazard(df).values

    def predict_survival_function(self, X: np.ndarray, times: np.ndarray = None) -> np.ndarray:
        df = pd.DataFrame(X, columns=self.feature_names)
        if times is not None:
            return self.model.predict_survival_function(df, times=times).values
        return self.model.predict_survival_function(df).values

    def get_coefficients(self) -> pd.DataFrame:
        return self.model.params_.reset_index().rename(
            columns={"index": "feature", "coef": "coefficient"}
        )


class RandomSurvivalForestModel:
    def __init__(self, params: dict = None):
        self.params = params or RSF_PARAMS
        try:
            from sksurv.ensemble import RandomSurvivalForest
            self.model = RandomSurvivalForest(**self.params)
            self._use_sksurv = True
        except ImportError:
            self._use_sksurv = False
            warnings.warn("scikit-survival not found, RSF unavailable")

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> None:
        if not self._use_sksurv:
            return
        y = np.array(
            [(bool(e), t) for e, t in zip(event, time)],
            dtype=[("event", bool), ("time", float)],
        )
        self.model.fit(X, y)

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if not self._use_sksurv:
            return np.zeros(X.shape[0])
        return self.model.predict(X)

    def predict_survival_function(self, X: np.ndarray) -> list:
        if not self._use_sksurv:
            return []
        return self.model.predict_survival_function(X)


class BenchmarkEquations:
    @staticmethod
    def ckd_pc_score(
        age: np.ndarray,
        sex_female: np.ndarray,
        egfr: np.ndarray,
        proteinuria: np.ndarray,
        diabetes: np.ndarray,
        systolic_bp: np.ndarray,
        ldl: np.ndarray,
        current_smoker: np.ndarray,
        hdl: np.ndarray,
    ) -> np.ndarray:
        log_age = np.log(age)
        log_egfr = np.log(np.clip(egfr, 1, None))
        log_proteinuria = np.log(np.clip(proteinuria, 1, None))
        log_sbp = np.log(systolic_bp)
        log_ldl = np.log(np.clip(ldl, 1, None))
        log_hdl = np.log(np.clip(hdl, 1, None))

        male_lp = (
            12.344 * log_age
            + 11.853 * log_egfr
            + 2.019 * log_proteinuria
            + 1.767 * log_sbp
            - 1.764 * log_ldl
            - 7.990 * log_hdl
            + 0.661 * current_smoker
            + 0.661 * diabetes
        )
        female_lp = (
            12.344 * log_age
            + 11.853 * log_egfr
            + 2.019 * log_proteinuria
            + 1.767 * log_sbp
            - 1.764 * log_ldl
            - 7.990 * log_hdl
            + 0.661 * current_smoker
            + 0.661 * diabetes
            - 0.329
        )

        lp = np.where(sex_female == 1, female_lp, male_lp)
        risk = 1 - np.exp(-np.exp(lp - 23.9388))
        return np.clip(risk, 0, 1)

    @staticmethod
    def kfre_4var(
        age: np.ndarray,
        sex_female: np.ndarray,
        egfr: np.ndarray,
        uacr: np.ndarray,
    ) -> np.ndarray:
        alpha_2yr = -0.2201
        alpha_5yr = -0.3013
        beta_age = 0.0155
        beta_male = 0.4510
        beta_egfr = -0.0274
        beta_log_uacr = 0.5849

        log_uacr = np.log(np.clip(uacr, 0.1, None))
        lp = beta_age * age + beta_male * (1 - sex_female) + beta_egfr * egfr + beta_log_uacr * log_uacr
        risk_5yr = 1 - np.exp(-np.exp(alpha_5yr + lp))
        return np.clip(risk_5yr, 0, 1)

    @staticmethod
    def kfre_8var(
        age: np.ndarray,
        sex_female: np.ndarray,
        egfr: np.ndarray,
        uacr: np.ndarray,
        albumin: np.ndarray,
        phosphate: np.ndarray,
        bicarbonate: np.ndarray,
        calcium: np.ndarray,
    ) -> np.ndarray:
        log_uacr = np.log(np.clip(uacr, 0.1, None))
        lp = (
            0.0155 * age
            + 0.4510 * (1 - sex_female)
            - 0.0274 * egfr
            + 0.5849 * log_uacr
            - 0.3040 * albumin
            + 0.1550 * phosphate
            - 0.1970 * bicarbonate
            - 0.2890 * calcium
        )
        risk_5yr = 1 - np.exp(-np.exp(-0.4013 + lp))
        return np.clip(risk_5yr, 0, 1)
