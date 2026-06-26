import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('..')
from config import (
    LOGISTIC_PARAMS, FINEGRAY_PARAMS, MAX_NOMOGRAM_FEATURES,
    NOMOGRAM_FEATURES_TASK1, NOMOGRAM_FEATURES_TASK2,
    NOMOGRAM_RANGES_TASK1, NOMOGRAM_RANGES_TASK2,
    FOLLOW_UP_YEARS, RANDOM_SEED
)


class NomogramScorer:
    def __init__(self, feature_ranges: Dict, max_points: int = 100):
        self.feature_ranges = feature_ranges
        self.max_points = max_points
        self.point_allocations = {}
        self.coefficients = {}
        self.intercept = 0.0
        self.baseline_risk = None
        self.score_to_risk = {}

    def fit_logistic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler

        model = LogisticRegression(
            penalty="l2",
            C=0.1,
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
        model.fit(X_scaled, y)
        self.model = model
        self.feature_names = feature_names

        coefs = model.coef_[0]
        self.intercept = model.intercept_[0]
        self.coefficients = {f: c for f, c in zip(feature_names, coefs)}

        self._compute_point_allocations(feature_names)

    def fit_cox(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        feature_names: List[str],
        eval_time: float = None,
    ) -> None:
        eval_time = eval_time or FOLLOW_UP_YEARS
        df = pd.DataFrame(X, columns=feature_names)
        df["duration"] = time
        df["event"] = event

        cox = CoxPHFitter(penalizer=FINEGRAY_PARAMS["penalizer"])
        cox.fit(df, duration_col="duration", event_col="event")
        self.cox_model = cox
        self.feature_names = feature_names
        self.eval_time = eval_time

        coefs_series = cox.params_
        self.coefficients = {f: float(coefs_series[f]) for f in feature_names if f in coefs_series}

        self._compute_point_allocations(feature_names)
        self._build_cox_risk_table(X, time, event, feature_names)

    def _compute_point_allocations(self, feature_names: List[str]) -> None:
        max_coef = max(abs(c) for c in self.coefficients.values()) + 1e-8

        for feat in feature_names:
            if feat not in self.feature_ranges:
                continue
            coef = self.coefficients.get(feat, 0.0)
            feat_range = self.feature_ranges[feat]
            min_val, max_val = feat_range

            if max_val == min_val:
                max_pts = 0
            else:
                max_pts = int(abs(coef) / max_coef * self.max_points)

            direction = "higher" if coef > 0 else "lower"
            self.point_allocations[feat] = {
                "min_val": min_val,
                "max_val": max_val,
                "max_points": max_pts,
                "coefficient": coef,
                "direction": direction,
            }

    def _build_cox_risk_table(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        feature_names: List[str],
    ) -> None:
        df = pd.DataFrame(X, columns=feature_names)
        sf = self.cox_model.predict_survival_function(df, times=[self.eval_time])
        risks = 1 - sf.values.flatten()

        scores = self.compute_total_score(X, feature_names)
        score_risk_pairs = list(zip(scores, risks))
        score_risk_pairs.sort()

        score_bins = np.arange(0, self.max_points * len(feature_names) + 10, 10)
        for s in score_bins:
            nearby = [(sc, r) for sc, r in score_risk_pairs if abs(sc - s) <= 10]
            if nearby:
                self.score_to_risk[s] = np.mean([r for _, r in nearby])

    def compute_feature_points(self, value: float, feature: str) -> int:
        if feature not in self.point_allocations:
            return 0
        alloc = self.point_allocations[feature]
        min_val = alloc["min_val"]
        max_val = alloc["max_val"]
        max_pts = alloc["max_points"]
        coef = alloc["coefficient"]

        value_clipped = np.clip(value, min_val, max_val)

        if max_val == min_val:
            return 0

        if coef > 0:
            pts = (value_clipped - min_val) / (max_val - min_val) * max_pts
        else:
            pts = (max_val - value_clipped) / (max_val - min_val) * max_pts

        return int(round(pts))

    def compute_total_score(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        scores = np.zeros(X.shape[0])
        for j, feat in enumerate(feature_names):
            for i in range(X.shape[0]):
                scores[i] += self.compute_feature_points(X[i, j], feat)
        return scores

    def score_to_probability_logistic(self, total_score: float) -> float:
        max_possible = sum(alloc["max_points"] for alloc in self.point_allocations.values())
        if max_possible == 0:
            return 0.5
        log_odds = self.intercept + total_score / max_possible * sum(
            abs(c) for c in self.coefficients.values()
        )
        prob = 1 / (1 + np.exp(-log_odds))
        return float(prob)

    def score_to_probability_cox(self, total_score: float) -> float:
        score_keys = np.array(list(self.score_to_risk.keys()))
        if len(score_keys) == 0:
            return 0.0
        nearest_key = score_keys[np.argmin(np.abs(score_keys - total_score))]
        return float(self.score_to_risk[nearest_key])

    def predict_probability(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        if hasattr(self, "model"):
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)[:, 1]
        elif hasattr(self, "cox_model"):
            df = pd.DataFrame(X, columns=feature_names)
            sf = self.cox_model.predict_survival_function(df, times=[self.eval_time])
            return 1 - sf.values.flatten()
        return np.zeros(X.shape[0])

    def generate_nomogram_table(self) -> pd.DataFrame:
        records = []
        for feat, alloc in self.point_allocations.items():
            records.append({
                "predictor": feat,
                "clinical_range_min": alloc["min_val"],
                "clinical_range_max": alloc["max_val"],
                "points_min": 0,
                "points_max": alloc["max_points"],
                "direction": alloc["direction"],
                "coefficient": alloc["coefficient"],
            })
        return pd.DataFrame(records).sort_values("points_max", ascending=False)

    def generate_score_risk_table(
        self,
        score_values: List[float] = None,
        task: str = "task2",
    ) -> pd.DataFrame:
        if score_values is None:
            score_values = [50, 100, 150, 175, 200, 250, 300, 350, 400]

        records = []
        for s in score_values:
            if task == "task2":
                risk = self.score_to_probability_logistic(s)
            else:
                risk = self.score_to_probability_cox(s)
            records.append({
                "total_score": s,
                "predicted_risk": risk,
                "predicted_risk_pct": f"{risk * 100:.1f}%",
            })
        return pd.DataFrame(records)


def build_task2_nomogram(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> NomogramScorer:
    scorer = NomogramScorer(feature_ranges=NOMOGRAM_RANGES_TASK2)
    scorer.fit_logistic(X, y, feature_names)
    return scorer


def build_task1_nomogram(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    feature_names: List[str],
) -> NomogramScorer:
    scorer = NomogramScorer(feature_ranges=NOMOGRAM_RANGES_TASK1)
    scorer.fit_cox(X, time, event, feature_names)
    return scorer
