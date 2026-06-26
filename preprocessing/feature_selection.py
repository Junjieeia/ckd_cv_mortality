import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Tuple
import sys
sys.path.append('..')
from config import RANDOM_SEED


def lasso_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv: int = 5,
    task: str = "classification",
) -> List[str]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if task == "classification":
        from sklearn.linear_model import LogisticRegressionCV
        model = LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            cv=cv,
            max_iter=2000,
            random_state=RANDOM_SEED,
        )
        model.fit(X_scaled, y)
        coefs = np.abs(model.coef_[0])
    else:
        model = LassoCV(cv=cv, max_iter=5000, random_state=RANDOM_SEED)
        model.fit(X_scaled, y)
        coefs = np.abs(model.coef_)

    selected = [feature_names[i] for i, c in enumerate(coefs) if c > 0]
    return selected


def boruta_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_estimators: int = 200,
    max_iter: int = 100,
    task: str = "classification",
) -> List[str]:
    np.random.seed(RANDOM_SEED)
    n_samples, n_features = X.shape

    X_shadow = X.copy()
    for col in range(n_features):
        np.random.shuffle(X_shadow[:, col])

    X_aug = np.hstack([X, X_shadow])

    if task == "classification":
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=7,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=7,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

    rf.fit(X_aug, y)
    importances = rf.feature_importances_

    real_imp = importances[:n_features]
    shadow_imp = importances[n_features:]
    shadow_max = shadow_imp.max()

    selected = [feature_names[i] for i, imp in enumerate(real_imp) if imp > shadow_max]
    return selected


def compute_vif(X: np.ndarray, feature_names: List[str], threshold: float = 10.0) -> Tuple[List[str], pd.DataFrame]:
    vif_data = []
    for i in range(X.shape[1]):
        vif_val = variance_inflation_factor(X, i)
        vif_data.append({"feature": feature_names[i], "vif": vif_val})
    vif_df = pd.DataFrame(vif_data)

    selected = vif_df[vif_df["vif"] < threshold]["feature"].tolist()
    return selected, vif_df


def combined_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task: str = "classification",
    vif_threshold: float = 10.0,
) -> Tuple[List[str], dict]:
    lasso_selected = lasso_feature_selection(X, y, feature_names, task=task)
    boruta_selected = boruta_feature_selection(X, y, feature_names, task=task)

    union_selected = list(set(lasso_selected) | set(boruta_selected))
    union_indices = [feature_names.index(f) for f in union_selected]
    X_union = X[:, union_indices]

    vif_selected, vif_df = compute_vif(X_union, union_selected, threshold=vif_threshold)

    report = {
        "lasso_selected": lasso_selected,
        "boruta_selected": boruta_selected,
        "union_before_vif": union_selected,
        "after_vif_removal": vif_selected,
        "vif_table": vif_df,
    }

    return vif_selected, report


def get_lasso_coefficient_path(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_alphas: int = 100,
) -> pd.DataFrame:
    from sklearn.linear_model import lasso_path
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas, coefs, _ = lasso_path(X_scaled, y, n_alphas=n_alphas)
    coef_df = pd.DataFrame(
        coefs.T,
        columns=feature_names,
    )
    coef_df["alpha"] = alphas
    return coef_df
