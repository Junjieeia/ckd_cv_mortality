import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from typing import Dict, List, Callable, Tuple, Optional
import warnings
import sys
sys.path.append('..')
from config import N_FOLDS, RANDOM_SEED, MIN_SENSITIVITY


def cross_validate_classifier(
    model_class,
    model_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    min_sensitivity: float = MIN_SENSITIVITY,
) -> Dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_aucs, fold_thresholds = [], []
    train_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)

        if hasattr(model, "predict_proba"):
            y_prob_tr = model.predict_proba(X_tr)[:, 1]
            y_prob_val = model.predict_proba(X_val)[:, 1]
        else:
            y_prob_tr = model.predict(X_tr)
            y_prob_val = model.predict(X_val)

        train_aucs.append(roc_auc_score(y_tr, y_prob_tr))
        val_auc = roc_auc_score(y_val, y_prob_val)
        fold_aucs.append(val_auc)

        threshold = _select_threshold_cv(y_val, y_prob_val, min_sensitivity)
        fold_thresholds.append(threshold)

    return {
        "mean_val_auc": np.mean(fold_aucs),
        "std_val_auc": np.std(fold_aucs),
        "fold_aucs": fold_aucs,
        "mean_train_auc": np.mean(train_aucs),
        "mean_threshold": np.mean(fold_thresholds),
        "fold_thresholds": fold_thresholds,
    }


def cross_validate_survival(
    model_class,
    model_params: dict,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_folds: int = N_FOLDS,
) -> Dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_cis = []
    train_cis = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, event)):
        X_tr, X_val = X[train_idx], X[val_idx]
        t_tr, t_val = time[train_idx], time[val_idx]
        e_tr, e_val = event[train_idx], event[val_idx]

        model = model_class(**model_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, t_tr, e_tr)

        risk_tr = model.predict_risk(X_tr)
        risk_val = model.predict_risk(X_val)

        train_ci = concordance_index(t_tr, -risk_tr, e_tr)
        val_ci = concordance_index(t_val, -risk_val, e_val)
        train_cis.append(train_ci)
        fold_cis.append(val_ci)

    return {
        "mean_val_ci": np.mean(fold_cis),
        "std_val_ci": np.std(fold_cis),
        "fold_cis": fold_cis,
        "mean_train_ci": np.mean(train_cis),
    }


def _select_threshold_cv(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_sensitivity: float = MIN_SENSITIVITY,
) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = tpr >= min_sensitivity
    if valid.sum() == 0:
        return float(thresholds[np.argmax(tpr - fpr)])
    youden = tpr - fpr
    youden[~valid] = -np.inf
    return float(thresholds[np.argmax(youden)])


class BayesianHPO:
    def __init__(
        self,
        search_space: Dict,
        objective: Callable,
        n_trials: int = 50,
        direction: str = "maximize",
        random_state: int = RANDOM_SEED,
    ):
        self.search_space = search_space
        self.objective = objective
        self.n_trials = n_trials
        self.direction = direction
        self.random_state = random_state
        self.results = []

    def optimize(self) -> Tuple[Dict, float]:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def optuna_objective(trial):
                params = {}
                for param_name, param_config in self.search_space.items():
                    if param_config["type"] == "int":
                        params[param_name] = trial.suggest_int(
                            param_name, param_config["low"], param_config["high"]
                        )
                    elif param_config["type"] == "float":
                        params[param_name] = trial.suggest_float(
                            param_name, param_config["low"], param_config["high"],
                            log=param_config.get("log", False),
                        )
                    elif param_config["type"] == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config["choices"]
                        )
                return self.objective(params)

            direction = "maximize" if self.direction == "maximize" else "minimize"
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            study = optuna.create_study(direction=direction, sampler=sampler)
            study.optimize(optuna_objective, n_trials=self.n_trials, show_progress_bar=False)

            best_params = study.best_params
            best_value = study.best_value
            self.results = [
                {"params": t.params, "value": t.value} for t in study.trials
            ]
            return best_params, best_value

        except ImportError:
            return self._random_search()

    def _random_search(self) -> Tuple[Dict, float]:
        rng = np.random.RandomState(self.random_state)
        best_params = None
        best_value = -np.inf if self.direction == "maximize" else np.inf

        for _ in range(self.n_trials):
            params = {}
            for param_name, param_config in self.search_space.items():
                if param_config["type"] == "int":
                    params[param_name] = int(rng.randint(param_config["low"], param_config["high"] + 1))
                elif param_config["type"] == "float":
                    if param_config.get("log", False):
                        log_low = np.log(param_config["low"])
                        log_high = np.log(param_config["high"])
                        params[param_name] = float(np.exp(rng.uniform(log_low, log_high)))
                    else:
                        params[param_name] = float(rng.uniform(param_config["low"], param_config["high"]))
                elif param_config["type"] == "categorical":
                    params[param_name] = rng.choice(param_config["choices"])

            value = self.objective(params)
            if (self.direction == "maximize" and value > best_value) or \
               (self.direction == "minimize" and value < best_value):
                best_value = value
                best_params = params.copy()

            self.results.append({"params": params, "value": value})

        return best_params, best_value

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)


XGBOOST_SEARCH_SPACE = {
    "n_estimators": {"type": "int", "low": 200, "high": 800},
    "max_depth": {"type": "int", "low": 3, "high": 8},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    "reg_lambda": {"type": "float", "low": 0.5, "high": 5.0},
}

LGBM_SEARCH_SPACE = {
    "n_estimators": {"type": "int", "low": 200, "high": 800},
    "num_leaves": {"type": "int", "low": 31, "high": 127},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
    "feature_fraction": {"type": "float", "low": 0.5, "high": 1.0},
}

DEEPHIT_SEARCH_SPACE = {
    "lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "dropout": {"type": "float", "low": 0.1, "high": 0.5},
    "alpha": {"type": "float", "low": 0.1, "high": 0.5},
    "sigma": {"type": "float", "low": 0.05, "high": 0.5},
    "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
}
