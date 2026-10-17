from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_squared_log_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

import model_tuning as tuning


DATA_DIR = Path.cwd()
RANDOM_STATE = tuning.RANDOM_STATE
PERMUTATION_REPEATS = 3


def target_metric(target: str, actual: np.ndarray, prediction: np.ndarray) -> float:
    if target == "Smoking":
        return float(roc_auc_score(actual, prediction))
    prediction = np.clip(prediction, 1, 1000)
    return float(mean_squared_log_error(actual, prediction) ** 0.5)


def permutation_importance_on_fold(
    target: str,
    features: pd.DataFrame,
    actual: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    params: dict[str, Any],
) -> dict[str, dict[str, float]]:
    model_params = {
        **params,
        "random_seed": RANDOM_STATE + 1,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    train_x = features.iloc[train_indices]
    valid_x = features.iloc[valid_indices]
    train_y = actual[train_indices]
    valid_y = actual[valid_indices]

    if target == "Smoking":
        model = CatBoostClassifier(
            **model_params,
            loss_function="Logloss",
            eval_metric="AUC",
        )
        model.fit(
            train_x,
            train_y,
            cat_features=[
                column
                for column in tuning.CATEGORICAL_COLUMNS
                if column in features.columns
            ],
            verbose=False,
        )

        def predict(frame: pd.DataFrame) -> np.ndarray:
            return model.predict_proba(frame)[:, 1]

    else:
        model = CatBoostRegressor(**model_params, loss_function="RMSE")
        model.fit(
            train_x,
            np.log1p(train_y),
            cat_features=[
                column
                for column in tuning.CATEGORICAL_COLUMNS
                if column in features.columns
            ],
            verbose=False,
        )

        def predict(frame: pd.DataFrame) -> np.ndarray:
            return np.expm1(model.predict(frame))

    baseline_prediction = predict(valid_x)
    baseline_metric = target_metric(target, valid_y, baseline_prediction)
    rng = np.random.default_rng(RANDOM_STATE)
    importance_values: dict[str, list[float]] = {
        column: [] for column in features.columns
    }

    for column in features.columns:
        for _ in range(PERMUTATION_REPEATS):
            permuted = valid_x.copy()
            values = permuted[column].to_numpy(copy=True)
            permuted[column] = rng.permutation(values)
            permuted_metric = target_metric(target, valid_y, predict(permuted))
            if target == "Smoking":
                importance_values[column].append(
                    baseline_metric - permuted_metric
                )
            else:
                importance_values[column].append(
                    permuted_metric - baseline_metric
                )
        print(f"{target}: permuted {column}")

    return {
        column: {
            "mean_metric_degradation": float(np.mean(values)),
            "std_metric_degradation": float(np.std(values, ddof=1)),
        }
        for column, values in importance_values.items()
    }


def evaluate_features_on_folds(
    target: str,
    features: pd.DataFrame,
    actual: np.ndarray,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any],
) -> dict[str, Any]:
    predictions = np.full(len(features), np.nan, dtype=float)
    fold_metrics: list[float] = []
    cat_features = [
        column
        for column in tuning.CATEGORICAL_COLUMNS
        if column in features.columns
    ]

    for fold_number, (train_indices, valid_indices) in enumerate(fold_splits, start=2):
        model_params = {
            **params,
            "random_seed": RANDOM_STATE + fold_number,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
        }
        if target == "Smoking":
            model = CatBoostClassifier(
                **model_params,
                loss_function="Logloss",
                eval_metric="AUC",
            )
            model.fit(
                features.iloc[train_indices],
                actual[train_indices],
                cat_features=cat_features,
                verbose=False,
            )
            prediction = model.predict_proba(features.iloc[valid_indices])[:, 1]
        else:
            model = CatBoostRegressor(**model_params, loss_function="RMSE")
            model.fit(
                features.iloc[train_indices],
                np.log1p(actual[train_indices]),
                cat_features=cat_features,
                verbose=False,
            )
            prediction = np.expm1(model.predict(features.iloc[valid_indices]))

        predictions[valid_indices] = prediction
        fold_metrics.append(
            target_metric(target, actual[valid_indices], prediction)
        )
        print(f"{target}: comparison fold {fold_number}/5")

    evaluated_indices = np.concatenate(
        [valid_indices for _, valid_indices in fold_splits]
    )
    if np.isnan(predictions[evaluated_indices]).any():
        raise ValueError("The four-fold comparison did not produce complete OOF values.")

    return {
        "pooled_oof": target_metric(
            target, actual[evaluated_indices], predictions[evaluated_indices]
        ),
        "fold_mean": float(np.mean(fold_metrics)),
        "fold_std": float(np.std(fold_metrics, ddof=1)),
        "fold_metrics": fold_metrics,
        "evaluated_rows": int(len(evaluated_indices)),
    }


def run_feature_selection() -> dict[str, Any]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    train_base = tuning.prepare_base_features(train)
    missing_source_columns = [
        column
        for column in train.columns
        if column not in tuning.TARGETS + (tuning.ID_COLUMN,) + tuning.EXCLUDED_COLUMNS
        and train[column].isna().any()
    ]
    features = tuning.add_engineered_features(
        train_base, train, missing_source_columns
    )
    targets = {
        "Smoking": train["Smoking"].to_numpy(),
        "Gamma_GT": train["Gamma_GT"].to_numpy(),
    }
    splitters = {
        "Smoking": StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        ),
        "Gamma_GT": KFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        ),
    }
    splits = {
        target: list(
            splitter.split(features, target_values)
            if target == "Smoking"
            else splitter.split(features)
        )
        for target, (splitter, target_values) in (
            ("Smoking", (splitters["Smoking"], targets["Smoking"])),
            ("Gamma_GT", (splitters["Gamma_GT"], targets["Gamma_GT"])),
        )
    }

    run_record_path = DATA_DIR / "best_run.json"
    run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
    selection: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Held-out-fold permutation importance on fold 1, followed by "
            "full-versus-reduced comparison on folds 2-5."
        ),
        "importance_fold": 1,
        "evaluation_folds": [2, 3, 4, 5],
        "permutation_repeats": PERMUTATION_REPEATS,
        "metric_direction": {
            "Smoking": "higher AUC is better",
            "Gamma_GT": "lower RMSLE is better",
        },
        "targets": {},
    }

    for target in ("Smoking", "Gamma_GT"):
        selected = run_record["selected"][target]
        params = selected["params"]
        actual = targets[target]
        importance = permutation_importance_on_fold(
            target,
            features,
            actual,
            splits[target][0][0],
            splits[target][0][1],
            params,
        )
        dropped_features = sorted(
            column
            for column, result in importance.items()
            if result["mean_metric_degradation"] <= 0
        )
        kept_features = [column for column in features.columns if column not in dropped_features]
        if not kept_features:
            raise ValueError(f"Feature selection removed every feature for {target}.")

        comparison_splits = splits[target][1:]
        full_result = evaluate_features_on_folds(
            target, features, actual, comparison_splits, params
        )
        reduced_result = evaluate_features_on_folds(
            target, features[kept_features], actual, comparison_splits, params
        )
        if target == "Smoking":
            reduced_wins = (
                reduced_result["pooled_oof"] > full_result["pooled_oof"]
                and reduced_result["fold_mean"] >= full_result["fold_mean"]
            )
        else:
            reduced_wins = (
                reduced_result["pooled_oof"] < full_result["pooled_oof"]
                and reduced_result["fold_mean"] <= full_result["fold_mean"]
            )

        selection["targets"][target] = {
            "full_feature_count": len(features.columns),
            "reduced_feature_count": len(kept_features),
            "dropped_features": dropped_features,
            "permutation_importance": importance,
            "full_feature_metrics": full_result,
            "reduced_feature_metrics": reduced_result,
            "reduced_feature_set_selected": bool(reduced_wins),
        }
        print(
            f"{target}: dropped {len(dropped_features)}/{len(features.columns)} "
            f"non-positive-importance features; reduced candidate "
            f"{'wins' if reduced_wins else 'does not win'} the four comparison folds."
        )
        print(
            f"  full={full_result['pooled_oof']:.5f}, "
            f"reduced={reduced_result['pooled_oof']:.5f}"
        )

        if reduced_wins:
            selected["dropped_features"] = dropped_features
            selected["feature_selection"] = {
                "method": "fold-1 permutation importance, validated on folds 2-5",
                "importance_fold": 1,
                "evaluation_folds": [2, 3, 4, 5],
            }
        else:
            selected.pop("dropped_features", None)
            selected.pop("feature_selection", None)

    run_record["feature_selection"] = selection
    run_record_path.write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    selection_path = DATA_DIR / "feature_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Saved feature-selection report: {selection_path}")

    if any(
        value["reduced_feature_set_selected"]
        for value in selection["targets"].values()
    ):
        tuning.predict_selected_models()
    else:
        print(
            "The reduced sets did not outperform the full selected feature set; "
            "the current submission was left unchanged."
        )
    return selection


if __name__ == "__main__":
    run_feature_selection()
