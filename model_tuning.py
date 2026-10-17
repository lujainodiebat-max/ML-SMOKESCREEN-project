from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_squared_log_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor


DATA_DIR = Path.cwd()
RANDOM_STATE = 42
ID_COLUMN = "ID"
TARGETS = ("Smoking", "Gamma_GT")
CATEGORICAL_COLUMNS = ("Visit_Quarter", "Screening_Center", "Workplace_ID")
EXCLUDED_COLUMNS = ("Urine_Plate", "POC_Batch")
BASELINE_OOF = {
    "Smoking_AUC": 0.94672,
    "Gamma_GT_RMSLE": 0.28553,
    "Combined_OOF_score": 0.66351,
}

CATBOOST_CANDIDATES = {
    "Smoking": [
        {
            "name": "CatBoost baseline settings + engineered features",
            "params": {
                "iterations": 563,
                "depth": 6,
                "learning_rate": 0.04,
                "l2_leaf_reg": 3.0,
            },
        },
        {
            "name": "CatBoost tuned depth 5",
            "params": {
                "iterations": 900,
                "depth": 5,
                "learning_rate": 0.03,
                "l2_leaf_reg": 5.0,
            },
        },
        {
            "name": "CatBoost tuned depth 7",
            "params": {
                "iterations": 1000,
                "depth": 7,
                "learning_rate": 0.03,
                "l2_leaf_reg": 7.0,
            },
        },
    ],
    "Gamma_GT": [
        {
            "name": "CatBoost baseline settings + engineered features",
            "params": {
                "iterations": 715,
                "depth": 6,
                "learning_rate": 0.04,
                "l2_leaf_reg": 3.0,
            },
        },
        {
            "name": "CatBoost tuned depth 5",
            "params": {
                "iterations": 1000,
                "depth": 5,
                "learning_rate": 0.03,
                "l2_leaf_reg": 5.0,
            },
        },
        {
            "name": "CatBoost tuned depth 7",
            "params": {
                "iterations": 1200,
                "depth": 7,
                "learning_rate": 0.025,
                "l2_leaf_reg": 7.0,
            },
        },
    ],
}

XGBOOST_PARAMS = {
    "n_estimators": 900,
    "max_depth": 4,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "n_jobs": -1,
}


def prepare_base_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.drop(columns=[column for column in TARGETS if column in frame]).drop(
        columns=[ID_COLUMN]
    ).copy()

    for eye in ("Eyesight_L", "Eyesight_R"):
        blind_column = f"{eye}_Blind"
        features[blind_column] = features[eye].eq(9.9).astype("int8")
        features.loc[features[eye].eq(9.9), eye] = np.nan

    ldl_was_missing = features["LDL"].isna()
    lipid_inputs_observed = features[
        ["Cholesterol_Total", "HDL", "Triglyceride"]
    ].notna().all(axis=1)
    calculated_ldl = (
        features["Cholesterol_Total"]
        - features["HDL"]
        - features["Triglyceride"] / 5
    ).clip(lower=0)
    derive_ldl = ldl_was_missing & lipid_inputs_observed
    features.loc[derive_ldl, "LDL"] = calculated_ldl.loc[derive_ldl]
    features["LDL_Was_Missing"] = ldl_was_missing.astype("int8")

    urine_measurements = [
        "Urine_Creatinine",
        "Urine_Cotinine_ng_mL",
        "Urine_EtG_ng_mL",
    ]
    features["Urine_Panel_Missing"] = (
        features[urine_measurements].isna().all(axis=1).astype("int8")
    )

    features = features.drop(
        columns=[column for column in EXCLUDED_COLUMNS if column in features]
    )
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].astype("string").fillna("Missing").astype(str)
    return features


def add_engineered_features(
    features: pd.DataFrame,
    raw_frame: pd.DataFrame,
    missing_source_columns: list[str],
) -> pd.DataFrame:
    engineered = features.copy()
    for column in missing_source_columns:
        indicator = f"{column}_Was_Missing"
        if indicator not in engineered:
            engineered[indicator] = raw_frame[column].isna().astype("int8")

    ratio_inputs = (
        ("AST_ALT_Ratio", "AST", "ALT"),
        ("POC_GGT_ALT_Ratio", "POC_GGT", "ALT"),
        ("Triglyceride_HDL_Ratio", "Triglyceride", "HDL"),
        ("Urine_Cotinine_Creatinine_Ratio", "Urine_Cotinine_ng_mL", "Urine_Creatinine"),
        ("Waist_Height_Ratio", "Waist_cm", "Height_cm"),
    )
    for output, numerator, denominator in ratio_inputs:
        divisor = engineered[denominator].where(engineered[denominator].ne(0))
        engineered[output] = engineered[numerator] / divisor

    engineered["Non_HDL_Cholesterol"] = (
        engineered["Cholesterol_Total"] - engineered["HDL"]
    )
    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    return engineered


def to_xgboost_categories(
    training: pd.DataFrame, testing: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    train_copy = training.copy()
    test_copy = testing.copy() if testing is not None else None
    for column in CATEGORICAL_COLUMNS:
        values = train_copy[column]
        if test_copy is not None:
            values = pd.concat([values, test_copy[column]], ignore_index=True)
        categories = pd.Index(values.dropna().unique())
        train_copy[column] = pd.Categorical(train_copy[column], categories=categories)
        if test_copy is not None:
            test_copy[column] = pd.Categorical(test_copy[column], categories=categories)
    return train_copy, test_copy


def rmsle(actual: np.ndarray, prediction: np.ndarray) -> float:
    clipped_prediction = np.clip(prediction, 1, 1000)
    return float(mean_squared_log_error(actual, clipped_prediction) ** 0.5)


def summarize_oof(
    target: str, oof: np.ndarray, fold_metrics: list[float]
) -> dict[str, float]:
    if target == "Smoking":
        pooled = float(roc_auc_score(smoking_target, oof))
        return {
            "pooled_oof": pooled,
            "fold_mean": float(np.mean(fold_metrics)),
            "fold_std": float(np.std(fold_metrics, ddof=1)),
        }

    pooled = rmsle(gamma_target, oof)
    return {
        "pooled_oof": pooled,
        "fold_mean": float(np.mean(fold_metrics)),
        "fold_std": float(np.std(fold_metrics, ddof=1)),
    }


def fit_catboost_oof(
    target: str, features: pd.DataFrame, params: dict[str, Any]
) -> tuple[np.ndarray, list[float]]:
    y = smoking_target if target == "Smoking" else gamma_target
    splitter = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        if target == "Smoking"
        else KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    )
    oof = np.zeros(len(features), dtype=float)
    fold_metrics: list[float] = []
    for fold, (fit_indices, valid_indices) in enumerate(
        splitter.split(features, y) if target == "Smoking" else splitter.split(features),
        start=1,
    ):
        model_params = {
            **params,
            "random_seed": RANDOM_STATE + fold,
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
                features.iloc[fit_indices],
                y[fit_indices],
                cat_features=list(CATEGORICAL_COLUMNS),
                verbose=False,
            )
            prediction = model.predict_proba(features.iloc[valid_indices])[:, 1]
            fold_metrics.append(float(roc_auc_score(y[valid_indices], prediction)))
        else:
            model = CatBoostRegressor(**model_params, loss_function="RMSE")
            model.fit(
                features.iloc[fit_indices],
                np.log1p(y[fit_indices]),
                cat_features=list(CATEGORICAL_COLUMNS),
                verbose=False,
            )
            prediction = np.expm1(model.predict(features.iloc[valid_indices]))
            fold_metrics.append(rmsle(y[valid_indices], prediction))
        oof[valid_indices] = prediction
        print(f"{target}: CatBoost fold {fold}/5 ({params['depth']=})")
    return oof, fold_metrics


def fit_xgboost_oof(
    target: str, features: pd.DataFrame
) -> tuple[np.ndarray, list[float]]:
    y = smoking_target if target == "Smoking" else gamma_target
    splitter = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        if target == "Smoking"
        else KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    )
    oof = np.zeros(len(features), dtype=float)
    fold_metrics: list[float] = []
    split_indices = splitter.split(features, y) if target == "Smoking" else splitter.split(features)
    for fold, (fit_indices, valid_indices) in enumerate(split_indices, start=1):
        if target == "Smoking":
            model = XGBClassifier(
                **XGBOOST_PARAMS,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=RANDOM_STATE + fold,
            )
            model.fit(
                features.iloc[fit_indices],
                y[fit_indices],
                verbose=False,
            )
            prediction = model.predict_proba(features.iloc[valid_indices])[:, 1]
            fold_metrics.append(float(roc_auc_score(y[valid_indices], prediction)))
        else:
            model = XGBRegressor(
                **XGBOOST_PARAMS,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=RANDOM_STATE + fold,
            )
            model.fit(
                features.iloc[fit_indices],
                np.log1p(y[fit_indices]),
                verbose=False,
            )
            prediction = np.expm1(model.predict(features.iloc[valid_indices]))
            fold_metrics.append(rmsle(y[valid_indices], prediction))
        oof[valid_indices] = prediction
        print(f"{target}: tuned XGBoost fold {fold}/5")
    return oof, fold_metrics


def combined_score(smoking_auc: float, gamma_rmsle: float) -> float:
    return float(0.5 * (2 * smoking_auc - 1) + 0.5 * (1 - gamma_rmsle / 0.5041))


def predict_selected_models() -> dict[str, Any]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    record_path = DATA_DIR / "best_run.json"
    run_record = json.loads(record_path.read_text(encoding="utf-8"))
    if "selected" not in run_record:
        raise ValueError(f"Selected model configuration is missing from {record_path}.")
    if "ggt_batch_correction" in run_record:
        from ggt_batch_correction import predict_saved_batch_corrected_submission

        return predict_saved_batch_corrected_submission()

    smoking_target_local = train["Smoking"].to_numpy()
    gamma_target_local = train["Gamma_GT"].to_numpy()
    train_base = prepare_base_features(train)
    test_base = prepare_base_features(test)
    missing_source_columns = [
        column
        for column in train.columns
        if column not in TARGETS + (ID_COLUMN,) + EXCLUDED_COLUMNS
        and train[column].isna().any()
    ]
    train_engineered = add_engineered_features(
        train_base, train, missing_source_columns
    )
    test_engineered = add_engineered_features(
        test_base, test, missing_source_columns
    )
    if train_engineered.columns.tolist() != test_engineered.columns.tolist():
        raise ValueError("Training and test feature columns do not match.")

    feature_sets = {
        "original": (train_base, test_base),
        "engineered": (train_engineered, test_engineered),
    }
    predictions: dict[str, np.ndarray] = {}
    for target in TARGETS:
        selected = run_record["selected"].get(target)
        if not selected:
            raise ValueError(f"No selected model configuration recorded for {target}.")
        if selected["model"] != "CatBoost":
            raise ValueError(
                f"Predict-only currently supports CatBoost, not {selected['model']}."
            )
        feature_set = selected["feature_set"]
        if feature_set not in feature_sets:
            raise ValueError(f"Unknown feature set {feature_set!r} for {target}.")
        training, testing = feature_sets[feature_set]
        dropped_features = selected.get("dropped_features", [])
        unknown_dropped = set(dropped_features) - set(training.columns)
        if unknown_dropped:
            raise ValueError(
                f"Unknown dropped features for {target}: {sorted(unknown_dropped)}"
            )
        if dropped_features:
            training = training.drop(columns=dropped_features)
            testing = testing.drop(columns=dropped_features)
        cat_features = [
            column for column in CATEGORICAL_COLUMNS if column in training.columns
        ]
        params = selected["params"]
        common_params = {
            **params,
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
        }

        if target == "Smoking":
            model = CatBoostClassifier(
                **common_params,
                loss_function="Logloss",
                eval_metric="AUC",
            )
            model.fit(
                training,
                smoking_target_local,
                cat_features=cat_features,
                verbose=False,
            )
            predictions[target] = model.predict_proba(testing)[:, 1]
            print(
                "Smoking test predictions generated as positive-class "
                "probabilities with predict_proba."
            )
        else:
            model = CatBoostRegressor(**common_params, loss_function="RMSE")
            model.fit(
                training,
                np.log1p(gamma_target_local),
                cat_features=cat_features,
                verbose=False,
            )
            predictions[target] = np.expm1(model.predict(testing)).clip(1, 1000)

    submission = sample_submission.copy()
    submission[ID_COLUMN] = test[ID_COLUMN].to_numpy()
    submission["Smoking"] = predictions["Smoking"]
    submission["Gamma_GT"] = predictions["Gamma_GT"]
    if submission.columns.tolist() != [ID_COLUMN, "Smoking", "Gamma_GT"]:
        raise ValueError("Submission columns do not match the required schema.")
    if submission.shape != (len(test), 3):
        raise ValueError(f"Unexpected submission shape: {submission.shape}.")
    if not submission[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("Submission IDs do not match test IDs in order.")
    prediction_values = submission[["Smoking", "Gamma_GT"]].to_numpy()
    if not np.isfinite(prediction_values).all():
        raise ValueError("Submission contains non-finite predictions.")
    if not submission["Smoking"].between(0, 1).all():
        raise ValueError("Smoking probabilities must be between 0 and 1.")
    if submission["Smoking"].nunique() < 2:
        raise ValueError("Smoking probabilities do not vary.")
    if not submission["Gamma_GT"].between(1, 1000).all():
        raise ValueError("Gamma_GT predictions must be between 1 and 1000.")

    submission_path = DATA_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    verified = pd.read_csv(submission_path)
    if verified.shape != (len(test), 3):
        raise ValueError("Saved submission row count changed during serialization.")
    if not verified[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("Saved submission IDs do not match test IDs.")
    if not np.allclose(
        verified[["Smoking", "Gamma_GT"]].to_numpy(),
        submission[["Smoking", "Gamma_GT"]].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Saved predictions do not match the generated predictions.")

    run_record["submission"] = {
        **run_record.get("submission", {}),
        "file": str(submission_path),
        "rows": int(len(verified)),
        "smoking_prediction_type": "positive-class probability",
        "smoking_prediction_method": "CatBoostClassifier.predict_proba[:, 1]",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "smoking_prediction_min": float(verified["Smoking"].min()),
        "smoking_prediction_max": float(verified["Smoking"].max()),
        "gamma_gt_prediction_min": float(verified["Gamma_GT"].min()),
        "gamma_gt_prediction_max": float(verified["Gamma_GT"].max()),
    }
    record_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(f"Saved first submission: {submission_path}")
    print(f"Verified test rows: {len(verified)}")
    print(
        "Smoking probability range:",
        (float(verified["Smoking"].min()), float(verified["Smoking"].max())),
    )
    print(
        "Gamma_GT prediction range:",
        (float(verified["Gamma_GT"].min()), float(verified["Gamma_GT"].max())),
    )
    print(f"Updated selected-model record: {record_path}")
    return run_record


def run_model_search() -> dict[str, Any]:
    global smoking_target, gamma_target

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    smoking_target = train["Smoking"].to_numpy()
    gamma_target = train["Gamma_GT"].to_numpy()

    baseline_train = prepare_base_features(train)
    baseline_test = prepare_base_features(test)
    missing_source_columns = [
        column
        for column in train.columns
        if column not in TARGETS + (ID_COLUMN,) + EXCLUDED_COLUMNS
        and train[column].isna().any()
    ]
    engineered_train = add_engineered_features(
        baseline_train, train, missing_source_columns
    )
    engineered_test = add_engineered_features(
        baseline_test, test, missing_source_columns
    )
    engineered_xgb_train, engineered_xgb_test = to_xgboost_categories(
        engineered_train, engineered_test
    )

    assert baseline_train.columns.tolist() == baseline_test.columns.tolist()
    assert engineered_train.columns.tolist() == engineered_test.columns.tolist()
    assert "Smoking" not in engineered_train and "Gamma_GT" not in engineered_train
    assert ID_COLUMN not in engineered_train

    candidate_results: dict[str, list[dict[str, Any]]] = {
        "Smoking": [
            {
                "name": "CatBoost baseline (saved five-fold OOF)",
                "model": "CatBoost",
                "feature_set": "original",
                "params": {"iterations": 563, "depth": 6, "learning_rate": 0.04},
                "metric": BASELINE_OOF["Smoking_AUC"],
                "fold_mean": 0.94673,
                "fold_std": 0.00271,
            }
        ],
        "Gamma_GT": [
            {
                "name": "CatBoost baseline (saved five-fold OOF)",
                "model": "CatBoost",
                "feature_set": "original",
                "params": {"iterations": 715, "depth": 6, "learning_rate": 0.04},
                "metric": BASELINE_OOF["Gamma_GT_RMSLE"],
                "fold_mean": 0.28548,
                "fold_std": 0.00587,
            }
        ],
    }

    for target, candidates in CATBOOST_CANDIDATES.items():
        for candidate in candidates:
            oof, fold_metrics = fit_catboost_oof(
                target, engineered_train, candidate["params"]
            )
            summary = summarize_oof(target, oof, fold_metrics)
            candidate_results[target].append(
                {
                    "name": candidate["name"],
                    "model": "CatBoost",
                    "feature_set": "engineered",
                    "params": candidate["params"],
                    "metric": summary["pooled_oof"],
                    "fold_mean": summary["fold_mean"],
                    "fold_std": summary["fold_std"],
                }
            )

        xgb_oof, xgb_fold_metrics = fit_xgboost_oof(
            target, engineered_xgb_train
        )
        xgb_summary = summarize_oof(target, xgb_oof, xgb_fold_metrics)
        candidate_results[target].append(
            {
                "name": "Tuned XGBoost + engineered features",
                "model": "XGBoost",
                "feature_set": "engineered",
                "params": XGBOOST_PARAMS,
                "metric": xgb_summary["pooled_oof"],
                "fold_mean": xgb_summary["fold_mean"],
                "fold_std": xgb_summary["fold_std"],
            }
        )

    metric_ascending = {"Smoking": False, "Gamma_GT": True}
    selected = {
        target: (
            max(candidates, key=lambda result: result["metric"])
            if not metric_ascending[target]
            else min(candidates, key=lambda result: result["metric"])
        )
        for target, candidates in candidate_results.items()
    }
    selected_smoking_auc = float(selected["Smoking"]["metric"])
    selected_gamma_rmsle = float(selected["Gamma_GT"]["metric"])
    selected_score = combined_score(selected_smoking_auc, selected_gamma_rmsle)

    print("\nCandidate results (pooled OOF; fold mean +/- sample std):")
    for target, candidates in candidate_results.items():
        direction = "higher is better" if target == "Smoking" else "lower is better"
        print(f"\n{target} ({direction})")
        for result in candidates:
            print(
                f"  {result['name']}: {result['metric']:.5f} "
                f"(fold {result['fold_mean']:.5f} +/- {result['fold_std']:.5f})"
            )
    print(
        f"\nSelected combined OOF score: {selected_score:.5f}; "
        f"saved CatBoost baseline: {BASELINE_OOF['Combined_OOF_score']:.5f}"
    )

    final_train_features = {
        "original": baseline_train,
        "engineered": engineered_train,
    }
    final_test_features = {
        "original": baseline_test,
        "engineered": engineered_test,
    }
    final_xgb_train, final_xgb_test = to_xgboost_categories(
        engineered_train, engineered_test
    )
    final_xgb_features = {"engineered": (final_xgb_train, final_xgb_test)}
    predictions: dict[str, np.ndarray] = {}

    for target, result in selected.items():
        feature_set = result["feature_set"]
        if result["model"] == "CatBoost":
            training = final_train_features[feature_set]
            testing = final_test_features[feature_set]
            params = result["params"]
            if target == "Smoking":
                model = CatBoostClassifier(
                    **params,
                    loss_function="Logloss",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                )
                model.fit(
                    training,
                    smoking_target,
                    cat_features=list(CATEGORICAL_COLUMNS),
                    verbose=False,
                )
                predictions[target] = model.predict_proba(testing)[:, 1]
            else:
                model = CatBoostRegressor(
                    **params,
                    loss_function="RMSE",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                )
                model.fit(
                    training,
                    np.log1p(gamma_target),
                    cat_features=list(CATEGORICAL_COLUMNS),
                    verbose=False,
                )
                predictions[target] = np.expm1(model.predict(testing)).clip(1, 1000)
        else:
            training, testing = final_xgb_features[feature_set]
            if target == "Smoking":
                model = XGBClassifier(
                    **XGBOOST_PARAMS,
                    objective="binary:logistic",
                    eval_metric="auc",
                    random_state=RANDOM_STATE,
                )
                model.fit(training, smoking_target, verbose=False)
                predictions[target] = model.predict_proba(testing)[:, 1]
            else:
                model = XGBRegressor(
                    **XGBOOST_PARAMS,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=RANDOM_STATE,
                )
                model.fit(training, np.log1p(gamma_target), verbose=False)
                predictions[target] = np.expm1(model.predict(testing)).clip(1, 1000)

    submission = sample_submission.copy()
    submission[ID_COLUMN] = test[ID_COLUMN].to_numpy()
    submission["Smoking"] = predictions["Smoking"]
    submission["Gamma_GT"] = predictions["Gamma_GT"]
    assert submission.columns.tolist() == [ID_COLUMN, "Smoking", "Gamma_GT"]
    assert submission.shape == (len(test), 3)
    assert submission[ID_COLUMN].equals(test[ID_COLUMN])
    assert np.isfinite(submission[["Smoking", "Gamma_GT"]].to_numpy()).all()
    assert submission["Smoking"].between(0, 1).all()
    assert submission["Gamma_GT"].between(1, 1000).all()
    assert submission["Smoking"].nunique() > 1

    submission_path = DATA_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    verified_submission = pd.read_csv(submission_path)
    assert verified_submission.shape == (len(test), 3)
    assert verified_submission.columns.tolist() == [ID_COLUMN, "Smoking", "Gamma_GT"]
    assert verified_submission[ID_COLUMN].equals(test[ID_COLUMN])
    assert np.isfinite(verified_submission[["Smoking", "Gamma_GT"]].to_numpy()).all()

    run_record: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation": {
            "folds": 5,
            "random_state": RANDOM_STATE,
            "smoking_split": "StratifiedKFold(shuffle=True)",
            "gamma_split": "KFold(shuffle=True)",
            "selection_note": (
                "Exploratory model selection on these folds; the folds are not an "
                "untouched final estimate. The saved baseline used the same folds."
            ),
        },
        "saved_baseline": BASELINE_OOF,
        "feature_engineering": {
            "missingness_flags": "raw source columns with training missing values",
            "lab_features": [
                "AST_ALT_Ratio",
                "POC_GGT_ALT_Ratio",
                "Triglyceride_HDL_Ratio",
                "Urine_Cotinine_Creatinine_Ratio",
                "Waist_Height_Ratio",
                "Non_HDL_Cholesterol",
            ],
        },
        "candidate_results": candidate_results,
        "selected": {
            "Smoking": selected["Smoking"],
            "Gamma_GT": selected["Gamma_GT"],
            "combined_oof_score": selected_score,
        },
        "submission": {
            "file": str(submission_path),
            "rows": int(len(verified_submission)),
            "smoking_prediction_min": float(verified_submission["Smoking"].min()),
            "smoking_prediction_max": float(verified_submission["Smoking"].max()),
            "gamma_gt_prediction_min": float(verified_submission["Gamma_GT"].min()),
            "gamma_gt_prediction_max": float(verified_submission["Gamma_GT"].max()),
        },
    }
    record_path = DATA_DIR / "best_run.json"
    record_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(f"\nSaved submission: {submission_path}")
    print(f"Saved best-run record: {record_path}")
    print(f"Submission rows: {len(verified_submission)}")
    print("Selected models:", {target: item["name"] for target, item in selected.items()})
    return run_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Refit the saved selected configurations and rebuild submission.csv.",
    )
    arguments = parser.parse_args()
    if arguments.predict_only:
        predict_selected_models()
    else:
        run_model_search()
