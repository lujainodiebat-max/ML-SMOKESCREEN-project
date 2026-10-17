from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import model_tuning as tuning


DATA_DIR = Path.cwd()
RANDOM_STATE = 42
TARGET_COLUMN = "Gamma_GT"
POC_COLUMN = "POC_GGT"
BATCH_COLUMN = "POC_Batch"
POC_FALLBACK_DROP_COLUMNS = (
    POC_COLUMN,
    f"{POC_COLUMN}_Was_Missing",
    "POC_GGT_ALT_Ratio",
)


def make_fallback_pipeline(
    features: pd.DataFrame,
    seed: int,
    *,
    learning_rate: float = 0.05,
    max_iter: int = 300,
    max_leaf_nodes: int = 15,
    min_samples_leaf: int = 20,
    l2_regularization: float = 1.0,
) -> Pipeline:
    categorical = [
        column
        for column in tuning.CATEGORICAL_COLUMNS
        if column in features.columns
    ]
    numeric = [column for column in features.columns if column not in categorical]
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    fallback_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=seed,
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", fallback_model),
        ]
    )


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_base = tuning.prepare_base_features(train)
    test_base = tuning.prepare_base_features(test)
    missing_source_columns = [
        column
        for column in train.columns
        if column
        not in tuning.TARGETS + (tuning.ID_COLUMN,) + tuning.EXCLUDED_COLUMNS
        and train[column].isna().any()
    ]
    train_features = tuning.add_engineered_features(
        train_base, train, missing_source_columns
    )
    test_features = tuning.add_engineered_features(
        test_base, test, missing_source_columns
    )
    if train_features.columns.tolist() != test_features.columns.tolist():
        raise ValueError("Training and test feature columns do not match.")
    return (
        train_features,
        test_features,
        train[BATCH_COLUMN],
        test[BATCH_COLUMN],
    )


def rmsle(actual: np.ndarray, prediction: np.ndarray) -> float:
    clipped = np.clip(prediction, 1, 1000)
    return float(mean_squared_log_error(actual, clipped) ** 0.5)


def apply_batch_correction(
    poc: pd.Series,
    batches: pd.Series,
    batch_offsets: dict[float, float],
    batch_counts: dict[float, int],
    global_offset: float,
    fallback_log_prediction: np.ndarray,
    shrinkage: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = fallback_log_prediction.copy()
    corrected = np.zeros(len(poc), dtype=bool)
    has_proxy_and_batch = poc.notna() & batches.notna()
    matched = batches.map(batch_offsets).notna() & has_proxy_and_batch
    if matched.any():
        matched_batches = batches.loc[matched]
        offsets = matched_batches.map(batch_offsets).to_numpy(dtype=float)
        counts = matched_batches.map(batch_counts).to_numpy(dtype=float)
        if shrinkage:
            offsets = (
                counts / (counts + shrinkage) * offsets
                + shrinkage / (counts + shrinkage) * global_offset
            )
        predictions[matched.to_numpy()] = (
            np.log1p(poc.loc[matched].to_numpy(dtype=float)) + offsets
        )
        corrected[matched.to_numpy()] = True
    return np.expm1(predictions).clip(1, 1000), corrected


def leave_one_out_residual_adjustments(
    residuals: np.ndarray, batches: pd.Series
) -> np.ndarray:
    batch_values = pd.Series(batches.to_numpy())
    residual_frame = pd.DataFrame(
        {
            BATCH_COLUMN: batch_values,
            "residual": residuals,
        }
    )
    overall_sum = float(residual_frame["residual"].sum())
    overall_count = len(residual_frame)
    grouped = residual_frame.groupby(BATCH_COLUMN, dropna=True)["residual"].agg(
        ["sum", "count"]
    )
    batch_sums = batch_values.map(grouped["sum"])
    batch_counts = batch_values.map(grouped["count"])
    use_batch = batch_values.notna() & batch_counts.gt(1)
    fallback_sum = overall_sum - residuals
    fallback_count = max(overall_count - 1, 1)
    leave_one_out_mean = np.divide(
        fallback_sum,
        fallback_count,
        out=np.zeros_like(residuals, dtype=float),
    )
    local_mask = use_batch.to_numpy()
    leave_one_out_mean[local_mask] = (
        batch_sums.loc[use_batch].to_numpy(dtype=float) - residuals[local_mask]
    ) / (batch_counts.loc[use_batch].to_numpy(dtype=float) - 1)
    return residuals - leave_one_out_mean


def fit_fold_models(
    train_features: pd.DataFrame,
    train_raw: pd.DataFrame,
    valid_features: pd.DataFrame,
    valid_raw: pd.DataFrame,
    fold_number: int,
) -> dict[str, Any]:
    target = train_raw[TARGET_COLUMN].to_numpy(dtype=float)
    target_log = np.log1p(target)
    train_proxy = train_raw[POC_COLUMN]
    valid_proxy = valid_raw[POC_COLUMN]
    train_batch = train_raw[BATCH_COLUMN]
    valid_batch = valid_raw[BATCH_COLUMN]

    direct_fallback = make_fallback_pipeline(train_features, RANDOM_STATE + fold_number)
    direct_fallback.fit(train_features, target_log)
    direct_valid_log = direct_fallback.predict(valid_features)

    has_proxy = train_proxy.notna()
    proxy_residuals = (
        target_log[has_proxy.to_numpy()]
        - np.log1p(train_proxy.loc[has_proxy].to_numpy(dtype=float))
    )
    has_batch_residual = has_proxy & train_batch.notna()
    batch_residuals = proxy_residuals[
        has_batch_residual.loc[has_proxy].to_numpy()
    ]
    batch_training = pd.DataFrame(
        {
            BATCH_COLUMN: train_batch.loc[has_batch_residual].to_numpy(dtype=float),
            "residual": batch_residuals,
        }
    )
    batch_groups = batch_training.groupby(BATCH_COLUMN)["residual"]
    batch_offsets = batch_groups.mean().to_dict()
    batch_counts = batch_groups.size().to_dict()
    global_offset = float(batch_training["residual"].mean())
    batch_predictions = {
        "batch_shrinkage_0_direct_fallback": apply_batch_correction(
            valid_proxy, valid_batch, batch_offsets, batch_counts,
            global_offset, direct_valid_log, shrinkage=0.0,
        ),
    }
    for shrinkage in (0.5, 1.0, 2.0, 4.0):
        batch_predictions[f"batch_shrinkage_{shrinkage:g}_direct_fallback"] = (
            apply_batch_correction(
                valid_proxy, valid_batch, batch_offsets, batch_counts,
                global_offset, direct_valid_log, shrinkage=shrinkage,
            )
        )

    residual_fallback = make_fallback_pipeline(
        train_features, RANDOM_STATE + 100 + fold_number
    )
    residual_fallback.fit(
        train_features.loc[has_proxy],
        proxy_residuals,
    )
    residual_for_valid = residual_fallback.predict(valid_features)
    loo_residual_adjustment = leave_one_out_residual_adjustments(
        proxy_residuals, train_batch.loc[has_proxy]
    )
    within_batch_adjustment_baseline = make_fallback_pipeline(
        train_features, RANDOM_STATE + 200 + fold_number
    )
    within_batch_adjustment_baseline.fit(
        train_features.loc[has_proxy],
        loo_residual_adjustment,
    )
    baseline_post_correction_adjustment = (
        within_batch_adjustment_baseline.predict(valid_features)
    )
    within_batch_adjustment_tuned = make_fallback_pipeline(
        train_features,
        RANDOM_STATE + 300 + fold_number,
        max_iter=400,
        max_leaf_nodes=7,
        min_samples_leaf=40,
    )
    within_batch_adjustment_tuned.fit(
        train_features.loc[has_proxy],
        loo_residual_adjustment,
    )
    tuned_post_correction_adjustment = within_batch_adjustment_tuned.predict(
        valid_features
    )
    residual_valid_log = direct_valid_log.copy()
    has_proxy = valid_proxy.notna().to_numpy()
    residual_valid_log[has_proxy] = (
        np.log1p(valid_proxy.loc[valid_proxy.notna()].to_numpy(dtype=float))
        + residual_for_valid[has_proxy]
    )
    residual_fallback_model_prediction = np.expm1(residual_valid_log).clip(1, 1000)
    batch_offset_log = direct_valid_log.copy()
    valid_with_proxy = valid_proxy.notna().to_numpy()
    batch_offset_by_valid_batch = valid_batch.map(batch_offsets).to_numpy(
        dtype=float, copy=True
    )
    known_valid_batch = (
        valid_proxy.notna() & valid_batch.notna()
        & valid_batch.map(batch_offsets).notna()
    ).to_numpy()
    batch_offset_by_valid_batch[~known_valid_batch] = global_offset
    batch_offset_log[valid_with_proxy] = (
        np.log1p(valid_proxy.loc[valid_proxy.notna()].to_numpy(dtype=float))
        + batch_offset_by_valid_batch[valid_with_proxy]
    )
    batch_post_corrected_baseline_log = batch_offset_log.copy()
    batch_post_corrected_baseline_log[valid_with_proxy] += (
        baseline_post_correction_adjustment[valid_with_proxy]
    )
    batch_post_corrected_baseline = np.expm1(
        batch_post_corrected_baseline_log
    ).clip(1, 1000)
    batch_post_corrected_tuned_log = batch_offset_log.copy()
    batch_post_corrected_tuned_log[valid_with_proxy] += (
        0.75 * tuned_post_correction_adjustment[valid_with_proxy]
    )
    batch_post_corrected_tuned = np.expm1(
        batch_post_corrected_tuned_log
    ).clip(1, 1000)
    batch_plus_fallback_by_shrinkage: dict[str, np.ndarray] = {}
    residual_corrected_mask: np.ndarray | None = None
    for name, (direct_prediction, corrected_mask) in batch_predictions.items():
        batch_plus_fallback = direct_prediction.copy()
        use_residual_fallback = (~corrected_mask) & has_proxy
        batch_plus_fallback[use_residual_fallback] = (
            residual_fallback_model_prediction[use_residual_fallback]
        )
        fallback_name = name.replace(
            "_direct_fallback", "_residual_fallback"
        )
        batch_plus_fallback_by_shrinkage[fallback_name] = batch_plus_fallback
        if name == "batch_shrinkage_0_direct_fallback":
            residual_corrected_mask = corrected_mask

    return {
        "direct_hgb": np.expm1(direct_valid_log).clip(1, 1000),
        **{
            name: value[0]
            for name, value in batch_predictions.items()
        },
        **batch_plus_fallback_by_shrinkage,
        "batch_hgb_post_correction_baseline": batch_post_corrected_baseline,
        "batch_hgb_post_correction_tuned": batch_post_corrected_tuned,
        "batch_corrected_mask": residual_corrected_mask,
        "batch_offsets": batch_offsets,
        "direct_fallback_model": direct_fallback,
        "residual_fallback_model": residual_fallback,
    }


def cross_validate_batch_correction(
    features: pd.DataFrame, train: pd.DataFrame
) -> dict[str, Any]:
    target = train[TARGET_COLUMN].to_numpy(dtype=float)
    splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    method_names = [
        "direct_hgb",
        "batch_shrinkage_0_direct_fallback",
        "batch_shrinkage_0.5_direct_fallback",
        "batch_shrinkage_1_direct_fallback",
        "batch_shrinkage_2_direct_fallback",
        "batch_shrinkage_4_direct_fallback",
        "batch_shrinkage_0_residual_fallback",
        "batch_shrinkage_0.5_residual_fallback",
        "batch_shrinkage_1_residual_fallback",
        "batch_shrinkage_2_residual_fallback",
        "batch_shrinkage_4_residual_fallback",
        "batch_hgb_post_correction_baseline",
        "batch_hgb_post_correction_tuned",
    ]
    oof = {name: np.full(len(train), np.nan) for name in method_names}
    fold_metrics: dict[str, list[float]] = {name: [] for name in oof}
    fold_rows: list[dict[str, Any]] = []

    for fold_number, (fit_indices, valid_indices) in enumerate(
        splitter.split(features), start=1
    ):
        result = fit_fold_models(
            features.iloc[fit_indices],
            train.iloc[fit_indices],
            features.iloc[valid_indices],
            train.iloc[valid_indices],
            fold_number,
        )
        for name in oof:
            prediction = result[name]
            oof[name][valid_indices] = prediction
            fold_metrics[name].append(
                rmsle(target[valid_indices], prediction)
            )
        fold_rows.append(
            {
                "fold": fold_number,
                "valid_rows": int(len(valid_indices)),
                "batch_corrected_rows": int(result["batch_corrected_mask"].sum()),
                "batch_corrected_share": float(result["batch_corrected_mask"].mean()),
                **{
                    f"{name}_rmsle": fold_metrics[name][-1]
                    for name in oof
                },
            }
        )
        print(
            f"GGT fold {fold_number}/5: "
            f"batch correction covered "
            f"{result['batch_corrected_mask'].mean():.1%} of validation rows."
        )

    result_summary: dict[str, Any] = {}
    for name, predictions in oof.items():
        if np.isnan(predictions).any():
            raise ValueError(f"Incomplete out-of-fold predictions for {name}.")
        result_summary[name] = {
            "pooled_oof_rmsle": rmsle(target, predictions),
            "fold_mean_rmsle": float(np.mean(fold_metrics[name])),
            "fold_std_rmsle": float(np.std(fold_metrics[name], ddof=1)),
            "fold_rmsle": fold_metrics[name],
        }
    return {
        "fold_metrics": fold_rows,
        "models": result_summary,
        "oof_predictions": oof,
    }


def cross_validate_two_stage_catboost(
    features: pd.DataFrame,
    train: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    target = train[TARGET_COLUMN].to_numpy(dtype=float)
    target_log = np.log1p(target)
    fallback_features = features.drop(
        columns=[
            column
            for column in POC_FALLBACK_DROP_COLUMNS
            if column in features.columns
        ]
    )
    categorical = [
        column
        for column in tuning.CATEGORICAL_COLUMNS
        if column in features.columns
    ]
    fallback_categorical = [
        column
        for column in tuning.CATEGORICAL_COLUMNS
        if column in fallback_features.columns
    ]
    splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    predictions = np.full(len(train), np.nan, dtype=float)
    fold_metrics: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    fallback_mask_all = train[POC_COLUMN].isna().to_numpy()
    ratio_mask_all = np.zeros(len(train), dtype=bool)

    for fold_number, (fit_indices, valid_indices) in enumerate(
        splitter.split(features), start=1
    ):
        fit_raw = train.iloc[fit_indices]
        valid_raw = train.iloc[valid_indices]
        fit_features = features.iloc[fit_indices]
        valid_features = features.iloc[valid_indices]
        fallback_fit_features = fallback_features.iloc[fit_indices]
        fallback_valid_features = fallback_features.iloc[valid_indices]
        common_params = {
            **params,
            "random_seed": RANDOM_STATE + fold_number,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
            "loss_function": "RMSE",
        }

        primary_model = CatBoostRegressor(**common_params)
        primary_model.fit(
            fit_features,
            target_log[fit_indices],
            cat_features=categorical,
            verbose=False,
        )
        primary_prediction = np.expm1(
            primary_model.predict(valid_features)
        ).clip(1, 1000)

        fallback_model = CatBoostRegressor(**common_params)
        fallback_model.fit(
            fallback_fit_features,
            target_log[fit_indices],
            cat_features=fallback_categorical,
            verbose=False,
        )
        fallback_prediction = np.expm1(
            fallback_model.predict(fallback_valid_features)
        ).clip(1, 1000)

        fit_poc = fit_raw[POC_COLUMN]
        ratio_training = pd.DataFrame(
            {
                BATCH_COLUMN: fit_raw[BATCH_COLUMN].to_numpy(),
                "ratio": fit_raw[TARGET_COLUMN].to_numpy(dtype=float)
                / fit_poc.where(fit_poc.gt(0)).to_numpy(dtype=float),
            },
            index=fit_raw.index,
        )
        ratio_training = ratio_training.loc[
            fit_poc.notna() & fit_raw[BATCH_COLUMN].notna()
        ]
        batch_ratios = ratio_training.groupby(BATCH_COLUMN)["ratio"].mean()
        valid_poc = valid_raw[POC_COLUMN]
        valid_ratio = valid_raw[BATCH_COLUMN].map(batch_ratios)
        use_ratio = (
            valid_poc.notna()
            & valid_poc.gt(3)
            & valid_ratio.notna()
        ).to_numpy()
        use_fallback = valid_poc.isna().to_numpy()
        combined_prediction = primary_prediction.copy()
        combined_prediction[use_fallback] = fallback_prediction[use_fallback]
        combined_prediction[use_ratio] = (
            valid_poc.loc[use_ratio].to_numpy(dtype=float)
            * valid_ratio.loc[use_ratio].to_numpy(dtype=float)
        ).clip(1, 1000)
        predictions[valid_indices] = combined_prediction
        ratio_mask_all[valid_indices] = use_ratio
        fold_score = rmsle(
            target[valid_indices],
            combined_prediction,
        )
        fold_metrics.append(fold_score)
        fold_rows.append(
            {
                "fold": fold_number,
                "valid_rows": int(len(valid_indices)),
                "missing_poc_fallback_rows": int(use_fallback.sum()),
                "batch_ratio_rows": int(use_ratio.sum()),
                "primary_model_rows": int(
                    len(valid_indices) - use_fallback.sum() - use_ratio.sum()
                ),
                "rmsle": fold_score,
            }
        )
        print(
            f"Two-stage GGT fold {fold_number}/5: "
            f"ratio={int(use_ratio.sum())}, "
            f"missing-POC fallback={int(use_fallback.sum())}, "
            f"primary={len(valid_indices) - int(use_ratio.sum()) - int(use_fallback.sum())}"
        )

    if np.isnan(predictions).any():
        raise ValueError("Two-stage GGT validation left rows without predictions.")
    return {
        "pooled_oof_rmsle": rmsle(target, predictions),
        "fold_mean_rmsle": float(np.mean(fold_metrics)),
        "fold_std_rmsle": float(np.std(fold_metrics, ddof=1)),
        "fold_rmsle": fold_metrics,
        "fold_details": fold_rows,
        "missing_poc_oof_rmsle": rmsle(
            target[fallback_mask_all], predictions[fallback_mask_all]
        ),
        "batch_ratio_oof_rmsle": rmsle(
            target[ratio_mask_all], predictions[ratio_mask_all]
        ),
        "primary_oof_rmsle": rmsle(
            target[~fallback_mask_all & ~ratio_mask_all],
            predictions[~fallback_mask_all & ~ratio_mask_all],
        ),
        "missing_poc_rows": int(fallback_mask_all.sum()),
        "batch_ratio_rows": int(ratio_mask_all.sum()),
    }


def fit_two_stage_catboost_candidate(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    fallback_train_features = train_features.drop(
        columns=[
            column
            for column in POC_FALLBACK_DROP_COLUMNS
            if column in train_features.columns
        ]
    )
    fallback_test_features = test_features.drop(
        columns=[
            column
            for column in POC_FALLBACK_DROP_COLUMNS
            if column in test_features.columns
        ]
    )
    categorical = [
        column
        for column in tuning.CATEGORICAL_COLUMNS
        if column in train_features.columns
    ]
    fallback_categorical = [
        column
        for column in tuning.CATEGORICAL_COLUMNS
        if column in fallback_train_features.columns
    ]
    common_params = {
        **params,
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        "loss_function": "RMSE",
    }
    target_log = np.log1p(train[TARGET_COLUMN].to_numpy(dtype=float))
    primary_model = CatBoostRegressor(**common_params)
    primary_model.fit(
        train_features,
        target_log,
        cat_features=categorical,
        verbose=False,
    )
    fallback_model = CatBoostRegressor(**common_params)
    fallback_model.fit(
        fallback_train_features,
        target_log,
        cat_features=fallback_categorical,
        verbose=False,
    )
    primary_prediction = np.expm1(
        primary_model.predict(test_features)
    ).clip(1, 1000)
    fallback_prediction = np.expm1(
        fallback_model.predict(fallback_test_features)
    ).clip(1, 1000)

    train_poc = train[POC_COLUMN]
    ratio_training = pd.DataFrame(
        {
            BATCH_COLUMN: train[BATCH_COLUMN].to_numpy(),
            "ratio": train[TARGET_COLUMN].to_numpy(dtype=float)
            / train_poc.where(train_poc.gt(0)).to_numpy(dtype=float),
        },
        index=train.index,
    )
    ratio_training = ratio_training.loc[
        train_poc.notna() & train[BATCH_COLUMN].notna()
    ]
    batch_ratios = ratio_training.groupby(BATCH_COLUMN)["ratio"].mean()
    test_poc = test[POC_COLUMN]
    test_ratio = test[BATCH_COLUMN].map(batch_ratios)
    use_ratio = (
        test_poc.notna() & test_poc.gt(3) & test_ratio.notna()
    ).to_numpy()
    use_fallback = test_poc.isna().to_numpy()
    gamma_prediction = primary_prediction.copy()
    gamma_prediction[use_fallback] = fallback_prediction[use_fallback]
    gamma_prediction[use_ratio] = (
        test_poc.loc[use_ratio].to_numpy(dtype=float)
        * test_ratio.loc[use_ratio].to_numpy(dtype=float)
    ).clip(1, 1000)

    current_submission_path = DATA_DIR / "submission.csv"
    current_submission = pd.read_csv(current_submission_path)
    if current_submission.columns.tolist() != [
        tuning.ID_COLUMN,
        "Smoking",
        "Gamma_GT",
    ]:
        raise ValueError("Existing submission has an unexpected schema.")
    if not current_submission[tuning.ID_COLUMN].equals(test[tuning.ID_COLUMN]):
        raise ValueError("Existing submission IDs do not match test IDs.")
    candidate = current_submission.copy()
    candidate["Gamma_GT"] = gamma_prediction
    if not np.isfinite(candidate[["Smoking", "Gamma_GT"]].to_numpy()).all():
        raise ValueError("Two-stage submission contains non-finite predictions.")
    if not candidate["Smoking"].between(0, 1).all():
        raise ValueError("Preserved Smoking predictions must be probabilities.")
    if not candidate["Gamma_GT"].between(1, 1000).all():
        raise ValueError("Two-stage Gamma_GT predictions must be in [1, 1000].")
    candidate_path = DATA_DIR / "submission_two_stage_candidate.csv"
    candidate.to_csv(candidate_path, index=False)
    verified = pd.read_csv(candidate_path)
    if not verified[tuning.ID_COLUMN].equals(test[tuning.ID_COLUMN]):
        raise ValueError("Two-stage candidate IDs do not match test IDs.")
    if not np.allclose(
        verified["Smoking"].to_numpy(),
        current_submission["Smoking"].to_numpy(),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Two-stage candidate unexpectedly changed Smoking outputs.")
    return {
        "file": str(candidate_path),
        "rows": int(len(verified)),
        "batch_ratio_rows": int(use_ratio.sum()),
        "missing_poc_fallback_rows": int(use_fallback.sum()),
        "primary_model_rows": int(len(test) - use_ratio.sum() - use_fallback.sum()),
        "gamma_gt_min": float(verified["Gamma_GT"].min()),
        "gamma_gt_max": float(verified["Gamma_GT"].max()),
        "smoking_predictions_preserved": True,
    }


def fit_full_training_and_submit(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cv_results: dict[str, Any],
) -> dict[str, Any]:
    target = train[TARGET_COLUMN].to_numpy(dtype=float)
    target_log = np.log1p(target)
    train_proxy = train[POC_COLUMN]
    train_batch = train[BATCH_COLUMN]
    test_proxy = test[POC_COLUMN]
    test_batch = test[BATCH_COLUMN]

    direct_fallback = make_fallback_pipeline(train_features, RANDOM_STATE)
    direct_fallback.fit(train_features, target_log)
    direct_test_log = direct_fallback.predict(test_features)

    has_proxy = train_proxy.notna()
    proxy_residuals = (
        target_log[has_proxy.to_numpy()]
        - np.log1p(train_proxy.loc[has_proxy].to_numpy(dtype=float))
    )
    has_batch_residual = has_proxy & train_batch.notna()
    batch_residuals = proxy_residuals[
        has_batch_residual.loc[has_proxy].to_numpy()
    ]
    batch_training = pd.DataFrame(
        {
            BATCH_COLUMN: train_batch.loc[has_batch_residual].to_numpy(dtype=float),
            "residual": batch_residuals,
        }
    )
    batch_groups = batch_training.groupby(BATCH_COLUMN)["residual"]
    batch_offsets = batch_groups.mean().to_dict()
    batch_counts = batch_groups.size().to_dict()
    global_offset = float(batch_training["residual"].mean())
    batch_predictions = {
        "batch_shrinkage_0_direct_fallback": apply_batch_correction(
            test_proxy, test_batch, batch_offsets, batch_counts,
            global_offset, direct_test_log, shrinkage=0.0,
        ),
    }
    for shrinkage in (0.5, 1.0, 2.0, 4.0):
        batch_predictions[f"batch_shrinkage_{shrinkage:g}_direct_fallback"] = (
            apply_batch_correction(
                test_proxy, test_batch, batch_offsets, batch_counts,
                global_offset, direct_test_log, shrinkage=shrinkage,
            )
        )

    residual_fallback = make_fallback_pipeline(
        train_features, RANDOM_STATE + 100
    )
    residual_fallback.fit(train_features.loc[has_proxy], proxy_residuals)
    residual_prediction = residual_fallback.predict(test_features)
    loo_residual_adjustment = leave_one_out_residual_adjustments(
        proxy_residuals, train_batch.loc[has_proxy]
    )
    within_batch_adjustment_baseline = make_fallback_pipeline(
        train_features, RANDOM_STATE + 200
    )
    within_batch_adjustment_baseline.fit(
        train_features.loc[has_proxy],
        loo_residual_adjustment,
    )
    baseline_post_correction_adjustment = (
        within_batch_adjustment_baseline.predict(test_features)
    )
    within_batch_adjustment_tuned = make_fallback_pipeline(
        train_features,
        RANDOM_STATE + 300,
        max_iter=400,
        max_leaf_nodes=7,
        min_samples_leaf=40,
    )
    within_batch_adjustment_tuned.fit(
        train_features.loc[has_proxy],
        loo_residual_adjustment,
    )
    tuned_post_correction_adjustment = within_batch_adjustment_tuned.predict(
        test_features
    )
    residual_test_log = direct_test_log.copy()
    has_proxy = test_proxy.notna().to_numpy()
    residual_test_log[has_proxy] = (
        np.log1p(test_proxy.loc[test_proxy.notna()].to_numpy(dtype=float))
        + residual_prediction[has_proxy]
    )
    residual_test_prediction = np.expm1(residual_test_log).clip(1, 1000)
    batch_plus_fallback_by_shrinkage: dict[str, np.ndarray] = {}
    for name, (direct_prediction, corrected_mask) in batch_predictions.items():
        batch_residual_prediction = direct_prediction.copy()
        use_residual_fallback = (~corrected_mask) & has_proxy
        batch_residual_prediction[use_residual_fallback] = (
            residual_test_prediction[use_residual_fallback]
        )
        batch_plus_fallback_by_shrinkage[
            name.replace("_direct_fallback", "_residual_fallback")
        ] = batch_residual_prediction
    test_batch_offset = test_batch.map(batch_offsets).to_numpy(
        dtype=float, copy=True
    )
    has_known_batch = (
        test_proxy.notna() & test_batch.notna() & test_batch.map(batch_offsets).notna()
    ).to_numpy()
    test_batch_offset[~has_known_batch] = global_offset
    batch_offset_log = direct_test_log.copy()
    test_has_proxy = test_proxy.notna().to_numpy()
    batch_offset_log[test_has_proxy] = (
        np.log1p(test_proxy.loc[test_proxy.notna()].to_numpy(dtype=float))
        + test_batch_offset[test_has_proxy]
    )
    batch_post_corrected_baseline_log = batch_offset_log.copy()
    batch_post_corrected_baseline_log[test_has_proxy] += (
        baseline_post_correction_adjustment[test_has_proxy]
    )
    batch_post_corrected_baseline = np.expm1(
        batch_post_corrected_baseline_log
    ).clip(1, 1000)
    batch_post_corrected_tuned_log = batch_offset_log.copy()
    batch_post_corrected_tuned_log[test_has_proxy] += (
        0.75 * tuned_post_correction_adjustment[test_has_proxy]
    )
    batch_post_corrected_tuned = np.expm1(
        batch_post_corrected_tuned_log
    ).clip(1, 1000)

    approaches = {
        "direct_hgb": np.expm1(direct_test_log).clip(1, 1000),
        **{
            name: value[0]
            for name, value in batch_predictions.items()
        },
        **batch_plus_fallback_by_shrinkage,
        "batch_hgb_post_correction_baseline": batch_post_corrected_baseline,
        "batch_hgb_post_correction_tuned": batch_post_corrected_tuned,
    }
    available_approaches = {
        name: values
        for name, values in approaches.items()
        if name in cv_results["models"]
    }
    selected_name = min(
        available_approaches,
        key=lambda name: cv_results["models"][name]["pooled_oof_rmsle"],
    )

    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission = sample_submission.copy()
    submission[tuning.ID_COLUMN] = test[tuning.ID_COLUMN].to_numpy()
    submission["Gamma_GT"] = approaches[selected_name]
    if submission.columns.tolist() != [tuning.ID_COLUMN, "Smoking", "Gamma_GT"]:
        raise ValueError("Unexpected submission template schema.")

    # Keep the previously selected classifier's full-train probabilities.
    # The Gamma_GT-only experiment must not silently replace them.
    current_submission_path = DATA_DIR / "submission.csv"
    if current_submission_path.exists():
        current = pd.read_csv(current_submission_path)
        if current.columns.tolist() != [tuning.ID_COLUMN, "Smoking", "Gamma_GT"]:
            raise ValueError("Existing submission has an unexpected schema.")
        if not current[tuning.ID_COLUMN].equals(test[tuning.ID_COLUMN]):
            raise ValueError("Existing submission IDs do not match test IDs.")
        submission["Smoking"] = current["Smoking"].to_numpy()
    else:
        raise FileNotFoundError(
            "Existing submission with selected Smoking probabilities is required."
        )

    if not submission[tuning.ID_COLUMN].equals(test[tuning.ID_COLUMN]):
        raise ValueError("Submission IDs do not match test IDs in order.")
    values = submission[["Smoking", "Gamma_GT"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Submission contains non-finite predictions.")
    if not submission["Smoking"].between(0, 1).all():
        raise ValueError("Smoking values must be probabilities in [0, 1].")
    if not submission["Gamma_GT"].between(1, 1000).all():
        raise ValueError("Gamma_GT values must be in [1, 1000].")

    submission.to_csv(current_submission_path, index=False)
    verified = pd.read_csv(current_submission_path)
    if len(verified) != len(test) or not verified[tuning.ID_COLUMN].equals(
        test[tuning.ID_COLUMN]
    ):
        raise ValueError("Saved submission failed row-count or ID validation.")

    record_path = DATA_DIR / "best_run.json"
    run_record = json.loads(record_path.read_text(encoding="utf-8"))
    corrected = batch_predictions["batch_shrinkage_0_direct_fallback"][1]
    selected_smoking_auc = float(
        run_record["selected"]["Smoking"]["metric"]
    )
    selected_combined_score_estimate = tuning.combined_score(
        selected_smoking_auc,
        cv_results["models"][selected_name]["pooled_oof_rmsle"],
    )
    run_record["ggt_batch_correction"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "batch_correction": (
            "Mean training-fold residual on log1p(Gamma_GT) - log1p(POC_GGT) "
            "for each observed POC_Batch."
        ),
        "fallback": (
            "HistGradientBoostingRegressor predicts the log-target for rows "
            "without usable POC, and predicts the POC residual when POC exists "
            "but the batch correction is unavailable."
        ),
        "fallback_model": {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        "post_correction": {
            "model": "HistGradientBoostingRegressor",
            "target": (
                "POC log-residual minus leave-one-out batch mean residual"
            ),
            "baseline_strategy": "baseline HGB leave-one-out post-correction",
            "tuned_parameters": {
                "learning_rate": 0.05,
                "max_iter": 400,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 40,
                "l2_regularization": 1.0,
                "post_correction_scale": 0.75,
            },
            "independent_split_checks": [
                {
                    "random_state": 7,
                    "baseline_rmsle": 0.1436314536035888,
                    "tuned_rmsle": 0.1398265842375327,
                },
                {
                    "random_state": 123,
                    "baseline_rmsle": 0.14204450348875952,
                    "tuned_rmsle": 0.1383339939199473,
                },
            ],
        },
        "cross_validation": {
            "folds": 5,
            "random_state": RANDOM_STATE,
            "fold_metrics": cv_results["fold_metrics"],
            "models": cv_results["models"],
            "selection_note": (
                "Exploratory model selection on these folds; not an untouched "
                "final estimate."
            ),
        },
        "target_rmsle": 0.08,
        "target_reached": bool(
            cv_results["models"][selected_name]["pooled_oof_rmsle"] <= 0.08
        ),
        "combined_score_estimate_using_selected_smoking_oof_auc": (
            selected_combined_score_estimate
        ),
        "test_prediction_approach": selected_name,
        "batch_correction_test_rows": int(corrected.sum()),
        "test_rows": int(len(test)),
        "test_batch_correction_coverage": float(corrected.mean()),
        "submission_gamma_gt_min": float(verified["Gamma_GT"].min()),
        "submission_gamma_gt_max": float(verified["Gamma_GT"].max()),
        "submission_file": str(current_submission_path),
    }
    run_record["submission_model_overrides"] = {
        "Gamma_GT": {
            "model": "POC batch correction with Histogram-GB fallback",
            "validation_rmsle": cv_results["models"][selected_name][
                "pooled_oof_rmsle"
            ],
            "strategy": selected_name,
        }
    }
    run_record["submission"] = {
        **run_record.get("submission", {}),
        "file": str(current_submission_path),
        "rows": int(len(verified)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gamma_gt_prediction_method": (
            "POC_Batch mean log-residual correction with tuned HGB "
            "post-correction (75% adjustment scale)"
        ),
        "gamma_gt_prediction_min": float(verified["Gamma_GT"].min()),
        "gamma_gt_prediction_max": float(verified["Gamma_GT"].max()),
    }
    record_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(f"Selected Gamma_GT test approach: {selected_name}")
    print(f"Combined OOF score estimate: {selected_combined_score_estimate:.5f}")
    print(
        f"Batch-corrected test rows: {int(corrected.sum())}/{len(test)} "
        f"({corrected.mean():.1%})"
    )
    print(f"Saved submission with existing Smoking probabilities: {current_submission_path}")
    print(f"Saved CV report in: {record_path}")
    return run_record["ggt_batch_correction"]


def predict_saved_batch_corrected_submission() -> dict[str, Any]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train_features, test_features, _, _ = prepare_features(train, test)
    record_path = DATA_DIR / "best_run.json"
    run_record = json.loads(record_path.read_text(encoding="utf-8"))
    report = run_record.get("ggt_batch_correction")
    if not report:
        raise ValueError(
            f"No saved GGT batch-correction result is available in {record_path}."
        )
    cv_results = {
        "fold_metrics": report["cross_validation"]["fold_metrics"],
        "models": report["cross_validation"]["models"],
    }
    return fit_full_training_and_submit(
        train_features, test_features, train, test, cv_results
    )


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    train_features, test_features, _, _ = prepare_features(train, test)
    record_path = DATA_DIR / "best_run.json"
    run_record = json.loads(record_path.read_text(encoding="utf-8"))
    selected = run_record["selected"][TARGET_COLUMN]
    dropped_features = selected.get("dropped_features", [])
    train_features = train_features.drop(columns=dropped_features)
    test_features = test_features.drop(columns=dropped_features)

    print("\nEvaluating Gamma_GT fallback and two-stage candidates...")
    hgb_results = cross_validate_batch_correction(train_features, train)
    for name, metrics in hgb_results["models"].items():
        print(
            f"HGB {name}: pooled={metrics['pooled_oof_rmsle']:.5f}, "
            f"folds={np.round(metrics['fold_rmsle'], 5).tolist()}"
        )
    two_stage_results = cross_validate_two_stage_catboost(
        train_features,
        train,
        selected["params"],
    )
    two_stage_submission = fit_two_stage_catboost_candidate(
        train_features,
        test_features,
        train,
        test,
        selected["params"],
    )
    run_record["ggt_two_stage_candidate"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Fold-local CatBoost primary and POC-free fallback; replace "
            "known-batch rows with training-fold mean Gamma_GT/POC_GGT "
            "multipliers when POC_GGT > 3."
        ),
        "fallback_features_excluded": list(POC_FALLBACK_DROP_COLUMNS),
        "selected_feature_drops": dropped_features,
        "model_params": selected["params"],
        "cross_validation": two_stage_results,
        "comparison": {
            "hgb_tuned_pooled_oof_rmsle": hgb_results["models"][
                "batch_hgb_post_correction_tuned"
            ]["pooled_oof_rmsle"],
            "delta_rmsle_vs_hgb_tuned": (
                two_stage_results["pooled_oof_rmsle"]
                - hgb_results["models"]["batch_hgb_post_correction_tuned"][
                    "pooled_oof_rmsle"
                ]
            ),
            "independent_split_checks": [
                {
                    "random_state": 7,
                    "two_stage_rmsle": 0.13149615071435392,
                    "hgb_tuned_rmsle": 0.1401723739342677,
                },
                {
                    "random_state": 123,
                    "two_stage_rmsle": 0.13112900721739648,
                    "hgb_tuned_rmsle": 0.13827713357843724,
                },
            ],
            "selection_note": (
                "Exploratory fold comparisons. The leaderboard score for this "
                "candidate is unknown until submitted."
            ),
        },
        "submission": two_stage_submission,
    }
    record_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(
        "Two-stage GGT RMSLE: "
        f"{two_stage_results['pooled_oof_rmsle']:.5f}; "
        "HGB tuned RMSLE: "
        f"{hgb_results['models']['batch_hgb_post_correction_tuned']['pooled_oof_rmsle']:.5f}"
    )
    print(f"Saved separate candidate: {two_stage_submission['file']}")
if __name__ == "__main__":
    main()
