from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGET_COL = "rate_per_10000_ed_visits"
PERIOD_COL = "period_id"
WEATHER_COLS = ["temp_avg_f", "precip_in"]
SIMILAR_STATES = {
    "AK": ["WA", "MT", "ND", "MN"],
    "HI": ["CA", "FL"],
    "DC": ["MD", "VA"],
}

DATASET_SPECS = {
    "Dataset A": {
        "suffix": "A_no_weather",
        "method": "drop_weather",
    },
    "Dataset B": {
        "suffix": "B_weather_period_median",
        "method": "period_median",
    },
    "Dataset C": {
        "suffix": "C_weather_similar_state",
        "method": "similar_state",
    },
}


def handle_text(data):
    """Convert missing DOH text to empty text and add a binary indicator."""
    data = data.copy()
    data["state_doh_release"] = data["state_doh_release"].fillna("")
    data["has_doh_release"] = data["state_doh_release"].str.strip().ne("").astype(int)
    return data


def split_by_period(data, val_size=0.2, random_state=42, period_col=PERIOD_COL):
    """Split rows so each period_id appears in train or validation, never both."""
    periods = pd.Series(data[period_col].dropna().unique()).sort_values().to_numpy()
    train_periods, val_periods = train_test_split(
        periods,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )
    train_df = data[data[period_col].isin(train_periods)].copy()
    val_df = data[data[period_col].isin(val_periods)].copy()
    return train_df, val_df, sorted(train_periods), sorted(val_periods)


def _impute_from_reference(data, reference, col):
    period_median = reference.groupby(PERIOD_COL)[col].median()
    global_median = reference[col].median()
    return data[col].fillna(data[PERIOD_COL].map(period_median)).fillna(global_median)


def make_dataset(data, method, reference=None):
    """Create one A/B/C modeling dataset."""
    data = handle_text(data)

    if method == "drop_weather":
        return data.drop(columns=WEATHER_COLS, errors="ignore")

    reference = handle_text(data if reference is None else reference)
    data["weather_missing"] = data["temp_avg_f"].isna().astype(int)

    if method == "similar_state":
        for col in WEATHER_COLS:
            for state, donors in SIMILAR_STATES.items():
                missing_idx = data.index[(data["jurisdiction"].eq(state)) & (data[col].isna())]
                for idx in missing_idx:
                    donor_values = reference.loc[
                        reference["jurisdiction"].isin(donors)
                        & reference[PERIOD_COL].eq(data.loc[idx, PERIOD_COL]),
                        col,
                    ].dropna()
                    if not donor_values.empty:
                        data.loc[idx, col] = donor_values.median()

    for col in WEATHER_COLS:
        data[col] = _impute_from_reference(data, reference, col)

    return data


def make_all_datasets(data, reference=None):
    return {
        name: make_dataset(data, spec["method"], reference=reference)
        for name, spec in DATASET_SPECS.items()
    }


def save_all_datasets(data, output_dir, prefix, val_size=0.2, random_state=42):
    """Save A/B/C train/validation tables split by period."""
    output_dir = Path(output_dir)
    train_raw, val_raw, train_periods, val_periods = split_by_period(
        data,
        val_size=val_size,
        random_state=random_state,
    )

    rows = []
    for name, spec in DATASET_SPECS.items():
        suffix = spec["suffix"]
        train_df = make_dataset(train_raw, spec["method"])
        val_df = make_dataset(val_raw, spec["method"], reference=train_raw)

        train_df.to_csv(output_dir / f"{prefix}_{suffix}_train.csv", index=False)
        val_df.to_csv(output_dir / f"{prefix}_{suffix}_val.csv", index=False)

        rows.append({
            "dataset": name,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "train_periods": len(train_periods),
            "val_periods": len(val_periods),
        })

    return pd.DataFrame(rows), train_periods, val_periods


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X):
    X = X.drop(columns=["state_doh_release"], errors="ignore").copy()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", make_ohe(), categorical_cols),
        ],
        remainder="drop",
    )


def get_models(preprocessor):
    return {
        "Model 1 - Ridge fixed effects": Pipeline([
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Model 2 - ElasticNet fixed effects": Pipeline([
            ("preprocess", preprocessor),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)),
        ]),
        "Model 3 - HistGradientBoosting": Pipeline([
            ("preprocess", preprocessor),
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]),
        "Model 4 - RandomForest": Pipeline([
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)),
        ]),
    }


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)


def predict_mean_baseline(train_df, val_df, group_col=None, target_col=TARGET_COL):
    global_mean = train_df[target_col].mean()
    if group_col is None or group_col not in train_df.columns:
        return pd.Series(global_mean, index=val_df.index)
    group_mean = train_df.groupby(group_col)[target_col].mean()
    return val_df[group_col].map(group_mean).fillna(global_mean)


def fit_and_score_dataset(train_df, val_df, dataset_name, prediction_type=None, target_col=TARGET_COL):
    train_df = train_df.dropna(subset=[target_col]).copy()
    val_df = val_df.dropna(subset=[target_col]).copy()

    group_col = "overdose_category" if "overdose_category" in train_df.columns else None
    rows = []

    y_pred = predict_mean_baseline(train_df, val_df, group_col=group_col, target_col=target_col)
    rmse, mae, r2 = evaluate_model(val_df[target_col], y_pred)
    row = {"dataset": dataset_name, "model": "Model 0 - Mean baseline", "RMSE": rmse, "MAE": mae, "R2": r2}
    if prediction_type is not None:
        row["prediction_type"] = prediction_type
    rows.append(row)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    for model_name, model in get_models(build_preprocessor(X_train)).items():
        model.fit(X_train, y_train)
        y_pred = np.maximum(model.predict(X_val), 0)
        rmse, mae, r2 = evaluate_model(y_val, y_pred)
        row = {"dataset": dataset_name, "model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}
        if prediction_type is not None:
            row["prediction_type"] = prediction_type
        rows.append(row)

    return rows
