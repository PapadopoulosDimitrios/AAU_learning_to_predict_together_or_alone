
# %%

from scipy.stats import gaussian_kde, norm
from sklearn.ensemble import RandomForestRegressor
import re

def get_holiday_calendar(country):
    if country == "Germany":
        return holidays.Germany()
    elif country == "Ireland":
        return holidays.Ireland()
    elif country == "Portugal":
        return holidays.Portugal()
    elif country == "Denmark":
        return holidays.Denmark()
    else:
        return holidays.Germany()


def add_calendar_features(df, country):
    df = df.copy()

    df["minute"] = df["ds"].dt.minute
    df["hour"] = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["day_of_year"] = df["ds"].dt.dayofyear
    df["week"] = df["ds"].dt.isocalendar().week.astype(int)
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    holiday_calendar = get_holiday_calendar(country)
    df["holiday"] = df["ds"].dt.normalize().map(
        lambda x: 1 if x in holiday_calendar else 0
    )

    minute_period = 60
    hour_period = 24
    week_period = 7
    month_period = 12
    year_period = 365.25

    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / minute_period)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / minute_period)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / hour_period)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / hour_period)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["day_of_week"] / week_period)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["day_of_week"] / week_period)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["day_of_year"] / year_period)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["day_of_year"] / year_period)
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / week_period)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / week_period)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / month_period)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / month_period)

    return df


def select_top_correlated_weather_lags(
    train_df,
    weather_cols,
    forecast_horizon,
    top_k_per_weather=6,
    max_weather_lag=None,
):
    if max_weather_lag is None:
        max_weather_lag = forecast_horizon * 2

    df = train_df.copy().sort_values(["unique_id", "ds"]).reset_index(drop=True)

    selected_weather_lag_features = []

    for col in weather_cols:
        corr_rows = []
        g = df.groupby("unique_id")[col]

        for lag in range(forecast_horizon, max_weather_lag + 1):
            lag_name = f"{col}_lag_{lag}"
            lag_values = g.shift(lag)

            tmp = pd.DataFrame({
                "y": df["y"].values,
                lag_name: lag_values.values
            }).dropna()

            if len(tmp) < 10:
                corr = 0.0
            else:
                corr = tmp["y"].corr(tmp[lag_name])
                if pd.isna(corr):
                    corr = 0.0

            corr_rows.append({
                "feature": lag_name,
                "abs_corr": abs(corr),
            })

        corr_df = pd.DataFrame(corr_rows).sort_values("abs_corr", ascending=False)
        best_feats = corr_df.head(top_k_per_weather)["feature"].tolist()
        selected_weather_lag_features.extend(best_feats)

    return sorted(selected_weather_lag_features)


def add_selected_weather_lags(df, selected_weather_lag_features):
    df = df.copy().sort_values(["unique_id", "ds"]).reset_index(drop=True)

    for feat in selected_weather_lag_features:
        m = re.fullmatch(r"(.+)_lag_(\d+)", feat)
        if m:
            base_col = m.group(1)
            lag = int(m.group(2))
            df[feat] = df.groupby("unique_id")[base_col].shift(lag)

    return df


def get_feature_groups():
    future_known_features = [
        "minute", "hour", "day_of_week", "day_of_year", "week", "month", "year",
        "is_weekend", "holiday",
        "minute_sin", "minute_cos",
        "hour_sin", "hour_cos",
        "dayofweek_sin", "dayofweek_cos",
        "dayofyear_sin", "dayofyear_cos",
        "week_sin", "week_cos",
        "month_sin", "month_cos",
    ]
    return future_known_features


def build_feature_enriched_df(
    df_all,
    home_cols,
    weather_cols,
    country,
    forecast_horizon,
    top_k_per_weather=6,
    max_weather_lag=None,
    train_end_for_selection=None,
):
    if max_weather_lag is None:
        max_weather_lag = forecast_horizon * 2

    df_nf = build_global_nf_df(df_all, home_cols, weather_cols)
    df_nf = add_calendar_features(df_nf, country=country)

    if train_end_for_selection is None:
        raise ValueError("train_end_for_selection must be provided.")

    train_only_df = df_nf[df_nf["ds"] < pd.Timestamp(train_end_for_selection)].copy()

    selected_weather_lag_features = select_top_correlated_weather_lags(
        train_df=train_only_df,
        weather_cols=weather_cols,
        forecast_horizon=forecast_horizon,
        top_k_per_weather=top_k_per_weather,
        max_weather_lag=max_weather_lag,
    )

    df_nf = add_selected_weather_lags(
        df=df_nf,
        selected_weather_lag_features=selected_weather_lag_features,
    )

    return df_nf, selected_weather_lag_features


def _mad(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return np.median(np.abs(x - med))


def empirical_bayes_threshold_from_importance(
    importances,
    feature_names,
    alpha=0.20,
    transform="log1p",
    central_prop=0.80,
    random_state=42,
):
    imp = np.asarray(importances, dtype=float)
    if np.any(imp < 0):
        raise ValueError("Importances must be non-negative.")

    names = np.asarray(feature_names)
    if len(names) != len(imp):
        raise ValueError("feature_names and importances must have the same length.")

    rng = np.random.default_rng(random_state)
    eps = 1e-12
    imp_j = imp + eps * rng.normal(size=len(imp))

    if transform == "log1p":
        z = np.log1p(np.maximum(imp_j, 0.0))
    elif transform is None:
        z = imp_j.copy()
    else:
        raise ValueError("transform must be 'log1p' or None")

    q_low = (1.0 - central_prop) / 2.0
    q_high = 1.0 - q_low
    lo, hi = np.quantile(z, [q_low, q_high])
    z_central = z[(z >= lo) & (z <= hi)]

    mu0 = np.median(z_central)
    sigma0 = 1.4826 * _mad(z_central)
    sigma0 = max(sigma0, 1e-6)

    if len(np.unique(z)) < 2:
        f_z = np.ones_like(z)
    else:
        kde = gaussian_kde(z)
        f_z = kde.evaluate(z)

    f0_z = norm.pdf(z, loc=mu0, scale=sigma0)

    pi0 = np.mean(z <= mu0 + sigma0)
    pi0 = float(np.clip(pi0, 0.50, 0.99))

    local_fdr = np.clip(pi0 * f0_z / np.maximum(f_z, 1e-12), 0.0, 1.0)
    selected = local_fdr <= alpha

    results_df = pd.DataFrame({
        "feature": names,
        "importance_raw": imp,
        "importance_transformed": z,
        "local_fdr": local_fdr,
        "selected": selected,
    }).sort_values(
        by=["selected", "importance_raw", "local_fdr"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    if results_df["selected"].any():
        threshold_raw = results_df.loc[results_df["selected"], "importance_raw"].min()
    else:
        threshold_raw = np.inf

    return results_df, threshold_raw


def select_exogenous_features(
    train_df,
    future_known_features,
    historical_lagged_features,
    alpha=0.20,
    rf_params=None,
    fallback_top_k=25,
):
    candidate_features = future_known_features + historical_lagged_features

    feat_df = train_df[["unique_id", "ds", "y"] + candidate_features].copy()
    feat_df = feat_df.dropna().reset_index(drop=True)

    X = feat_df[candidate_features].copy()
    y = feat_df["y"].values

    if rf_params is None:
        rf_params = {
            "n_estimators": 500,
            "random_state": 42,
            "n_jobs": -1,
            "max_features": "sqrt",
        }

    rf = RandomForestRegressor(**rf_params)
    rf.fit(X, y)

    importance_df, threshold_raw = empirical_bayes_threshold_from_importance(
        importances=rf.feature_importances_,
        feature_names=X.columns.tolist(),
        alpha=alpha,
        transform="log1p",
        central_prop=0.80,
        random_state=42,
    )

    selected_features = importance_df.loc[importance_df["selected"], "feature"].tolist()

    if len(selected_features) == 0:
        selected_features = (
            importance_df.sort_values("importance_raw", ascending=False)
            .head(min(fallback_top_k, len(importance_df)))["feature"]
            .tolist()
        )
        importance_df["selected"] = importance_df["feature"].isin(selected_features)
        threshold_raw = importance_df.loc[
            importance_df["feature"].isin(selected_features), "importance_raw"
        ].min()

    selected_future_known = [f for f in selected_features if f in future_known_features]
    selected_historical_lags = [f for f in selected_features if f in historical_lagged_features]

    print(f"Empirical-Bayes threshold (raw importance): {threshold_raw:.8f}")
    print(f"Selected future-known exogenous features: {len(selected_future_known)}")
    print(f"Selected historical lagged exogenous features: {len(selected_historical_lags)}")

    return selected_future_known, selected_historical_lags, importance_df


def keep_only_required_columns(df, selected_future_known, selected_historical_lags):
    keep_cols = ["unique_id", "ds", "y"] + selected_future_known + selected_historical_lags
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()

# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from neuralforecast import NeuralForecast
import optuna
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from pathlib import Path
import pathlib
import math
import holidays
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import random
import torch
import warnings
import xgboost as xgb
import lightgbm as lgb
import torch
torch.set_float32_matmul_precision("medium")
from neuralforecast.models import MLP
from sklearn.metrics import root_mean_squared_error
from neuralforecast.losses.pytorch import MSE


# ============================================================
# GLOBAL SETTINGS FOR LIGHTWEIGHT HPO
# ============================================================
MAX_STEPS = 500
VAL_CHECK_STEPS = MAX_STEPS // 10


def split_train_val_test_global(df_nf, test_start, forecast_horizon, training_size, val_days=3, freq_minutes=15):
    """
    Split a long NeuralForecast dataframe into train / validation / test per series.

    Validation starts `val_days` before test_start and ends right before test_start.
    Test spans `forecast_horizon` steps starting at test_start.
    Training is the last `training_size` rows before validation starts.
    """
    test_start = pd.Timestamp(test_start)
    val_start = test_start - pd.Timedelta(days=val_days)
    test_end = test_start + pd.Timedelta(minutes=freq_minutes * forecast_horizon)

    train_parts = []
    val_parts = []
    test_parts = []

    expected_val_len = val_days * (24 * 60 // freq_minutes)

    for uid, g in df_nf.groupby("unique_id"):
        g = g.sort_values("ds").reset_index(drop=True)

        # train: everything before validation starts, keep only last training_size rows
        train_candidates = g[g["ds"] < val_start].copy()
        train_df_uid = train_candidates.iloc[-training_size:].copy()

        # validation: from val_start until just before test_start
        val_df_uid = g[(g["ds"] >= val_start) & (g["ds"] < test_start)].copy()

        # test: from test_start for forecast_horizon steps
        test_df_uid = g[(g["ds"] >= test_start) & (g["ds"] < test_end)].copy()

        # safety checks
        if len(train_df_uid) != training_size:
            raise ValueError(f"{uid}: expected {training_size} training rows, got {len(train_df_uid)}")

        if len(val_df_uid) != expected_val_len:
            raise ValueError(f"{uid}: expected {expected_val_len} validation rows, got {len(val_df_uid)}")

        if len(test_df_uid) != forecast_horizon:
            raise ValueError(f"{uid}: expected {forecast_horizon} test rows, got {len(test_df_uid)}")

        train_parts.append(train_df_uid)
        val_parts.append(val_df_uid)
        test_parts.append(test_df_uid)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, val_df, test_df


def build_global_nf_df(df_all, home_cols, weather_cols=None):
    """
    Convert a wide household load dataframe into NeuralForecast long format.

    Parameters
    ----------
    df_all : pd.DataFrame
        Index must be DatetimeIndex, columns include home_* and optional weather cols.
    home_cols : list
        List of household columns, e.g. ['home_1', 'home_2', ...]
    weather_cols : list or None
        Optional list of weather columns to attach to every home/timestamp row.

    Returns
    -------
    df_nf : pd.DataFrame
        Columns: unique_id, ds, y, [weather columns...]
    """
    if not isinstance(df_all.index, pd.DatetimeIndex):
        raise ValueError("df_all index must be a DatetimeIndex.")

    # keep time as a normal column
    df_base = df_all.reset_index().rename(columns={"timestamp": "ds"})

    # wide -> long for homes
    df_nf = df_base.melt(
        id_vars=["ds"],
        value_vars=home_cols,
        var_name="unique_id",
        value_name="y"
    )

    # add weather columns if requested
    if weather_cols is not None and len(weather_cols) > 0:
        weather_df = df_base[["ds"] + weather_cols].copy()
        df_nf = df_nf.merge(weather_df, on="ds", how="left")

    # sort for safety
    df_nf = df_nf.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return df_nf

def build_daily_profile_matrix(df_all, home_cols):
    df_tmp = df_all.copy()
    df_tmp["slot"] = df_tmp.index.hour * 4 + df_tmp.index.minute // 15

    profiles = []
    for home in home_cols:
        avg_profile = df_tmp.groupby("slot")[home].mean()
        avg_profile.name = home
        profiles.append(avg_profile)

    profile_df = pd.concat(profiles, axis=1).T
    profile_df.index.name = "home"

    return profile_df



def rolling_forecasting_validation_predictions(
    train_df,
    val_df,
    h,
    model_params,
    future_known_features,
    historical_lagged_features,
    freq="15min"
):
    rolling_train_df = train_df.copy()
    val_predictions = []

    val_starts = sorted(val_df["ds"].unique())[::h]

    for window_start in val_starts:

        model = MLP(
            h=h,
            input_size=model_params["input_size"],
            futr_exog_list=future_known_features,
            hist_exog_list=historical_lagged_features,
            num_layers=model_params["num_layers"],
            hidden_size=model_params["hidden_size"],
            batch_size=model_params["batch_size"],
            learning_rate=model_params["learning_rate"],
            max_steps=MAX_STEPS,
            val_check_steps=min(VAL_CHECK_STEPS, MAX_STEPS),
            scaler_type=model_params["scaler_type"],
            random_seed=42,
            loss=MSE(),
        )

        nf = NeuralForecast(models=[model], freq=freq)
        nf.fit(df=rolling_train_df)

        next_val_chunk = val_df[
            (val_df["ds"] >= window_start) &
            (val_df["ds"] < window_start + pd.Timedelta(minutes=15 * h))
        ].copy()

        futr_df = next_val_chunk[["unique_id", "ds"] + future_known_features].copy()

        preds = nf.predict(futr_df=futr_df)
        val_predictions.append(preds)

        rolling_train_df = pd.concat([rolling_train_df, next_val_chunk], ignore_index=True)
        rolling_train_df = rolling_train_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return pd.concat(val_predictions, ignore_index=True)




def compute_average_rmse_per_cluster(val_df, val_preds_df, pred_col="MLP"):
    val_compare_df = val_df.merge(val_preds_df, on=["unique_id", "ds"], how="left")

    rmse_rows = []
    for uid, g in val_compare_df.groupby("unique_id"):
        rmse_rows.append({
            "unique_id": uid,
            "RMSE": root_mean_squared_error(g["y"], g[pred_col])
        })

    rmse_per_home = pd.DataFrame(rmse_rows).sort_values("RMSE").reset_index(drop=True)
    avg_rmse_cluster = rmse_per_home["RMSE"].mean()

    return avg_rmse_cluster, rmse_per_home, val_compare_df

def objective(trial):

    model_params = {
        "num_layers": trial.suggest_int("num_layers", 2, 4, step=1),
        "hidden_size": trial.suggest_int("hidden_size", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "input_size": trial.suggest_categorical("input_size", [96, 192, 288, 672]),
        "scaler_type": trial.suggest_categorical("scaler_type", ["standard", "robust"]),
    }

    try:
        val_preds_df = rolling_forecasting_validation_predictions(
            train_df=train_df,
            val_df=val_df,
            h=forecast_horizon,
            model_params=model_params,
            future_known_features=selected_future_known,
            historical_lagged_features=selected_historical_lags,
            freq="15min"
        )

        avg_rmse_cluster, _, _ = compute_average_rmse_per_cluster(
            val_df=val_df,
            val_preds_df=val_preds_df,
            pred_col="MLP"
        )

        return avg_rmse_cluster

    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")

# %% [markdown]
# # start

# %%

project_path = r"C:\Users\CR58XM\Documents\GitHub\AAU_learning_to_predict_together_or_alone"
days_json_path = pathlib.Path(project_path) / "dataset_days.json"

countries = ["Germany", "Ireland", "Portugal"]
#countries = ["Germany"]
#countries = ["Denmark"]

days = ["day1", "day2", "day3", "day4", "day5"]
#days = ["day1"]



weather_cols = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "direct_radiation",
#    "price_eur_kwh"
]


forecast_horizon = 96
training_size = 96 * 7 * 3 * 2
feature_selection = True
plot_forecast = True
hyperparameter_opt = True
opt_trials = 20

# -------------------------
# Read JSON with forecast days
# -------------------------
with open(days_json_path, "r") as f:
    dataset_days = json.load(f)

# -------------------------
# Loop over countries
# -------------------------
for country in countries:
    print(f"\n{'#'*100}")
    print(f"COUNTRY: {country}")
    print(f"{'#'*100}")

    dataset_path = pathlib.Path(project_path) / "DataCleaning" / "clean" / f"dataset_{country}.csv"

    df_all = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    df_all = df_all.set_index("timestamp")
    df_all = df_all.sort_index()

    home_cols = [col for col in df_all.columns if col.startswith("home_")]

    print(f"Detected {len(home_cols)} homes for {country}.")
    print(home_cols)

    # make the dataset of each country clustered
    df_nf = build_global_nf_df(df_all, home_cols, weather_cols)
    profile_df = build_daily_profile_matrix(df_all, home_cols)

    print(profile_df.head())
    print(profile_df.shape)   # should be (28, 96)

    scaler = StandardScaler()
    X_profile = scaler.fit_transform(profile_df)

    results = []
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X_profile)
        score = silhouette_score(X_profile, labels)

        results.append({"k": k, "silhouette_score": score})

    results_df = pd.DataFrame(results).sort_values("silhouette_score", ascending=False)
    print(results_df)

    best_k = int(results_df.iloc[0]["k"])
    best_score = results_df.iloc[0]["silhouette_score"]

    print(f"Best k: {best_k}")
    print(f"Best silhouette score: {best_score:.4f}")

    best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    best_labels = best_kmeans.fit_predict(X_profile)

    cluster_profile_df = profile_df.copy()
    cluster_profile_df["cluster"] = best_labels

    print(cluster_profile_df["cluster"].sort_values())
    # finish clustering
    
    # -------------------------
    # Loop over days
    # -------------------------
    for day_name in days:
        selected_day = dataset_days[country][day_name]
        date = f"{selected_day} 00:00:00"
        forecast_end_date = str(pd.Timestamp(date) + pd.Timedelta(days=1))

        print(f"\n{'='*100}")
        print(f"Running {country} - {day_name}")
        print(f"Forecast start: {date}")
        print(f"Forecast end:   {forecast_end_date}")
        print(f"{'='*100}")


        clusters=best_k
        cluster_test_preds_list = [] # for the predictions
        for cluster in range(clusters):
            print(cluster)

            # --------------------------------------------------
            # iterate over clusters
            # --------------------------------------------------
            selected_cluster = cluster   # change to 1 if you want the other one later

            cluster_homes = cluster_profile_df.index[cluster_profile_df["cluster"] == selected_cluster].tolist()

            print(f"Selected cluster: {selected_cluster}")
            print(f"Number of homes in cluster: {len(cluster_homes)}")
            print("Homes in cluster:")
            print(cluster_homes)


            # --------------------------------------------------
            # subset original dataframe to homes in this cluster
            # --------------------------------------------------
            cluster_cols = cluster_homes + weather_cols
            df_cluster_wide = df_all[cluster_cols].copy()

            print("\nCluster-wide dataframe head:")
            print(df_cluster_wide.head())


            val_start_for_selection = pd.Timestamp(date) - pd.Timedelta(days=3)

            df_cluster_nf, selected_weather_lag_features = build_feature_enriched_df(
                df_all=df_cluster_wide,
                home_cols=cluster_homes,
                weather_cols=weather_cols,
                country=country,
                forecast_horizon=forecast_horizon,
                top_k_per_weather=6,
                max_weather_lag=forecast_horizon * 2,
                train_end_for_selection=val_start_for_selection,
            )

            print("\nCluster long-format dataset:")
            print(df_cluster_nf.head(10))

            print("\nColumns:")
            print(df_cluster_nf.columns.tolist())

            print("\nShape:")
            print(df_cluster_nf.shape)

            print("\nUnique homes in long dataset:")
            print(df_cluster_nf['unique_id'].unique())

            train_df, val_df, test_df = split_train_val_test_global(
                df_nf=df_cluster_nf,
                test_start=date,
                forecast_horizon=forecast_horizon,
                training_size=training_size,
                val_days=3,
                freq_minutes=15
            )

            for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
                start = df['ds'].min()
                end = df['ds'].max()
                print(f"{name}: {start} to {end} (Shape: {df.shape})")

            future_known_candidates = get_feature_groups()

            selected_future_known, selected_historical_lags, importance_df = select_exogenous_features(
                train_df=train_df,
                future_known_features=future_known_candidates,
                historical_lagged_features=selected_weather_lag_features,
                alpha=0.20,
                rf_params={
                    "n_estimators": 500,
                    "random_state": 42,
                    "n_jobs": -1,
                    "max_features": "sqrt",
                },
                fallback_top_k=25,
            )

            print("\nSelected future-known features:")
            print(selected_future_known)

            print("\nSelected historical lagged weather features:")
            print(selected_historical_lags)

            print("\nTop feature importances:")
            print(importance_df.head(20))

            train_df = keep_only_required_columns(train_df, selected_future_known, selected_historical_lags)
            val_df = keep_only_required_columns(val_df, selected_future_known, selected_historical_lags)
            test_df = keep_only_required_columns(test_df, selected_future_known, selected_historical_lags)


            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=opt_trials, show_progress_bar=True)

            print("Best avg RMSE:", study.best_value)
            print("Best params:", study.best_params)


            # now we keep the best parameters and we predict the test
            best_params = study.best_params

            train_val_df = pd.concat([train_df, val_df], ignore_index=True)
            train_val_df = train_val_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


            final_mlp = MLP(
                h=forecast_horizon,
                input_size=best_params["input_size"],
                futr_exog_list=selected_future_known,
                hist_exog_list=selected_historical_lags,
                num_layers=best_params["num_layers"],
                hidden_size=best_params["hidden_size"],
                batch_size=best_params["batch_size"],
                learning_rate=best_params["learning_rate"],
                max_steps=MAX_STEPS,
                val_check_steps=min(VAL_CHECK_STEPS, MAX_STEPS),
                scaler_type=best_params["scaler_type"],
                random_seed=42,
                loss=MSE(),
            )

            nf_final = NeuralForecast(
                models=[final_mlp],
                freq="15min",
            )

            nf_final.fit(df=train_val_df)

            futr_df_test = test_df[["unique_id", "ds"] + selected_future_known].copy()
            test_preds_df = nf_final.predict(futr_df=futr_df_test)
            
            test_preds_wide = test_preds_df.pivot(
                index="ds",
                columns="unique_id",
                values="MLP"
            ).sort_index()
            cluster_test_preds_list.append(test_preds_wide)
        final_test_preds_wide = pd.concat(cluster_test_preds_list, axis=1).sort_index()


        # --------------------------------------------------
        # save final combined test predictions
        # --------------------------------------------------
        save_dir = pathlib.Path(project_path) / "Outputs" / "Global models" / "MLP"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"prediction_MLP_{day_name}_{country}.csv"

        final_test_preds_wide.to_csv(save_path, index=True)

        print(f"Saved final_test_preds_wide to: {save_path}")




# %%
import time

start_time = time.time()



project_path = r"C:\Users\CR58XM\Documents\GitHub\AAU_learning_to_predict_together_or_alone"
days_json_path = pathlib.Path(project_path) / "dataset_days.json"

#countries = ["Germany", "Ireland", "Portugal"]
#countries = ["Germany"]
countries = ["Denmark"]

days = ["day1", "day2", "day3", "day4", "day5"]
#days = ["day1"]



weather_cols = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "direct_radiation",
    "price_eur_kwh"
]


forecast_horizon = 96
training_size = 96 * 7 * 3 * 2
feature_selection = True
plot_forecast = True
hyperparameter_opt = True
opt_trials = 20

# -------------------------
# Read JSON with forecast days
# -------------------------
with open(days_json_path, "r") as f:
    dataset_days = json.load(f)

# -------------------------
# Loop over countries
# -------------------------
for country in countries:
    print(f"\n{'#'*100}")
    print(f"COUNTRY: {country}")
    print(f"{'#'*100}")

    dataset_path = pathlib.Path(project_path) / "DataCleaning" / "clean" / f"dataset_{country}.csv"

    df_all = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    df_all = df_all.set_index("timestamp")
    df_all = df_all.sort_index()

    home_cols = [col for col in df_all.columns if col.startswith("home_")]

    print(f"Detected {len(home_cols)} homes for {country}.")
    print(home_cols)

    # make the dataset of each country clustered
    df_nf = build_global_nf_df(df_all, home_cols, weather_cols)
    profile_df = build_daily_profile_matrix(df_all, home_cols)

    print(profile_df.head())
    print(profile_df.shape)   # should be (28, 96)

    scaler = StandardScaler()
    X_profile = scaler.fit_transform(profile_df)

    results = []
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X_profile)
        score = silhouette_score(X_profile, labels)

        results.append({"k": k, "silhouette_score": score})

    results_df = pd.DataFrame(results).sort_values("silhouette_score", ascending=False)
    print(results_df)

    best_k = int(results_df.iloc[0]["k"])
    best_score = results_df.iloc[0]["silhouette_score"]

    print(f"Best k: {best_k}")
    print(f"Best silhouette score: {best_score:.4f}")

    best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    best_labels = best_kmeans.fit_predict(X_profile)

    cluster_profile_df = profile_df.copy()
    cluster_profile_df["cluster"] = best_labels

    print(cluster_profile_df["cluster"].sort_values())
    # finish clustering
    
    # -------------------------
    # Loop over days
    # -------------------------
    for day_name in days:
        selected_day = dataset_days[country][day_name]
        date = f"{selected_day} 00:00:00"
        forecast_end_date = str(pd.Timestamp(date) + pd.Timedelta(days=1))

        print(f"\n{'='*100}")
        print(f"Running {country} - {day_name}")
        print(f"Forecast start: {date}")
        print(f"Forecast end:   {forecast_end_date}")
        print(f"{'='*100}")


        clusters=best_k
        cluster_test_preds_list = [] # for the predictions
        for cluster in range(clusters):
            print(cluster)

            # --------------------------------------------------
            # iterate over clusters
            # --------------------------------------------------
            selected_cluster = cluster   # change to 1 if you want the other one later

            cluster_homes = cluster_profile_df.index[cluster_profile_df["cluster"] == selected_cluster].tolist()

            print(f"Selected cluster: {selected_cluster}")
            print(f"Number of homes in cluster: {len(cluster_homes)}")
            print("Homes in cluster:")
            print(cluster_homes)


            # --------------------------------------------------
            # subset original dataframe to homes in this cluster
            # --------------------------------------------------
            cluster_cols = cluster_homes + weather_cols
            df_cluster_wide = df_all[cluster_cols].copy()

            print("\nCluster-wide dataframe head:")
            print(df_cluster_wide.head())


            val_start_for_selection = pd.Timestamp(date) - pd.Timedelta(days=3)

            df_cluster_nf, selected_weather_lag_features = build_feature_enriched_df(
                df_all=df_cluster_wide,
                home_cols=cluster_homes,
                weather_cols=weather_cols,
                country=country,
                forecast_horizon=forecast_horizon,
                top_k_per_weather=6,
                max_weather_lag=forecast_horizon * 2,
                train_end_for_selection=val_start_for_selection,
            )

            print("\nCluster long-format dataset:")
            print(df_cluster_nf.head(10))

            print("\nColumns:")
            print(df_cluster_nf.columns.tolist())

            print("\nShape:")
            print(df_cluster_nf.shape)

            print("\nUnique homes in long dataset:")
            print(df_cluster_nf['unique_id'].unique())

            train_df, val_df, test_df = split_train_val_test_global(
                df_nf=df_cluster_nf,
                test_start=date,
                forecast_horizon=forecast_horizon,
                training_size=training_size,
                val_days=3,
                freq_minutes=15
            )

            for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
                start = df['ds'].min()
                end = df['ds'].max()
                print(f"{name}: {start} to {end} (Shape: {df.shape})")

            future_known_candidates = get_feature_groups()

            selected_future_known, selected_historical_lags, importance_df = select_exogenous_features(
                train_df=train_df,
                future_known_features=future_known_candidates,
                historical_lagged_features=selected_weather_lag_features,
                alpha=0.20,
                rf_params={
                    "n_estimators": 500,
                    "random_state": 42,
                    "n_jobs": -1,
                    "max_features": "sqrt",
                },
                fallback_top_k=25,
            )

            print("\nSelected future-known features:")
            print(selected_future_known)

            print("\nSelected historical lagged weather features:")
            print(selected_historical_lags)

            print("\nTop feature importances:")
            print(importance_df.head(20))

            train_df = keep_only_required_columns(train_df, selected_future_known, selected_historical_lags)
            val_df = keep_only_required_columns(val_df, selected_future_known, selected_historical_lags)
            test_df = keep_only_required_columns(test_df, selected_future_known, selected_historical_lags)


            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=opt_trials, show_progress_bar=True)

            print("Best avg RMSE:", study.best_value)
            print("Best params:", study.best_params)


            # now we keep the best parameters and we predict the test
            best_params = study.best_params

            train_val_df = pd.concat([train_df, val_df], ignore_index=True)
            train_val_df = train_val_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


            final_mlp = MLP(
                h=forecast_horizon,
                input_size=best_params["input_size"],
                futr_exog_list=selected_future_known,
                hist_exog_list=selected_historical_lags,
                num_layers=best_params["num_layers"],
                hidden_size=best_params["hidden_size"],
                batch_size=best_params["batch_size"],
                learning_rate=best_params["learning_rate"],
                max_steps=MAX_STEPS,
                val_check_steps=min(VAL_CHECK_STEPS, MAX_STEPS),
                scaler_type=best_params["scaler_type"],
                random_seed=42,
                loss=MSE(),
            )

            nf_final = NeuralForecast(
                models=[final_mlp],
                freq="15min",
            )

            nf_final.fit(df=train_val_df)

            futr_df_test = test_df[["unique_id", "ds"] + selected_future_known].copy()
            test_preds_df = nf_final.predict(futr_df=futr_df_test)
            
            test_preds_wide = test_preds_df.pivot(
                index="ds",
                columns="unique_id",
                values="MLP"
            ).sort_index()
            cluster_test_preds_list.append(test_preds_wide)
        final_test_preds_wide = pd.concat(cluster_test_preds_list, axis=1).sort_index()


        # --------------------------------------------------
        # save final combined test predictions
        # --------------------------------------------------
        save_dir = pathlib.Path(project_path) / "Outputs" / "Global models" / "MLP"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"prediction_MLP_{day_name}_{country}.csv"

        final_test_preds_wide.to_csv(save_path, index=True)

        print(f"Saved final_test_preds_wide to: {save_path}")


end_time = time.time()

total_seconds = end_time - start_time

# %%
print(f"Time taken: {total_seconds:.4f} seconds")


file_path = r"C:\Users\CR58XM\Documents\GitHub\AAU_learning_to_predict_together_or_alone\time_spend.json"

# 1. Load existing JSON
with open(file_path, "r") as f:
    data = json.load(f)

# 2. Add TCN inside "Global"
data["Global"]["MLP"] = total_seconds

# 3. Save back (without disturbing structure)
with open(file_path, "w") as f:
    json.dump(data, f, indent=4)

# %% [markdown]
# # end 


