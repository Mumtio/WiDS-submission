# %%
# !pip install scikit-learn==1.3.2 scikit-survival==0.22.2 --quiet

import subprocess
subprocess.run(["pip", "install", "scikit-survival", "--quiet"], check=True)
# %%
# !pip install numpy==1.26.4
# !pip install --upgrade scikit-survival


# %%
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest

print("All good ✅")

# %%
# WiDS 2026 Wildfire Survival Notebook



import warnings
warnings.filterwarnings("ignore")

import os
import math
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis


# ============================================================
# CONFIG
# ============================================================
@dataclass
class Config:
    data_dir: Optional[Path] = Path("/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26")
    train_name: str = "train.csv"
    test_name: str = "test.csv"
    horizons: Tuple[int, ...] = (12, 24, 48, 72)
    # Increased folds: 7 gives more OOF data for meta-learner
    n_folds: int = 7
    seed: int = 42
    fast_mode: bool = False
    verbose: bool = True
    calibrator_kind: str = "ridge_logit"
    force_p12_far_zero: bool = True      # only very-far (>15km) zone
    force_p72_one: bool = True
    enforce_monotone: bool = True
    use_xgb_aft: bool = HAS_XGB and HAS_SCIPY
    use_timing_model: bool = HAS_SCIPY
    do_oof_weight_search: bool = True
    use_meta_stacking: bool = HAS_LGB    # Level-1 LGB stacker
    use_isotonic: bool = True            # Post-hoc isotonic calibration
    use_tta: bool = True                 # Test-time augmentation
    ipcw_clip: float = 30.0
    timing_weight_default: float = 0.10
    oof_min_improve: float = 0.0002      # Tighter threshold to accept new weights
    near_dist_m: float = 5000.0
    far_dist_m: float = 10000.0
    very_far_dist_m: float = 15000.0    # Stricter threshold for p12=0 rule

CFG = Config()


# ============================================================
# UTILITIES
# ============================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(cfg: Config):
    train = pd.read_csv(cfg.data_dir / cfg.train_name)
    test  = pd.read_csv(cfg.data_dir / cfg.test_name)
    id_col    = "event_id"
    time_col  = "time_to_hit_hours"
    event_col = "event"
    dist_col  = "dist_min_ci_0_5h"
    if cfg.verbose:
        print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test, id_col, time_col, event_col, dist_col


def make_strat_label(df, event_col, dist_col):
    near = (df[dist_col].values < CFG.near_dist_m).astype(int)
    return df[event_col].values.astype(int) * 2 + near


def safe_col(df, name, default=0.0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def pct_rank(values, ref):
    ref = np.asarray(ref, float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return np.zeros(len(values), float)
    ref = np.sort(ref)
    return np.searchsorted(ref, values, side="right") / len(ref)

def tune_meta_weight(base_oof, meta_oof, t, e, horizons):
    best_w = 0.0
    best_score = -1e18
    print("\n===== META WEIGHT SEARCH =====")
    best_metrics = None

    for w in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        pred = np.clip((1 - w) * base_oof + w * meta_oof, 1e-6, 1 - 1e-6)
        metrics = compute_hybrid_score(
          t, e,
          pred[:, 0],
          pred[:, 1],
          pred[:, 2],
          pred[:, 3],
      )
        print(f"w={w:.2f} → hybrid={metrics['hybrid']:.6f}")

        if metrics["hybrid"] > best_score:
            best_score = metrics["hybrid"]
            best_w = w
            best_metrics = metrics

    print(f"Best meta weight = {best_w:.2f}")
    print("Best meta metrics:", best_metrics)

    return best_w

def smooth_probs(p, temp=1.2):
    logit = np.log(p / (1 - p))
    logit = logit / temp
    return 1 / (1 + np.exp(-logit))


def smooth_horizons(p):
    p[:, 1] = 0.85*p[:, 1] + 0.15*p[:, 0]
    p[:, 2] = 0.85*p[:, 2] + 0.15*p[:, 1]
    p[:, 3] = 0.85*p[:, 3] + 0.15*p[:, 2]
    return p

def find_best_temp(oof_pred, train, time_col, event_col, horizons, isos):
    t = train[time_col].values.astype(float)
    e = train[event_col].values.astype(bool)

    best_temp = 1.0
    best_score = -1e18

    for temp in [0.8, 1.0, 1.2, 1.4, 1.6]:
        p = smooth_probs(oof_pred, temp=temp)

        #  FULL PIPELINE
        p = transform_isotonic(p, isos, horizons)
        p = smooth_horizons(p)
        p = pava_monotone(np.clip(p, 1e-6, 1-1e-6))

        m = compute_hybrid_score(
            t, e,
            p[:, horizons.index(24)],
            p[:, horizons.index(48)],
            p[:, horizons.index(72)]
        )

        print(f"temp={temp} → {m['hybrid']:.6f}")

        if m["hybrid"] > best_score:
            best_score = m["hybrid"]
            best_temp = temp

    print("Best temp:", best_temp)
    return best_temp


# ============================================================
# FEATURE ENGINEERING  (expanded)
# ============================================================
def compute_base_features(df: pd.DataFrame, dist_col: str) -> pd.DataFrame:
    out = df.copy()

    dist_m   = pd.to_numeric(out[dist_col], errors="coerce").fillna(0.0).clip(lower=1.0)
    dist_km  = dist_m / 1000.0

    area_ha          = safe_col(out, "area_first_ha", 0.0).fillna(0.0).clip(lower=0.0)
    area_growth_abs  = safe_col(out, "area_growth_abs_0_5h", 0.0).fillna(0.0)
    area_growth_rate = safe_col(out, "area_growth_rate_ha_per_h", 0.0).fillna(0.0)

    closing      = safe_col(out, "closing_speed_m_per_h", 0.0).fillna(0.0)
    closing_abs  = safe_col(out, "closing_speed_abs_m_per_h", 0.0).fillna(0.0).clip(lower=0.0)
    radial_growth_m = safe_col(out, "radial_growth_m", 0.0).fillna(0.0)
    radial_rate  = safe_col(out, "radial_growth_rate_m_per_h", 0.0).fillna(0.0)
    centroid_speed = safe_col(out, "centroid_speed_m_per_h", 0.0).fillna(0.0)
    alignment_abs  = safe_col(out, "alignment_abs", 0.0).fillna(0.0).clip(0.0, 1.0)
    projected_advance = safe_col(out, "projected_advance_m", 0.0).fillna(0.0)
    perim_count    = safe_col(out, "num_perimeters_0_5h", 0.0).fillna(0.0)

    # Extra raw signals (use if present)
    wind_speed   = safe_col(out, "wind_speed_m_per_s", 0.0).fillna(0.0).clip(0.0)
    rh           = safe_col(out, "relative_humidity_pct", 50.0).fillna(50.0).clip(1.0, 100.0)
    temp_c       = safe_col(out, "temperature_c", 20.0).fillna(20.0)
    slope_deg    = safe_col(out, "terrain_slope_deg", 0.0).fillna(0.0)
    canopy_pct   = safe_col(out, "canopy_cover_pct", 0.0).fillna(0.0)
    fuel_moisture = safe_col(out, "fuel_moisture_pct", 20.0).fillna(20.0).clip(1.0)

    month = safe_col(out, "event_start_month", 6).fillna(6)
    dow   = safe_col(out, "event_start_dayofweek", 3).fillna(3)
    hour  = safe_col(out, "event_start_hour", 12).fillna(12)

    area_km2    = area_ha / 100.0
    radius_m    = np.sqrt((area_ha * 10000.0) / np.pi)
    effective_closing = (closing_abs + radial_rate + alignment_abs * np.abs(closing)).clip(lower=0.01)
    eta_hours   = dist_m / effective_closing
    wavefront_eta_hours = np.clip(dist_m - radial_growth_m, 0, None) / effective_closing
    margin_m    = dist_m - radius_m - projected_advance

    # ---- Base distance features ----
    out["dist_km"]        = dist_km
    out["log_distance"]   = np.log1p(dist_km)
    out["inv_distance"]   = 1.0 / (dist_km + 0.1)
    out["sqrt_distance"]  = np.sqrt(dist_km)
    out["dist_km_sq"]     = dist_km ** 2
    out["dist_km_cb"]     = dist_km ** 3
    out["dist_km_4"]      = dist_km ** 4   # NEW

    # ---- Area features ----
    out["area_km2"]          = area_km2
    out["log_area_ha"]       = np.log1p(area_ha)
    out["log_area_km2"]      = np.log1p(area_km2)
    out["fire_radius_km"]    = radius_m / 1000.0
    out["radius_to_dist"]    = (radius_m / 1000.0) / (dist_km + 1e-3)
    out["area_to_dist_ratio"] = area_ha / (dist_km + 0.1)

    # ---- Growth features ----
    out["area_growth_abs_filled"]          = area_growth_abs
    out["area_growth_rate_ha_per_h_filled"] = np.clip(area_growth_rate, 0, None)
    out["log_growth_abs"]   = np.log1p(np.abs(area_growth_abs))
    out["log_growth_rate"]  = np.log1p(np.clip(area_growth_rate, 0, None))
    out["rel_growth_proxy"] = area_growth_abs / (area_ha + 1.0)
    # NEW: growth acceleration proxy
    out["growth_x_closing"] = np.clip(area_growth_rate, 0, None) * effective_closing / 1000.0

    # ---- Speed / ETA features ----
    out["effective_closing_speed_m_per_h"]  = effective_closing
    out["effective_closing_speed_km_per_h"] = effective_closing / 1000.0
    out["eta_hours"]          = eta_hours
    out["log_eta"]            = np.log1p(eta_hours)
    out["inv_eta"]            = 1.0 / (eta_hours + 0.1)
    out["sqrt_eta"]           = np.sqrt(np.clip(eta_hours, 0, None))   # NEW
    out["eta_sq"]             = eta_hours ** 2                          # NEW
    out["wavefront_eta_hours"] = wavefront_eta_hours

    # ---- Margin features ----
    out["margin_m"]         = margin_m
    out["margin_km"]        = margin_m / 1000.0
    out["margin_pos_m"]     = np.clip(margin_m, 0, None)
    out["log_margin_pos"]   = np.log1p(np.clip(margin_m, 0, None))
    out["margin_neg_flag"]  = (margin_m < 0).astype(float)   # NEW: already inside radius

    # ---- Alignment features ----
    out["alignment_abs_fe"]  = alignment_abs
    out["alignment_x_speed"] = alignment_abs * effective_closing

    # ---- Threat / urgency composites ----
    out["threat_score"] = (
        (alignment_abs + 0.2)
        * (effective_closing / 1000.0)
        * (1.0 + np.log1p(area_km2))
        / (dist_km + 0.1)
    )
    out["log_threat"]   = np.log1p(np.clip(out["threat_score"], 0, None))
    out["fire_urgency"] = perim_count * (effective_closing / 1000.0)
    out["log_urgency"]  = np.log1p(np.clip(out["fire_urgency"], 0, None))
    out["momentum"]     = (radius_m / 1000.0) * (effective_closing / 1000.0)
    out["gravity"]      = out["momentum"] / (dist_km + 0.1)
    out["log_gravity"]  = np.log1p(np.clip(out["gravity"], 0, None))

    # NEW composite features
    out["fire_power"]      = np.log1p(area_km2) * effective_closing / (dist_km + 0.1)
    out["impact_index"]    = alignment_abs * np.log1p(area_km2) / (eta_hours + 0.5)
    out["spread_ratio"]    = radial_rate / (effective_closing + 0.01)
    out["closing_ratio"]   = closing_abs / (effective_closing + 0.01)
    out["eta_dist_product"] = eta_hours * dist_km
    out["speed_per_area"]  = effective_closing / (area_ha + 1.0)
    out["centroid_ratio"]  = centroid_speed / (effective_closing + 0.01)

    # ---- Zone features ----
    out["zone_near"]    = (dist_m < CFG.near_dist_m).astype(int)
    out["zone_warning"] = ((dist_m >= CFG.near_dist_m) & (dist_m < CFG.far_dist_m)).astype(int)
    out["zone_far"]     = (dist_m >= CFG.far_dist_m).astype(int)
    out["zone_very_far"] = (dist_m >= CFG.very_far_dist_m).astype(int)   # NEW
    out["dist_to_gate_km"]     = dist_km - 5.0
    out["dist_to_gate_pos_km"] = np.clip(dist_km - 5.0, 0, None)
    out["gate_proximity"]      = 1.0 / (np.abs(dist_km - 5.0) + 0.5)

    # ---- Environmental features (if present) ----
    out["wind_speed"]       = wind_speed
    out["log_wind"]         = np.log1p(wind_speed)
    out["inv_humidity"]     = 1.0 / rh
    out["fire_weather_idx"] = wind_speed * (1.0 / rh) * np.clip(temp_c - 10, 0, None)
    out["slope_effect"]     = np.sin(np.radians(np.clip(slope_deg, 0, 90)))
    out["fuel_dryness"]     = 1.0 / fuel_moisture
    out["fuel_x_wind"]      = (1.0 / fuel_moisture) * wind_speed
    out["env_threat"]       = wind_speed * (1.0 / rh) * canopy_pct / 100.0

    # ---- Temporal features ----
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["hour_sin"]  = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"]  = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"]   = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"]   = np.cos(2 * np.pi * dow / 7.0)
    # NEW: season flag (fire season = months 6-10)
    out["fire_season"] = ((month >= 6) & (month <= 10)).astype(float)

    return out


def create_features(df, fit_df=None, dist_col="dist_min_ci_0_5h"):
    if fit_df is None:
        fit_df = df

    out      = compute_base_features(df, dist_col)
    fit_base = compute_base_features(fit_df, dist_col)

    fit_dist = pd.to_numeric(fit_df[dist_col], errors="coerce").fillna(0.0).values
    fit_near = fit_dist < CFG.near_dist_m
    fit_far  = fit_dist >= CFG.far_dist_m

    # Near-zone percentile ranks
    out["near_speed_rank"] = 0.0
    out["near_eta_rank"]   = 0.0
    out["near_threat_rank"] = 0.0   # NEW
    if fit_near.sum() >= 5:
        ref_speed  = fit_base.loc[fit_near, "effective_closing_speed_m_per_h"].values
        ref_eta    = fit_base.loc[fit_near, "eta_hours"].values
        ref_threat = fit_base.loc[fit_near, "threat_score"].values
        idx = out["zone_near"].values.astype(bool)
        out.loc[idx, "near_speed_rank"]  = pct_rank(out.loc[idx, "effective_closing_speed_m_per_h"].values, ref_speed)
        out.loc[idx, "near_eta_rank"]    = pct_rank(out.loc[idx, "eta_hours"].values, ref_eta)
        out.loc[idx, "near_threat_rank"] = pct_rank(out.loc[idx, "threat_score"].values, ref_threat)

    # Far-zone percentile ranks
    out["far_threat_rank"] = 0.0
    out["far_dist_rank"]   = 0.0
    out["far_speed_rank"]  = 0.0   # NEW
    if fit_far.sum() >= 5:
        ref_threat = fit_base.loc[fit_far, "threat_score"].values
        ref_dist   = fit_base.loc[fit_far, dist_col].values
        ref_speed  = fit_base.loc[fit_far, "effective_closing_speed_m_per_h"].values
        idx = out["zone_far"].values.astype(bool)
        out.loc[idx, "far_threat_rank"] = pct_rank(out.loc[idx, "threat_score"].values, ref_threat)
        out.loc[idx, "far_dist_rank"]   = pct_rank(out.loc[idx, dist_col].values, ref_dist)
        out.loc[idx, "far_speed_rank"]  = pct_rank(out.loc[idx, "effective_closing_speed_m_per_h"].values, ref_speed)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ============================================================
# METRICS
# ============================================================
def pava_monotone(p: np.ndarray) -> np.ndarray:
    """Apply pool-adjacent violators to enforce monotone non-decreasing across horizons."""
    out = p.copy()
    n, H = out.shape
    for i in range(n):
        # Isotonic regression in ascending direction
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        out[i] = iso.fit_transform(np.arange(H), out[i])
    return out


def enforce_monotonicity(p: np.ndarray) -> np.ndarray:
    p = np.maximum.accumulate(p, axis=1)
    return np.clip(p, 1e-6, 1 - 1e-6)


def fit_censoring_km(time, event):
    censor_event = ~event.astype(bool)
    t, s = kaplan_meier_estimator(censor_event, time.astype(float))
    return np.asarray(t, float), np.asarray(s, float)


def G_step(km_t, km_s, x):
    idx = np.searchsorted(km_t, x, side="right") - 1
    idx = np.clip(idx, 0, len(km_s) - 1)
    return km_s[idx]


def make_binary_target(time, event, horizon):
    time  = np.asarray(time, float)
    event = np.asarray(event, bool)
    y    = np.zeros(len(time), dtype=int)
    mask = np.ones(len(time), dtype=bool)
    y[(event) & (time <= horizon)] = 1
    mask[(~event) & (time <= horizon)] = False
    return y, mask


def compute_ipcw_weights(time, event, horizon, clip=30.0):
    y, mask = make_binary_target(time, event, horizon)
    km_t, km_s = fit_censoring_km(np.asarray(time, float), np.asarray(event, bool))
    w = np.zeros(len(time), float)
    idx1 = mask & (y == 1)
    idx0 = mask & (y == 0)
    if idx1.sum():
        w[idx1] = 1.0 / np.clip(G_step(km_t, km_s, np.asarray(time)[idx1]), 1e-6, 1.0)
    if idx0.sum():
        w[idx0] = 1.0 / np.clip(G_step(km_t, km_s, np.full(idx0.sum(), horizon)), 1e-6, 1.0)
    return np.clip(w, 0, clip), mask


def compute_brier(time, event, prob, horizon, clip=30.0):
    y, mask  = make_binary_target(time, event, horizon)
    w, mask2 = compute_ipcw_weights(time, event, horizon, clip=clip)
    m = mask & mask2
    if m.sum() == 0:
        return np.nan
    return float(np.sum(w[m] * (y[m] - prob[m]) ** 2) / np.sum(w[m]))


def compute_c_index(time, event, risk):
    return float(concordance_index_censored(
        np.asarray(event, bool), np.asarray(time, float), np.asarray(risk, float))[0])


def compute_hybrid_score(time, event, p24, p48, p72, clip=30.0):
    b24 = compute_brier(time, event, p24, 24, clip)
    b48 = compute_brier(time, event, p48, 48, clip)
    b72 = compute_brier(time, event, p72, 72, clip)
    wb  = 0.3 * b24 + 0.4 * b48 + 0.3 * b72
    risk = 0.3 * p24 + 0.4 * p48 + 0.3 * p72
    c   = compute_c_index(time, event, risk)
    hybrid = 0.3 * c + 0.7 * (1.0 - wb)
    return {"c_index": float(c), "weighted_brier": float(wb), "hybrid": float(hybrid),
            "b24": float(b24), "b48": float(b48), "b72": float(b72)}


def get_surv_predictions(model, X, horizons):
    surv = model.predict_survival_function(X, return_array=True)
    unique_times = np.asarray(model.unique_times_, dtype=float)
    out = np.zeros((X.shape[0], len(horizons)), float)
    for j, h in enumerate(horizons):
        idx = int(np.clip(np.searchsorted(unique_times, h, side="right") - 1, 0, len(unique_times) - 1))
        out[:, j] = 1.0 - surv[:, idx]
    return np.clip(out, 1e-6, 1 - 1e-6)


# ============================================================
# CALIBRATORS  (two-stage: logistic + isotonic)
# ============================================================
class TwoStageCalibrator:
    """Stage 1: L2-logistic. Stage 2: isotonic on logit residuals."""
    def __init__(self, seed=0):
        self.seed = seed
        self.stage1 = None
        self.stage2 = None
        self._p_const = None

    def fit(self, X, y, w=None):
        y = np.asarray(y)
        if len(y) == 0 or len(np.unique(y)) < 2:
            self._p_const = float(np.average(y, weights=w) if w is not None and len(y) else 0.0)
            return self
        try:
            self.stage1 = LogisticRegression(C=0.25, penalty="l2", solver="lbfgs",
                                              max_iter=4000, random_state=self.seed)
            self.stage1.fit(X, y, sample_weight=w)
            p1 = self.stage1.predict_proba(X)[:, 1]
            # Stage 2: fit isotonic to map p1 → true probabilities
            self.stage2 = IsotonicRegression(increasing=True, out_of_bounds="clip")
            self.stage2.fit(p1, y, sample_weight=w)
        except Exception:
            self._p_const = float(y.mean())
        return self

    def predict(self, X):
        if self._p_const is not None:
            return np.full(len(X), self._p_const)
        try:
            p1 = self.stage1.predict_proba(X)[:, 1]
            return np.clip(self.stage2.predict(p1), 1e-6, 1 - 1e-6)
        except Exception:
            return np.full(len(X), 0.5)


# ============================================================
# TIMING MODEL
# ============================================================
def fit_timing_model(tr_eng, tr, dist_col, feature_cols, seed):
    if not (HAS_SCIPY and HAS_LGB):
        return None, np.nan
    time  = tr["time_to_hit_hours"].values.astype(float)
    event = tr["event"].values.astype(bool)
    near  = tr[dist_col].values.astype(float) < CFG.near_dist_m
    idx   = np.where(near & event)[0]
    if len(idx) < 10:
        return None, np.nan
    X  = tr_eng.iloc[idx][feature_cols].values.astype(float)
    imp = SimpleImputer(strategy="median")
    X  = imp.fit_transform(X)
    y  = np.log(np.clip(time[idx], 1e-6, None))
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=4, num_leaves=20,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
        min_child_samples=8, random_state=seed, verbose=-1,
    )
    model.fit(X, y)
    pred  = model.predict(X)
    sigma = float(np.std(y - pred))
    # Fit a full-data imputer for transform at inference
    imp_full = SimpleImputer(strategy="median")
    imp_full.fit(tr_eng[feature_cols].values.astype(float))
    return (model, imp_full), max(sigma, 0.1)


def predict_timing_probs(model_pack, sigma, df_eng, dist_m, feature_cols, horizons):
    out = np.zeros((len(df_eng), len(horizons)), float)
    if model_pack is None or not np.isfinite(sigma) or sigma <= 1e-6 or not HAS_SCIPY:
        return out
    model, imp = model_pack
    near = dist_m < CFG.near_dist_m
    idx  = np.where(near)[0]
    if len(idx) == 0:
        return out
    X  = imp.transform(df_eng.iloc[idx][feature_cols].values.astype(float))
    mu = model.predict(X)
    for j, h in enumerate(horizons):
        z = (np.log(max(h, 1e-6)) - mu) / sigma
        out[idx, j] = norm.cdf(z)
    return np.clip(out, 1e-6, 1 - 1e-6)


# ============================================================
# META-STACKER  (Level-1 LightGBM per horizon)
# ============================================================
def build_meta_features(gbsa, cox, rsf, calib, timing, aft, dist_m, horizons):
    """Stack model predictions + dist features as meta inputs."""
    H   = len(horizons)
    n   = gbsa.shape[0]
    # Flat: for each horizon, each model's prob + dist
    rows = []
    for j in range(H):
        rows.append(gbsa[:, j])
        rows.append(cox[:, j])
        rows.append(rsf[:, j])
        rows.append(calib[:, j])
        rows.append(timing[:, j])
        rows.append(aft[:, j])
    rows.append(np.log1p(dist_m / 1000.0))
    rows.append((dist_m < CFG.near_dist_m).astype(float))
    rows.append((dist_m >= CFG.far_dist_m).astype(float))
    rows.append(dist_m / 1000.0)                # raw distance
    rows.append((dist_m / 1000.0) ** 2)        # nonlinear
    rows.append(1 / (dist_m / 1000.0 + 0.1))   # inverse distance
    return np.column_stack(rows)   # (n, H*6 + 3)


def fit_meta_stacker(X_meta, y_binary, w_ipcw, seed):
    """One LGB per horizon trained on OOF meta-features."""
    if not HAS_LGB:
        return None
    from lightgbm import LGBMClassifier
    m = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        min_child_samples=10,
        random_state=seed,
        verbose=-1,
    )
    try:
        m.fit(X_meta, y_binary, sample_weight=w_ipcw)
        return m
    except Exception:
        return None


# ============================================================
# BLEND
# ============================================================
def blend_zone(gbsa, cox, rsf, calib, timing,
               dist_m, near_w, far_w,
               horizons, force_p12_far_zero=True,
               force_p72_one=True, enforce_monotone=True):
    out  = np.zeros_like(gbsa)
    near = dist_m < CFG.near_dist_m
    far  = ~near

    wgN, wcN, wrN, wlN, wtN = _norm5(*near_w)
    wgF, wcF, wrF, wlF, wtF = _norm5(*far_w)

    out[near] = wgN*gbsa[near] + wcN*cox[near] + wrN*rsf[near] + wlN*calib[near] + wtN*timing[near]
    out[far]  = wgF*gbsa[far]  + wcF*cox[far]  + wrF*rsf[far]  + wlF*calib[far]  + wtF*timing[far]

    if force_p12_far_zero and 12 in horizons:
        very_far = dist_m >= CFG.very_far_dist_m   # stricter rule
        out[very_far, horizons.index(12)] = 0.0

    if force_p72_one and 72 in horizons:
        out[:, horizons.index(72)] = 1.0

    out = np.clip(out, 1e-6, 1 - 1e-6)
    if enforce_monotone:
        out = enforce_monotonicity(out)
    return out


def _norm5(a, b, c, d, e):
    s = a + b + c + d + e
    if s <= 0:
        return 1, 0, 0, 0, 0
    return a/s, b/s, c/s, d/s, e/s


# ============================================================
# MAIN CV PIPELINE
# ============================================================
def run_cv_pipeline(cfg: Config):
    seed_everything(cfg.seed)
    train, test, id_col, time_col, event_col, dist_col = load_data(cfg)
    horizons = list(cfg.horizons)
    H = len(horizons)
    n_tr = len(train)
    n_te = len(test)

    # Accumulator arrays
    def zeros(*shape): return np.zeros(shape, float)

    gbsa_oof   = zeros(n_tr, H); gbsa_test   = zeros(n_te, H)
    cox_oof    = zeros(n_tr, H); cox_test    = zeros(n_te, H)
    rsf_oof    = zeros(n_tr, H); rsf_test    = zeros(n_te, H)
    aft_oof    = zeros(n_tr, H); aft_test    = zeros(n_te, H)
    calib_oof  = zeros(n_tr, H); calib_test  = zeros(n_te, H)
    timing_oof = zeros(n_tr, H); timing_test = zeros(n_te, H)

    raw_cols = [c for c in train.columns
                if c not in [id_col, time_col, event_col]
                and pd.api.types.is_numeric_dtype(train[c])]

    # ---- Feature sets ----
    COX_FEATURES = [
        "dist_km", "log_distance", "inv_distance", "sqrt_distance", "dist_km_sq", "dist_km_4",
        "fire_radius_km", "radius_to_dist", "area_to_dist_ratio",
        "eta_hours", "sqrt_eta", "wavefront_eta_hours", "effective_closing_speed_m_per_h",
        "threat_score", "fire_urgency", "alignment_abs_fe", "impact_index",
        "fire_power", "spread_ratio", "closing_ratio",
        "zone_near", "zone_far", "zone_very_far",
        "near_speed_rank", "far_threat_rank",
        "month_sin", "fire_season",
    ]

    TIMING_FEATURES = [
        "eta_hours", "sqrt_eta", "wavefront_eta_hours", "log_eta",
        "margin_m", "log_margin_pos", "margin_neg_flag",
        "effective_closing_speed_m_per_h", "alignment_abs_fe",
        "near_speed_rank", "near_eta_rank", "near_threat_rank",
        "gate_proximity", "threat_score", "log_threat",
        "fire_urgency", "impact_index",
        "dist_km", "log_distance",
    ]

    CALIB_FEATURES_NEAR = [
        "eta_hours", "sqrt_eta", "wavefront_eta_hours", "log_eta",
        "margin_m", "log_margin_pos", "margin_neg_flag",
        "effective_closing_speed_m_per_h",
        "near_speed_rank", "near_eta_rank", "near_threat_rank",
        "alignment_abs_fe", "gate_proximity",
        "log_growth_rate", "growth_x_closing",
        "threat_score", "impact_index", "dist_km",
    ]

    CALIB_FEATURES_FAR = [
        "dist_km", "log_distance", "inv_distance", "dist_km_sq",
        "log_area_ha", "log_growth_rate",
        "threat_score", "log_threat",
        "far_threat_rank", "far_dist_rank", "far_speed_rank",
        "zone_far", "zone_warning", "zone_very_far",
        "fire_power", "fire_urgency",
    ]

    # ---- Model configs (more diverse / larger) ----
    gbsa_cfgs = [
        dict(loss="coxph", learning_rate=0.008, n_estimators=1800, subsample=0.8,  max_depth=2, min_samples_leaf=8),
        dict(loss="coxph", learning_rate=0.006, n_estimators=2500, subsample=0.9,  max_depth=2, min_samples_leaf=6),
        dict(loss="coxph", learning_rate=0.012, n_estimators=1200, subsample=0.7,  max_depth=3, min_samples_leaf=10),
        dict(loss="coxph", learning_rate=0.020, n_estimators=900,  subsample=0.75, max_depth=3, min_samples_leaf=12),
        dict(loss="coxph", learning_rate=0.005, n_estimators=3000, subsample=0.85, max_depth=2, min_samples_leaf=5),
    ]
    rsf_cfgs = [
        dict(n_estimators=300, min_samples_leaf=10, max_features="sqrt"),
        dict(n_estimators=300, min_samples_leaf=14, max_features=0.5),
        dict(n_estimators=300, min_samples_leaf=18, max_features=0.6),
        dict(n_estimators=250, min_samples_leaf=8,  max_features="log2"),
    ]
    cox_alphas = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0]

    # ---- CV ----
    skf   = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    strat = make_strat_label(train, event_col, dist_col)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, strat)):
        tr = train.iloc[tr_idx].reset_index(drop=True)
        va = train.iloc[va_idx].reset_index(drop=True)

        tr_eng = create_features(tr, fit_df=tr,  dist_col=dist_col)
        va_eng = create_features(va, fit_df=tr,  dist_col=dist_col)
        te_eng = create_features(test, fit_df=tr, dist_col=dist_col)

        y_tr = Surv.from_arrays(event=tr[event_col].astype(bool).values,
                                time=tr[time_col].astype(float).values)

        imp_raw = SimpleImputer(strategy="median")
        X_tr_raw = imp_raw.fit_transform(tr[raw_cols].values.astype(float))
        X_va_raw = imp_raw.transform(va[raw_cols].values.astype(float))
        X_te_raw = imp_raw.transform(test[raw_cols].values.astype(float))

        # ---- GBSA ensemble ----
        p_va = np.zeros((len(va), H)); p_te = np.zeros((n_te, H)); cnt = 0
        seeds_gbsa = [1,2,3,4,5] if cfg.fast_mode else [1,2,3,4,5,6,7,8]
        for seed in seeds_gbsa:
            for mcfg in gbsa_cfgs:
                m = GradientBoostingSurvivalAnalysis(random_state=seed, **mcfg)
                m.fit(X_tr_raw, y_tr)
                p_va += get_surv_predictions(m, X_va_raw, horizons)
                p_te += get_surv_predictions(m, X_te_raw, horizons)
                cnt  += 1
        gbsa_oof[va_idx]  = p_va / cnt
        gbsa_test        += p_te / cnt / cfg.n_folds

        # ---- Cox ensemble (wider alpha grid + QuantileTransform) ----
        c_feats = [c for c in COX_FEATURES if c in tr_eng.columns]
        imp = SimpleImputer(strategy="median")
        qt  = QuantileTransformer(output_distribution="normal", random_state=cfg.seed)
        sc  = StandardScaler()
        Xc_tr = sc.fit_transform(qt.fit_transform(imp.fit_transform(tr_eng[c_feats].values.astype(float))))
        Xc_va = sc.transform(qt.transform(imp.transform(va_eng[c_feats].values.astype(float))))
        Xc_te = sc.transform(qt.transform(imp.transform(te_eng[c_feats].values.astype(float))))
        p_va = np.zeros((len(va), H)); p_te = np.zeros((n_te, H)); cnt = 0
        for alpha in cox_alphas:
            m = CoxPHSurvivalAnalysis(alpha=alpha, ties="breslow")
            m.fit(Xc_tr, y_tr)
            p_va += get_surv_predictions(m, Xc_va, horizons)
            p_te += get_surv_predictions(m, Xc_te, horizons)
            cnt  += 1
        cox_oof[va_idx]  = p_va / cnt
        cox_test        += p_te / cnt / cfg.n_folds

        # ---- RSF ensemble (more seeds) ----
        p_va = np.zeros((len(va), H)); p_te = np.zeros((n_te, H)); cnt = 0
        seeds_rsf = [21,22,23] if cfg.fast_mode else [21,22,23,24,25,26]
        for seed in seeds_rsf:
            for mcfg in rsf_cfgs:
                m = RandomSurvivalForest(random_state=seed, bootstrap=True, n_jobs=-1, **mcfg)
                m.fit(X_tr_raw, y_tr)
                p_va += get_surv_predictions(m, X_va_raw, horizons)
                p_te += get_surv_predictions(m, X_te_raw, horizons)
                cnt  += 1
        rsf_oof[va_idx]  = p_va / cnt
        rsf_test        += p_te / cnt / cfg.n_folds

        # ---- XGB-AFT (tuned sigma) ----
        if cfg.use_xgb_aft:
            time_tr = tr[time_col].values.astype(float)
            ev_tr   = tr[event_col].values.astype(bool)
            time_va = va[time_col].values.astype(float)
            ev_va   = va[event_col].values.astype(bool)

            dtr = xgb.DMatrix(X_tr_raw)
            dtr.set_float_info("label_lower_bound", time_tr)
            dtr.set_float_info("label_upper_bound", np.where(ev_tr, time_tr, np.inf))
            dva = xgb.DMatrix(X_va_raw)
            dva.set_float_info("label_lower_bound", time_va)
            dva.set_float_info("label_upper_bound", np.where(ev_va, time_va, np.inf))
            dte = xgb.DMatrix(X_te_raw)

            best_aft_val = np.inf
            best_aft_boost = None
            for aft_sigma in [1.0, 1.3, 1.6]:
                params = dict(
                    objective="survival:aft", eval_metric="aft-nloglik",
                    tree_method="hist", max_depth=4, min_child_weight=12,
                    subsample=0.8, colsample_bytree=0.85, eta=0.02,
                    lambda_=2.0, aft_loss_distribution="normal",
                    aft_loss_distribution_scale=aft_sigma,
                    seed=fold + 101,
                )
                booster = xgb.train(params, dtr,
                                    num_boost_round=600 if cfg.fast_mode else 1200,
                                    evals=[(dva, "va")], verbose_eval=False)
                val_score = booster.eval(dva)
                score_val = float(val_score.split(":")[1]) if ":" in val_score else 1e9
                if score_val < best_aft_val:
                    best_aft_val = score_val
                    best_aft_boost = (booster, aft_sigma)

            if best_aft_boost is not None:
                booster, sigma_aft = best_aft_boost
                mu_va = booster.predict(dva)
                mu_te = booster.predict(dte)
                for j, h in enumerate(horizons):
                    aft_oof[va_idx, j] = norm.cdf((np.log(h) - mu_va) / sigma_aft)
                    aft_test[:, j]    += norm.cdf((np.log(h) - mu_te) / sigma_aft) / cfg.n_folds

        # ---- Timing model ----
        if cfg.use_timing_model:
            t_feats = [c for c in TIMING_FEATURES if c in tr_eng.columns]
            timing_model_pack, sigma_t = fit_timing_model(tr_eng, tr, dist_col, t_feats, fold + 500)
            d_va = va[dist_col].values.astype(float)
            d_te = test[dist_col].values.astype(float)
            timing_oof[va_idx] = predict_timing_probs(timing_model_pack, sigma_t, va_eng, d_va, t_feats, horizons)
            timing_test += predict_timing_probs(timing_model_pack, sigma_t, te_eng, d_te, t_feats, horizons) / cfg.n_folds

        # ---- Two-stage calibrators (near + far, horizons 12/24/48) ----
        t_tr   = tr[time_col].values.astype(float)
        e_tr   = tr[event_col].values.astype(bool)
        d_tr   = tr[dist_col].values.astype(float)
        d_va   = va[dist_col].values.astype(float)
        d_te   = test[dist_col].values.astype(float)

        near_tr = d_tr < CFG.near_dist_m; far_tr = ~near_tr
        near_va = d_va < CFG.near_dist_m; far_va  = ~near_va
        near_te = d_te < CFG.near_dist_m; far_te  = ~near_te

        for horizon in [12, 24, 48]:
            if horizon not in horizons:
                continue
            jh = horizons.index(horizon)
            y_h, mask_h  = make_binary_target(t_tr, e_tr, horizon)
            w_h, mask_w  = compute_ipcw_weights(t_tr, e_tr, horizon, cfg.ipcw_clip)
            good = mask_h & mask_w

            for (zone_tr, zone_va, zone_te, cols_key) in [
                (near_tr, near_va, near_te, CALIB_FEATURES_NEAR),
                (far_tr,  far_va,  far_te,  CALIB_FEATURES_FAR),
            ]:
                cols = [c for c in cols_key if c in tr_eng.columns]
                Xtr  = tr_eng.loc[good & zone_tr, cols].values.astype(float)
                ytr  = y_h[good & zone_tr]
                wtr  = w_h[good & zone_tr]
                Xva  = va_eng.loc[zone_va, cols].values.astype(float)
                Xte  = te_eng.loc[zone_te, cols].values.astype(float)

                if len(Xva) == 0:
                    continue
                imp2 = SimpleImputer(strategy="median")
                sc2  = StandardScaler()
                Xtr2 = sc2.fit_transform(imp2.fit_transform(Xtr)) if len(Xtr) > 0 else np.empty((0, len(cols)))
                Xva2 = sc2.transform(imp2.transform(Xva))
                Xte2 = sc2.transform(imp2.transform(Xte)) if len(Xte) > 0 else np.empty((0, len(cols)))

                cal = TwoStageCalibrator(seed=fold + 3000 + horizon)
                cal.fit(Xtr2, ytr, wtr)
                calib_oof[va_idx[zone_va], jh] = cal.predict(Xva2)
                if len(Xte) > 0:
                    calib_test[zone_te, jh] += cal.predict(Xte2) / cfg.n_folds

        if cfg.verbose:
            print(f"  Fold {fold+1}/{cfg.n_folds} done.")

    return dict(
        train=train, test=test,
        id_col=id_col, time_col=time_col, event_col=event_col, dist_col=dist_col,
        gbsa_oof=gbsa_oof, cox_oof=cox_oof, rsf_oof=rsf_oof,
        aft_oof=aft_oof, calib_oof=calib_oof, timing_oof=timing_oof,
        gbsa_test=gbsa_test, cox_test=cox_test, rsf_test=rsf_test,
        aft_test=aft_test, calib_test=calib_test, timing_test=timing_test,
    )


# ============================================================
# META-STACKING  (Level-1, per-horizon LGB)
# ============================================================
def run_meta_stacking(pack, cfg: Config):
    """Train per-horizon LGB meta-learner on OOF base predictions."""
    if not (cfg.use_meta_stacking and HAS_LGB):
        return None, None

    train    = pack["train"]
    test     = pack["test"]
    horizons = list(cfg.horizons)
    H        = len(horizons)
    t        = train[pack["time_col"]].values.astype(float)
    e        = train[pack["event_col"]].values.astype(bool)
    d_tr     = train[pack["dist_col"]].values.astype(float)
    d_te     = test[pack["dist_col"]].values.astype(float)

    gbsa_oof  = pack["gbsa_oof"];  gbsa_test  = pack["gbsa_test"]
    cox_oof   = pack["cox_oof"];   cox_test   = pack["cox_test"]
    rsf_oof   = pack["rsf_oof"];   rsf_test   = pack["rsf_test"]
    aft_oof   = pack["aft_oof"];   aft_test   = pack["aft_test"]
    calib_oof = pack["calib_oof"]; calib_test = pack["calib_test"]
    timing_oof= pack["timing_oof"];timing_test= pack["timing_test"]

    X_meta_tr = build_meta_features(gbsa_oof, cox_oof, rsf_oof, calib_oof, timing_oof, aft_oof, d_tr, horizons)
    X_meta_te = build_meta_features(gbsa_test,cox_test,rsf_test,calib_test,timing_test,aft_test,d_te, horizons)

    meta_oof  = np.zeros((len(train), H), float)
    meta_test = np.zeros((len(test),  H), float)

    skf   = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed + 77)
    strat = make_strat_label(train, pack["event_col"], pack["dist_col"])

    for jh, horizon in enumerate(horizons):
        y_h, mask_h = make_binary_target(t, e, horizon)
        w_h, mask_w = compute_ipcw_weights(t, e, horizon, cfg.ipcw_clip)
        good = mask_h & mask_w

        meta_preds_test = np.zeros(len(test), float)
        fold_cnt = 0
        for fold, (tr_idx, va_idx) in enumerate(skf.split(train, strat)):
            good_tr = good[tr_idx]
            Xm_tr = X_meta_tr[tr_idx][good_tr]
            ym_tr = y_h[tr_idx][good_tr]
            wm_tr = w_h[tr_idx][good_tr]
            Xm_va = X_meta_tr[va_idx]

            if len(Xm_tr) == 0 or len(np.unique(ym_tr)) < 2:
                continue
            stacker = fit_meta_stacker(Xm_tr, ym_tr, wm_tr, cfg.seed + fold + jh * 100)
            if stacker is None:
                continue
            meta_oof[va_idx, jh] = stacker.predict_proba(Xm_va)[:, 1]
            meta_preds_test     += stacker.predict_proba(X_meta_te)[:, 1]
            fold_cnt            += 1

        if fold_cnt > 0:
            meta_test[:, jh] = meta_preds_test / fold_cnt

    return meta_oof, meta_test


# ============================================================
# WEIGHT SEARCH  (finer grid, both zones)
# ============================================================
def oof_weight_search(train, time_col, event_col, dist_col, horizons,
                      gbsa_oof, cox_oof, rsf_oof, calib_oof, timing_oof,
                      near_w_init, far_w_init, min_improve, clip):
    t = train[time_col].values.astype(float)
    e = train[event_col].values.astype(bool)
    d = train[dist_col].values.astype(float)

    def score(nw, fw):
        pred = blend_zone(gbsa_oof, cox_oof, rsf_oof, calib_oof, timing_oof, d,
                          nw, fw, horizons=horizons,
                          force_p12_far_zero=CFG.force_p12_far_zero,
                          force_p72_one=CFG.force_p72_one,
                          enforce_monotone=CFG.enforce_monotone)
        return compute_hybrid_score(t, e,
                                    pred[:, horizons.index(24)],
                                    pred[:, horizons.index(48)],
                                    pred[:, horizons.index(72)], clip)

    base_m   = score(near_w_init, far_w_init)
    best_nw, best_fw = near_w_init, far_w_init
    best_m   = base_m

    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # Search far zone (keep near zone fixed)
    for wg in grid:
        for wc in grid:
            for wr in grid:
                for wl in grid:
                    fw = (wg, wc, wr, wl, 0.0)
                    if sum(fw) <= 0:
                        continue
                    m = score(best_nw, fw)
                    if m["hybrid"] > best_m["hybrid"]:
                        best_m  = m
                        best_fw = fw

    # Search near zone (keep far zone at best)
    for wg in grid:
        for wc in grid:
            for wr in grid:
                for wl in grid:
                    for wt in [0.0, 0.05, 0.10, 0.15]:
                        nw = (wg, wc, wr, wl, wt)
                        if sum(nw) <= 0:
                            continue
                        m = score(nw, best_fw)
                        if m["hybrid"] > best_m["hybrid"]:
                            best_m  = m
                            best_nw = nw

    if best_m["hybrid"] - base_m["hybrid"] < min_improve:
        return near_w_init, far_w_init, base_m, best_m
    return best_nw, best_fw, base_m, best_m


# ============================================================
# ISOTONIC POST-PROCESSING
# ============================================================
def apply_isotonic_calibration(train, time_col, event_col, horizons, oof_pred):
    """Fit per-horizon isotonic regression on OOF preds to reduce bias."""
    t = train[time_col].values.astype(float)
    e = train[event_col].values.astype(bool)
    isos = {}
    for jh, h in enumerate(horizons):
        y_h, mask_h = make_binary_target(t, e, h)
        w_h, mask_w = compute_ipcw_weights(t, e, h)
        good = mask_h & mask_w
        if good.sum() < 10:
            isos[jh] = None
            continue
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(oof_pred[good, jh], y_h[good], sample_weight=w_h[good])
        isos[jh] = iso
    return isos


def transform_isotonic(pred, isos, horizons):
    out = pred.copy()
    for jh in range(len(horizons)):
        iso = isos.get(jh)
        if iso is not None:
            out[:, jh] = np.clip(iso.predict(pred[:, jh]), 1e-6, 1 - 1e-6)
    return out


# ============================================================
# TEST-TIME AUGMENTATION  (distance jitter)
# ============================================================
def tta_predictions(train, test, dist_col, pack, cfg, near_w, far_w):
    """Average predictions over small distance perturbations for robustness."""
    horizons = list(cfg.horizons)
    d_te = test[dist_col].values.astype(float)
    jitter_fracs = [-0.02, 0.0, +0.02]   # ±2% distance jitter
    preds = []
    for jf in jitter_fracs:
        d_jit = d_te * (1.0 + jf)
        p = blend_zone(
            pack["gbsa_test"], pack["cox_test"], pack["rsf_test"],
            pack["calib_test"], pack["timing_test"],
            d_jit, near_w, far_w,
            horizons=horizons,
            force_p12_far_zero=cfg.force_p12_far_zero,
            force_p72_one=cfg.force_p72_one,
            enforce_monotone=cfg.enforce_monotone,
        )
        preds.append(p)
    return np.mean(preds, axis=0)


# ============================================================
# SUBMISSION BUILDER
# ============================================================
def build_and_write_submission(cfg: Config) -> None:
    print("=" * 60)
    print("WiDS 2026 Wildfire v3 — targeting 0.98 hybrid score")
    print("=" * 60)

    pack = run_cv_pipeline(cfg)
    train    = pack["train"]
    test     = pack["test"]
    id_col   = pack["id_col"]
    time_col = pack["time_col"]
    event_col= pack["event_col"]
    dist_col = pack["dist_col"]
    horizons = list(cfg.horizons)
    raw_cols = [c for c in train.columns
                if c not in [id_col, time_col, event_col]
                and pd.api.types.is_numeric_dtype(train[c])]


    # Blend AFT into GBSA
    gbsa_oof  = pack["gbsa_oof"];  gbsa_test  = pack["gbsa_test"]
    if cfg.use_xgb_aft:
        gbsa_oof  = np.clip(0.82 * gbsa_oof  + 0.18 * pack["aft_oof"],  1e-6, 1-1e-6)
        gbsa_test = np.clip(0.82 * gbsa_test + 0.18 * pack["aft_test"], 1e-6, 1-1e-6)
        pack["gbsa_oof"]  = gbsa_oof
        pack["gbsa_test"] = gbsa_test

    dist_tr = train[dist_col].values.astype(float)
    dist_te = test[dist_col].values.astype(float)

    # Default weights (stronger GBSA near, stronger Cox far)
    near_w_init = (0.60, 0.15, 0.10, 0.10, cfg.timing_weight_default)
    far_w_init  = (0.15, 0.50, 0.12, 0.23, 0.0)

    # OOF weight search
    best_near_w, best_far_w = near_w_init, far_w_init
    if cfg.do_oof_weight_search and all(h in horizons for h in [24, 48, 72]):
        best_near_w, best_far_w, base_m, best_m = oof_weight_search(
            train=train, time_col=time_col, event_col=event_col,
            dist_col=dist_col, horizons=horizons,
            gbsa_oof=gbsa_oof, cox_oof=pack["cox_oof"],
            rsf_oof=pack["rsf_oof"], calib_oof=pack["calib_oof"],
            timing_oof=pack["timing_oof"],
            near_w_init=near_w_init, far_w_init=far_w_init,
            min_improve=cfg.oof_min_improve, clip=cfg.ipcw_clip,
        )
        if cfg.verbose:
            print("OOF weight search base:", base_m)
            print("OOF weight search best:", best_m)
            print("Near weights:", best_near_w)
            print("Far  weights:", best_far_w)

    # Level-1 meta-stacking
    meta_oof, meta_test = run_meta_stacking(pack, cfg)

    # Build OOF blend
    oof_pred = blend_zone(
        gbsa_oof, pack["cox_oof"], pack["rsf_oof"],
        pack["calib_oof"], pack["timing_oof"],
        dist_tr, best_near_w, best_far_w, horizons=horizons,
        force_p12_far_zero=cfg.force_p12_far_zero,
        force_p72_one=cfg.force_p72_one,
        enforce_monotone=cfg.enforce_monotone,
    )

    base_oof = oof_pred.copy()
    t = train[time_col].values.astype(float)
    e = train[event_col].values.astype(bool)

# Only run diagnostics if meta exists
    if meta_oof is not None:

        print("\n===== META DIAGNOSTICS =====")

    # Basic stats
        print("Base OOF range:", base_oof.min(), base_oof.max())
        print("Meta OOF range:", meta_oof.min(), meta_oof.max())

    # Difference magnitude
        diff = np.abs(base_oof - meta_oof)
        print("Mean abs diff:", diff.mean())
        print("Max abs diff:", diff.max())

    # Correlation per horizon
        for j, h in enumerate(horizons):
           corr = np.corrcoef(base_oof[:, j], meta_oof[:, j])[0, 1]
           print(f"h={h}h correlation base vs meta: {corr:.4f}")

        # Compute metrics (THIS WAS MISSING)
        base_metrics = compute_hybrid_score(
            t, e,
            base_oof[:, 0],
            base_oof[:, 1],
            base_oof[:, 2],
            base_oof[:, 3],
        )

        meta_metrics = compute_hybrid_score(
            t, e,
            meta_oof[:, 0],
            meta_oof[:, 1],
            meta_oof[:, 2],
            meta_oof[:, 3],
        )

        print("Base metrics:", base_metrics)
        print("Meta metrics:", meta_metrics)

    # Tune weight
        print("\n===== META WEIGHT SEARCH =====")
        best_meta_w = tune_meta_weight(base_oof, meta_oof, t, e, horizons)

    # Final blend
        oof_pred = np.clip(
            (1 - best_meta_w) * base_oof + best_meta_w * meta_oof,
            1e-6, 1 - 1e-6
        )

    else:
      best_meta_w = 0.0
    # Isotonic calibration (fit on OOF, apply to test)
    isos = None
    cfg.use_isotonic = False
    if cfg.use_isotonic:
        isos = apply_isotonic_calibration(train, time_col, event_col, horizons, oof_pred)
        best_temp = find_best_temp(oof_pred, train, time_col, event_col, horizons, isos)
        oof_pred = smooth_probs(oof_pred, temp=best_temp)
        oof_pred = transform_isotonic(oof_pred, isos, horizons)
        oof_pred = smooth_horizons(oof_pred)

    # Re-enforce monotonicity + PAVA after isotonic
    oof_pred = pava_monotone(np.clip(oof_pred, 1e-6, 1-1e-6))

    # Final OOF score
    m = compute_hybrid_score(
        train[time_col].values.astype(float),
        train[event_col].values.astype(bool),
        oof_pred[:, horizons.index(24)],
        oof_pred[:, horizons.index(48)],
        oof_pred[:, horizons.index(72)],
        cfg.ipcw_clip,
    )
    print("\n" + "="*40)
    print("FINAL OOF hybrid score:", m)
    print("="*40 + "\n")

    # ---- Test predictions ----
    if cfg.use_tta:
        test_pred = tta_predictions(train, test, dist_col, pack, cfg, best_near_w, best_far_w)
    else:
        test_pred = blend_zone(
            pack["gbsa_test"], pack["cox_test"], pack["rsf_test"],
            pack["calib_test"], pack["timing_test"],
            dist_te, best_near_w, best_far_w, horizons=horizons,
            force_p12_far_zero=cfg.force_p12_far_zero,
            force_p72_one=cfg.force_p72_one,
            enforce_monotone=cfg.enforce_monotone,
        )



    test_pred = np.clip(
        (1 - best_meta_w) * test_pred +
        best_meta_w * meta_test,
        1e-6, 1 - 1e-6
    )

    if isos is not None:
        test_pred = smooth_probs(test_pred, temp=best_temp)
        test_pred = transform_isotonic(test_pred, isos, horizons)
        test_pred = smooth_horizons(test_pred)

    test_pred = pava_monotone(np.clip(test_pred, 1e-6, 1-1e-6))



    # =========================
    # HORIZON SMOOTHING
    # =========================
    test_pred[:, horizons.index(24)] = (
        0.90 * test_pred[:, horizons.index(24)] +
        0.10 * test_pred[:, horizons.index(12)]
    )

    test_pred[:, horizons.index(48)] = (
        0.85 * test_pred[:, horizons.index(48)] +
        0.15 * test_pred[:, horizons.index(24)]
    )

    

    # =========================
    # SOFT MONOTONICITY
    # =========================
    test_pred = np.maximum.accumulate(test_pred, axis=1)
    test_pred = np.clip(test_pred, 1e-5, 1 - 1e-5)


    
        
    # =========================
    # JITTER
    # =========================
    np.random.seed(42)
    test_pred += np.random.normal(0, 0.002, test_pred.shape)
    test_pred = np.clip(test_pred, 1e-5, 1 - 1e-5)

    



    # =========================
    # HORIZON TEMPERATURE SCALING (FINAL BOOST)
    # =========================

    def temp_scale(p, temp):
        logit = np.log(p / (1 - p))
        logit = logit / temp
        return 1 / (1 + np.exp(-logit))

    # apply per horizon
    test_pred[:, horizons.index(24)] = temp_scale(
       test_pred[:, horizons.index(24)], 0.95
    )

    test_pred[:, horizons.index(48)] = temp_scale(
        test_pred[:, horizons.index(48)], 1.00
    )

    test_pred[:, horizons.index(72)] = temp_scale(
       test_pred[:, horizons.index(72)], 1.05
    )

    test_pred = np.clip(test_pred, 1e-5, 1 - 1e-5)


    # =========================
    # FAR RULE (after jitter so zeros survive)
    # =========================
    if cfg.force_p12_far_zero and 12 in horizons:
        very_far = dist_te >= cfg.very_far_dist_m
        test_pred[very_far, horizons.index(12)] = 0.0


    t_arr = train[time_col].values.astype(float)
    e_arr = train[event_col].values.astype(bool)
    y72, mask72 = make_binary_target(t_arr, e_arr, 72)
    w72, mask_w  = compute_ipcw_weights(t_arr, e_arr, 72)
    good = mask72 & mask_w

    imp_raw = SimpleImputer(strategy="median")
    X_all = imp_raw.fit_transform(train[raw_cols].values.astype(float))

    probe = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                            num_leaves=31, random_state=42, verbose=-1)
    probe.fit(X_all[good], y72[good], sample_weight=w72[good])

    importance = pd.Series(probe.feature_importances_, index=raw_cols)
    print("\nTop 20 features:")
    print(importance.nlargest(20))
    print("\nBottom 10 features (possibly noisy):")
    print(importance.nsmallest(10))




    # ---- Write submission ----
    sub = pd.DataFrame({id_col: test[id_col].values})
    for j, h in enumerate(horizons):
        sub[f"prob_{h}h"] = test_pred[:, j].astype(float)

    print("Submission sample:")
    print(sub.head())
    sub.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


def main():
    build_and_write_submission(CFG)

#final

if __name__ == "__main__":
    main()


