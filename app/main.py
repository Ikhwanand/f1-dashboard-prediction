from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import streamlit as st
from joblib import load

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
MODELS_DIR = REPO_ROOT / "models"
METADATA_PATH = MODELS_DIR / "f1_models_metadata.json"

MODEL_TITLES = {
    "position_model": "Finishing Position (SVR)",
    "points_model": "Scored Points (Extra Trees)",
}
MODEL_OUTPUT_COLUMNS = {
    "position_model": "predicted_finish_position",
    "points_model": "predicted_points",
}
MODEL_DISPLAY_ORDER: Sequence[str] = tuple(MODEL_TITLES.keys())

FEATURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "SpeedI1_mean": {
        "label": "Speed I1 mean (km/h)",
        "min": 250.0,
        "max": 360.0,
        "value": 305.0,
        "step": 0.5,
        "dtype": "float",
        "format": "%.1f",
        "help": "Average speed in sector 1.",
    },
    "SpeedI1_max": {
        "label": "Speed I1 max (km/h)",
        "min": 260.0,
        "max": 370.0,
        "value": 320.0,
        "step": 0.5,
        "dtype": "float",
        "format": "%.1f",
        "help": "Peak speed reached in sector 1.",
    },
    "SpeedI2_mean": {
        "label": "Speed I2 mean (km/h)",
        "min": 250.0,
        "max": 360.0,
        "value": 300.0,
        "step": 0.5,
        "dtype": "float",
        "format": "%.1f",
        "help": "Average speed in sector 2.",
    },
    "SpeedI2_max": {
        "label": "Speed I2 max (km/h)",
        "min": 260.0,
        "max": 370.0,
        "value": 318.0,
        "step": 0.5,
        "dtype": "float",
        "format": "%.1f",
        "help": "Peak speed reached in sector 2.",
    },
    "SpeedFL_mean": {
        "label": "Speed FL mean (km/h)",
        "min": 260.0,
        "max": 370.0,
        "value": 315.0,
        "step": 0.5,
        "dtype": "float",
        "format": "%.1f",
        "help": "Average speed on the fastest lap.",
    },
    "SpeedFL_max": {
        "label": "Speed FL max (km/h)",
        "min": 270.0,
        "max": 380.0,
        "value": 330.0,
        "step": 0.5,
        "dtype": "float",
        "format": "%.1f",
        "help": "Maximum speed on the fastest lap.",
    },
    "SpeedST_mean": {
        "label": "Speed straight mean (km/h)",
        "min": 260.0,
        "max": 370.0,
        "value": 320.0,
        "step": 0.5,
        "dtype": "float",
        "format": "%.1f",
        "help": "Average top speed on the main straight.",
    },
    "SpeedST_max": {
        "label": "Speed straight max (km/h)",
        "min": 270.0,
        "max": 380.0,
        "value": 336.0,
        "step": 0.5,
        "dtype": "float",
        "format": "%.1f",
        "help": "Maximum speed on the main straight.",
    },
    "Position_mean": {
        "label": "Average race position",
        "min": 1,
        "max": 20,
        "value": 8,
        "step": 1,
        "dtype": "int",
        "help": "Mean position across the race.",
    },
    "Position_std": {
        "label": "Position std dev",
        "min": 0.0,
        "max": 8.0,
        "value": 2.0,
        "step": 0.1,
        "dtype": "float",
        "format": "%.2f",
        "help": "Standard deviation of race position.",
    },
    "Position_min": {
        "label": "Best position reached",
        "min": 1,
        "max": 20,
        "value": 4,
        "step": 1,
        "dtype": "int",
        "help": "Lowest (best) position achieved.",
    },
    "Position_max": {
        "label": "Worst position reached",
        "min": 1,
        "max": 20,
        "value": 12,
        "step": 1,
        "dtype": "int",
        "help": "Highest (worst) position reached.",
    },
    "lap_time_cv": {
        "label": "Lap-time coefficient of variation",
        "min": 0.0,
        "max": 0.15,
        "value": 0.04,
        "step": 0.005,
        "dtype": "float",
        "format": "%.3f",
        "help": "Lap-time variation (lower is more consistent).",
    },
    "position_changes": {
        "label": "Net position changes",
        "min": -15,
        "max": 15,
        "value": 1,
        "step": 1,
        "dtype": "int",
        "help": "Total gain/loss of places during the race.",
    },
    "fastest_lap_percentage": {
        "label": "Fastest-lap percentile %",
        "min": 0.0,
        "max": 100.0,
        "value": 12.0,
        "step": 1.0,
        "dtype": "float",
        "format": "%.0f",
        "help": "Share of laps within 105% of the absolute fastest lap.",
    },
    "top_10_laps_percentage": {
        "label": "Top-10 laps %",
        "min": 0.0,
        "max": 100.0,
        "value": 45.0,
        "step": 1.0,
        "dtype": "float",
        "format": "%.0f",
        "help": "Percentage of laps that ranked in the top ten.",
    },
    "TyreLife_mean": {
        "label": "Tyre life mean (laps)",
        "min": 0.0,
        "max": 40.0,
        "value": 18.0,
        "step": 0.5,
        "dtype": "float",
        "format": "%.1f",
        "help": "Average stint length in laps.",
    },
    "TyreLife_max": {
        "label": "Tyre life max (laps)",
        "min": 0.0,
        "max": 60.0,
        "value": 30.0,
        "step": 0.5,
        "dtype": "float",
        "format": "%.1f",
        "help": "Longest stint completed.",
    },
    "pit_stops": {
        "label": "Pit stops",
        "min": 0,
        "max": 6,
        "value": 2,
        "step": 1,
        "dtype": "int",
        "help": "Total pit stops made.",
    },
    "position_volatility": {
        "label": "Position volatility",
        "min": 0.0,
        "max": 8.0,
        "value": 1.5,
        "step": 0.1,
        "dtype": "float",
        "format": "%.2f",
        "help": "Rolling std of position changes.",
    },
    "SpeedI1_mean_advantage": {
        "label": "Sector 1 mean Δ (km/h)",
        "min": -15.0,
        "max": 15.0,
        "value": 0.5,
        "step": 0.1,
        "dtype": "float",
        "format": "%.1f",
        "help": "Mean sector-1 speed advantage vs field.",
    },
    "SpeedI2_mean_advantage": {
        "label": "Sector 2 mean Δ (km/h)",
        "min": -15.0,
        "max": 15.0,
        "value": 0.3,
        "step": 0.1,
        "dtype": "float",
        "format": "%.1f",
        "help": "Mean sector-2 speed advantage vs field.",
    },
    "SpeedFL_mean_advantage": {
        "label": "Fastest lap mean Δ (km/h)",
        "min": -15.0,
        "max": 15.0,
        "value": 1.0,
        "step": 0.1,
        "dtype": "float",
        "format": "%.1f",
        "help": "Mean fastest-lap speed advantage.",
    },
    "SpeedST_mean_advantage": {
        "label": "Straight-line mean Δ (km/h)",
        "min": -15.0,
        "max": 15.0,
        "value": 0.8,
        "step": 0.1,
        "dtype": "float",
        "format": "%.1f",
        "help": "Mean straight-line speed advantage.",
    },
}

FEATURE_SECTIONS: Sequence[tuple[str, Sequence[str]]] = (
    (
        "Sector & Straight-Line Pace",
        (
            "SpeedI1_mean",
            "SpeedI1_max",
            "SpeedI2_mean",
            "SpeedI2_max",
            "SpeedFL_mean",
            "SpeedFL_max",
            "SpeedST_mean",
            "SpeedST_max",
        ),
    ),
    (
        "Race Consistency",
        (
            "Position_mean",
            "Position_std",
            "Position_min",
            "Position_max",
            "lap_time_cv",
            "position_changes",
            "position_volatility",
        ),
    ),
    (
        "Strategy & Tyre Management",
        (
            "TyreLife_mean",
            "TyreLife_max",
            "pit_stops",
            "fastest_lap_percentage",
            "top_10_laps_percentage",
        ),
    ),
    (
        "Relative Pace Advantage",
        (
            "SpeedI1_mean_advantage",
            "SpeedI2_mean_advantage",
            "SpeedFL_mean_advantage",
            "SpeedST_mean_advantage",
        ),
    ),
)

DEFAULT_FEATURE_VALUES = {key: cfg["value"] for key, cfg in FEATURE_CONFIG.items()}


def load_metadata() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def hydrate_models(metadata: Dict[str, Any]) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for key, info in metadata.items():
        if not key.endswith("_model"):
            continue
        model_filename = Path(info["filename"]).name
        model_path = MODELS_DIR / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")
        models[key] = load(model_path)
    return models


@st.cache_resource(show_spinner="Loading trained models…")
def get_assets() -> tuple[Dict[str, Any], Dict[str, Any]]:
    metadata = load_metadata()
    models = hydrate_models(metadata)
    return metadata, models


def coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feature, cfg in FEATURE_CONFIG.items():
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
        else:
            df[feature] = cfg["value"]
    df = df.fillna({col: DEFAULT_FEATURE_VALUES[col] for col in DEFAULT_FEATURE_VALUES})
    return df


def run_inference(
    models: Dict[str, Any],
    metadata: Dict[str, Any],
    df: pd.DataFrame,
    selected_keys: Sequence[str] | None = None,
) -> Dict[str, pd.Series]:
    outputs: Dict[str, pd.Series] = {}
    for key in MODEL_DISPLAY_ORDER:
        if selected_keys and key not in selected_keys:
            continue
        if key not in models or key not in metadata:
            continue
        features = metadata[key]["features_used"]
        missing = [col for col in features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing features for {MODEL_TITLES[key]}: {missing}")
        preds = models[key].predict(df[features])
        outputs[key] = pd.Series(preds, index=df.index)
    return outputs


def render_sidebar(metadata: Dict[str, Any]) -> None:
    st.sidebar.title("How to use")
    st.sidebar.markdown("1. Masukkan telemetry di tab **Single Prediction** atau unggah CSV di tab **Batch**.")
    st.sidebar.markdown("2. Klik *Predict* untuk menjalankan pipeline.")
    st.sidebar.markdown("3. Unduh hasilnya dan gunakan untuk dashboard Anda.")
    model_count = sum(1 for key in metadata if key.endswith("_model"))
    st.sidebar.info(f"{model_count} model ditemukan di folder `models/`.")
    st.sidebar.code("streamlit run app/main.py", language="bash")


def render_single_prediction_tab(metadata: Dict[str, Any], models: Dict[str, Any]) -> None:
    st.subheader("Single prediction")
    st.caption("Sesuaikan input untuk satu skenario lap / pembalap, lalu jalankan kedua model sekaligus.")
    with st.form(key="single_prediction_form"):
        feature_values: Dict[str, Any] = {}
        for section_title, features in FEATURE_SECTIONS:
            st.markdown(f"#### {section_title}")
            cols = st.columns(2)
            for idx, feature in enumerate(features):
                cfg = FEATURE_CONFIG[feature]
                column = cols[idx % 2]
                dtype = cfg["dtype"]
                if dtype == "int":
                    feature_values[feature] = column.number_input(
                        cfg["label"],
                        min_value=int(cfg["min"]),
                        max_value=int(cfg["max"]),
                        value=int(cfg["value"]),
                        step=int(cfg.get("step", 1)),
                        help=cfg.get("help"),
                    )
                else:
                    kwargs = {
                        "min_value": float(cfg["min"]),
                        "max_value": float(cfg["max"]),
                        "value": float(cfg["value"]),
                        "step": float(cfg.get("step", 0.1)),
                        "help": cfg.get("help"),
                    }
                    if cfg.get("format"):
                        kwargs["format"] = cfg["format"]
                    feature_values[feature] = column.number_input(
                        cfg["label"],
                        **kwargs,
                    )
        submitted = st.form_submit_button("Predict scenario", type="primary")
    if not submitted:
        return
    input_df = pd.DataFrame([feature_values])
    try:
        predictions = run_inference(models, metadata, input_df)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.success("Inference finished.")
    metric_cols = st.columns(len(predictions))
    for idx, key in enumerate(MODEL_DISPLAY_ORDER):
        if key not in predictions:
            continue
        value = predictions[key].iloc[0]
        formatted = f"{value:.1f}" if key == "position_model" else f"{value:.2f}"
        metric_cols[idx].metric(MODEL_TITLES[key], formatted)
    with st.expander("Feature payload"):
        st.dataframe(input_df.T, use_container_width=True)


def render_batch_prediction_tab(metadata: Dict[str, Any], models: Dict[str, Any]) -> None:
    st.subheader("Batch predictions")
    st.caption("Unggah CSV dengan kolom fitur yang sama seperti saat training, lalu download hasilnya.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if not uploaded:
        st.info("Belum ada file diunggah.")
        return
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Gagal membaca CSV: {exc}")
        return
    st.write(f"📄 {len(raw_df)} baris terdeteksi.")
    st.dataframe(raw_df.head(), use_container_width=True)
    df = coerce_features(raw_df)
    missing_cols = [col for col in FEATURE_CONFIG if col not in raw_df.columns]
    if missing_cols:
        st.warning(
            "Kolom berikut tidak ditemukan di CSV, sehingga diisi default dari slider single prediction: "
            + ", ".join(missing_cols)
        )
    options = [MODEL_TITLES[key] for key in MODEL_DISPLAY_ORDER if key in models]
    selection = st.multiselect(
        "Pilih model yang ingin dijalankan",
        options=options,
        default=options,
    )
    if not selection:
        st.info("Pilih minimal satu model.")
        return
    key_map = {MODEL_TITLES[key]: key for key in MODEL_DISPLAY_ORDER}
    selected_keys = [key_map[label] for label in selection]
    if st.button("Predict batch", type="primary"):
        try:
            predictions = run_inference(models, metadata, df, selected_keys)
        except ValueError as exc:
            st.error(str(exc))
            return
        result_df = df.copy()
        for key, series in predictions.items():
            result_df[MODEL_OUTPUT_COLUMNS[key]] = series
        st.success("Batch inference selesai.")
        st.dataframe(result_df.head(), use_container_width=True)
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions CSV",
            data=csv_bytes,
            file_name="f1_batch_predictions.csv",
            mime="text/csv",
        )


def render_model_cards_tab(metadata: Dict[str, Any]) -> None:
    st.subheader("Model cards")
    for key in MODEL_DISPLAY_ORDER:
        if key not in metadata:
            continue
        info = metadata[key]
        with st.container(border=True):
            st.markdown(f"### {MODEL_TITLES[key]}")
            st.markdown(f"**Artifact**: `{Path(info['filename']).name}`")
            st.markdown(
                f"**Pipeline steps**: {', '.join(info.get('pipeline_steps', []))}"
            )
            st.markdown("**Features used**")
            st.code(", ".join(info.get("features_used", [])), language="text")
            performance = info.get("performance", {})
            scalar_metrics = {
                k: v for k, v in performance.items() if not isinstance(v, dict)
            }
            if scalar_metrics:
                perf_df = (
                    pd.DataFrame.from_dict(scalar_metrics, orient="index", columns=["value"])
                    .round(4)
                )
                st.dataframe(perf_df, use_container_width=True, height=180)
            if "Best_Params" in performance:
                st.markdown("**Best params**")
                st.json(performance["Best_Params"])


def main() -> None:
    st.set_page_config(
        page_title="F1 Dashboard Prediction",
        page_icon="🏎️",
        layout="wide",
    )
    st.title("F1 Race Outcome Predictor")
    st.write(
        "Antarmuka Streamlit untuk menjalankan model posisi finish dan poin F1 yang sudah kamu latih."
    )
    try:
        metadata, models = get_assets()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    render_sidebar(metadata)
    tab_single, tab_batch, tab_cards = st.tabs(
        ["Single Prediction", "Batch Prediction", "Model Cards"]
    )
    with tab_single:
        render_single_prediction_tab(metadata, models)
    with tab_batch:
        render_batch_prediction_tab(metadata, models)
    with tab_cards:
        render_model_cards_tab(metadata)


if __name__ == "__main__":
    main()