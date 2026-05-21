
import json
from pathlib import Path
import numpy as np
import joblib
import streamlit as st
import torch
import torch.nn as nn
from theoretical_model import theoretical_hche_model

# Setup
st.set_page_config(page_title="HCHE Performance Dashboard", layout="wide")
BASE_DIR = Path(__file__).resolve().parent

DISPLAY_SCALE = {
    "effectiveness": 1.0,
    "pressure_drop_total_Pa": 1 / 1000,   # Pa → kPa
    "overall_heat_transfer_coefficient_U_W_m2K": 1 / 1000  # W/m²K → kW/m²K
}

DISPLAY_LABELS = {
    "effectiveness": "Eff",
    "pressure_drop_total_Pa": "ΔP (kPa)",
    "overall_heat_transfer_coefficient_U_W_m2K": "U (kW/m²K)"
}

TARGET_KEYS = [
    "effectiveness",
    "pressure_drop_total_Pa",
    "overall_heat_transfer_coefficient_U_W_m2K"
]

# -----------------------------
# PARAM CONFIG
# -----------------------------
PARAM_INFO = {
    "time_s": {"label": "Time", "unit": "s", "default": 1000.0},
    "hot_inlet_temp_C": {"label": "Hot Inlet Temp", "unit": "°C", "default": 80.0},
    "cold_inlet_temp_C": {"label": "Cold Inlet Temp", "unit": "°C", "default": 30.0},
    "hot_mass_flow_kg_s": {"label": "Hot Mass Flow", "unit": "kg/s", "default": 0.12},
    "cold_mass_flow_kg_s": {"label": "Cold Mass Flow", "unit": "kg/s", "default": 0.18},
    "hot_density_kg_m3": {"label": "Hot Density", "unit": "kg/m³", "default": 980.0},
    "cold_density_kg_m3": {"label": "Cold Density", "unit": "kg/m³", "default": 995.0},
    "hot_viscosity_Pa_s": {"label": "Hot Viscosity", "unit": "Pa·s", "default": 0.00045},
    "cold_viscosity_Pa_s": {"label": "Cold Viscosity", "unit": "Pa·s", "default": 0.00075},
    "hot_specific_heat_J_kgK": {"label": "Cp Hot", "unit": "J/kgK", "default": 4180.0},
    "cold_specific_heat_J_kgK": {"label": "Cp Cold", "unit": "J/kgK", "default": 4180.0},
    "hot_thermal_conductivity_W_mK": {"label": "k Hot", "unit": "W/mK", "default": 0.64},
    "cold_thermal_conductivity_W_mK": {"label": "k Cold", "unit": "W/mK", "default": 0.60},
    "tube_diameter_m": {"label": "Tube Diameter", "unit": "m", "default": 0.012},
    "coil_diameter_m": {"label": "Coil Diameter", "unit": "m", "default": 0.18},
    "coil_pitch_m": {"label": "Coil Pitch", "unit": "m", "default": 0.025},
    "number_of_turns": {"label": "Turns", "unit": "-", "default": 12},
    "shell_diameter_m": {"label": "Shell Diameter", "unit": "m", "default": 0.30},
    "pr_hot": {"label": "Pr Hot", "unit": "-", "default": 3.5},
    "pr_cold": {"label": "Pr Cold", "unit": "-", "default": 5.2},
}

# -----------------------------
# DERIVED FEATURES
# -----------------------------
def compute_derived_features(raw):
    eps = 1e-12

    d = raw["tube_diameter_m"]
    D = raw["coil_diameter_m"]
    p = raw["coil_pitch_m"]
    N = raw["number_of_turns"]
    Ds = raw["shell_diameter_m"]

    mh = raw["hot_mass_flow_kg_s"]
    mc = raw["cold_mass_flow_kg_s"]

    rho_h = raw["hot_density_kg_m3"]
    rho_c = raw["cold_density_kg_m3"]

    coil_length = N * np.sqrt((np.pi * D) ** 2 + p ** 2)

    A_tube = np.pi * d**2 / 4
    A_shell = np.pi * (Ds**2 - D**2) / 4

    v_h = mh / (rho_h * A_tube + eps)
    v_c = mc / (rho_c * A_shell + eps)

    Re_h = rho_h * v_h * d / (raw["hot_viscosity_Pa_s"] + eps)
    Re_c = rho_c * v_c * (Ds - D) / (raw["cold_viscosity_Pa_s"] + eps)

    C_h = mh * raw["hot_specific_heat_J_kgK"]
    C_c = mc * raw["cold_specific_heat_J_kgK"]

    return {
        **raw,
        "coil_length_m": coil_length,
        "tube_cross_section_area_m2": A_tube,
        "shell_cross_section_area_m2": A_shell,
        "hot_velocity_m_s": v_h,
        "cold_velocity_m_s": v_c,
        "re_hot": Re_h,
        "re_cold": Re_c,
        "dean_number": Re_h * np.sqrt(d / (D + eps)),
        "c_min_W_K": min(C_h, C_c),
        "c_max_W_K": max(C_h, C_c),
        "capacity_ratio": min(C_h, C_c) / (max(C_h, C_c) + eps),
        "hydraulic_diameter_m": Ds - D,
        "tube_surface_area_m2": np.pi * d * coil_length,
        "pitch_ratio": p / d,
        "curvature_ratio": d / D,
        "hot_heat_capacity_rate_W_K": C_h,
        "cold_heat_capacity_rate_W_K": C_c,
    }

# -----------------------------
# MODELS
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.network(x)


class PINN(MLP):
    pass


class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 128)
        self.block = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.out = nn.Linear(128, out_dim)

    def forward(self, x):
        h = torch.relu(self.fc(x))
        return self.out(torch.relu(self.block(h) + h))


class TransformerTabular(nn.Module):
    def __init__(self, in_dim, out_dim, d_model=64):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(in_dim * d_model, out_dim)

    def forward(self, x):
        x = self.embed(x.unsqueeze(-1))
        x = self.encoder(x)
        return self.fc(x.flatten(1))


# -----------------------------
# LOAD MODELS
# -----------------------------
def safe_load(model, path):
    if not Path(path).exists():
        return None
    try:
        sd = torch.load(path, map_location="cpu")
        sd = {k.replace("network.", "").replace("net.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        return model
    except:
        return None


@st.cache_resource
def load_artifacts():
    config = json.load(open(BASE_DIR / "model_config.json"))
    x_scaler, y_scaler = joblib.load(BASE_DIR / "x_scaler.pkl"), joblib.load(BASE_DIR / "y_scaler.pkl")

    in_d, out_d = config["input_dim"], config["output_dim"]

    models = {
        "MLP": safe_load(MLP(in_d, out_d), BASE_DIR / "mlp_model.pth"),
        "PINN": safe_load(PINN(in_d, out_d), BASE_DIR / "pinn_model.pth"),
        "Tabular Transformer": safe_load(ResNet(in_d, out_d), BASE_DIR / "tabular_transformer_model.pth"),
        "PINN Transformer": safe_load(TransformerTabular(in_d, out_d), BASE_DIR / "transformer_model.pth"),
    }

    for m in models.values():
        if m:
            m.eval()

    return config, x_scaler, y_scaler, models


# -----------------------------
# UI
# -----------------------------
st.title("🌡️ HCHE Performance Dashboard")

config, x_scaler, y_scaler, models = load_artifacts()

col1, col2 = st.columns(2)
raw = {}

keys = list(PARAM_INFO.keys())
for i, k in enumerate(keys):
    target = col1 if i < len(keys) // 2 else col2
    raw[k] = target.number_input(
        f"{PARAM_INFO[k]['label']} ({PARAM_INFO[k]['unit']})",
        value=PARAM_INFO[k]["default"],
    )

if st.button("Generate Predictions", type="primary"):
    full = compute_derived_features(raw)

    x = np.array([[full[f] for f in config["input_features"]]], dtype=np.float32)
    xt = torch.tensor(x_scaler.transform(x), dtype=torch.float32)

    results = {"Theoretical": theoretical_hche_model(full)}

    for name, model in models.items():
        if model:
            with torch.no_grad():
                pred = model(xt).numpy()[0]
                results[name] = y_scaler.inverse_transform(pred.reshape(1, -1))[0]

    st.subheader("Model Comparison")

    cols = st.columns(len(results))

    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"**{name}**")

            # Apply clamping logic to effectiveness
            raw_eff = res[0]
            if raw_eff >= 1.0:
                eff = 0.9
            elif raw_eff <= 0.0:
                eff = 0.05
            else:
                eff = raw_eff * DISPLAY_SCALE["effectiveness"]

            dp = res[1] * DISPLAY_SCALE["pressure_drop_total_Pa"]
            u = res[2] * DISPLAY_SCALE["overall_heat_transfer_coefficient_U_W_m2K"]

            st.metric("Eff", f"{eff:.3f}")
            st.metric("ΔP (kPa)", f"{dp:.3f}")
            st.metric("U (kW/m²K)", f"{u:.3f}")

    with st.expander("View Derived Engineering Features"):
        st.json({k: v for k, v in full.items() if k not in raw})