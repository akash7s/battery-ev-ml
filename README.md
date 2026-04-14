# Battery Health Intelligence System
### ML-Based SOH, RUL Prediction & Solar Range Analysis Across Urban EV Profiles

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/AkashR7/battery-health-intelligence)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Inspired by Siemens Digital Industries Software — 100 AI-Powered Engineering Use Cases (2026)**  
> Pages 13, 16 & 19: Real-time battery monitoring for 2-wheelers | SoH & RUL prediction | SOC optimisation

---

## Project Overview

This project builds a complete battery health intelligence pipeline that predicts **State of Health (SOH)** and **Remaining Useful Life (RUL)** of lithium-ion batteries across three urban electric vehicle operating profiles — 3-Wheeler e-rickshaw (tropical), 4-Wheeler passenger EV (temperate), and cold climate reference — with an integrated solar range extension calculator.

The work directly supports the research context of solar-electric hybrid vehicle development, providing the battery intelligence layer required for effective energy management in retrofitted e-rickshaw platforms.

---

## Live Dashboard

**[Open Interactive Dashboard](https://huggingface.co/spaces/AkashR7/battery-health-intelligence)**

The dashboard includes four interactive modules:
- **Battery Health Monitor** — real-time SOH & RUL prediction with health status
- **Degradation Trajectory** — full lifecycle predicted vs actual curves
- **Vehicle Profile Comparison** — 3-wheeler vs 4-wheeler vs cold climate
- **Solar Range Calculator** — range extension and battery life impact of solar integration

---

## Key Results

| Model | Test R² | Test MAE | Industrial Benchmark |
|-------|---------|----------|---------------------|
| SOH Prediction | **0.9762** | 0.0081 Ahr | Siemens: ~98% (Pg. 19) |
| RUL Prediction | **0.9114** | ~33 cycles | — |

**Validated on unseen batteries using leave-battery-out evaluation** — the correct methodology for generalisation testing in battery health prediction.

---

## Key Findings

### 1. Cold Climate Causes 2.5× Faster Battery Degradation
Contrary to naive Arrhenius expectations, batteries at 4°C degrade **2.5× faster** than tropical batteries at 43°C due to lithium plating mechanisms at low temperatures.

| Profile | Temperature | Degradation Rate | Avg Final SOH |
|---------|------------|-----------------|---------------|
| 3-Wheeler (Tropical) | 43°C | 2.064 × 10⁻³/cycle | 0.853 |
| 4-Wheeler (Temperate) | 24°C | 2.078 × 10⁻³/cycle | 0.839 |
| Cold Climate Reference | 4°C | **5.147 × 10⁻³/cycle** | 0.720 |

### 2. Solar Integration Disproportionately Benefits E-Rickshaws
At 600 W/m² irradiance with a 1.2m² rooftop panel:

| Vehicle | Range Extension | Battery Life Extension | CO₂ Saved |
|---------|----------------|----------------------|-----------|
| E-Rickshaw (48V, 100Ah) | **+28.8 km/day** | **+54%** | 215 kg/year |
| Passenger EV (400V, 60Ah) | +5.5 km/day | +12% | 287 kg/year |

### 3. Model Generalises Across All Operating Profiles
| Profile | SOH R² | SOH MAE |
|---------|--------|---------|
| 3-Wheeler Tropical | 0.9989 | 0.0007 |
| 4-Wheeler Temperate | 0.9832 | 0.0060 |
| Cold Climate | 0.9993 | 0.0018 |

---

## Project Architecture

```
battery-ev-ml/
├── notebooks/
│   ├── 01_eda.ipynb              # Data loading, parsing, EDA
│   ├── 02_results.ipynb          # Model training & evaluation
│   ├── 03_vehicle_profiles.ipynb # Vehicle profile comparison
│   └── 04_solar_analysis.ipynb   # Solar integration analysis
├── data/
│   └── processed/                # Feature-engineered datasets
├── models/
│   ├── rf_soh_model.pkl          # SOH Random Forest model
│   └── rf_rul_norm_model.pkl     # RUL Random Forest model
├── figures/                      # All 12 publication-quality figures
├── app.py                        # Gradio dashboard
└── requirements.txt
```

---

## Methodology

### Data
- **Source:** NASA Prognostics Center of Excellence Battery Dataset
- **Scale:** 2,542 discharge cycles across 34 batteries
- **Evaluation:** Leave-battery-out split — train on 30 batteries, test on 4 core NASA batteries (B0005, B0006, B0007, B0018)

### Feature Engineering
7 physics-informed features per discharge cycle:

| Feature | Physical Meaning |
|---------|-----------------|
| `SOH` | Current capacity / peak capacity |
| `capacity_fade` | Absolute capacity lost from peak |
| `cycle_norm` | Fraction of battery life elapsed |
| `capacity_rolling_5` | Smoothed 5-cycle capacity trend |
| `capacity_delta` | Rate of degradation (1st derivative) |
| `capacity_accel` | Acceleration of degradation (2nd derivative) |
| `temperature` | Operating temperature (Arrhenius factor) |

### Models
- **Algorithm:** Random Forest Regressor (200 trees, max_depth=15)
- **SOH:** Dominated by capacity_fade (importance: 0.950)
- **RUL:** Dominated by cycle_norm (importance: 0.766)

---

## Quick Start

```bash
git clone https://github.com/akash7s/battery-ev-ml.git
cd battery-ev-ml
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860` in your browser.

---

## Future Work

- **Neural ODEs** for physics-constrained thermal runaway prediction (Siemens use case, Page 33)
- **LSTM** for sample-level SOC prediction within discharge curves
- **Digital Twin** integration — packaging the model as an FMU for HIL deployment
- **Pack-level generalisation** — extending from cell-level NASA data to multi-cell EV pack data

---

## References

- Siemens Digital Industries Software. *100 AI-Powered Engineering Use Cases* (2026) — Pages 13, 16, 19, 33
- NASA Prognostics Center of Excellence. *Battery Dataset* (Saha & Goebel, 2007)
- NREL Global Solar Atlas — Guwahati, Assam irradiance data
- Chen et al. *Neural Ordinary Differential Equations* (NeurIPS 2018)

---

## Author

**Akash Sarma**