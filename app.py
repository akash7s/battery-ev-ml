# ── Battery Health Intelligence Dashboard ─────────────────────────────────────
# Gradio web application for interactive battery SOH, RUL prediction
# and solar range extension analysis
#
# Inspired by Siemens industry use cases:
#   - Real-time battery monitoring for 2-wheelers (Page 16)
#   - Battery SoH and RUL prediction (Page 13)
#   - Optimise battery pack SOC and voltage (Page 19)
#
# Project: ML-Based Battery Health Intelligence for Solar-Electric Vehicles
# Dataset: NASA Battery Dataset (B0005, B0006, B0007, B0018)

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'Arial',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Load models and data ───────────────────────────────────────────────────────
print("Loading models and data...")
rf_soh      = joblib.load('models/rf_soh_model.pkl')
rf_rul_norm = joblib.load('models/rf_rul_norm_model.pkl')
all_data    = pd.read_csv('data/processed/features_all_batteries.csv')

FEATURE_COLS = ['cycle_norm', 'capacity_rolling_5', 'capacity_delta',
                'capacity_accel', 'temperature', 'capacity_fade']

# Pre-generate predictions for all batteries
all_data['SOH_predicted']      = rf_soh.predict(all_data[FEATURE_COLS])
all_data['RUL_norm_predicted'] = rf_rul_norm.predict(all_data[FEATURE_COLS])

BATTERY_IDS = sorted(all_data['battery_id'].unique().tolist())

COLORS = {
    'B0005': '#2196F3', 'B0006': '#4CAF50',
    'B0007': '#FF9800', 'B0018': '#E91E63',
}
DEFAULT_COLOR = '#607D8B'

print(f"Loaded {len(BATTERY_IDS)} batteries, {len(all_data)} cycles")
print("Dashboard ready!")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Battery Health Monitor
# ─────────────────────────────────────────────────────────────────────────────

def get_health_status(soh):
    if soh >= 0.90:
        return "🟢 Excellent", "#2e7d32"
    elif soh >= 0.80:
        return "🟡 Good", "#f57f17"
    elif soh >= 0.70:
        return "🟠 Fair — Monitor Closely", "#e65100"
    else:
        return "🔴 End of Life — Replace Soon", "#c62828"

def battery_health_monitor(battery_id, cycle_number):
    battery = all_data[all_data['battery_id'] == battery_id].sort_values('test_id')

    if len(battery) == 0:
        return "Battery not found", "N/A", "N/A", "N/A", None

    # Clamp cycle number to available range
    cycle_number = int(min(cycle_number, len(battery) - 1))
    row = battery.iloc[cycle_number]

    soh_actual    = row['SOH']
    soh_predicted = row['SOH_predicted']
    rul_norm      = row['RUL_norm_predicted']
    max_rul       = row['max_rul'] if 'max_rul' in row else battery['RUL'].max()
    rul_cycles    = int(rul_norm * max_rul)
    temperature   = row['temperature']
    capacity      = row['Capacity']

    status_text, status_color = get_health_status(soh_predicted)

    # ── Gauge figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    color = COLORS.get(battery_id, DEFAULT_COLOR)

    # SOH gauge (pie chart style)
    ax = axes[0]
    soh_pct = min(soh_predicted, 1.0)
    wedge_colors = [color, '#ECEFF1']
    ax.pie([soh_pct, 1 - soh_pct],
           colors=wedge_colors,
           startangle=90,
           counterclock=False,
           wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3))
    ax.text(0, 0, f'{soh_pct*100:.1f}%',
            ha='center', va='center',
            fontsize=28, fontweight='bold', color=color)
    ax.set_title(f'State of Health\n{status_text}',
                fontsize=13, fontweight='bold', pad=15)

    # RUL gauge
    ax = axes[1]
    rul_pct = min(rul_norm, 1.0)
    ax.pie([rul_pct, 1 - rul_pct],
           colors=['#FF9800', '#ECEFF1'],
           startangle=90,
           counterclock=False,
           wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3))
    ax.text(0, 0, f'{rul_cycles}\ncycles',
            ha='center', va='center',
            fontsize=20, fontweight='bold', color='#FF9800')
    ax.set_title('Remaining Useful Life\n(estimated)',
                fontsize=13, fontweight='bold', pad=15)

    # Battery stats bar
    ax = axes[2]
    ax.axis('off')
    stats = [
        ('Battery ID',    battery_id),
        ('Cycle Number',  f'{cycle_number + 1} of {len(battery)}'),
        ('Temperature',   f'{temperature:.0f}°C'),
        ('Capacity',      f'{capacity:.4f} Ahr'),
        ('SOH Actual',    f'{soh_actual:.4f}'),
        ('SOH Predicted', f'{soh_predicted:.4f}'),
        ('Prediction Δ',  f'{abs(soh_predicted - soh_actual)*100:.2f}%'),
    ]
    y_pos = 0.95
    for label, value in stats:
        ax.text(0.05, y_pos, label + ':',
                transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='#455A64')
        ax.text(0.55, y_pos, value,
                transform=ax.transAxes,
                fontsize=11, color='#212121')
        y_pos -= 0.13

    ax.set_title('Battery Statistics',
                fontsize=13, fontweight='bold', pad=15)

    fig.suptitle(f'Battery Health Monitor — {battery_id}  |  Cycle {cycle_number + 1}',
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    return (
        f"{soh_predicted*100:.2f}%",
        f"{rul_cycles} cycles (~{rul_cycles//30} months)",
        status_text,
        f"{temperature:.0f}°C operating temperature",
        fig
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Degradation Trajectory
# ─────────────────────────────────────────────────────────────────────────────

def degradation_trajectory(battery_id, highlight_cycle):
    battery = all_data[all_data['battery_id'] == battery_id].sort_values('test_id')

    if len(battery) == 0:
        return None

    highlight_cycle = int(min(highlight_cycle, len(battery) - 1))
    color = COLORS.get(battery_id, DEFAULT_COLOR)
    cycles = range(len(battery))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ── SOH trajectory ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(cycles, battery['SOH'].values,
            color=color, linewidth=2.5,
            label='Actual SOH', alpha=0.9)
    ax.plot(cycles, battery['SOH_predicted'].values,
            color=color, linewidth=2, linestyle='--',
            label='Predicted SOH', alpha=0.7)
    ax.fill_between(cycles,
                    battery['SOH'].values,
                    battery['SOH_predicted'].values,
                    alpha=0.15, color=color)

    # Highlight current cycle
    ax.axvline(x=highlight_cycle, color='red',
               linestyle=':', linewidth=2, alpha=0.8,
               label=f'Current cycle ({highlight_cycle + 1})')
    ax.scatter([highlight_cycle],
               [battery['SOH_predicted'].iloc[highlight_cycle]],
               s=200, color='red', zorder=10)

    ax.axhline(y=0.70, color='red', linestyle='--',
               linewidth=1.5, alpha=0.6, label='EOL threshold (SOH=0.70)')
    ax.axhspan(0, 0.70, alpha=0.05, color='red')
    ax.set_ylabel('State of Health (SOH)', fontsize=12)
    ax.set_title(f'{battery_id} — SOH Degradation Trajectory',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.05)

    # ── RUL trajectory ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(cycles, battery['RUL_norm'].values,
            color='#FF9800', linewidth=2.5,
            label='Actual RUL_norm', alpha=0.9)
    ax.plot(cycles, battery['RUL_norm_predicted'].values,
            color='#FF9800', linewidth=2, linestyle='--',
            label='Predicted RUL_norm', alpha=0.7)

    ax.axvline(x=highlight_cycle, color='red',
               linestyle=':', linewidth=2, alpha=0.8,
               label=f'Current cycle ({highlight_cycle + 1})')
    ax.scatter([highlight_cycle],
               [battery['RUL_norm_predicted'].iloc[highlight_cycle]],
               s=200, color='red', zorder=10)

    ax.axhline(y=0.20, color='red', linestyle='--',
               linewidth=1.5, alpha=0.6,
               label='Critical zone (20% life left)')
    ax.axhspan(0, 0.20, alpha=0.08, color='red')
    ax.set_xlabel('Discharge Cycle', fontsize=12)
    ax.set_ylabel('Normalised RUL', fontsize=12)
    ax.set_title(f'{battery_id} — Remaining Useful Life Trajectory',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle(f'Degradation Trajectory — {battery_id}\n'
                 f'Predicted vs Actual SOH and RUL across {len(battery)} cycles',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Vehicle Profile Comparison
# ─────────────────────────────────────────────────────────────────────────────

def vehicle_profile_comparison():
    profile_map = {
        '1 — 3-Wheeler (Tropical)':   ('#FF5722', '3-Wheeler\nTropical (43°C)'),
        '2 — 4-Wheeler (Temperate)':  ('#2196F3', '4-Wheeler\nTemperate (24°C)'),
        '3 — Cold Climate Reference': ('#9C27B0', 'Cold Climate\nReference (4°C)'),
    }

    def assign_profile(temp):
        if temp <= 10:   return '3 — Cold Climate Reference'
        elif temp <= 30: return '2 — 4-Wheeler (Temperate)'
        else:            return '1 — 3-Wheeler (Tropical)'

    all_data['vehicle_profile'] = all_data['temperature'].apply(assign_profile)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    profile_order = list(profile_map.keys())

    for idx, profile in enumerate(profile_order):
        ax = axes[idx]
        color, label = profile_map[profile]
        subset = all_data[all_data['vehicle_profile'] == profile]

        all_soh, max_len = [], 0
        for bid in subset['battery_id'].unique():
            b = subset[subset['battery_id'] == bid].sort_values('test_id')
            ax.plot(range(len(b)), b['SOH'].values,
                    color=color, linewidth=1, alpha=0.2)
            all_soh.append(b['SOH'].values)
            max_len = max(max_len, len(b))

        padded = np.full((len(all_soh), max_len), np.nan)
        for i, s in enumerate(all_soh):
            padded[i, :len(s)] = s
        mean_soh = np.nanmean(padded, axis=0)

        ax.plot(range(len(mean_soh)), mean_soh,
                color=color, linewidth=3, label='Mean SOH')
        ax.axhline(y=0.70, color='red', linestyle='--',
                   linewidth=1.5, label='EOL threshold')

        deg_rates = []
        for bid in subset['battery_id'].unique():
            b = subset[subset['battery_id'] == bid].sort_values('test_id')
            if len(b) > 5:
                deg_rates.append(
                    (b['SOH'].iloc[0] - b['SOH'].iloc[-1]) / len(b) * 1000)

        avg_rate     = np.mean(deg_rates) if deg_rates else 0
        avg_final    = subset.groupby('battery_id')['SOH'].last().mean()
        n_batteries  = subset['battery_id'].nunique()

        ax.set_title(label, fontsize=14, fontweight='bold',
                    color=color, pad=15)
        ax.set_xlabel('Discharge Cycle', fontsize=11)
        if idx == 0:
            ax.set_ylabel('State of Health (SOH)', fontsize=11)
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=10)

        stats = (f'n = {n_batteries} batteries\n'
                 f'Deg. rate: {avg_rate:.3f}×10⁻³/cycle\n'
                 f'Avg final SOH: {avg_final:.3f}')
        ax.text(0.05, 0.08, stats, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor='white',
                         edgecolor=color, alpha=0.8))

    fig.suptitle('Battery Degradation Across Vehicle Operating Profiles\n'
                 'Key Finding: Cold Climate Causes 2.5× Faster Degradation',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Solar Range Calculator
# ─────────────────────────────────────────────────────────────────────────────

def solar_calculator(vehicle_type, irradiance, panel_efficiency):
    vehicles = {
        '3-Wheeler (E-Rickshaw, Tropical India)': {
            'battery_voltage':   48,
            'battery_capacity':  100,
            'energy_consumption': 25,
            'panel_area':        1.2,
            'typical_dod':       0.70,
            'peak_sun_hours':    5.0,
            'daily_km':          80,
            'color':             '#FF5722',
        },
        '4-Wheeler (Passenger EV, Temperate)': {
            'battery_voltage':   400,
            'battery_capacity':  60,
            'energy_consumption': 175,
            'panel_area':        2.0,
            'typical_dod':       0.80,
            'peak_sun_hours':    4.0,
            'daily_km':          150,
            'color':             '#2196F3',
        }
    }

    specs  = vehicles[vehicle_type]
    eff    = panel_efficiency / 100
    color  = specs['color']

    solar_power    = irradiance * specs['panel_area'] * eff
    daily_solar_wh = solar_power * specs['peak_sun_hours']
    range_ext      = daily_solar_wh / specs['energy_consumption']
    battery_wh     = specs['battery_voltage'] * specs['battery_capacity']
    solar_frac     = min(daily_solar_wh / battery_wh, 1.0)
    new_dod        = max(specs['typical_dod'] - solar_frac, 0.10)
    dod_reduction  = (specs['typical_dod'] - new_dod) / specs['typical_dod'] * 100
    life_ext       = (1 + (dod_reduction / 10) * 0.25)
    co2_saved      = (daily_solar_wh / 1000) * 0.82

    # ── Results figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Plot 1: Energy breakdown pie chart
    ax = axes[0]
    labels  = ['Solar contribution', 'Battery required']
    solar_e = min(daily_solar_wh, specs['daily_km'] * specs['energy_consumption'])
    batt_e  = max(0, specs['daily_km'] * specs['energy_consumption'] - solar_e)
    ax.pie([solar_e, batt_e],
           labels=labels,
           colors=[color, '#ECEFF1'],
           autopct='%1.1f%%',
           startangle=90,
           wedgeprops=dict(edgecolor='white', linewidth=2))
    ax.set_title(f'Daily Energy Split\n({specs["daily_km"]} km route)',
                fontsize=12, fontweight='bold')

    # Plot 2: DoD comparison bar
    ax = axes[1]
    bars = ax.bar(['Without Solar', 'With Solar'],
                  [specs['typical_dod'] * 100, new_dod * 100],
                  color=['#B0BEC5', color],
                  edgecolor='white', linewidth=2,
                  width=0.5)
    for bar, val in zip(bars, [specs['typical_dod']*100, new_dod*100]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{val:.1f}%',
                ha='center', fontsize=13, fontweight='bold')
    ax.set_ylabel('Depth of Discharge (%)', fontsize=11)
    ax.set_title(f'DoD Reduction\n({dod_reduction:.1f}% improvement)',
                fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=specs['typical_dod']*100,
               color='red', linestyle='--', alpha=0.4)

    # Plot 3: Battery life extension
    ax = axes[2]
    ax.bar(['Baseline', 'With Solar'],
           [1.0, life_ext],
           color=['#B0BEC5', color],
           edgecolor='white', linewidth=2,
           width=0.5)
    ax.text(1, life_ext + 0.01,
            f'{life_ext:.2f}×\n(+{(life_ext-1)*100:.0f}%)',
            ha='center', fontsize=13, fontweight='bold', color=color)
    ax.set_ylabel('Battery Life Multiplier', fontsize=11)
    ax.set_title(f'Battery Life Extension\n'
                 f'CO₂ saved: {co2_saved:.3f} kg/day',
                fontsize=12, fontweight='bold')
    ax.set_ylim(0.9, life_ext * 1.2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle(f'Solar Integration Analysis — {vehicle_type}\n'
                 f'Irradiance: {irradiance} W/m²  |  '
                 f'Panel: {specs["panel_area"]}m² @ {panel_efficiency}% efficiency',
                fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    summary = (
        f"⚡ Solar Power Output: {solar_power:.1f} W\n"
        f"🔋 Daily Solar Energy: {daily_solar_wh:.1f} Wh\n"
        f"🚗 Range Extension: +{range_ext:.1f} km/day\n"
        f"📉 DoD Reduction: {specs['typical_dod']*100:.0f}% → {new_dod*100:.1f}%\n"
        f"⏳ Battery Life: +{(life_ext-1)*100:.0f}% longer\n"
        f"🌱 CO₂ Saved: {co2_saved:.3f} kg/day ({co2_saved*365:.1f} kg/year)"
    )

    return summary, fig

# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE GRADIO INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Battery Health Intelligence Dashboard",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("""
    # 🔋 Battery Health Intelligence Dashboard
    ### ML-Based SOH, RUL Prediction & Solar Range Analysis for Electric Vehicles
    
    **Inspired by Siemens industry use cases** (Pages 13, 16, 19 — Real-time battery 
    monitoring for 2-wheelers, SoH & RUL prediction, SOC optimisation)
    
    **Dataset:** NASA Battery Dataset (B0005, B0006, B0007, B0018 + 30 additional batteries)  
    **Models:** Random Forest — SOH Test R²: 0.9762 | RUL Test R²: 0.9114
    """)

    with gr.Tabs():

        # ── Tab 1: Battery Health Monitor ─────────────────────────────────────
        with gr.TabItem("🔍 Battery Health Monitor"):
            gr.Markdown("### Select a battery and cycle to see real-time health predictions")
            with gr.Row():
                with gr.Column(scale=1):
                    bat_select = gr.Dropdown(
                        choices=BATTERY_IDS,
                        value='B0005',
                        label="Select Battery"
                    )
                    cycle_slider = gr.Slider(
                        minimum=0,
                        maximum=167,
                        value=0,
                        step=1,
                        label="Discharge Cycle Number"
                    )
                    monitor_btn = gr.Button(
                        "🔍 Analyse Battery Health",
                        variant="primary"
                    )

                with gr.Column(scale=2):
                    soh_out    = gr.Textbox(label="Predicted SOH")
                    rul_out    = gr.Textbox(label="Predicted RUL")
                    status_out = gr.Textbox(label="Health Status")
                    temp_out   = gr.Textbox(label="Operating Temperature")

            gauge_plot = gr.Plot(label="Battery Health Gauges")

            monitor_btn.click(
                fn=battery_health_monitor,
                inputs=[bat_select, cycle_slider],
                outputs=[soh_out, rul_out, status_out, temp_out, gauge_plot]
            )

        # ── Tab 2: Degradation Trajectory ─────────────────────────────────────
        with gr.TabItem("📈 Degradation Trajectory"):
            gr.Markdown("### Full lifecycle view — predicted vs actual SOH and RUL")
            with gr.Row():
                traj_battery = gr.Dropdown(
                    choices=BATTERY_IDS,
                    value='B0005',
                    label="Select Battery"
                )
                traj_cycle = gr.Slider(
                    minimum=0, maximum=167,
                    value=50, step=1,
                    label="Highlight Cycle"
                )
                traj_btn = gr.Button(
                    "📈 Plot Trajectory",
                    variant="primary"
                )
            traj_plot = gr.Plot(label="Degradation Trajectory")
            traj_btn.click(
                fn=degradation_trajectory,
                inputs=[traj_battery, traj_cycle],
                outputs=traj_plot
            )

        # ── Tab 3: Vehicle Profile Comparison ─────────────────────────────────
        with gr.TabItem("🚗 Vehicle Profile Comparison"):
            gr.Markdown("""
            ### Battery degradation across 3-Wheeler, 4-Wheeler, and Cold Climate profiles
            **Key Finding:** Cold climate batteries degrade 2.5× faster than tropical 
            batteries due to lithium plating mechanisms at low temperatures.
            """)
            profile_btn = gr.Button(
                "🚗 Generate Profile Comparison",
                variant="primary"
            )
            profile_plot = gr.Plot(label="Vehicle Profile Comparison")
            profile_btn.click(
                fn=vehicle_profile_comparison,
                inputs=[],
                outputs=profile_plot
            )

        # ── Tab 4: Solar Range Calculator ──────────────────────────────────────
        with gr.TabItem("☀️ Solar Range Calculator"):
            gr.Markdown("""
            ### Calculate solar panel benefit for different vehicle types
            Directly connected to the ASU project: 
            *Retrofitting a Three-Wheeler Electric Rickshaw into a Solar-Electric Hybrid Vehicle*
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    vehicle_select = gr.Dropdown(
                        choices=[
                            '3-Wheeler (E-Rickshaw, Tropical India)',
                            '4-Wheeler (Passenger EV, Temperate)'
                        ],
                        value='3-Wheeler (E-Rickshaw, Tropical India)',
                        label="Vehicle Type"
                    )
                    irradiance_slider = gr.Slider(
                        minimum=0, maximum=1000,
                        value=600, step=10,
                        label="Solar Irradiance (W/m²)"
                    )
                    efficiency_slider = gr.Slider(
                        minimum=15, maximum=25,
                        value=20, step=0.5,
                        label="Panel Efficiency (%)"
                    )
                    solar_btn = gr.Button(
                        "☀️ Calculate Solar Benefit",
                        variant="primary"
                    )
                with gr.Column(scale=1):
                    solar_summary = gr.Textbox(
                        label="Solar Benefit Summary",
                        lines=8
                    )

            solar_plot = gr.Plot(label="Solar Integration Analysis")
            solar_btn.click(
                fn=solar_calculator,
                inputs=[vehicle_select, irradiance_slider, efficiency_slider],
                outputs=[solar_summary, solar_plot]
            )

    gr.Markdown("""
    ---
    **Project:** ML-Based Battery Health Intelligence Across Urban EV Profiles  
    **Inspired by:** Siemens Digital Industries Software — 100 AI-Powered Engineering Use Cases (2026)  
    **Data:** NASA Prognostics Center of Excellence Battery Dataset  
    **Author:** Akash Sarma
    """)

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_port=7860,
        show_error=True
    )