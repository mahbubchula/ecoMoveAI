import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Load data
DATA_PATH = Path("data/analysis_results.json")
OUTPUT_DIR = Path("publication/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data["results"])
metadata = data["metadata"]

# Set Q1 Journal Aesthetics
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "axes.labelweight": "bold",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.edgecolor": "#444444",
    "grid.color": "#e0e0e0"
})

def format_bn(x, pos):
    return f'${x/1e9:.1f}B'

def format_mn(x, pos):
    return f'${x/1e6:.0f}M'

# 1. Advanced Heatmap
def plot_heatmap():
    pivot_df = df.pivot(index="ai_efficiency_rate", columns="policy_scenario_name", values="total_economic_benefit_usd_yr")
    pivot_df.index = [f"{int(x*100)}%" for x in pivot_df.index]
    
    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(pivot_df / 1e9, annot=True, fmt=".2f", cmap="RdYlGn", center=0, 
                    cbar_kws={'label': 'Net Annual Benefit (Billion USD/year)'},
                    linewidths=0.5, linecolor='white')
    plt.title("Scenario Matrix: Total Economic Benefit")
    plt.ylabel("AI Efficiency Target (%)")
    plt.xlabel("Policy Ambition Level")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_heatmap.png")
    plt.close()

# 2. Impact Curve: Emissions reduction
def plot_emissions_reduction():
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="ai_efficiency_rate", y="emissions_reduced_ton_yr", marker='o', 
                color='#0f766e', linewidth=3, markersize=10)
    plt.title("Transport CO2 Abatement Potential")
    plt.xlabel("AI Efficiency Improvement (%)")
    plt.ylabel("Annual Emissions Reduced (Tons CO2)")
    plt.xticks(metadata["ai_efficiency_rates"], [f"{int(x*100)}%" for x in metadata["ai_efficiency_rates"]])
    # Add annotation for max reduction
    max_val = df["emissions_reduced_ton_yr"].max()
    plt.annotate(f'Max Abatement:\n{max_val/1e6:.2f}M tons/yr', 
                 xy=(0.25, max_val), xytext=(0.18, max_val*0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_emissions_reduction.png")
    plt.close()

# 3. Policy Sensitivity: Carbon Pricing
def plot_total_benefit():
    plt.figure(figsize=(11, 7))
    palette = sns.color_palette("viridis", len(metadata["ai_efficiency_rates"]))
    for i, rate in enumerate(metadata["ai_efficiency_rates"]):
        subset = df[df["ai_efficiency_rate"] == rate]
        sns.lineplot(data=subset, x="carbon_price_usd_per_ton", y="total_economic_benefit_usd_yr", 
                    label=f"Eff: {int(rate*100)}%", marker='s', markersize=8, color=palette[i], linewidth=2)
    
    plt.axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.6)
    plt.title("Economic Resilience across Carbon Price Trajectories")
    plt.xlabel("Internalized Carbon Price (USD/ton CO2)")
    plt.ylabel("Net Policy Benefit (USD/year)")
    plt.legend(title="AI Efficiency (%)", loc='upper left', frameon=True, shadow=True)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_bn))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_total_benefit.png")
    plt.close()

# 4. MAC Curve with Policy Anchors
def plot_mac():
    plt.figure(figsize=(10, 7))
    unique_df = df.drop_duplicates("ai_efficiency_rate").sort_values("ai_efficiency_rate").copy()
    unique_df["marginal_abatement_cost_usd_per_ton"] = unique_df["marginal_abatement_cost_usd_per_ton"].fillna(0)
    
    colors = sns.color_palette("YlOrBr", len(unique_df))
    bars = plt.bar([f"{int(x*100)}%" for x in unique_df["ai_efficiency_rate"]], 
            unique_df["marginal_abatement_cost_usd_per_ton"], color=colors, edgecolor='black', alpha=0.8)
    
    # Reference markers for carbon prices
    price_levels = [50, 100, 150]
    labels = ["Mod. Carbon Price", "Amb. Carbon Price", "Net-Zero Standard"]
    colors_ref = ["#f59e0b", "#d97706", "#991b1b"]
    
    for lvl, label, clr in zip(price_levels, labels, colors_ref):
        plt.axhline(lvl, color=clr, linestyle='--', linewidth=1.5, alpha=0.7)
        if len(unique_df) > 0:
            plt.text(len(unique_df)-1.2, lvl + 5, label, color=clr, fontweight='bold', fontsize=10)

    plt.title("Marginal Abatement Cost (MAC) vs. Climate Policy Anchors")
    plt.xlabel("Realized AI Efficiency Target (%)")
    plt.ylabel("Marginal Abatement Cost (USD/ton CO2)")
    
    # Value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., height + 10, f'${int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_mac.png")
    plt.close()

# 5. Policy Portfolio Comparison
def plot_policy_overview():
    max_eff = max(metadata["ai_efficiency_rates"])
    subset = df[df["ai_efficiency_rate"] == max_eff]
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=subset, x="policy_scenario_name", y="total_economic_benefit_usd_yr", palette="Spectral", edgecolor='black')
    plt.title(f"Inter-Policy Comparison at {int(max_eff*100)}% Conversion Efficiency")
    plt.ylabel("Total Economic Benefit (USD/year)")
    plt.xlabel("Policy Instrument Mix")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_bn))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_policy_overview.png")
    plt.close()

# 6. High-Fidelity Waterfall Decomposition
def plot_waterfall():
    best_scenario = df[(df["ai_efficiency_rate"] == 0.25) & (df["policy_scenario_name"] == "Net-Zero Aligned")].iloc[0]
    
    components = {
        "Fuel Savings": best_scenario["fuel_cost_savings_usd_yr"],
        "Carbon Value": best_scenario["carbon_cost_savings_usd_yr"],
        "Tax Savings": best_scenario["fuel_tax_savings_usd_yr"],
        "Congestion Red.": best_scenario["congestion_charge_savings_usd_yr"],
        "AI Subsidy": best_scenario["ai_subsidy_usd_yr"] or 0,
        "System Costs": -best_scenario["ai_implementation_cost_usd_yr"]
    }
    
    names = list(components.keys())
    values = list(components.values())
    
    cumulative = 0
    tops = []
    bottoms = []
    for val in values:
        if val >= 0:
            bottoms.append(cumulative)
            tops.append(cumulative + val)
        else:
            tops.append(cumulative)
            bottoms.append(cumulative + val)
        cumulative += val
    
    plt.figure(figsize=(12, 8))
    colors = ['#10b981' if v >= 0 else '#ef4444' for v in values]
    plt.bar(names, [tops[i] - bottoms[i] for i in range(len(names))], bottom=bottoms, color=colors, edgecolor='black')
    plt.bar(["Net Outcome"], [cumulative], color='#3b82f6', edgecolor='black', hatch='//')
    
    plt.title("Benefit Waterfall: Structural Decomposition of Returns\n(25% Efficiency | Net-Zero Policy)", pad=20)
    plt.ylabel("Monetized Value (USD/year)")
    plt.xticks(rotation=15, fontweight='bold')
    plt.axhline(0, color='black', linewidth=1.5, alpha=0.5)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_bn))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_waterfall.png")
    plt.close()

# 7. Q1 Tornado Sensitivity Diagram (The requested "Advanced" Figure)
def plot_tornado_sensitivity():
    # Base configuration for sensitivity: 15% efficiency, Ambitious Policy (representative Q1 case)
    base_eff = 0.15
    base_policy_name = "Ambitious Policy"
    base_row = df[(df["ai_efficiency_rate"] == base_eff) & (df["policy_scenario_name"] == base_policy_name)].iloc[0]
    base_benefit = base_row["total_economic_benefit_usd_yr"]
    
    # Fixed parameters from metadata
    E0 = metadata["baseline"]["co2_emissions_ton_yr"]
    C0 = metadata["baseline"]["fuel_cost_usd_yr"]
    Cimpl = metadata["ai_implementation_cost_usd_yr"]
    
    # Active policy params for Ambitious Policy Row
    Pc = base_row["carbon_price_usd_per_ton"]
    tau_f = base_row["fuel_tax_rate"]
    S = base_row["ai_subsidy_usd_yr"] or 0
    
    def calc_benefit(eff, pc, e0, c0, tf, c_imp, s):
        de = e0 * eff
        dc = c0 * eff
        dk = de * pc
        dt = dc * tf
        return (dk + dc + dt) - (c_imp - s)

    params = {
        "AI Efficiency Rate": {"val": base_eff, "calc": lambda x: calc_benefit(x, Pc, E0, C0, tau_f, Cimpl, S)},
        "Baseline Emissions": {"val": E0, "calc": lambda x: calc_benefit(base_eff, Pc, x, C0, tau_f, Cimpl, S)},
        "Fuel Cost Unit": {"val": C0, "calc": lambda x: calc_benefit(base_eff, Pc, E0, x, tau_f, Cimpl, S)},
        "Carbon Price ($/ton)": {"val": Pc, "calc": lambda x: calc_benefit(base_eff, x, E0, C0, tau_f, Cimpl, S)},
        "Implementation Cost": {"val": Cimpl, "calc": lambda x: calc_benefit(base_eff, Pc, E0, C0, tau_f, x, S)},
        "Fuel Tax Rate (%)": {"val": tau_f, "calc": lambda x: calc_benefit(base_eff, Pc, E0, C0, x, Cimpl, S)}
    }
    
    results = []
    for name, config in params.items():
        low_val = config["val"] * 0.8  # -20%
        high_val = config["val"] * 1.2 # +20%
        low_res = config["calc"](low_val)
        high_res = config["calc"](high_val)
        results.append({
            "Parameter": name,
            "Low": low_res - base_benefit,
            "High": high_res - base_benefit,
            "Range": abs(high_res - low_res)
        })
    
    sens_df = pd.DataFrame(results).sort_values("Range", ascending=True)
    
    plt.figure(figsize=(12, 8))
    # Low Swings
    plt.barh(sens_df["Parameter"], sens_df["Low"], color='#ef4444', label='-20% Variation', alpha=0.8, edgecolor='black')
    # High Swings
    plt.barh(sens_df["Parameter"], sens_df["High"], color='#10b981', label='+20% Variation', alpha=0.8, edgecolor='black')
    
    plt.axvline(0, color='black', linewidth=1.5)
    plt.title("Tornado Diagram: Sensitivity of Annual Net Benefit\n(Relative to Ambitious Policy Outcome at 15% Eff)", pad=20)
    plt.xlabel("Change in Total Economic Benefit (USD/year)")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_mn))
    plt.legend(frameon=True, loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_sensitivity.png")
    plt.close()

# 8. Advanced Cost Savings Breakdown (Comprehensive ROI)
def plot_cost_breakdown():
    # Representative case: 15% efficiency, average across policy scenarios
    subset = df[df["ai_efficiency_rate"] == 0.15].copy()
    
    # Fill NaN with 0 for components
    cols = ["fuel_cost_savings_usd_yr", "carbon_cost_savings_usd_yr", "fuel_tax_savings_usd_yr", 
            "congestion_charge_savings_usd_yr", "ai_subsidy_usd_yr"]
    for col in cols:
        if col in subset.columns:
            subset[col] = subset[col].fillna(0)
    
    components = {
        "Fuel Savings": subset["fuel_cost_savings_usd_yr"].mean(),
        "Carbon Abatement": subset["carbon_cost_savings_usd_yr"].mean(),
        "Tax Shield": subset["fuel_tax_savings_usd_yr"].mean(),
        "Congestion Reduction": subset["congestion_charge_savings_usd_yr"].mean(),
        "AI Policy Subsidy": subset["ai_subsidy_usd_yr"].mean()
    }
    
    # Sort by value for better visual flow
    sorted_components = dict(sorted(components.items(), key=lambda item: item[1] if not np.isnan(item[1]) else 0, reverse=True))
    labels = list(sorted_components.keys())
    values = [sorted_components[l] for l in labels]
    total = sum(v for v in values if not np.isnan(v))
    
    plt.figure(figsize=(13, 8))
    colors = sns.color_palette("coolwarm", len(labels))
    bars = plt.barh(labels, values, color=colors, edgecolor='black', alpha=0.95)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if not np.isnan(width) and total > 0:
            percentage = (width / total) * 100
            label_text = f'${width/1e6:.1f}M ({percentage:.1f}%)'
            plt.text(width + total*0.015, bar.get_y() + bar.get_height()/2, 
                     label_text, va='center', fontweight='bold', fontsize=12)
    
    plt.title("Structural Decomposition of Gross Economic Benefits\n(Weighted Mean at 15% System Efficiency)", pad=30, fontsize=18)
    plt.xlabel("Annual Monetized Value (USD/year)", fontsize=14)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_mn))
    
    if total > 0:
        plt.xlim(0, max(values) * 1.4)  # Make room for labels
    
    # Despine and clean up
    sns.despine(left=True, bottom=False)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save both names for compatibility
    plt.savefig(OUTPUT_DIR / "cost_savings_breakdown.png")
    plt.savefig(OUTPUT_DIR / "fig_cost_breakdown.png")
    plt.close()

if __name__ == "__main__":
    print("Regenerating advanced Q1-Standard figures...")
    plot_heatmap()
    plot_emissions_reduction()
    plot_total_benefit()
    plot_mac()
    plot_policy_overview()
    plot_waterfall()
    plot_tornado_sensitivity()
    plot_cost_breakdown()
    print(f"Success! High-fidelity assets saved to {OUTPUT_DIR}")
