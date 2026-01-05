"""
EcoMoveAI - Reporting utilities.

Purpose
-------
Generate report-ready tables and figures (CSV/PNG) from evaluation outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_report_assets(payload: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """
    Write report-ready CSV tables and PNG figures.
    """

    if "results" not in payload:
        raise ValueError("payload must include results.")

    _ensure_dir(output_dir)
    df = pd.DataFrame(payload["results"])

    scenario_csv = output_dir / "scenario_results.csv"
    df.to_csv(scenario_csv, index=False)

    numeric_columns = [
        "total_economic_benefit_usd_yr",
        "carbon_cost_savings_usd_yr",
        "fuel_cost_savings_usd_yr",
        "fuel_tax_savings_usd_yr",
        "congestion_charge_savings_usd_yr",
        "emissions_reduced_ton_yr",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    policy_summary = (
        df.groupby(["policy_scenario_name", "carbon_price_usd_per_ton"], as_index=False)
        .agg({
            "total_economic_benefit_usd_yr": "mean",
            "carbon_cost_savings_usd_yr": "mean",
            "fuel_cost_savings_usd_yr": "mean",
            "fuel_tax_savings_usd_yr": "mean",
            "congestion_charge_savings_usd_yr": "mean",
            "emissions_reduced_ton_yr": "mean",
        })
        .sort_values("carbon_price_usd_per_ton")
    )
    policy_csv = output_dir / "summary_by_policy.csv"
    policy_summary.to_csv(policy_csv, index=False)

    efficiency_summary = (
        df.groupby("ai_efficiency_rate", as_index=False)
        .agg({
            "total_economic_benefit_usd_yr": "mean",
            "carbon_cost_savings_usd_yr": "mean",
            "fuel_cost_savings_usd_yr": "mean",
            "fuel_tax_savings_usd_yr": "mean",
            "congestion_charge_savings_usd_yr": "mean",
            "emissions_reduced_ton_yr": "mean",
        })
        .sort_values("ai_efficiency_rate")
    )
    efficiency_csv = output_dir / "summary_by_efficiency.csv"
    efficiency_summary.to_csv(efficiency_csv, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")

    figure_paths: dict[str, Path] = {}

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for rate, group in df.groupby("ai_efficiency_rate"):
        group = group.sort_values("carbon_price_usd_per_ton")
        ax1.plot(
            group["carbon_price_usd_per_ton"],
            group["total_economic_benefit_usd_yr"],
            marker="o",
            label=f"AI efficiency {rate:.2f}",
        )
    ax1.set_title("Total Economic Benefit vs Carbon Price")
    ax1.set_xlabel("Carbon price (USD/ton CO2)")
    ax1.set_ylabel("Total economic benefit (USD/year)")
    ax1.legend(frameon=False)
    fig1.tight_layout()
    fig1_path = output_dir / "total_benefit_by_carbon_price.png"
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)
    figure_paths["total_benefit_by_carbon_price"] = fig1_path

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(
        efficiency_summary["ai_efficiency_rate"].astype(float),
        efficiency_summary["emissions_reduced_ton_yr"].astype(float),
        color="#2A9D8F",
    )
    ax2.set_title("Emissions Reduced by AI Efficiency")
    ax2.set_xlabel("AI efficiency rate")
    ax2.set_ylabel("Emissions reduced (ton CO2/year)")
    fig2.tight_layout()
    fig2_path = output_dir / "emissions_reduced_by_efficiency.png"
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)
    figure_paths["emissions_reduced_by_efficiency"] = fig2_path

    max_efficiency = df["ai_efficiency_rate"].max()
    subset = df[df["ai_efficiency_rate"] == max_efficiency].copy()
    subset = subset.sort_values("carbon_price_usd_per_ton")
    for col in [
        "carbon_cost_savings_usd_yr",
        "fuel_cost_savings_usd_yr",
        "fuel_tax_savings_usd_yr",
        "congestion_charge_savings_usd_yr",
    ]:
        if col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors="coerce").fillna(0.0)

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    x = range(len(subset))
    bottom = None
    for label, color in [
        ("carbon_cost_savings_usd_yr", "#264653"),
        ("fuel_cost_savings_usd_yr", "#2A9D8F"),
        ("fuel_tax_savings_usd_yr", "#E9C46A"),
        ("congestion_charge_savings_usd_yr", "#F4A261"),
    ]:
        values = subset[label].astype(float)
        ax3.bar(x, values, bottom=bottom, label=label.replace("_", " "))
        bottom = values if bottom is None else bottom + values
    ax3.set_title("Cost Savings Breakdown (Max AI Efficiency)")
    ax3.set_xlabel("Policy scenario")
    ax3.set_ylabel("Savings (USD/year)")
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(subset["policy_scenario_name"], rotation=20, ha="right")
    ax3.legend(frameon=False, fontsize=8)
    fig3.tight_layout()
    fig3_path = output_dir / "cost_savings_breakdown.png"
    fig3.savefig(fig3_path, dpi=200)
    plt.close(fig3)
    figure_paths["cost_savings_breakdown"] = fig3_path

    return {
        "scenario_results_csv": scenario_csv,
        "summary_by_policy_csv": policy_csv,
        "summary_by_efficiency_csv": efficiency_csv,
        **figure_paths,
    }
