"""
EcoMoveAI Streamlit Dashboard

Policy-grade interface for scenario evaluation and interpretation.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from core.memory import append_memory_note, format_memory_notes, load_memory_notes
from core.rag import build_index, load_documents_from_dir, search
from core.reporting import write_report_assets
from main import build_results, load_config_from_dict, summarize_results

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_USER_AGENT = "EcoMoveAI/1.0"


def _apply_theme() -> None:
    """Apply a premium, policy-grade UI theme with clear hierarchy."""
    st.set_page_config(
        page_title="EcoMoveAI | Mahbub Hassan",
        page_icon="AI",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
        :root {
            --ink: #0b1220;
            --muted: #54627a;
            --accent: #06b6d4;
            --accent-2: #f97316;
            --accent-3: #22c55e;
            --accent-soft: #cffafe;
            --panel: #ffffff;
            --canvas: #f1f6ff;
            --border: rgba(15, 23, 42, 0.12);
            --shadow: 0 20px 38px rgba(15, 23, 42, 0.12);
            --glow: 0 0 45px rgba(6, 182, 212, 0.25);
        }
        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
        }
        .appview-container {
            background:
                radial-gradient(circle at 15% 10%, rgba(6, 182, 212, 0.18), transparent 42%),
                radial-gradient(circle at 85% 15%, rgba(249, 115, 22, 0.18), transparent 40%),
                radial-gradient(circle at 70% 85%, rgba(34, 197, 94, 0.12), transparent 45%),
                linear-gradient(180deg, #f8fbff 0%, #eef6ff 60%, #f6fbff 100%);
            background-size: 140% 140%;
            animation: gradientShift 18s ease infinite;
        }
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        .hero {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.2rem;
            align-items: stretch;
            animation: fadeIn 0.8s ease both;
        }
        @media (max-width: 900px) {
            .hero { grid-template-columns: 1fr; }
        }
        .brand-card {
            position: relative;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), #f0f7ff);
            border: 1px solid rgba(6, 182, 212, 0.2);
            border-radius: 16px;
            padding: 1.5rem 1.8rem;
            box-shadow: var(--shadow), var(--glow);
            overflow: hidden;
        }
        .brand-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            height: 4px;
            width: 100%;
            background: linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent-3));
        }
        .brand-card::after {
            content: "";
            position: absolute;
            top: -60px;
            right: -40px;
            width: 160px;
            height: 160px;
            background: radial-gradient(circle, rgba(6, 182, 212, 0.35), transparent 70%);
            opacity: 0.9;
            animation: glowPulse 6s ease-in-out infinite;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-top: 0.4rem;
        }
        .hero-subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin-top: 0.4rem;
        }
        .hero-tagline {
            color: var(--ink);
            font-size: 1.05rem;
            margin-top: 0.6rem;
        }
        .hero-meta {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.8rem;
        }
        .hero-panel {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(6, 182, 212, 0.2);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(8px);
        }
        .hero-panel h4 {
            margin: 0 0 0.5rem 0;
            font-size: 1rem;
        }
        .hero-panel ul {
            list-style: none;
            padding-left: 0;
            margin: 0;
            color: var(--muted);
        }
        .hero-panel li {
            padding: 0.3rem 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            background: linear-gradient(120deg, rgba(6, 182, 212, 0.2), rgba(34, 197, 94, 0.2));
            border: 1px solid rgba(6, 182, 212, 0.35);
            color: #0b3550;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .icon-inline {
            width: 14px;
            height: 14px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
            color: var(--accent);
        }
        .icon-inline svg {
            width: 14px;
            height: 14px;
            stroke: currentColor;
        }
        .icon-accent-2 { color: var(--accent-2); }
        .icon-accent-3 { color: var(--accent-3); }
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--ink);
            margin-bottom: 0.25rem;
        }
        .section-helper {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.8rem;
        }
        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
            border: 1px solid rgba(6, 182, 212, 0.14);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: var(--shadow);
            animation: fadeUp 0.6s ease both;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 22px 40px rgba(15, 23, 42, 0.16);
        }
        .metric-label {
            font-size: 0.78rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-number {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.3rem;
            font-weight: 600;
            margin-top: 0.4rem;
        }
        .metric-unit {
            font-size: 0.75rem;
            color: var(--muted);
            margin-top: 0.2rem;
        }
        .flow-card {
            position: relative;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), #f1fbff);
            border: 1px solid rgba(6, 182, 212, 0.18);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: var(--shadow);
            overflow: hidden;
        }
        .flow-card::after {
            content: "";
            position: absolute;
            inset: auto -30% -80% -30%;
            height: 120%;
            background: radial-gradient(circle, rgba(6, 182, 212, 0.18), transparent 60%);
        }
        .flow-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.8rem;
        }
        .flow-step {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.6rem 0.9rem;
            border-radius: 999px;
            border: 1px solid rgba(6, 182, 212, 0.2);
            background: #f8fdff;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .flow-step svg {
            width: 14px;
            height: 14px;
            stroke: currentColor;
        }
        .flow-step.active {
            background: linear-gradient(120deg, var(--accent), var(--accent-3));
            color: #ffffff;
            border-color: transparent;
            box-shadow: var(--glow);
            animation: pulseGlow 2.2s ease-in-out infinite;
        }
        .flow-line {
            flex: 1;
            height: 2px;
            background: linear-gradient(90deg, rgba(6, 182, 212, 0.2), rgba(34, 197, 94, 0.3));
            min-width: 40px;
        }
        .step-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid rgba(6, 182, 212, 0.16);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow);
        }
        .stButton>button {
            background: linear-gradient(120deg, var(--accent), var(--accent-2));
            color: #ffffff;
            border: none;
            padding: 0.6rem 1.4rem;
            border-radius: 999px;
            font-weight: 600;
            box-shadow: 0 12px 24px rgba(6, 182, 212, 0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 28px rgba(6, 182, 212, 0.32);
            color: #ffffff;
        }
        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
            color: var(--muted);
            background: rgba(148, 163, 184, 0.25);
            border-radius: 10px 10px 0 0;
            margin-right: 6px;
            padding: 8px 14px;
        }
        .stTabs [aria-selected="true"] {
            color: var(--ink);
            background: var(--panel);
            border: 1px solid var(--border);
            border-bottom: 3px solid var(--accent);
            box-shadow: 0 -10px 20px rgba(6, 182, 212, 0.12);
        }
        @keyframes gradientShift {
            0% { background-position: 0% 0%; }
            50% { background-position: 90% 70%; }
            100% { background-position: 0% 0%; }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes glowPulse {
            0% { transform: translateY(0); opacity: 0.6; }
            50% { transform: translateY(6px); opacity: 1; }
            100% { transform: translateY(0); opacity: 0.6; }
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 rgba(6, 182, 212, 0.0); }
            50% { box-shadow: 0 0 22px rgba(6, 182, 212, 0.35); }
            100% { box-shadow: 0 0 0 rgba(6, 182, 212, 0.0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _clean_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _parse_rates(text: str) -> list[float]:
    rates = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        rates.append(float(item))
    if not rates:
        raise ValueError("Provide at least one AI efficiency rate.")
    return rates


def _build_config(
    baseline: dict[str, Any],
    ai_rates: list[float],
    policy_rows: list[dict[str, Any]],
    ai_cost: float | None,
    policy_defaults: dict[str, Any],
    assumptions: list[str],
) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "ai_efficiency_rates": ai_rates,
        "policy": {
            "carbon_price_scenarios": policy_rows,
            "ai_implementation_cost_usd_yr": ai_cost,
            "policy_defaults": policy_defaults,
        },
        "assumptions": assumptions,
    }


def _format_number(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{decimals}f}"


def _format_range(min_value: Optional[float], max_value: Optional[float], decimals: int = 2) -> str:
    if min_value is None or max_value is None:
        return "n/a"
    return f"{min_value:,.{decimals}f} - {max_value:,.{decimals}f}"


def _animate_metric(container: st.delta_generator.DeltaGenerator, label: str, value: float, unit: str) -> None:
    steps = 10
    for step in range(1, steps + 1):
        current = value * step / steps
        container.markdown(
            _metric_card_html(label, _format_number(current), unit),
            unsafe_allow_html=True,
        )
        time.sleep(0.015)


def _metric_card_html(label: str, value_text: str, unit: str) -> str:
    return (
        "<div class=\"metric-card\">"
        f"<div class=\"metric-label\">{label}</div>"
        f"<div class=\"metric-number\">{value_text}</div>"
        f"<div class=\"metric-unit\">{unit}</div>"
        "</div>"
    )


def _render_metric_card(
    label: str,
    value_text: str,
    unit: str,
    animate_value: Optional[float] = None,
    animate: bool = False,
) -> None:
    container = st.empty()
    if animate and animate_value is not None:
        _animate_metric(container, label, animate_value, unit)
    container.markdown(_metric_card_html(label, value_text, unit), unsafe_allow_html=True)


def _display_exec_panel(summary: dict[str, Any], payload: dict[str, Any]) -> None:
    """Render the executive intelligence panel with animated metrics."""
    metadata = payload.get("metadata", {})
    baseline = metadata.get("baseline", {})

    baseline_co2 = baseline.get("co2_emissions_ton_yr")
    co2_reduced_max = summary["total_economic_benefit_usd_yr"].get("max")
    emissions_reduced_max = max(
        (row.get("emissions_reduced_ton_yr", 0.0) for row in payload.get("results", [])),
        default=None,
    )
    ai_range = summary.get("ai_efficiency_rate_range", {})
    carbon_range = summary.get("carbon_price_range_usd_per_ton", {})
    benefit_max = summary["total_economic_benefit_usd_yr"].get("max")

    signature = (
        baseline_co2,
        emissions_reduced_max,
        benefit_max,
        ai_range.get("min"),
        ai_range.get("max"),
        carbon_range.get("min"),
        carbon_range.get("max"),
    )
    animate = st.session_state.get("last_summary_signature") != signature
    st.session_state.last_summary_signature = signature

    st.markdown("<div class=\"section-title\">Executive Intelligence Panel</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-helper\">Key indicators summarizing baseline conditions and scenario outcomes.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class=\"metric-grid\">", unsafe_allow_html=True)
    _render_metric_card(
        "Baseline CO2",
        _format_number(baseline_co2, 1),
        "ton CO2 per year",
        animate_value=float(baseline_co2) if baseline_co2 is not None else None,
        animate=animate,
    )
    _render_metric_card(
        "AI efficiency range",
        _format_range(
            ai_range.get("min", 0.0) * 100 if ai_range.get("min") is not None else None,
            ai_range.get("max", 0.0) * 100 if ai_range.get("max") is not None else None,
            1,
        ),
        "percent",
    )
    _render_metric_card(
        "CO2 reduced (max)",
        _format_number(emissions_reduced_max, 1),
        "ton CO2 per year",
        animate_value=float(emissions_reduced_max) if emissions_reduced_max is not None else None,
        animate=animate,
    )
    _render_metric_card(
        "Economic benefit (max)",
        _format_number(benefit_max, 2),
        "USD per year",
        animate_value=float(benefit_max) if benefit_max is not None else None,
        animate=animate,
    )
    _render_metric_card(
        "Carbon price range",
        _format_range(carbon_range.get("min"), carbon_range.get("max"), 2),
        "USD per ton CO2",
    )
    _render_metric_card(
        "Scenario count",
        f"{summary['scenario_count']}",
        "total scenarios",
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _ensure_session_state() -> None:
    if "baseline" not in st.session_state:
        st.session_state.baseline = {
            "co2_emissions_ton_yr": 100.0,
            "fuel_cost_usd_yr": 1000.0,
            "vkt_km_yr": None,
        }
    if "ai_rates_text" not in st.session_state:
        st.session_state.ai_rates_text = "0.05,0.10,0.20"
    if "policy_defaults" not in st.session_state:
        st.session_state.policy_defaults = {
            "fuel_tax_rate": 0.0,
            "ai_subsidy_usd_yr": None,
            "congestion_charge_usd_yr": 0.0,
            "congestion_charge_usd_per_km": None,
        }
    if "policy_table" not in st.session_state:
        st.session_state.policy_table = pd.DataFrame(
            [
                {
                    "name": "Low",
                    "carbon_price_usd_per_ton": 25.0,
                    "fuel_tax_rate": 0.0,
                    "ai_subsidy_usd_yr": None,
                    "congestion_charge_usd_yr": 0.0,
                    "congestion_charge_usd_per_km": None,
                },
                {
                    "name": "High",
                    "carbon_price_usd_per_ton": 100.0,
                    "fuel_tax_rate": 0.0,
                    "ai_subsidy_usd_yr": None,
                    "congestion_charge_usd_yr": 0.0,
                    "congestion_charge_usd_per_km": None,
                },
            ]
        )
    if "assumptions_text" not in st.session_state:
        st.session_state.assumptions_text = (
            "AI efficiency is scenario-based\n"
            "Annual static evaluation\n"
            "Road transport only"
        )
    if "ai_cost" not in st.session_state:
        st.session_state.ai_cost = 0.0
    if "payload" not in st.session_state:
        st.session_state.payload = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "results_df" not in st.session_state:
        st.session_state.results_df = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def _run_evaluation() -> None:
    try:
        baseline = st.session_state.baseline
        vkt_value = baseline.get("vkt_km_yr")
        if vkt_value == 0.0:
            vkt_value = None
        baseline = {
            "co2_emissions_ton_yr": baseline["co2_emissions_ton_yr"],
            "fuel_cost_usd_yr": baseline["fuel_cost_usd_yr"],
            "vkt_km_yr": vkt_value,
        }
        ai_rates = _parse_rates(st.session_state.ai_rates_text)
        policy_defaults = st.session_state.policy_defaults
        policy_rows = [
            {key: _clean_value(value) for key, value in row.items()}
            for row in st.session_state.policy_table.to_dict(orient="records")
        ]
        assumptions = [
            line.strip()
            for line in st.session_state.assumptions_text.split("\n")
            if line.strip()
        ]
        ai_cost_value = st.session_state.ai_cost
        ai_cost_value = None if ai_cost_value == 0.0 else ai_cost_value

        config_dict = _build_config(
            baseline=baseline,
            ai_rates=ai_rates,
            policy_rows=policy_rows,
            ai_cost=ai_cost_value,
            policy_defaults=policy_defaults,
            assumptions=assumptions,
        )
        config = load_config_from_dict(config_dict)
        payload = build_results(config)
        summary = summarize_results(payload)

        results_df = pd.DataFrame(payload["results"]).copy()
        results_df["ai_efficiency_rate"] = results_df["ai_efficiency_rate"].astype(float)
        results_df["carbon_price_usd_per_ton"] = results_df["carbon_price_usd_per_ton"].astype(float)

        st.session_state.payload = payload
        st.session_state.summary = summary
        st.session_state.results_df = results_df
    except Exception as exc:
        st.error(str(exc))


def _build_context_payload(
    payload: dict[str, Any],
    summary: dict[str, Any],
    rag_context: Optional[list[dict[str, Any]]],
    memory_notes: Optional[list[dict[str, Any]]],
) -> dict[str, Any]:
    compact_rows = [
        {
            "policy_scenario_name": row.get("policy_scenario_name"),
            "ai_efficiency_rate": row.get("ai_efficiency_rate"),
            "carbon_price_usd_per_ton": row.get("carbon_price_usd_per_ton"),
            "fuel_tax_rate": row.get("fuel_tax_rate"),
            "ai_subsidy_usd_yr": row.get("ai_subsidy_usd_yr"),
            "congestion_charge_usd_yr": row.get("congestion_charge_usd_yr"),
            "congestion_charge_usd_per_km": row.get("congestion_charge_usd_per_km"),
            "emissions_reduced_ton_yr": row.get("emissions_reduced_ton_yr"),
            "fuel_cost_savings_usd_yr": row.get("fuel_cost_savings_usd_yr"),
            "carbon_cost_savings_usd_yr": row.get("carbon_cost_savings_usd_yr"),
            "fuel_tax_savings_usd_yr": row.get("fuel_tax_savings_usd_yr"),
            "congestion_charge_savings_usd_yr": row.get("congestion_charge_savings_usd_yr"),
            "total_economic_benefit_usd_yr": row.get("total_economic_benefit_usd_yr"),
            "marginal_abatement_cost_usd_per_ton": row.get("marginal_abatement_cost_usd_per_ton"),
        }
        for row in payload.get("results", [])
    ]
    return {
        "scope_constraints": [
            "Road transport only",
            "CO2 only",
            "Annual static evaluation",
            "AI efficiency is scenario-based, not predictive",
            "No causal claims, no forecasting, no optimization",
        ],
        "summary": summary,
        "scenario_table": compact_rows,
        "rag_context": rag_context or [],
        "memory_notes": memory_notes or [],
    }


def _groq_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        GROQ_API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": GROQ_USER_AGENT,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") if exc.fp else str(exc)
        raise RuntimeError(f"Groq API request failed: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Groq API request failed: {exc}") from exc

    parsed = json.loads(body)
    choices = parsed.get("choices", [])
    if not choices:
        raise RuntimeError("Groq API returned no choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        raise RuntimeError("Groq API returned empty content.")
    return content.strip()


def render_header() -> None:
    """Header establishes product identity and scope for credibility."""
    st.markdown(
        """
        <div class="hero">
            <div class="brand-card">
                <div class="chip">
                    <span class="icon-inline" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 3v3m0 12v3m9-9h-3M6 12H3m14.364-6.364l-2.121 2.121M8.757 15.243l-2.121 2.121m0-12.728l2.121 2.121m8.486 8.486l2.121 2.121" />
                        </svg>
                    </span>
                    AI policy intelligence
                </div>
                <div class="hero-title">EcoMoveAI</div>
                <div class="hero-tagline">AI-Enabled Economic Evaluation of Transport Decarbonization Policies</div>
                <div class="hero-subtitle">Policy-grade decision support for transport decarbonization</div>
                <div class="hero-meta">Mahbub Hassan</div>
            </div>
            <div class="hero-panel">
                <h4>Scope</h4>
                <ul>
                    <li>
                        <span class="icon-inline icon-accent-2" aria-hidden="true">
                            <svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 01.553-.894L9 2m0 18l6-3m-6 3V2m6 15l5.447-2.724A1 1 0 0021 13.382V4.618a1 1 0 00-.553-.894L15 1m0 16V1m0 0L9 4" />
                            </svg>
                        </span>
                        Road transport only
                    </li>
                    <li>
                        <span class="icon-inline icon-accent-3" aria-hidden="true">
                            <svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M20 16.5a4.5 4.5 0 00-4.5-4.5h-.75a6 6 0 10-11.25 2.25A4 4 0 006 22h9a5 5 0 005-5.5z" />
                            </svg>
                        </span>
                        CO2 only
                    </li>
                    <li>
                        <span class="icon-inline" aria-hidden="true">
                            <svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M12 6v6l4 2" />
                                <circle cx="12" cy="12" r="8" />
                            </svg>
                        </span>
                        Annual static evaluation
                    </li>
                    <li>
                        <span class="icon-inline icon-accent-2" aria-hidden="true">
                            <svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M9 3v2m6-2v2m-6 14v2m6-2v2M4 9H2m2 6H2m20-6h-2m2 6h-2M6 6h12v12H6z" />
                            </svg>
                        </span>
                        Scenario-based AI efficiency
                    </li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flow_section() -> None:
    """Explain the analytical chain from baseline to policy outcome."""
    st.markdown("<div class=\"section-title\">AI - Economics - Policy Flow</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-helper\">How the system transforms inputs into policy-relevant insights.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
            <div class="flow-card">
            <div class="flow-row">
                <div class="flow-step active">
                    <svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <path d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z" />
                    </svg>
                    Baseline
                </div>
                <div class="flow-line"></div>
                <div class="flow-step">
                    <svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <path d="M13 3L4 14h7l-1 7 9-11h-7z" />
                    </svg>
                    AI efficiency
                </div>
                <div class="flow-line"></div>
                <div class="flow-step">
                    <svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <path d="M4 19V5m5 14V9m5 10V7m5 12V11" />
                    </svg>
                    Economic valuation
                </div>
                <div class="flow-line"></div>
                <div class="flow-step">
                    <svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <path d="M12 3l8 4v5c0 5-3.5 9-8 11-4.5-2-8-6-8-11V7l8-4z" />
                    </svg>
                    Policy outcome
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    """Overview tab focuses on executive insight density."""
    st.markdown("<div class=\"section-title\">Executive Overview</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-helper\">Core signals and top scenarios for fast situational awareness.</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.payload is None:
        st.info("Build scenarios and run the evaluation to populate the dashboard.")
        return

    _display_exec_panel(st.session_state.summary, st.session_state.payload)
    st.write("")

    render_flow_section()
    st.write("")

    results_df = st.session_state.results_df
    if results_df is None:
        return

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("<div class=\"section-title\">Top Scenarios</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class=\"section-helper\">Highest total economic benefit across AI efficiencies.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            results_df.sort_values("total_economic_benefit_usd_yr", ascending=False).head(6),
            use_container_width=True,
        )
    with col2:
        st.markdown("<div class=\"section-title\">Assumptions</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class=\"section-helper\">Documented evaluation context for reproducibility.</div>",
            unsafe_allow_html=True,
        )
        st.code(st.session_state.assumptions_text, language="text")

        st.markdown("<div class=\"section-title\">Policy Defaults</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class=\"section-helper\">Default policy parameters applied to scenarios.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame([st.session_state.policy_defaults]), use_container_width=True)


def render_builder() -> None:
    """Step-by-step configuration to reduce cognitive load."""
    st.markdown("<div class=\"section-title\">Scenario Builder</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-helper\">Enter baseline data, AI assumptions, and policy instruments.</div>",
        unsafe_allow_html=True,
    )

    step_tabs = st.tabs(
        [
            "1 Baseline",
            "2 AI Efficiency",
            "3 Policy Instruments",
            "4 Assumptions and Costs",
            "5 Review and Run",
        ]
    )

    with step_tabs[0]:
        st.markdown("<div class=\"step-card\">Baseline inputs</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.baseline["co2_emissions_ton_yr"] = st.number_input(
                "CO2 emissions (ton/yr)",
                min_value=0.0,
                value=float(st.session_state.baseline["co2_emissions_ton_yr"]),
                step=1.0,
            )
        with col2:
            st.session_state.baseline["fuel_cost_usd_yr"] = st.number_input(
                "Fuel cost (USD/yr)",
                min_value=0.0,
                value=float(st.session_state.baseline["fuel_cost_usd_yr"]),
                step=10.0,
            )
        with col3:
            vkt_value = st.session_state.baseline.get("vkt_km_yr") or 0.0
            st.session_state.baseline["vkt_km_yr"] = st.number_input(
                "VKT (km/yr, optional)",
                min_value=0.0,
                value=float(vkt_value),
                step=100.0,
            )

    with step_tabs[1]:
        st.markdown("<div class=\"step-card\">AI efficiency rates</div>", unsafe_allow_html=True)
        st.session_state.ai_rates_text = st.text_input(
            "Efficiency rates (comma separated)",
            value=st.session_state.ai_rates_text,
        )

    with step_tabs[2]:
        st.markdown("<div class=\"step-card\">Policy defaults</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        with cols[0]:
            st.session_state.policy_defaults["fuel_tax_rate"] = st.number_input(
                "Fuel tax rate (0-1)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.policy_defaults["fuel_tax_rate"] or 0.0),
                step=0.01,
            )
        with cols[1]:
            default_ai_subsidy = st.session_state.policy_defaults.get("ai_subsidy_usd_yr")
            default_ai_subsidy = 0.0 if default_ai_subsidy is None else float(default_ai_subsidy)
            st.session_state.policy_defaults["ai_subsidy_usd_yr"] = st.number_input(
                "AI subsidy (USD/yr, optional)",
                min_value=0.0,
                value=default_ai_subsidy,
                step=10.0,
            )
        with cols[2]:
            st.session_state.policy_defaults["congestion_charge_usd_yr"] = st.number_input(
                "Congestion charge (USD/yr)",
                min_value=0.0,
                value=float(st.session_state.policy_defaults["congestion_charge_usd_yr"] or 0.0),
                step=10.0,
            )
        with cols[3]:
            default_congestion_per_km = st.session_state.policy_defaults.get("congestion_charge_usd_per_km")
            default_congestion_per_km = 0.0 if default_congestion_per_km is None else float(default_congestion_per_km)
            st.session_state.policy_defaults["congestion_charge_usd_per_km"] = st.number_input(
                "Congestion charge (USD/km, optional)",
                min_value=0.0,
                value=default_congestion_per_km,
                step=0.01,
            )

        st.markdown("<div class=\"step-card\">Policy scenarios</div>", unsafe_allow_html=True)
        policy_table = st.data_editor(
            st.session_state.policy_table,
            use_container_width=True,
            num_rows="dynamic",
        )
        st.session_state.policy_table = policy_table

    with step_tabs[3]:
        st.markdown("<div class=\"step-card\">Assumptions and costs</div>", unsafe_allow_html=True)
        st.session_state.ai_cost = st.number_input(
            "AI implementation cost (USD/yr, optional)",
            min_value=0.0,
            value=float(st.session_state.ai_cost or 0.0),
            step=10.0,
        )
        st.session_state.assumptions_text = st.text_area(
            "Assumptions (one per line)",
            value=st.session_state.assumptions_text,
            height=140,
        )

    with step_tabs[4]:
        st.markdown("<div class=\"step-card\">Review configuration</div>", unsafe_allow_html=True)
        baseline_preview = pd.DataFrame([st.session_state.baseline])
        st.markdown("Baseline")
        st.dataframe(baseline_preview, use_container_width=True)
        st.markdown("Policy scenarios")
        st.dataframe(st.session_state.policy_table, use_container_width=True)
        st.markdown("AI efficiency rates")
        st.code(st.session_state.ai_rates_text, language="text")
        st.markdown("Assumptions")
        st.code(st.session_state.assumptions_text, language="text")

        st.button("Run Evaluation", on_click=_run_evaluation)
        if st.session_state.payload is not None:
            st.success("Evaluation complete. Navigate to Results or Reports.")


def render_results() -> None:
    st.markdown("<div class=\"section-title\">Results</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-helper\">Interactive charts and scenario tables for policy comparison.</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.payload is None:
        st.info("Run the evaluation to see results.")
        return

    results_df = st.session_state.results_df
    if results_df is None:
        return

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("<div class=\"section-title\">Benefit vs Carbon Price</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class=\"section-helper\">Tracks benefit sensitivity to carbon pricing.</div>",
            unsafe_allow_html=True,
        )
        fig1 = px.line(
            results_df,
            x="carbon_price_usd_per_ton",
            y="total_economic_benefit_usd_yr",
            color="ai_efficiency_rate",
            markers=True,
            labels={
                "carbon_price_usd_per_ton": "Carbon price (USD/ton CO2)",
                "total_economic_benefit_usd_yr": "Total economic benefit (USD/yr)",
                "ai_efficiency_rate": "AI efficiency",
            },
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("<div class=\"section-title\">Benefit Heatmap</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class=\"section-helper\">Comparison across policy and AI efficiency grids.</div>",
            unsafe_allow_html=True,
        )
        pivot = results_df.pivot_table(
            index="ai_efficiency_rate",
            columns="carbon_price_usd_per_ton",
            values="total_economic_benefit_usd_yr",
            aggfunc="mean",
        )
        fig2 = px.imshow(
            pivot,
            labels=dict(x="Carbon price", y="AI efficiency", color="Benefit"),
            aspect="auto",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns([1, 1])
    with col3:
        st.markdown("<div class=\"section-title\">Emissions Reduced</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class=\"section-helper\">Average emissions reduction by AI efficiency.</div>",
            unsafe_allow_html=True,
        )
        emissions_df = results_df.groupby("ai_efficiency_rate", as_index=False)[
            "emissions_reduced_ton_yr"
        ].mean()
        fig3 = px.bar(
            emissions_df,
            x="ai_efficiency_rate",
            y="emissions_reduced_ton_yr",
            labels={
                "ai_efficiency_rate": "AI efficiency",
                "emissions_reduced_ton_yr": "Emissions reduced (ton CO2/yr)",
            },
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class=\"section-title\">Savings Breakdown</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class=\"section-helper\">Economic value composition across instruments.</div>",
            unsafe_allow_html=True,
        )
        breakdown_cols = [
            "carbon_cost_savings_usd_yr",
            "fuel_cost_savings_usd_yr",
            "fuel_tax_savings_usd_yr",
            "congestion_charge_savings_usd_yr",
        ]
        breakdown = results_df.copy()
        breakdown = breakdown[breakdown_cols + ["policy_scenario_name", "ai_efficiency_rate"]]
        breakdown = breakdown.melt(
            id_vars=["policy_scenario_name", "ai_efficiency_rate"],
            value_vars=breakdown_cols,
            var_name="component",
            value_name="value",
        )
        fig4 = px.bar(
            breakdown,
            x="policy_scenario_name",
            y="value",
            color="component",
            labels={
                "policy_scenario_name": "Policy scenario",
                "value": "USD/yr",
            },
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<div class=\"section-title\">Scenario Table</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-helper\">Full scenario matrix for auditing and export.</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(results_df, use_container_width=True)


def render_policy_brief() -> None:
    st.markdown("<div class=\"section-title\">Policy Brief</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-helper\">Summary statements for decision meetings.</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.payload is None:
        st.info("Run the evaluation to generate a policy brief.")
        return

    summary = st.session_state.summary
    best = summary["best_scenario"]
    worst = summary["worst_scenario"]

    st.markdown(
        "- Best scenario: {name} at AI efficiency {rate} with total benefit {benefit} USD/yr".format(
            name=best.get("policy_scenario_name"),
            rate=best.get("ai_efficiency_rate"),
            benefit=_format_number(best.get("total_economic_benefit_usd_yr"), 2),
        )
    )
    st.markdown(
        "- Worst scenario: {name} at AI efficiency {rate} with total benefit {benefit} USD/yr".format(
            name=worst.get("policy_scenario_name"),
            rate=worst.get("ai_efficiency_rate"),
            benefit=_format_number(worst.get("total_economic_benefit_usd_yr"), 2),
        )
    )
    st.markdown(
        "- Carbon price range: {min} to {max} USD/ton CO2".format(
            min=_format_number(summary["carbon_price_range_usd_per_ton"]["min"], 2),
            max=_format_number(summary["carbon_price_range_usd_per_ton"]["max"], 2),
        )
    )

    results_df = st.session_state.results_df
    if results_df is None:
        return

    st.markdown("<div class=\"section-title\">Top Scenarios</div>", unsafe_allow_html=True)
    st.dataframe(
        results_df.sort_values("total_economic_benefit_usd_yr", ascending=False).head(5),
        use_container_width=True,
    )


def render_reports() -> None:
    st.markdown("<div class=\"section-title\">Reports</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-helper\">Export decision-ready tables and figures.</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.payload is None:
        st.info("Run the evaluation to generate exports.")
        return

    payload = st.session_state.payload
    results_df = st.session_state.results_df
    if results_df is None:
        return

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_data, file_name="scenario_results.csv")
    json_data = json.dumps(payload, indent=2).encode("utf-8")
    st.download_button("Download JSON", data=json_data, file_name="results.json")

    st.write("")
    if st.button("Build report assets"):
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_report_assets(payload, Path(tmpdir))
            st.success("Report assets generated.")
            for label, path in outputs.items():
                st.download_button(
                    f"Download {label}",
                    data=Path(path).read_bytes(),
                    file_name=path.name,
                )


def render_llm_studio() -> None:
    st.markdown("<div class=\"section-title\">LLM Studio</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-helper\">Interpret results with controlled, context-aware AI chat.</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.payload is None:
        st.info("Run the evaluation first to unlock context-aware chat.")
        return

    col1, col2 = st.columns([1.4, 1])
    with col2:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if api_key:
            st.caption("Groq API key loaded from Streamlit secrets.")
        else:
            st.warning("Groq API key not found in Streamlit secrets.")
        model = st.text_input("Model", value="llama-3.1-8b-instant")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        max_tokens = st.slider("Max tokens", 200, 1200, 700, 50)

        st.markdown("#### Optional RAG")
        rag_dir = st.text_input("RAG directory", value="data/knowledge_base")
        rag_top_k = st.slider("RAG top-k", 1, 6, 3, 1)
        rag_query = st.text_input(
            "RAG query",
            value="road transport decarbonization policy evaluation AI efficiency",
        )

        st.markdown("#### Memory Notes")
        memory_file = st.text_input("Memory file", value="data/memory/notes.jsonl")
        include_memory = st.checkbox("Include memory notes", value=True)
        if st.checkbox("Add a memory note"):
            note_title = st.text_input("Note title")
            note_content = st.text_area("Note content", height=80)
            note_tags = st.text_input("Tags (comma separated)")
            if st.button("Save note"):
                append_memory_note(
                    Path(memory_file),
                    title=note_title,
                    content=note_content,
                    tags=[tag.strip() for tag in note_tags.split(",") if tag.strip()],
                )
                st.success("Memory note saved.")

        if st.button("Reset chat"):
            st.session_state.chat_messages = []

    with col1:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask about the policy results")
        if user_input:
            if not api_key:
                st.error("Groq API key is required.")
                return

            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            rag_context = None
            if rag_dir:
                try:
                    documents = load_documents_from_dir(Path(rag_dir))
                    index = build_index(documents)
                    rag_context = search(index, rag_query, top_k=rag_top_k)
                except Exception as exc:
                    st.warning(f"RAG unavailable: {exc}")

            memory_notes = []
            if include_memory:
                memory_notes = format_memory_notes(
                    load_memory_notes(Path(memory_file), limit=5)
                )

            context_payload = _build_context_payload(
                payload=st.session_state.payload,
                summary=st.session_state.summary,
                rag_context=rag_context,
                memory_notes=memory_notes,
            )

            system_message = (
                "You are a policy analyst. Interpret the data without new calculations. "
                "No forecasts, no causality, no optimization. Do not reveal chain-of-thought. "
                "Provide concise, structured answers."
            )
            context_message = "Context data:\n" + json.dumps(context_payload, indent=2)

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": context_message},
            ] + st.session_state.chat_messages

            try:
                response = _groq_chat_completion(
                    messages=messages,
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                st.error(str(exc))
                return

            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)


def render_help() -> None:
    st.markdown("<div class=\"section-title\">Help</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-helper\">Short guide to configure, run, and interpret results.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "### Quick start\n"
        "- Enter baseline emissions and fuel cost.\n"
        "- Add AI efficiency rates and policy scenarios.\n"
        "- Click Run Evaluation to generate charts and tables.\n"
        "- Use the Reports tab for CSV and PNG exports."
    )
    st.markdown(
        "### Policy instruments\n"
        "- Carbon price (USD per ton CO2)\n"
        "- Fuel tax rate (share of fuel cost)\n"
        "- AI subsidy (USD per year)\n"
        "- Congestion charge (flat and per km)"
    )
    st.markdown(
        "### LLM chat\n"
        "- Provide a Groq API key.\n"
        "- Add local RAG documents in data/knowledge_base.\n"
        "- Add memory notes to keep policy context consistent."
    )
    st.markdown("### CLI quick reference")
    st.code(
        "python main.py --config path/to/config.json --output results.json --format json\n"
        "python main.py --config path/to/config.json --output results.csv --format csv\n"
        "python main.py --config path/to/config.json --output results.json --format json --report-dir reports",
        language="bash",
    )


def render_first_time_guide() -> None:
    st.markdown("<div class=\"section-title\">First-Time Guide</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-helper\">Starter values and guardrails for each input parameter.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class=\"step-card\">Use these starter values</div>", unsafe_allow_html=True)
    guide_rows = [
        {
            "Parameter": "CO2 emissions (ton/yr)",
            "First-time value": "100",
            "Notes": "Replace with your baseline inventory; keep non-negative.",
        },
        {
            "Parameter": "Fuel cost (USD/yr)",
            "First-time value": "1000",
            "Notes": "Use your baseline spend; keep non-negative.",
        },
        {
            "Parameter": "VKT (km/yr, optional)",
            "First-time value": "Leave blank",
            "Notes": "Only needed for per-km congestion charge.",
        },
        {
            "Parameter": "AI efficiency rates",
            "First-time value": "0.05, 0.10, 0.20",
            "Notes": "Values must be between 0 and 1 (5%-20% starter range).",
        },
        {
            "Parameter": "Fuel tax rate (0-1)",
            "First-time value": "0.00",
            "Notes": "Try 0.05 or 0.10 for sensitivity checks.",
        },
        {
            "Parameter": "Carbon price scenarios (USD/ton CO2)",
            "First-time value": "Low: 25, High: 100",
            "Notes": "Start with 2 scenarios; add more for richer comparisons.",
        },
        {
            "Parameter": "AI subsidy (USD/yr, optional)",
            "First-time value": "0",
            "Notes": "Set only if a subsidy is part of the policy.",
        },
        {
            "Parameter": "Congestion charge (USD/yr)",
            "First-time value": "0",
            "Notes": "Flat annual charge; keep 0 if unused.",
        },
        {
            "Parameter": "Congestion charge (USD/km, optional)",
            "First-time value": "0",
            "Notes": "Requires VKT; keep 0 if unused.",
        },
        {
            "Parameter": "AI implementation cost (USD/yr, optional)",
            "First-time value": "0",
            "Notes": "Set to annual program cost if known.",
        },
    ]
    st.dataframe(pd.DataFrame(guide_rows), use_container_width=True)
    st.info("Tip: Start with defaults, run once, then refine inputs using your data.")


def main() -> None:
    _apply_theme()
    _ensure_session_state()

    render_header()
    st.write("")

    tabs = st.tabs([
        "Overview",
        "Scenarios",
        "First-Time Guide",
        "Results",
        "Policy Brief",
        "Reports",
        "LLM Studio",
        "Help",
    ])
    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_builder()
    with tabs[2]:
        render_first_time_guide()
    with tabs[3]:
        render_results()
    with tabs[4]:
        render_policy_brief()
    with tabs[5]:
        render_reports()
    with tabs[6]:
        render_llm_studio()
    with tabs[7]:
        render_help()


if __name__ == "__main__":
    main()
