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
    """Apply a premium, sophisticated dark theme with professional aesthetics."""
    st.set_page_config(
        page_title="EcoMoveAI | Transport Decarbonization Intelligence",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');
        
        :root {
            /* Premium Dark Theme - Sophisticated Navy/Slate */
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --bg-elevated: #1a2332;
            
            /* Text Colors */
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            
            /* Accent Colors - Emerald/Teal Focus */
            --accent-primary: #10b981;
            --accent-secondary: #06b6d4;
            --accent-tertiary: #8b5cf6;
            --accent-warning: #f59e0b;
            --accent-success: #22c55e;
            --accent-danger: #ef4444;
            
            /* Gradients */
            --gradient-primary: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #8b5cf6 100%);
            --gradient-secondary: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            --gradient-glow: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 182, 212, 0.15), rgba(139, 92, 246, 0.1));
            
            /* Effects */
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
            --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.5);
            --shadow-glow: 0 0 60px rgba(16, 185, 129, 0.2);
            --border-subtle: 1px solid rgba(148, 163, 184, 0.1);
            --border-accent: 1px solid rgba(16, 185, 129, 0.3);
            
            /* Glassmorphism */
            --glass-bg: rgba(30, 41, 59, 0.7);
            --glass-border: rgba(148, 163, 184, 0.1);
            --glass-blur: blur(20px);
        }
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-primary);
            -webkit-font-smoothing: antialiased;
        }
        
        /* Main Container Background */
        .stApp {
            background: var(--bg-primary);
            background-image: 
                radial-gradient(ellipse at 20% 0%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 0%, rgba(6, 182, 212, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 100%, rgba(139, 92, 246, 0.05) 0%, transparent 50%);
        }
        
        .appview-container {
            background: transparent;
        }
        
        .main .block-container {
            padding: 2rem 3rem 4rem 3rem;
            max-width: 1400px;
        }
        
        /* Hero Section */
        .hero-section {
            display: grid;
            grid-template-columns: 1.8fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
            animation: slideUp 0.6s ease-out;
        }
        
        @media (max-width: 1024px) {
            .hero-section { grid-template-columns: 1fr; }
        }
        
        .hero-main {
            position: relative;
            background: var(--glass-bg);
            backdrop-filter: var(--glass-blur);
            border: var(--border-subtle);
            border-radius: 24px;
            padding: 2rem 2.5rem;
            overflow: hidden;
        }
        
        .hero-main::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-primary);
        }
        
        .hero-main::after {
            content: "";
            position: absolute;
            top: -100px;
            right: -100px;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.15), transparent 70%);
            animation: pulse 4s ease-in-out infinite;
        }
        
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 100px;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--accent-primary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 1rem;
        }
        
        .hero-badge svg {
            width: 14px;
            height: 14px;
        }
        
        .hero-title {
            font-size: 2.75rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            line-height: 1.1;
        }
        
        .hero-tagline {
            font-size: 1.1rem;
            color: var(--text-secondary);
            font-weight: 400;
            margin-bottom: 0.75rem;
            line-height: 1.5;
        }
        
        .hero-description {
            font-size: 0.95rem;
            color: var(--text-muted);
            line-height: 1.6;
        }
        
        .hero-author {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            margin-top: 1.25rem;
            padding: 8px 16px;
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 100px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .hero-author-avatar {
            width: 28px;
            height: 28px;
            background: var(--gradient-primary);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.7rem;
            color: white;
        }
        
        /* Scope Panel */
        .scope-panel {
            background: var(--glass-bg);
            backdrop-filter: var(--glass-blur);
            border: var(--border-subtle);
            border-radius: 20px;
            padding: 1.5rem;
        }
        
        .scope-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .scope-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .scope-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.08);
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        .scope-item:last-child {
            border-bottom: none;
        }
        
        .scope-icon {
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            font-size: 1rem;
        }
        
        .scope-icon.emerald { background: rgba(16, 185, 129, 0.15); }
        .scope-icon.cyan { background: rgba(6, 182, 212, 0.15); }
        .scope-icon.violet { background: rgba(139, 92, 246, 0.15); }
        .scope-icon.amber { background: rgba(245, 158, 11, 0.15); }
        
        /* Section Styling */
        .section-header {
            margin-bottom: 1rem;
        }
        
        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .section-title-icon {
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(16, 185, 129, 0.15);
            border-radius: 8px;
            font-size: 0.9rem;
        }
        
        .section-subtitle {
            font-size: 0.9rem;
            color: var(--text-muted);
        }
        
        /* Process Flow */
        .process-flow {
            background: var(--glass-bg);
            backdrop-filter: var(--glass-blur);
            border: var(--border-subtle);
            border-radius: 20px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
        }
        
        .flow-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        .flow-step {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 20px;
            background: var(--bg-tertiary);
            border: var(--border-subtle);
            border-radius: 100px;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-secondary);
            transition: all 0.3s ease;
        }
        
        .flow-step:hover {
            background: rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.3);
            color: var(--accent-primary);
            transform: translateY(-2px);
        }
        
        .flow-step.active {
            background: var(--gradient-primary);
            border-color: transparent;
            color: white;
            box-shadow: 0 8px 30px rgba(16, 185, 129, 0.3);
        }
        
        .flow-step-icon {
            font-size: 1.1rem;
        }
        
        .flow-arrow {
            color: var(--text-muted);
            font-size: 1.2rem;
        }
        
        /* Metric Cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        
        .metric-card {
            background: var(--glass-bg);
            backdrop-filter: var(--glass-blur);
            border: var(--border-subtle);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--gradient-primary);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            border-color: rgba(16, 185, 129, 0.3);
            box-shadow: var(--shadow-glow);
        }
        
        .metric-card:hover::before {
            opacity: 1;
        }
        
        .metric-label {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-family: 'Fira Code', monospace;
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }
        
        .metric-unit {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        
        /* Cards and Panels */
        .card {
            background: var(--glass-bg);
            backdrop-filter: var(--glass-blur);
            border: var(--border-subtle);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        }
        
        .card-title {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        /* Buttons */
        .stButton > button {
            background: var(--gradient-primary) !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 2rem !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            box-shadow: 0 8px 30px rgba(16, 185, 129, 0.25) !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 40px rgba(16, 185, 129, 0.35) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0) !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: var(--bg-secondary);
            padding: 6px;
            border-radius: 14px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            border-radius: 10px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            color: var(--text-muted) !important;
            border: none !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--gradient-primary) !important;
            color: white !important;
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            padding: 1.5rem 0;
        }
        
        /* Input Fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > div {
            background: var(--bg-secondary) !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
            padding: 0.75rem 1rem !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: var(--accent-primary) !important;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15) !important;
        }
        
        .stTextInput label,
        .stNumberInput label,
        .stTextArea label,
        .stSelectbox label {
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
        }
        
        /* Data Editor / Tables */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }
        
        [data-testid="stDataFrameResizable"] {
            background: var(--bg-secondary);
            border: var(--border-subtle);
            border-radius: 12px;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--bg-secondary) !important;
            border-right: 1px solid rgba(148, 163, 184, 0.1) !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: var(--text-secondary);
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: var(--bg-tertiary) !important;
            border-radius: 12px !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }
        
        .streamlit-expanderContent {
            background: var(--bg-secondary) !important;
            border: var(--border-subtle) !important;
            border-top: none !important;
            border-radius: 0 0 12px 12px !important;
        }
        
        /* Charts */
        .js-plotly-plot .plotly .main-svg {
            background: transparent !important;
        }
        
        /* Animations */
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 0.5;
                transform: scale(1);
            }
            50% {
                opacity: 0.8;
                transform: scale(1.05);
            }
        }
        
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--bg-tertiary);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
        
        /* Hide Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Status Messages */
        .stSuccess {
            background: rgba(34, 197, 94, 0.1) !important;
            border: 1px solid rgba(34, 197, 94, 0.3) !important;
            border-radius: 12px !important;
        }
        
        .stError {
            background: rgba(239, 68, 68, 0.1) !important;
            border: 1px solid rgba(239, 68, 68, 0.3) !important;
            border-radius: 12px !important;
        }
        
        .stWarning {
            background: rgba(245, 158, 11, 0.1) !important;
            border: 1px solid rgba(245, 158, 11, 0.3) !important;
            border-radius: 12px !important;
        }
        
        .stInfo {
            background: rgba(6, 182, 212, 0.1) !important;
            border: 1px solid rgba(6, 182, 212, 0.3) !important;
            border-radius: 12px !important;
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
        f"<div class=\"metric-value\">{value_text}</div>"
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

    st.markdown("<div class=\"section-title\"><span class=\"section-title-icon\">📈</span> Executive Intelligence Panel</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class=\"section-subtitle\">Key indicators summarizing baseline conditions and scenario outcomes.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class=\"metrics-grid\">", unsafe_allow_html=True)
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
        <div class="hero-section">
            <div class="hero-main">
                <div class="hero-badge">
                    🌿 AI Policy Intelligence Platform
                </div>
                <div class="hero-title">EcoMoveAI</div>
                <div class="hero-tagline">
                    AI-Enabled Economic Evaluation of Transport Decarbonization Policies
                </div>
                <div class="hero-description">
                    Policy-grade decision support system for analyzing the economic impacts of 
                    AI-enabled efficiency improvements in road transport CO₂ emissions.
                </div>
                <div class="hero-author">
                    <div class="hero-author-avatar">MH</div>
                    <span>Mahbub Hassan • Chulalongkorn University</span>
                </div>
            </div>
            <div class="scope-panel">
                <div class="scope-title">
                    📋 Analysis Scope
                </div>
                <ul class="scope-list">
                    <li class="scope-item">
                        <div class="scope-icon emerald">🚗</div>
                        <span>Road transport only</span>
                    </li>
                    <li class="scope-item">
                        <div class="scope-icon cyan">☁️</div>
                        <span>CO₂ emissions focus</span>
                    </li>
                    <li class="scope-item">
                        <div class="scope-icon violet">📅</div>
                        <span>Annual static evaluation</span>
                    </li>
                    <li class="scope-item">
                        <div class="scope-icon amber">🤖</div>
                        <span>Scenario-based AI efficiency</span>
                    </li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flow_section() -> None:
    """Explain the analytical chain from baseline to policy outcome."""
    st.markdown(
        """
        <div class="process-flow">
            <div class="flow-container">
                <div class="flow-step active">
                    <span class="flow-step-icon">📊</span>
                    Baseline Data
                </div>
                <span class="flow-arrow">→</span>
                <div class="flow-step">
                    <span class="flow-step-icon">🤖</span>
                    AI Efficiency
                </div>
                <span class="flow-arrow">→</span>
                <div class="flow-step">
                    <span class="flow-step-icon">💰</span>
                    Economic Valuation
                </div>
                <span class="flow-arrow">→</span>
                <div class="flow-step">
                    <span class="flow-step-icon">🎯</span>
                    Policy Outcome
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
