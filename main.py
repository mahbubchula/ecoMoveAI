"""
EcoMoveAI - Reproducible Evaluation Entrypoint

Purpose
-------
Runs deterministic, scenario-based economic evaluations for road-transport
CO2 emissions under policy scenarios defined in a user-provided config.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from core.ai_scenarios import AIScenario
from core.baseline import BaselineInputs, BaselineModel
from core.economics import CarbonPriceScenario, EconomicEvaluator, EconomicInputs
from core.memory import append_memory_note, format_memory_notes, load_memory_notes
from core.rag import build_index, load_documents_from_dir, search
from core.reporting import write_report_assets

UNITS = {
    "co2_emissions_ton_yr": "ton CO2/year",
    "fuel_cost_usd_yr": "USD/year",
    "vkt_km_yr": "km/year",
    "carbon_price_usd_per_ton": "USD/ton CO2",
    "ai_implementation_cost_usd_yr": "USD/year",
    "fuel_tax_rate": "share of fuel cost",
    "ai_subsidy_usd_yr": "USD/year",
    "congestion_charge_usd_yr": "USD/year",
    "congestion_charge_usd_per_km": "USD/km",
    "net_ai_cost_usd_yr": "USD/year",
    "marginal_abatement_cost_usd_per_ton": "USD/ton CO2",
}

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_USER_AGENT = "EcoMoveAI/1.0"

CSV_FIELDS = [
    "policy_scenario_name",
    "ai_efficiency_rate",
    "carbon_price_usd_per_ton",
    "fuel_tax_rate",
    "ai_subsidy_usd_yr",
    "congestion_charge_usd_yr",
    "congestion_charge_usd_per_km",
    "baseline_co2_emissions_ton_yr",
    "baseline_fuel_cost_usd_yr",
    "baseline_vkt_km_yr",
    "emissions_reduced_ton_yr",
    "new_emissions_ton_yr",
    "fuel_cost_savings_usd_yr",
    "new_fuel_cost_usd_yr",
    "baseline_carbon_cost_usd_yr",
    "new_carbon_cost_usd_yr",
    "carbon_cost_savings_usd_yr",
    "baseline_fuel_tax_cost_usd_yr",
    "new_fuel_tax_cost_usd_yr",
    "fuel_tax_savings_usd_yr",
    "baseline_congestion_charge_usd_yr",
    "new_congestion_charge_usd_yr",
    "congestion_charge_savings_usd_yr",
    "ai_implementation_cost_usd_yr",
    "net_ai_cost_usd_yr",
    "total_economic_benefit_usd_yr",
    "marginal_abatement_cost_usd_per_ton",
]


def _require_non_negative(name: str, value: float) -> None:
    if value is None:
        raise ValueError(f"{name} must not be None.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative. Got: {value}")


def _require_non_empty_str(name: str, value: str) -> None:
    if value is None or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_list_non_empty(name: str, values: Any) -> list[Any]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list.")
    return values


def _require_allowed_keys(
    section_name: str,
    data: dict[str, Any],
    required: set[str],
    optional: set[str],
) -> None:
    keys = set(data.keys())
    missing = required - keys
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{section_name} is missing required keys: {missing_list}")
    unknown = keys - required - optional
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"{section_name} has unsupported keys: {unknown_list}")


def _to_float(name: str, value: Any) -> float:
    if value is None:
        raise ValueError(f"{name} must not be None.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number. Got: {value}") from exc


def _parse_baseline(data: Any) -> BaselineInputs:
    if not isinstance(data, dict):
        raise ValueError("baseline must be an object.")

    _require_allowed_keys(
        "baseline",
        data,
        required={"co2_emissions_ton_yr", "fuel_cost_usd_yr"},
        optional={"vkt_km_yr"},
    )

    co2_emissions = _to_float("baseline.co2_emissions_ton_yr", data["co2_emissions_ton_yr"])
    fuel_cost = _to_float("baseline.fuel_cost_usd_yr", data["fuel_cost_usd_yr"])
    vkt = data.get("vkt_km_yr")
    vkt_value = None if vkt is None else _to_float("baseline.vkt_km_yr", vkt)

    inputs = BaselineInputs(
        co2_emissions_ton_yr=co2_emissions,
        fuel_cost_usd_yr=fuel_cost,
        vkt_km_yr=vkt_value,
    )
    inputs.validate()
    return inputs


def _parse_assumptions(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("assumptions must be a list of strings.")
    assumptions: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("assumptions entries must be strings.")
        _require_non_empty_str("assumptions entry", item)
        assumptions.append(item.strip())
    return assumptions


def _parse_policy(
    data: Any,
) -> tuple[list[CarbonPriceScenario], Optional[float], dict[str, Any]]:
    if not isinstance(data, dict):
        raise ValueError("policy must be an object.")

    _require_allowed_keys(
        "policy",
        data,
        required={"carbon_price_scenarios"},
        optional={
            "ai_implementation_cost_usd_yr",
            "policy_defaults",
        },
    )

    defaults = data.get("policy_defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("policy_defaults must be an object.")
    _require_allowed_keys(
        "policy_defaults",
        defaults,
        required=set(),
        optional={
            "fuel_tax_rate",
            "ai_subsidy_usd_yr",
            "congestion_charge_usd_yr",
            "congestion_charge_usd_per_km",
        },
    )

    default_fuel_tax_rate = _to_float(
        "policy_defaults.fuel_tax_rate", defaults.get("fuel_tax_rate", 0.0)
    )
    default_ai_subsidy = defaults.get("ai_subsidy_usd_yr")
    default_ai_subsidy_value = (
        None
        if default_ai_subsidy is None
        else _to_float("policy_defaults.ai_subsidy_usd_yr", default_ai_subsidy)
    )
    default_congestion_charge = _to_float(
        "policy_defaults.congestion_charge_usd_yr",
        defaults.get("congestion_charge_usd_yr", 0.0),
    )
    default_congestion_per_km = defaults.get("congestion_charge_usd_per_km")
    default_congestion_per_km_value = (
        None
        if default_congestion_per_km is None
        else _to_float(
            "policy_defaults.congestion_charge_usd_per_km",
            default_congestion_per_km,
        )
    )

    scenarios_raw = _require_list_non_empty("policy.carbon_price_scenarios", data["carbon_price_scenarios"])
    scenarios: list[CarbonPriceScenario] = []
    seen_names: set[str] = set()
    for scenario in scenarios_raw:
        if not isinstance(scenario, dict):
            raise ValueError("Each carbon_price_scenarios entry must be an object.")
        _require_allowed_keys(
            "carbon_price_scenarios entry",
            scenario,
            required={"name", "carbon_price_usd_per_ton"},
            optional={
                "fuel_tax_rate",
                "ai_subsidy_usd_yr",
                "congestion_charge_usd_yr",
                "congestion_charge_usd_per_km",
            },
        )
        name = scenario["name"]
        _require_non_empty_str("carbon_price_scenarios.name", name)
        if name in seen_names:
            raise ValueError(f"Duplicate carbon price scenario name: {name}")
        seen_names.add(name)
        price = _to_float("carbon_price_scenarios.carbon_price_usd_per_ton", scenario["carbon_price_usd_per_ton"])
        fuel_tax_rate = _to_float(
            "carbon_price_scenarios.fuel_tax_rate",
            scenario.get("fuel_tax_rate", default_fuel_tax_rate),
        )
        ai_subsidy = scenario.get("ai_subsidy_usd_yr", default_ai_subsidy_value)
        ai_subsidy_value = (
            None
            if ai_subsidy is None
            else _to_float("carbon_price_scenarios.ai_subsidy_usd_yr", ai_subsidy)
        )
        congestion_charge = _to_float(
            "carbon_price_scenarios.congestion_charge_usd_yr",
            scenario.get("congestion_charge_usd_yr", default_congestion_charge),
        )
        congestion_per_km = scenario.get(
            "congestion_charge_usd_per_km", default_congestion_per_km_value
        )
        congestion_per_km_value = (
            None
            if congestion_per_km is None
            else _to_float(
                "carbon_price_scenarios.congestion_charge_usd_per_km",
                congestion_per_km,
            )
        )
        carbon_scenario = CarbonPriceScenario(
            name=name,
            carbon_price_usd_per_ton=price,
            fuel_tax_rate=fuel_tax_rate,
            ai_subsidy_usd_yr=ai_subsidy_value,
            congestion_charge_usd_yr=congestion_charge,
            congestion_charge_usd_per_km=congestion_per_km_value,
        )
        carbon_scenario.validate()
        scenarios.append(carbon_scenario)

    ai_cost = data.get("ai_implementation_cost_usd_yr")
    ai_cost_value = None if ai_cost is None else _to_float("policy.ai_implementation_cost_usd_yr", ai_cost)
    if ai_cost_value is not None:
        _require_non_negative("policy.ai_implementation_cost_usd_yr", ai_cost_value)

    defaults_payload = {
        "fuel_tax_rate": default_fuel_tax_rate,
        "ai_subsidy_usd_yr": default_ai_subsidy_value,
        "congestion_charge_usd_yr": default_congestion_charge,
        "congestion_charge_usd_per_km": default_congestion_per_km_value,
    }

    return scenarios, ai_cost_value, defaults_payload


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Fully validated evaluation configuration.
    """

    baseline_inputs: BaselineInputs
    ai_efficiency_rates: list[float]
    carbon_price_scenarios: list[CarbonPriceScenario]
    ai_implementation_cost_usd_yr: Optional[float]
    assumptions: list[str]
    policy_defaults: dict[str, Any]

    def validate(self) -> None:
        self.baseline_inputs.validate()
        _require_list_non_empty("ai_efficiency_rates", self.ai_efficiency_rates)
        for rate in self.ai_efficiency_rates:
            AIScenario(rate)
        _require_list_non_empty("carbon_price_scenarios", self.carbon_price_scenarios)
        for scenario in self.carbon_price_scenarios:
            scenario.validate()
        if self.ai_implementation_cost_usd_yr is not None:
            _require_non_negative(
                "ai_implementation_cost_usd_yr",
                float(self.ai_implementation_cost_usd_yr),
            )


def load_config_from_dict(raw: dict[str, Any]) -> EvaluationConfig:
    """
    Load a configuration dictionary into a validated EvaluationConfig.
    """

    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object.")

    _require_allowed_keys(
        "config",
        raw,
        required={"baseline", "ai_efficiency_rates", "policy"},
        optional={"assumptions"},
    )

    baseline_inputs = _parse_baseline(raw["baseline"])
    ai_rates = _require_list_non_empty("ai_efficiency_rates", raw["ai_efficiency_rates"])
    ai_efficiency_rates = [_to_float("ai_efficiency_rates entry", rate) for rate in ai_rates]
    assumptions = _parse_assumptions(raw.get("assumptions"))

    policy_scenarios, ai_cost, policy_defaults = _parse_policy(raw["policy"])

    config = EvaluationConfig(
        baseline_inputs=baseline_inputs,
        ai_efficiency_rates=ai_efficiency_rates,
        carbon_price_scenarios=policy_scenarios,
        ai_implementation_cost_usd_yr=ai_cost,
        assumptions=assumptions,
        policy_defaults=policy_defaults,
    )
    config.validate()
    return config


def load_config(path: Path) -> EvaluationConfig:
    """
    Load a JSON configuration file into a validated EvaluationConfig.
    """

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    return load_config_from_dict(raw)


def build_results(config: EvaluationConfig) -> dict[str, Any]:
    """
    Run evaluation across AI efficiency and carbon price scenarios.
    """

    baseline_state = BaselineModel(config.baseline_inputs).compute()
    rows: list[dict[str, Any]] = []

    for efficiency_rate in config.ai_efficiency_rates:
        scenario_result = AIScenario(efficiency_rate).apply(baseline_state)
        for policy_scenario in config.carbon_price_scenarios:
            evaluator = EconomicEvaluator(
                EconomicInputs(
                    carbon_price_usd_per_ton=policy_scenario.carbon_price_usd_per_ton,
                    ai_implementation_cost_usd_yr=config.ai_implementation_cost_usd_yr,
                )
            )
            economic_result = evaluator.evaluate(baseline_state, scenario_result)

            rows.append(
                {
                    "policy_scenario_name": policy_scenario.name,
                    "ai_efficiency_rate": efficiency_rate,
                    "carbon_price_usd_per_ton": policy_scenario.carbon_price_usd_per_ton,
                    "fuel_tax_rate": policy_scenario.fuel_tax_rate,
                    "ai_subsidy_usd_yr": economic_result.ai_subsidy_usd_yr,
                    "congestion_charge_usd_yr": policy_scenario.congestion_charge_usd_yr,
                    "congestion_charge_usd_per_km": policy_scenario.congestion_charge_usd_per_km,
                    "baseline_co2_emissions_ton_yr": baseline_state.co2_emissions_ton_yr,
                    "baseline_fuel_cost_usd_yr": baseline_state.fuel_cost_usd_yr,
                    "baseline_vkt_km_yr": baseline_state.vkt_km_yr,
                    "emissions_reduced_ton_yr": scenario_result.emissions_reduced_ton_yr,
                    "new_emissions_ton_yr": scenario_result.new_emissions_ton_yr,
                    "fuel_cost_savings_usd_yr": scenario_result.fuel_cost_savings_usd_yr,
                    "new_fuel_cost_usd_yr": scenario_result.new_fuel_cost_usd_yr,
                    "baseline_carbon_cost_usd_yr": economic_result.baseline_carbon_cost_usd_yr,
                    "new_carbon_cost_usd_yr": economic_result.new_carbon_cost_usd_yr,
                    "carbon_cost_savings_usd_yr": economic_result.carbon_cost_savings_usd_yr,
                    "baseline_fuel_tax_cost_usd_yr": economic_result.baseline_fuel_tax_cost_usd_yr,
                    "new_fuel_tax_cost_usd_yr": economic_result.new_fuel_tax_cost_usd_yr,
                    "fuel_tax_savings_usd_yr": economic_result.fuel_tax_savings_usd_yr,
                    "baseline_congestion_charge_usd_yr": economic_result.baseline_congestion_charge_usd_yr,
                    "new_congestion_charge_usd_yr": economic_result.new_congestion_charge_usd_yr,
                    "congestion_charge_savings_usd_yr": economic_result.congestion_charge_savings_usd_yr,
                    "ai_implementation_cost_usd_yr": economic_result.ai_implementation_cost_usd_yr,
                    "net_ai_cost_usd_yr": economic_result.net_ai_cost_usd_yr,
                    "total_economic_benefit_usd_yr": economic_result.total_economic_benefit_usd_yr,
                    "marginal_abatement_cost_usd_per_ton": economic_result.marginal_abatement_cost_usd_per_ton,
                }
            )

    return {
        "metadata": {
            "assumptions": config.assumptions,
            "units": UNITS,
            "baseline": {
                "co2_emissions_ton_yr": baseline_state.co2_emissions_ton_yr,
                "fuel_cost_usd_yr": baseline_state.fuel_cost_usd_yr,
                "vkt_km_yr": baseline_state.vkt_km_yr,
            },
            "ai_efficiency_rates": config.ai_efficiency_rates,
            "policy_scenarios": [
                {
                    "name": scenario.name,
                    "carbon_price_usd_per_ton": scenario.carbon_price_usd_per_ton,
                    "fuel_tax_rate": scenario.fuel_tax_rate,
                    "ai_subsidy_usd_yr": scenario.ai_subsidy_usd_yr,
                    "congestion_charge_usd_yr": scenario.congestion_charge_usd_yr,
                    "congestion_charge_usd_per_km": scenario.congestion_charge_usd_per_km,
                }
                for scenario in config.carbon_price_scenarios
            ],
            "ai_implementation_cost_usd_yr": config.ai_implementation_cost_usd_yr,
            "policy_defaults": config.policy_defaults,
        },
        "results": rows,
    }


def summarize_results(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Build a compact, deterministic summary for interpretation only.
    """

    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        raise ValueError("results must be a non-empty list for summarization.")

    def _row_key(row: dict[str, Any]) -> float:
        value = row.get("total_economic_benefit_usd_yr")
        if value is None:
            return float("-inf")
        return float(value)

    best_row = max(results, key=_row_key)
    worst_row = min(results, key=_row_key)

    mac_values = [
        float(row["marginal_abatement_cost_usd_per_ton"])
        for row in results
        if row.get("marginal_abatement_cost_usd_per_ton") is not None
    ]

    carbon_prices = [
        float(row["carbon_price_usd_per_ton"])
        for row in results
        if row.get("carbon_price_usd_per_ton") is not None
    ]
    efficiency_rates = [
        float(row["ai_efficiency_rate"])
        for row in results
        if row.get("ai_efficiency_rate") is not None
    ]

    summary = {
        "scenario_count": len(results),
        "carbon_price_range_usd_per_ton": {
            "min": min(carbon_prices) if carbon_prices else None,
            "max": max(carbon_prices) if carbon_prices else None,
        },
        "ai_efficiency_rate_range": {
            "min": min(efficiency_rates) if efficiency_rates else None,
            "max": max(efficiency_rates) if efficiency_rates else None,
        },
        "total_economic_benefit_usd_yr": {
            "min": float(worst_row.get("total_economic_benefit_usd_yr")),
            "max": float(best_row.get("total_economic_benefit_usd_yr")),
        },
        "best_scenario": {
            "policy_scenario_name": best_row.get("policy_scenario_name"),
            "ai_efficiency_rate": best_row.get("ai_efficiency_rate"),
            "total_economic_benefit_usd_yr": best_row.get("total_economic_benefit_usd_yr"),
        },
        "worst_scenario": {
            "policy_scenario_name": worst_row.get("policy_scenario_name"),
            "ai_efficiency_rate": worst_row.get("ai_efficiency_rate"),
            "total_economic_benefit_usd_yr": worst_row.get("total_economic_benefit_usd_yr"),
        },
        "marginal_abatement_cost_usd_per_ton": {
            "min": min(mac_values) if mac_values else None,
            "max": max(mac_values) if mac_values else None,
        },
    }

    return summary


def build_interpretation_prompt(
    payload: dict[str, Any],
    rag_context: Optional[list[dict[str, Any]]] = None,
    memory_notes: Optional[list[dict[str, Any]]] = None,
) -> str:
    """
    Build a policy-interpretation prompt from deterministic results.
    """

    summary = summarize_results(payload)
    metadata = payload.get("metadata", {})
    assumptions = metadata.get("assumptions", [])

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

    prompt_payload = {
        "scope_constraints": [
            "Road transport only",
            "CO2 only",
            "Annual static evaluation",
            "AI efficiency is scenario-based, not predictive",
            "No causal claims, no forecasting, no optimization",
        ],
        "assumptions": assumptions,
        "summary": summary,
        "scenario_table": compact_rows,
        "rag_context": rag_context or [],
        "memory_notes": memory_notes or [],
    }

    return (
        "You are a policy analyst. Interpret the results without performing new "
        "calculations. Do not make causal claims or forecasts. Focus on practical "
        "policy insights, trade-offs, and caveats. Use only the provided data. "
        "Do not reveal chain-of-thought. Provide concise, structured insights.\n\n"
        f"DATA:\n{json.dumps(prompt_payload, indent=2)}"
    )


def get_groq_api_key() -> str:
    """
    Load Groq API key from environment without persisting it.
    """

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is required for LLM interpretation.")
    return key


def generate_interpretation(
    payload: dict[str, Any],
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 700,
    rag_context: Optional[list[dict[str, Any]]] = None,
    memory_notes: Optional[list[dict[str, Any]]] = None,
) -> str:
    """
    Generate an LLM interpretation of the deterministic results.
    """

    prompt = build_interpretation_prompt(
        payload=payload,
        rag_context=rag_context,
        memory_notes=memory_notes,
    )
    request_body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You interpret policy evaluation outputs. You do not compute "
                    "new numbers or infer causality."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    data = json.dumps(request_body).encode("utf-8")
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


def _resolve_output_format(format_arg: Optional[str], output_path: Optional[Path]) -> str:
    if format_arg:
        normalized = format_arg.lower()
        if normalized not in {"csv", "json"}:
            raise ValueError("output format must be 'csv' or 'json'.")
        return normalized
    if output_path and output_path.suffix.lower() in {".csv", ".json"}:
        return output_path.suffix.lower().lstrip(".")
    raise ValueError("output format must be specified when not inferable from path.")


def _normalize_row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in CSV_FIELDS:
        value = row.get(key)
        normalized[key] = "" if value is None else value
    return normalized


def write_output(payload: dict[str, Any], output_path: Optional[Path], output_format: str) -> None:
    """
    Write evaluation results as JSON or CSV.
    """

    if output_format == "json":
        content = json.dumps(payload, indent=2, sort_keys=True)
        if output_path:
            output_path.write_text(content, encoding="utf-8")
        else:
            print(content)
        return

    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("results must be a list for CSV output.")

    if output_path:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(_normalize_row_for_csv(row))
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row_for_csv(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EcoMoveAI deterministic policy evaluation runner."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to config JSON.")
    parser.add_argument("--output", type=Path, help="Output file path (CSV or JSON).")
    parser.add_argument(
        "--format",
        dest="output_format",
        help="Output format: csv or json. Required if not inferable from --output.",
    )
    parser.add_argument(
        "--interpret",
        action="store_true",
        help="Generate an LLM policy interpretation (no calculations).",
    )
    parser.add_argument(
        "--llm-model",
        help="Groq model name for interpretation. Required with --interpret.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="LLM temperature for interpretation (default: 0.0).",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=700,
        help="Max tokens for interpretation output (default: 700).",
    )
    parser.add_argument(
        "--interpret-output",
        type=Path,
        help="Optional file path to write interpretation text.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Directory to write report-ready tables and figures.",
    )
    parser.add_argument(
        "--rag-dir",
        type=Path,
        help="Directory of local documents for retrieval-augmented interpretation.",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=3,
        help="Number of RAG snippets to include (default: 3).",
    )
    parser.add_argument(
        "--rag-query",
        help="Custom RAG query string. If omitted, a default is used.",
    )
    parser.add_argument(
        "--memory-file",
        type=Path,
        default=Path("data/memory/notes.jsonl"),
        help="Path to JSONL memory notes file.",
    )
    parser.add_argument(
        "--memory-top-k",
        type=int,
        default=5,
        help="Number of memory notes to include (default: 5).",
    )
    parser.add_argument(
        "--memory-title",
        help="Title for a new memory note to append.",
    )
    parser.add_argument(
        "--memory-note",
        help="Content for a new memory note to append.",
    )
    parser.add_argument(
        "--memory-tags",
        help="Comma-separated tags for a new memory note.",
    )

    args = parser.parse_args()

    if args.interpret and not args.llm_model:
        raise ValueError("--llm-model is required when --interpret is set.")

    if args.interpret and args.output is None and args.interpret_output is None:
        raise ValueError(
            "--interpret-output is required when writing results to stdout."
        )

    if args.memory_note:
        if not args.memory_title:
            raise ValueError("--memory-title is required with --memory-note.")
        tags = [tag.strip() for tag in (args.memory_tags or "").split(",") if tag.strip()]
        append_memory_note(
            path=args.memory_file,
            title=args.memory_title,
            content=args.memory_note,
            tags=tags,
        )

    config = load_config(args.config)
    payload = build_results(config)
    output_format = _resolve_output_format(args.output_format, args.output)
    write_output(payload, args.output, output_format)

    if args.report_dir:
        write_report_assets(payload, args.report_dir)

    if args.interpret:
        rag_context = None
        if args.rag_dir:
            rag_query = args.rag_query or (
                "road transport decarbonization policy evaluation carbon price "
                "fuel tax subsidy congestion charge AI efficiency"
            )
            documents = load_documents_from_dir(args.rag_dir)
            index = build_index(documents)
            rag_context = search(index, rag_query, top_k=args.rag_top_k)

        memory_notes = format_memory_notes(
            load_memory_notes(args.memory_file, limit=args.memory_top_k)
        )

        interpretation = generate_interpretation(
            payload=payload,
            model=args.llm_model,
            api_key=get_groq_api_key(),
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            rag_context=rag_context,
            memory_notes=memory_notes,
        )
        if args.interpret_output:
            args.interpret_output.write_text(interpretation, encoding="utf-8")
        else:
            print(interpretation)


if __name__ == "__main__":
    main()
