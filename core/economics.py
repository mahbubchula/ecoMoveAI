"""
EcoMoveAI - Economics Module

Purpose
-------
Evaluates the economic implications of AI-enabled efficiency improvements
under transparent, scenario-based policy assumptions.

Design Principles
-----------------
- Deterministic calculations
- Explicit units
- No hidden defaults
- Reproducible, peer-review-ready logic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from core.ai_scenarios import AIScenarioResult
from core.baseline import BaselineState


def _require_non_negative(name: str, value: float) -> None:
    if value is None:
        raise ValueError(f"{name} must not be None.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative. Got: {value}")


def _require_non_empty(name: str, value: str) -> None:
    if value is None or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_rate_between_0_and_1(name: str, rate: float) -> None:
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"{name} must be between 0 and 1. Got: {rate}")


@dataclass(frozen=True)
class EconomicInputs:
    """
    Economic inputs for a single policy evaluation.

    Parameters
    ----------
    carbon_price_usd_per_ton : float
        Carbon price (USD per ton CO2).
    ai_implementation_cost_usd_yr : Optional[float]
        Annualized cost of AI implementation (USD/year). Optional but required
        to compute marginal abatement cost.
    fuel_tax_rate : float
        Fuel tax rate applied to fuel cost (0 to 1). Expressed as a share.
    ai_subsidy_usd_yr : Optional[float]
        Annual AI subsidy that offsets implementation cost (USD/year).
    congestion_charge_usd_yr : float
        Flat annual congestion charge (USD/year).
    congestion_charge_usd_per_km : Optional[float]
        Per-kilometer congestion charge (USD/km). Requires baseline VKT.
    """

    carbon_price_usd_per_ton: float
    ai_implementation_cost_usd_yr: Optional[float] = None
    fuel_tax_rate: float = 0.0
    ai_subsidy_usd_yr: Optional[float] = None
    congestion_charge_usd_yr: float = 0.0
    congestion_charge_usd_per_km: Optional[float] = None

    def validate(self) -> None:
        _require_non_negative(
            "carbon_price_usd_per_ton", float(self.carbon_price_usd_per_ton)
        )
        if self.ai_implementation_cost_usd_yr is not None:
            _require_non_negative(
                "ai_implementation_cost_usd_yr",
                float(self.ai_implementation_cost_usd_yr),
            )
        _require_rate_between_0_and_1("fuel_tax_rate", float(self.fuel_tax_rate))
        if self.ai_subsidy_usd_yr is not None:
            _require_non_negative(
                "ai_subsidy_usd_yr",
                float(self.ai_subsidy_usd_yr),
            )
        _require_non_negative(
            "congestion_charge_usd_yr", float(self.congestion_charge_usd_yr)
        )
        if self.congestion_charge_usd_per_km is not None:
            _require_non_negative(
                "congestion_charge_usd_per_km",
                float(self.congestion_charge_usd_per_km),
            )


@dataclass(frozen=True)
class CarbonPriceScenario:
    """
    Named carbon price scenario for policy evaluation.

    Parameters
    ----------
    name : str
        Human-readable scenario label.
    carbon_price_usd_per_ton : float
        Carbon price (USD per ton CO2).
    fuel_tax_rate : float
        Fuel tax rate applied to fuel cost (0 to 1). Expressed as a share.
    ai_subsidy_usd_yr : Optional[float]
        Annual AI subsidy that offsets implementation cost (USD/year).
    congestion_charge_usd_yr : float
        Flat annual congestion charge (USD/year).
    congestion_charge_usd_per_km : Optional[float]
        Per-kilometer congestion charge (USD/km). Requires baseline VKT.
    """

    name: str
    carbon_price_usd_per_ton: float
    fuel_tax_rate: float = 0.0
    ai_subsidy_usd_yr: Optional[float] = None
    congestion_charge_usd_yr: float = 0.0
    congestion_charge_usd_per_km: Optional[float] = None

    def validate(self) -> None:
        _require_non_empty("name", self.name)
        _require_non_negative(
            "carbon_price_usd_per_ton", float(self.carbon_price_usd_per_ton)
        )
        _require_rate_between_0_and_1("fuel_tax_rate", float(self.fuel_tax_rate))
        if self.ai_subsidy_usd_yr is not None:
            _require_non_negative(
                "ai_subsidy_usd_yr",
                float(self.ai_subsidy_usd_yr),
            )
        _require_non_negative(
            "congestion_charge_usd_yr", float(self.congestion_charge_usd_yr)
        )
        if self.congestion_charge_usd_per_km is not None:
            _require_non_negative(
                "congestion_charge_usd_per_km",
                float(self.congestion_charge_usd_per_km),
            )


@dataclass(frozen=True)
class EconomicResult:
    """
    Economic evaluation result for a given policy scenario.

    Parameters
    ----------
    carbon_price_usd_per_ton : float
        Applied carbon price.
    baseline_carbon_cost_usd_yr : float
        Baseline carbon cost (USD/year).
    new_carbon_cost_usd_yr : float
        Post-AI carbon cost (USD/year).
    carbon_cost_savings_usd_yr : float
        Carbon cost savings (USD/year).
    baseline_fuel_tax_cost_usd_yr : float
        Baseline fuel tax cost (USD/year).
    new_fuel_tax_cost_usd_yr : float
        Post-AI fuel tax cost (USD/year).
    fuel_tax_savings_usd_yr : float
        Fuel tax savings due to AI (USD/year).
    baseline_congestion_charge_usd_yr : float
        Baseline congestion charge (USD/year).
    new_congestion_charge_usd_yr : float
        Post-AI congestion charge (USD/year).
    congestion_charge_savings_usd_yr : float
        Congestion charge savings due to AI (USD/year).
    fuel_cost_savings_usd_yr : float
        Fuel cost savings (USD/year).
    total_economic_benefit_usd_yr : float
        Total benefit net of AI implementation cost if provided (USD/year).
    emissions_reduced_ton_yr : float
        Emissions reduced (tons/year).
    marginal_abatement_cost_usd_per_ton : Optional[float]
        AI implementation cost per ton reduced, if available.
    ai_implementation_cost_usd_yr : Optional[float]
        Annualized AI implementation cost used in evaluation.
    ai_subsidy_usd_yr : Optional[float]
        Annual AI subsidy applied to offset implementation cost.
    net_ai_cost_usd_yr : Optional[float]
        Net AI cost after subsidy (USD/year).
    """

    carbon_price_usd_per_ton: float
    baseline_carbon_cost_usd_yr: float
    new_carbon_cost_usd_yr: float
    carbon_cost_savings_usd_yr: float
    baseline_fuel_tax_cost_usd_yr: float
    new_fuel_tax_cost_usd_yr: float
    fuel_tax_savings_usd_yr: float
    baseline_congestion_charge_usd_yr: float
    new_congestion_charge_usd_yr: float
    congestion_charge_savings_usd_yr: float
    fuel_cost_savings_usd_yr: float
    total_economic_benefit_usd_yr: float
    emissions_reduced_ton_yr: float
    marginal_abatement_cost_usd_per_ton: Optional[float]
    ai_implementation_cost_usd_yr: Optional[float]
    ai_subsidy_usd_yr: Optional[float]
    net_ai_cost_usd_yr: Optional[float]


class EconomicEvaluator:
    """
    Economic evaluator for a single carbon price policy scenario.
    """

    def __init__(self, inputs: EconomicInputs):
        inputs.validate()
        self._inputs = inputs

    @property
    def inputs(self) -> EconomicInputs:
        return self._inputs

    def evaluate(
        self,
        baseline: BaselineState,
        scenario: AIScenarioResult,
    ) -> EconomicResult:
        """
        Evaluate economic outcomes for a given baseline and AI scenario.

        Parameters
        ----------
        baseline : BaselineState
            Baseline transport state.
        scenario : AIScenarioResult
            Result of applying AI efficiency to the baseline.

        Returns
        -------
        EconomicResult
        """

        carbon_price = float(self._inputs.carbon_price_usd_per_ton)

        baseline_carbon_cost = baseline.co2_emissions_ton_yr * carbon_price
        new_carbon_cost = scenario.new_emissions_ton_yr * carbon_price
        carbon_cost_savings = baseline_carbon_cost - new_carbon_cost

        fuel_cost_savings = scenario.fuel_cost_savings_usd_yr

        fuel_tax_rate = float(self._inputs.fuel_tax_rate)
        baseline_fuel_tax_cost = baseline.fuel_cost_usd_yr * fuel_tax_rate
        new_fuel_tax_cost = scenario.new_fuel_cost_usd_yr * fuel_tax_rate
        fuel_tax_savings = baseline_fuel_tax_cost - new_fuel_tax_cost

        congestion_charge_flat = float(self._inputs.congestion_charge_usd_yr)
        congestion_per_km = self._inputs.congestion_charge_usd_per_km
        if congestion_per_km is not None and baseline.vkt_km_yr is None:
            raise ValueError(
                "baseline.vkt_km_yr is required when congestion_charge_usd_per_km is set."
            )
        congestion_variable = (
            float(congestion_per_km) * float(baseline.vkt_km_yr)
            if congestion_per_km is not None
            else 0.0
        )
        baseline_congestion_charge = congestion_charge_flat + congestion_variable
        new_congestion_charge = baseline_congestion_charge
        congestion_charge_savings = baseline_congestion_charge - new_congestion_charge

        total_benefit = (
            carbon_cost_savings
            + fuel_cost_savings
            + fuel_tax_savings
            + congestion_charge_savings
        )

        ai_cost = self._inputs.ai_implementation_cost_usd_yr
        ai_subsidy = self._inputs.ai_subsidy_usd_yr
        net_ai_cost = None
        if ai_cost is not None or ai_subsidy is not None:
            net_ai_cost = float(ai_cost or 0.0) - float(ai_subsidy or 0.0)
            total_benefit -= net_ai_cost

        mac = None
        if net_ai_cost is not None:
            if scenario.emissions_reduced_ton_yr <= 0:
                raise ValueError(
                    "emissions_reduced_ton_yr must be positive to compute "
                    "marginal abatement cost."
                )
            mac = net_ai_cost / scenario.emissions_reduced_ton_yr

        return EconomicResult(
            carbon_price_usd_per_ton=float(carbon_price),
            baseline_carbon_cost_usd_yr=float(baseline_carbon_cost),
            new_carbon_cost_usd_yr=float(new_carbon_cost),
            carbon_cost_savings_usd_yr=float(carbon_cost_savings),
            baseline_fuel_tax_cost_usd_yr=float(baseline_fuel_tax_cost),
            new_fuel_tax_cost_usd_yr=float(new_fuel_tax_cost),
            fuel_tax_savings_usd_yr=float(fuel_tax_savings),
            baseline_congestion_charge_usd_yr=float(baseline_congestion_charge),
            new_congestion_charge_usd_yr=float(new_congestion_charge),
            congestion_charge_savings_usd_yr=float(congestion_charge_savings),
            fuel_cost_savings_usd_yr=float(fuel_cost_savings),
            total_economic_benefit_usd_yr=float(total_benefit),
            emissions_reduced_ton_yr=float(scenario.emissions_reduced_ton_yr),
            marginal_abatement_cost_usd_per_ton=float(mac) if mac is not None else None,
            ai_implementation_cost_usd_yr=float(ai_cost) if ai_cost is not None else None,
            ai_subsidy_usd_yr=float(ai_subsidy) if ai_subsidy is not None else None,
            net_ai_cost_usd_yr=float(net_ai_cost) if net_ai_cost is not None else None,
        )


def evaluate_carbon_price_scenarios(
    baseline: BaselineState,
    scenario: AIScenarioResult,
    scenarios: Iterable[CarbonPriceScenario],
    ai_implementation_cost_usd_yr: Optional[float] = None,
) -> list[EconomicResult]:
    """
    Evaluate multiple carbon price policy scenarios.

    Parameters
    ----------
    baseline : BaselineState
        Baseline transport state.
    scenario : AIScenarioResult
        Result of applying AI efficiency to the baseline.
    scenarios : Iterable[CarbonPriceScenario]
        Policy scenarios with distinct carbon prices.
    ai_implementation_cost_usd_yr : Optional[float]
        Annualized AI implementation cost used for all scenarios.

    Returns
    -------
    list[EconomicResult]
    """

    results: list[EconomicResult] = []
    for price_scenario in scenarios:
        price_scenario.validate()
        inputs = EconomicInputs(
            carbon_price_usd_per_ton=price_scenario.carbon_price_usd_per_ton,
            ai_implementation_cost_usd_yr=ai_implementation_cost_usd_yr,
            fuel_tax_rate=price_scenario.fuel_tax_rate,
            ai_subsidy_usd_yr=price_scenario.ai_subsidy_usd_yr,
            congestion_charge_usd_yr=price_scenario.congestion_charge_usd_yr,
            congestion_charge_usd_per_km=price_scenario.congestion_charge_usd_per_km,
        )
        evaluator = EconomicEvaluator(inputs)
        results.append(evaluator.evaluate(baseline, scenario))
    return results
