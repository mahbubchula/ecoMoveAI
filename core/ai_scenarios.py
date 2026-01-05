"""
EcoMoveAI — AI Efficiency Scenarios Module

Purpose
-------
Defines controlled AI-enabled efficiency scenarios that reduce emissions
and fuel consumption by a specified proportion.

Important Note (for academic use)
---------------------------------
AI efficiency is modeled as a *scenario parameter*, not as a predictive model.
This design choice ensures transparency and policy relevance while avoiding
unverifiable performance claims.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.baseline import BaselineState


def _require_rate_between_0_and_1(rate: float) -> None:
    if rate < 0.0 or rate > 1.0:
        raise ValueError(
            f"efficiency_rate must be between 0 and 1. Got: {rate}"
        )


@dataclass(frozen=True)
class AIScenarioResult:
    """
    Result of applying an AI efficiency scenario.

    Parameters
    ----------
    new_emissions_ton_yr : float
        Emissions after AI efficiency is applied.
    emissions_reduced_ton_yr : float
        Absolute emissions reduction.
    new_fuel_cost_usd_yr : float
        Fuel cost after AI efficiency is applied.
    fuel_cost_savings_usd_yr : float
        Absolute fuel cost savings.
    efficiency_rate : float
        Applied AI efficiency rate.
    """

    new_emissions_ton_yr: float
    emissions_reduced_ton_yr: float
    new_fuel_cost_usd_yr: float
    fuel_cost_savings_usd_yr: float
    efficiency_rate: float


class AIScenario:
    """
    AI efficiency scenario model.

    Example
    -------
    An efficiency_rate of 0.10 represents a 10% reduction in emissions
    and fuel use relative to the baseline.
    """

    def __init__(self, efficiency_rate: float):
        _require_rate_between_0_and_1(efficiency_rate)
        self._efficiency_rate = efficiency_rate

    @property
    def efficiency_rate(self) -> float:
        return self._efficiency_rate

    def apply(self, baseline: BaselineState) -> AIScenarioResult:
        """
        Apply AI efficiency to a baseline state.

        Parameters
        ----------
        baseline : BaselineState
            Validated baseline transport state.

        Returns
        -------
        AIScenarioResult
        """

        emissions_reduced = baseline.co2_emissions_ton_yr * self._efficiency_rate
        new_emissions = baseline.co2_emissions_ton_yr - emissions_reduced

        fuel_cost_savings = baseline.fuel_cost_usd_yr * self._efficiency_rate
        new_fuel_cost = baseline.fuel_cost_usd_yr - fuel_cost_savings

        return AIScenarioResult(
            new_emissions_ton_yr=float(new_emissions),
            emissions_reduced_ton_yr=float(emissions_reduced),
            new_fuel_cost_usd_yr=float(new_fuel_cost),
            fuel_cost_savings_usd_yr=float(fuel_cost_savings),
            efficiency_rate=float(self._efficiency_rate),
        )
