"""
EcoMoveAI — Baseline Module

Purpose
-------
Defines the baseline state of the transport system for annual (static) evaluation.
This module is intentionally deterministic and transparent to support reproducible
economic and policy analysis suitable for academic publication.

Conventions
-----------
- Emissions are expressed in tons of CO2 per year.
- Monetary values are expressed in USD per year.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _require_non_negative(name: str, value: float) -> None:
    if value is None:
        raise ValueError(f"{name} must not be None.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative. Got: {value}")


@dataclass(frozen=True)
class BaselineInputs:
    """
    Minimal baseline inputs for annual road-transport evaluation.

    Parameters
    ----------
    co2_emissions_ton_yr : float
        Baseline CO2 emissions (tons/year).
    fuel_cost_usd_yr : float
        Baseline fuel expenditure (USD/year). If unknown, set to 0 and document it.
    vkt_km_yr : Optional[float]
        Vehicle-kilometers traveled (km/year). Optional in v1; useful for extensions.
    """

    co2_emissions_ton_yr: float
    fuel_cost_usd_yr: float = 0.0
    vkt_km_yr: Optional[float] = None

    def validate(self) -> None:
        _require_non_negative("co2_emissions_ton_yr", float(self.co2_emissions_ton_yr))
        _require_non_negative("fuel_cost_usd_yr", float(self.fuel_cost_usd_yr))
        if self.vkt_km_yr is not None:
            _require_non_negative("vkt_km_yr", float(self.vkt_km_yr))


@dataclass(frozen=True)
class BaselineState:
    """
    Computed baseline state (a simple pass-through in v1, but standardized).

    Keeping this separate allows later upgrades (e.g., deriving fuel cost from fuel use)
    without breaking the public interface used by scenarios and evaluators.
    """

    co2_emissions_ton_yr: float
    fuel_cost_usd_yr: float
    vkt_km_yr: Optional[float] = None


class BaselineModel:
    """
    Baseline model wrapper.

    In v1, this primarily validates inputs and returns a standardized baseline state.
    """

    def __init__(self, inputs: BaselineInputs):
        inputs.validate()
        self._inputs = inputs

    @property
    def inputs(self) -> BaselineInputs:
        return self._inputs

    def compute(self) -> BaselineState:
        return BaselineState(
            co2_emissions_ton_yr=float(self._inputs.co2_emissions_ton_yr),
            fuel_cost_usd_yr=float(self._inputs.fuel_cost_usd_yr),
            vkt_km_yr=float(self._inputs.vkt_km_yr) if self._inputs.vkt_km_yr is not None else None,
        )
