import json
import tempfile
import unittest
from pathlib import Path

from core.ai_scenarios import AIScenario
from core.baseline import BaselineInputs, BaselineModel
from core.economics import (
    CarbonPriceScenario,
    EconomicEvaluator,
    EconomicInputs,
    evaluate_carbon_price_scenarios,
)
from main import build_results, load_config, load_config_from_dict, summarize_results


class TestEconomics(unittest.TestCase):
    def setUp(self) -> None:
        inputs = BaselineInputs(
            co2_emissions_ton_yr=100.0,
            fuel_cost_usd_yr=1000.0,
        )
        self.baseline = BaselineModel(inputs).compute()
        self.scenario = AIScenario(0.10).apply(self.baseline)

    def test_economic_evaluation_basic(self) -> None:
        evaluator = EconomicEvaluator(EconomicInputs(carbon_price_usd_per_ton=50.0))
        result = evaluator.evaluate(self.baseline, self.scenario)

        self.assertAlmostEqual(result.emissions_reduced_ton_yr, 10.0)
        self.assertAlmostEqual(result.baseline_carbon_cost_usd_yr, 5000.0)
        self.assertAlmostEqual(result.new_carbon_cost_usd_yr, 4500.0)
        self.assertAlmostEqual(result.carbon_cost_savings_usd_yr, 500.0)
        self.assertAlmostEqual(result.fuel_cost_savings_usd_yr, 100.0)
        self.assertAlmostEqual(result.fuel_tax_savings_usd_yr, 0.0)
        self.assertAlmostEqual(result.congestion_charge_savings_usd_yr, 0.0)
        self.assertAlmostEqual(result.total_economic_benefit_usd_yr, 600.0)
        self.assertIsNone(result.marginal_abatement_cost_usd_per_ton)

    def test_economic_evaluation_with_ai_cost_and_mac(self) -> None:
        evaluator = EconomicEvaluator(
            EconomicInputs(
                carbon_price_usd_per_ton=50.0,
                ai_implementation_cost_usd_yr=200.0,
            )
        )
        result = evaluator.evaluate(self.baseline, self.scenario)

        self.assertAlmostEqual(result.total_economic_benefit_usd_yr, 400.0)
        self.assertAlmostEqual(result.marginal_abatement_cost_usd_per_ton, 20.0)

    def test_economic_evaluation_with_fuel_tax_and_subsidy(self) -> None:
        evaluator = EconomicEvaluator(
            EconomicInputs(
                carbon_price_usd_per_ton=50.0,
                ai_implementation_cost_usd_yr=200.0,
                fuel_tax_rate=0.10,
                ai_subsidy_usd_yr=50.0,
            )
        )
        result = evaluator.evaluate(self.baseline, self.scenario)

        self.assertAlmostEqual(result.fuel_tax_savings_usd_yr, 10.0)
        self.assertAlmostEqual(result.net_ai_cost_usd_yr, 150.0)
        self.assertAlmostEqual(result.total_economic_benefit_usd_yr, 460.0)

    def test_evaluate_carbon_price_scenarios(self) -> None:
        scenarios = [
            CarbonPriceScenario(name="Low", carbon_price_usd_per_ton=25.0),
            CarbonPriceScenario(name="High", carbon_price_usd_per_ton=100.0),
        ]
        results = evaluate_carbon_price_scenarios(
            self.baseline,
            self.scenario,
            scenarios,
        )

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].carbon_price_usd_per_ton, 25.0)
        self.assertAlmostEqual(results[1].carbon_price_usd_per_ton, 100.0)

    def test_invalid_carbon_price_raises(self) -> None:
        with self.assertRaises(ValueError):
            EconomicInputs(carbon_price_usd_per_ton=-1.0).validate()


class TestPipeline(unittest.TestCase):
    def _write_config(self, config: dict) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_build_results(self) -> None:
        config = {
            "baseline": {
                "co2_emissions_ton_yr": 100.0,
                "fuel_cost_usd_yr": 1000.0,
            },
            "ai_efficiency_rates": [0.10],
            "policy": {
                "carbon_price_scenarios": [
                    {"name": "Base", "carbon_price_usd_per_ton": 50.0}
                ]
            },
        }
        config_path = self._write_config(config)
        evaluation_config = load_config(config_path)
        payload = build_results(evaluation_config)

        self.assertEqual(len(payload["results"]), 1)
        row = payload["results"][0]
        self.assertAlmostEqual(row["carbon_cost_savings_usd_yr"], 500.0)
        self.assertAlmostEqual(row["fuel_cost_savings_usd_yr"], 100.0)

    def test_load_config_from_dict(self) -> None:
        config = {
            "baseline": {
                "co2_emissions_ton_yr": 100.0,
                "fuel_cost_usd_yr": 1000.0,
            },
            "ai_efficiency_rates": [0.10],
            "policy": {
                "carbon_price_scenarios": [
                    {"name": "Base", "carbon_price_usd_per_ton": 50.0}
                ]
            },
        }
        evaluation_config = load_config_from_dict(config)
        payload = build_results(evaluation_config)
        summary = summarize_results(payload)

        self.assertEqual(summary["scenario_count"], 1)
        self.assertAlmostEqual(
            summary["total_economic_benefit_usd_yr"]["max"], 600.0
        )

    def test_policy_defaults_applied(self) -> None:
        config = {
            "baseline": {
                "co2_emissions_ton_yr": 100.0,
                "fuel_cost_usd_yr": 1000.0,
            },
            "ai_efficiency_rates": [0.10],
            "policy": {
                "carbon_price_scenarios": [
                    {"name": "Base", "carbon_price_usd_per_ton": 50.0}
                ],
                "policy_defaults": {
                    "fuel_tax_rate": 0.10,
                },
            },
        }
        evaluation_config = load_config_from_dict(config)
        payload = build_results(evaluation_config)
        row = payload["results"][0]

        self.assertAlmostEqual(row["fuel_tax_rate"], 0.10)
        self.assertAlmostEqual(row["fuel_tax_savings_usd_yr"], 10.0)

    def test_config_requires_fuel_cost(self) -> None:
        config = {
            "baseline": {
                "co2_emissions_ton_yr": 100.0,
            },
            "ai_efficiency_rates": [0.10],
            "policy": {
                "carbon_price_scenarios": [
                    {"name": "Base", "carbon_price_usd_per_ton": 50.0}
                ]
            },
        }
        config_path = self._write_config(config)
        with self.assertRaises(ValueError):
            load_config(config_path)


if __name__ == "__main__":
    unittest.main()
