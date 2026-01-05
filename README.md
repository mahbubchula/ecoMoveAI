# EcoMoveAI

EcoMoveAI is a deterministic policy evaluation tool for annual economic analysis
of AI-enabled efficiency improvements in road transport CO2 emissions. It is
transparent, scenario-based, and designed for reproducible results.

## Scope

- Road transport only
- CO2 only
- Annual (static) evaluation
- Single city or country
- AI as a scenario parameter (no prediction)

## What It Does

- Evaluates emissions and fuel cost changes under AI efficiency scenarios
- Applies policy instruments: carbon price, fuel tax, AI subsidy, congestion charge
- Computes carbon cost savings, total economic benefit, and marginal abatement cost
- Exports results to JSON/CSV and report-ready CSV/PNG assets
- Optional LLM-based policy interpretation (no calculations) with local RAG context

## What It Does NOT Do

- No real-time modeling
- No causal claims
- No black-box ML
- No optimization or traffic control

## Install

```bash
pip install -r requirements.txt
```

## Configuration (JSON)

All values must be explicit. Replace placeholders with real numbers.

```json
{
  "baseline": {
    "co2_emissions_ton_yr": "<float>",
    "fuel_cost_usd_yr": "<float>",
    "vkt_km_yr": null
  },
  "ai_efficiency_rates": ["<float between 0 and 1>"],
  "policy": {
    "carbon_price_scenarios": [
      {
        "name": "<string>",
        "carbon_price_usd_per_ton": "<float>",
        "fuel_tax_rate": "<float 0-1>",
        "ai_subsidy_usd_yr": "<float or null>",
        "congestion_charge_usd_yr": "<float>",
        "congestion_charge_usd_per_km": "<float or null>"
      }
    ],
    "ai_implementation_cost_usd_yr": "<float or null>",
    "policy_defaults": {
      "fuel_tax_rate": "<float 0-1>",
      "ai_subsidy_usd_yr": "<float or null>",
      "congestion_charge_usd_yr": "<float>",
      "congestion_charge_usd_per_km": "<float or null>"
    }
  },
  "assumptions": ["<string>"]
}
```

## Run (CLI)

```bash
python main.py --config path/to/config.json --output results.json --format json
python main.py --config path/to/config.json --output results.csv --format csv
```

## User Guide

1) Prepare inputs  
   - Define baseline emissions and fuel cost in a JSON config.
   - List AI efficiency rates (e.g., 0.05, 0.10, 0.20).
   - Add policy scenarios (carbon price, fuel tax, AI subsidy, congestion charge).
   - Optional: add assumptions for transparency.

2) Run the evaluation  
   - Use the CLI to generate JSON or CSV outputs.
   - Use `--report-dir` to create report-ready tables and figures.

3) Interpret results (optional)  
   - Set `GROQ_API_KEY` and run with `--interpret` and `--llm-model`.
   - Add local documents for RAG via `--rag-dir`.
   - Add or include memory notes for consistent policy context.

4) Use the dashboard  
   - Launch Streamlit for interactive scenario analysis and exports.

5) Validate outputs  
   - Review scenario tables and figures for consistency.
   - Re-run with updated assumptions if needed.

### Report-Ready Exports (CSV/PNG)

```bash
python main.py --config path/to/config.json \
  --output results.json --format json \
  --report-dir reports
```

### Optional LLM Interpretation (Groq)

Interpretation only. No calculations. The API key is read from environment and
never written to disk.

```bash
export GROQ_API_KEY="your_key"
python main.py --config path/to/config.json \
  --output results.json --format json \
  --interpret --llm-model llama-3.1-8b-instant \
  --rag-dir data/knowledge_base --rag-top-k 3 \
  --interpret-output interpretation.txt
```

### Memory Notes (Optional)

```bash
python main.py --config path/to/config.json \
  --output results.json --format json \
  --memory-title "Assumption update" \
  --memory-note "Fuel tax rate applies to baseline and AI scenarios." \
  --memory-tags policy,assumption
```

## Streamlit App

```bash
streamlit run streamlit_app.py
```

For Streamlit Cloud, set `GROQ_API_KEY` as a secret if you enable LLM interpretation.

## Docker

```bash
docker build -t ecomoveai .
docker run -p 8501:8501 ecomoveai
```

## Tests

```bash
python -m unittest discover -s tests
```
