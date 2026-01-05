# EcoMoveAI Architecture and Capabilities

## Purpose and Scope
EcoMoveAI is a deterministic, scenario-based evaluation system for annual
economic analysis of AI-enabled efficiency in road transport CO2 emissions.
It is designed for transparency, reproducibility, and policy-grade reporting.

Scope constraints:
- Road transport only
- CO2 only
- Annual static evaluation (no time-series dynamics)
- AI is a scenario parameter, not a predictive model
- No causal claims, forecasting, or optimization

## Architecture at a Glance
```
                +-------------------------+
Config JSON --->| main.py (CLI runner)    |--------------------+
                +-----------+-------------+                    |
                            |                                  |
                            v                                  v
                 +-------------------+               +-----------------------+
                 | Baseline Model    |               | Streamlit Dashboard   |
                 | core/baseline.py  |               | streamlit_app.py      |
                 +---------+---------+               +-----------------------+
                           |                                  |
                           v                                  |
                 +-------------------+                        |
                 | AI Scenarios      |                        |
                 | core/ai_scenarios |                        |
                 +---------+---------+                        |
                           |                                  |
                           v                                  |
                 +-------------------+                        |
                 | Economics Engine  |                        |
                 | core/economics.py |                        |
                 +---------+---------+                        |
                           |                                  |
                           v                                  |
                 +-------------------+                        |
                 | Results Payload   |<-----------------------+
                 +---------+---------+
                           |
          +----------------+------------------+
          |                                   |
          v                                   v
 +-------------------+              +------------------------+
 | Reporting         |              | Optional Interpretation|
 | core/reporting.py |              | Groq + RAG + Memory     |
 +-------------------+              +------------------------+
```

## Core Components

### CLI Orchestrator (`main.py`)
- Validates config schema and values.
- Builds baseline, applies AI efficiency scenarios, evaluates economics.
- Produces JSON/CSV outputs and optional report assets.
- Optional LLM interpretation with RAG + memory notes.

### Baseline Model (`core/baseline.py`)
- `BaselineInputs` validates baseline CO2, fuel cost, optional VKT.
- `BaselineModel` produces a standardized `BaselineState`.

### AI Efficiency Scenarios (`core/ai_scenarios.py`)
- `AIScenario` applies a rate in [0, 1] to baseline emissions and fuel cost.
- Produces `AIScenarioResult` with new emissions, savings, and reductions.

### Economics Engine (`core/economics.py`)
- `EconomicInputs` holds policy instrument inputs (carbon price, tax, subsidy).
- `EconomicEvaluator` computes:
  - Carbon cost savings
  - Fuel cost savings
  - Fuel tax savings
  - Congestion charge savings
  - Total economic benefit
  - Marginal abatement cost (if AI cost is provided)
- `CarbonPriceScenario` enables named policy scenarios.

### Reporting (`core/reporting.py`)
- Builds report-ready CSVs and PNG figures from results.
- Uses pandas and matplotlib for aggregation and charts.

### RAG Utilities (`core/rag.py`)
- Local TF-IDF retrieval over docs in `data/knowledge_base`.
- No external embeddings; deterministic and auditable.
- Returns snippets for LLM interpretation context.

### Memory Notes (`core/memory.py`)
- Append-only JSONL for policy notes and assumptions.
- Loaded into LLM context for consistent interpretation.

### Streamlit Dashboard (`streamlit_app.py`)
- Interactive scenario builder and results explorer.
- Uses core evaluation logic from `main.py`.
- Optional LLM chat with RAG and memory notes.

## Data Flow (Evaluation Pipeline)
1. **Config validation**: JSON is validated for required keys and types.
2. **Baseline state**: `BaselineInputs` -> `BaselineState`.
3. **AI scenarios**: For each efficiency rate, compute emissions/fuel savings.
4. **Policy scenarios**: For each AI scenario and policy scenario, evaluate
   economic outcomes.
5. **Results payload**: Structured metadata + scenario result rows.
6. **Exports**: JSON/CSV outputs; optional report assets (CSV/PNG).
7. **Optional interpretation**: Summarize results for LLM interpretation,
   enhanced by local RAG docs and memory notes.

## Configuration Contract (High Level)
Required sections:
- `baseline` with `co2_emissions_ton_yr` and `fuel_cost_usd_yr`
- `ai_efficiency_rates` list
- `policy.carbon_price_scenarios` list

Optional sections:
- `baseline.vkt_km_yr`
- `policy.ai_implementation_cost_usd_yr`
- `policy.policy_defaults` (fuel tax, AI subsidy, congestion charges)
- `assumptions` list

See `README.md` for full JSON example.

## Capabilities
- Deterministic evaluation across AI efficiency and policy scenarios.
- Supports:
  - Carbon price (USD/ton CO2)
  - Fuel tax rate (share of fuel cost)
  - AI subsidy (USD/year)
  - Congestion charge (USD/year) and per-km charge (requires VKT)
- Computes:
  - Emissions reduction, fuel savings, carbon cost savings
  - Fuel tax and congestion charge impacts
  - Total economic benefit
  - Marginal abatement cost (when AI cost is provided)
- Exports results to JSON/CSV and report-ready CSV/PNG assets.
- Interactive Streamlit dashboard for scenario building and visualization.
- Optional LLM interpretation (Groq API) with:
  - Deterministic summaries
  - Local RAG context
  - Memory notes for consistent framing
- Docker-ready runtime.

## Outputs
- `results.json` or `results.csv` (scenario matrix)
- Report assets when `--report-dir` is used:
  - `scenario_results.csv`
  - `summary_by_policy.csv`
  - `summary_by_efficiency.csv`
  - PNG charts for benefit, emissions, and savings breakdown

## Directory Layout
```
core/                Core logic (baseline, scenarios, economics, reporting, RAG)
data/
  knowledge_base/    Local docs for RAG
  memory/            JSONL memory notes
main.py              CLI entrypoint
streamlit_app.py     Dashboard
tests/               Unit tests
```

## Validation and Guardrails
- Strict input validation (non-negative values, required fields).
- AI efficiency rates must be in [0, 1].
- Per-km congestion charge requires baseline VKT.
- LLM interpretation is read-only over results (no new calculations).

## Non-Goals and Limits
- No real-time or dynamic modeling.
- No optimization or traffic control algorithms.
- No predictive ML for AI performance.
- No causal inference or forecasting.

## Testing
- Unit tests in `tests/test_core.py` cover:
  - Economic calculations
  - Scenario evaluation
  - Config validation and defaults

## Extension Points
- Add new policy instruments in `core/economics.py`.
- Extend reporting outputs in `core/reporting.py`.
- Enhance RAG sources by adding files to `data/knowledge_base`.
- Integrate additional UI panels in `streamlit_app.py`.
