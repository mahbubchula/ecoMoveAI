"""
EcoMoveAI core package.
"""

from core.ai_scenarios import AIScenario, AIScenarioResult
from core.baseline import BaselineInputs, BaselineModel, BaselineState
from core.economics import (
    CarbonPriceScenario,
    EconomicEvaluator,
    EconomicInputs,
    EconomicResult,
    evaluate_carbon_price_scenarios,
)
from core.memory import MemoryNote, append_memory_note, load_memory_notes
from core.rag import RAGDocument, RAGIndex, build_index, load_documents_from_dir, search
from core.reporting import write_report_assets

__all__ = [
    "AIScenario",
    "AIScenarioResult",
    "BaselineInputs",
    "BaselineModel",
    "BaselineState",
    "CarbonPriceScenario",
    "EconomicEvaluator",
    "EconomicInputs",
    "EconomicResult",
    "evaluate_carbon_price_scenarios",
    "MemoryNote",
    "append_memory_note",
    "load_memory_notes",
    "RAGDocument",
    "RAGIndex",
    "build_index",
    "load_documents_from_dir",
    "search",
    "write_report_assets",
]

__version__ = "0.2.0"
