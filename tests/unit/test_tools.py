"""Unit tests for agent tools."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── calculator ────────────────────────────────────────────────────────────────

from app.agent.tools.calculator import calculator, _safe_eval
import ast


class TestCalculator:
    def test_basic_addition(self):
        assert calculator.invoke({"expression": "2 + 3"}) == "5"

    def test_multiplication(self):
        assert calculator.invoke({"expression": "6 * 7"}) == "42"

    def test_float_result(self):
        result = calculator.invoke({"expression": "1 / 3"})
        assert float(result) == pytest.approx(1 / 3, rel=1e-6)

    def test_power(self):
        assert calculator.invoke({"expression": "2 ** 10"}) == "1024"

    def test_sqrt(self):
        assert calculator.invoke({"expression": "sqrt(144)"}) == "12"

    def test_pi_constant(self):
        result = float(calculator.invoke({"expression": "pi"}))
        assert result == pytest.approx(3.14159, rel=1e-4)

    def test_complex_expression(self):
        result = calculator.invoke({"expression": "sqrt(16) + 2 ** 3 - 1"})
        assert result == "11"

    def test_division_by_zero(self):
        result = calculator.invoke({"expression": "1 / 0"})
        assert "Error" in result

    def test_disallowed_function(self):
        result = calculator.invoke({"expression": "open('file')"})
        assert "Error" in result

    def test_invalid_expression(self):
        result = calculator.invoke({"expression": "not_a_function()"})
        assert "Error" in result

    def test_integer_float_formatting(self):
        # sqrt(4) = 2.0 → should format as "2" not "2.0"
        assert calculator.invoke({"expression": "sqrt(4)"}) == "2"


# ── retrieve_documents ────────────────────────────────────────────────────────

class TestRetrieveDocuments:
    @patch("app.agent.tools.retriever._get_retriever")
    def test_returns_formatted_results(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.similarity_search.return_value = [
            Document(
                page_content="FastAPI is a modern web framework.",
                metadata={"source": "docs.pdf", "page": 1},
            ),
            Document(
                page_content="It supports async endpoints.",
                metadata={"source": "docs.pdf", "page": 2},
            ),
        ]
        mock_get_retriever.return_value = mock_retriever

        from app.agent.tools.retriever import retrieve_documents
        result = retrieve_documents.invoke({"query": "web framework", "k": 2})

        assert "[1]" in result
        assert "docs.pdf" in result
        assert "FastAPI" in result

    @patch("app.agent.tools.retriever._get_retriever")
    def test_no_results(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.similarity_search.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from app.agent.tools.retriever import retrieve_documents
        result = retrieve_documents.invoke({"query": "nonexistent topic"})

        assert "No relevant documents" in result

    @patch("app.agent.tools.retriever._get_retriever")
    def test_k_capped_at_10(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.similarity_search.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from app.agent.tools.retriever import retrieve_documents
        retrieve_documents.invoke({"query": "test", "k": 100})

        # Should have been capped to 10
        call_kwargs = mock_retriever.similarity_search.call_args
        assert call_kwargs[1]["k"] == 10 or call_kwargs[0][1] == 10
