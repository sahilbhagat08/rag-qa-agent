"""Unit tests for agent factory and memory."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestCreateAgent:
    @patch("app.agent.agent.get_session_memory")
    @patch("app.agent.agent.ChatAnthropic")
    def test_create_agent_returns_executor(self, mock_llm_cls, mock_memory):
        """create_agent() returns an AgentExecutor without errors."""
        from langchain.agents import AgentExecutor
        from app.agent.agent import create_agent

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_memory.return_value = MagicMock()

        # Patch create_tool_calling_agent to avoid real LLM calls
        with patch("app.agent.agent.create_tool_calling_agent") as mock_create:
            mock_create.return_value = MagicMock()
            agent = create_agent("test-session-001")

        assert isinstance(agent, AgentExecutor)

    @patch("app.agent.agent.get_session_memory")
    @patch("app.agent.agent.ChatAnthropic")
    def test_different_sessions_get_different_memory(self, mock_llm_cls, mock_memory):
        """Each session_id gets its own memory instance."""
        from app.agent.agent import create_agent

        mock_llm_cls.return_value = MagicMock()
        mock_memory.side_effect = lambda sid: MagicMock(session_id=sid)

        with patch("app.agent.agent.create_tool_calling_agent") as mock_create:
            mock_create.return_value = MagicMock()
            create_agent("session-A")
            create_agent("session-B")

        calls = [c[0][0] for c in mock_memory.call_args_list]
        assert "session-A" in calls
        assert "session-B" in calls


class TestSessionMemory:
    @patch("app.agent.memory.RedisChatMessageHistory")
    def test_get_session_memory_uses_session_id(self, mock_history_cls):
        from app.agent.memory import get_session_memory

        mock_history_cls.return_value = MagicMock()
        get_session_memory("my-session-xyz")

        mock_history_cls.assert_called_once()
        call_kwargs = mock_history_cls.call_args[1]
        assert call_kwargs["session_id"] == "my-session-xyz"

    @patch("app.agent.memory.RedisChatMessageHistory")
    def test_clear_session_calls_history_clear(self, mock_history_cls):
        from app.agent.memory import clear_session_memory

        mock_history = MagicMock()
        mock_history_cls.return_value = mock_history
        clear_session_memory("session-to-clear")

        mock_history.clear.assert_called_once()
