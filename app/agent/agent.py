from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic

from app.agent.memory import get_session_memory
from app.agent.prompts import get_agent_prompt
from app.agent.tools.calculator import calculator
from app.agent.tools.retriever import retrieve_documents
from app.agent.tools.summarizer import summarize_document
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

TOOLS = [retrieve_documents, summarize_document, calculator]


def create_agent(session_id: str) -> AgentExecutor:
    """
    Factory that creates a fully wired AgentExecutor for a given session.

    The agent uses Claude's native tool calling (not ReAct string parsing),
    with Redis-backed conversation memory scoped to session_id.
    """
    settings = get_settings()
    logger.info("creating_agent", session_id=session_id)

    llm = ChatAnthropic(
        model=settings.ANTHROPIC_MODEL,
        api_key=settings.ANTHROPIC_API_KEY,
        streaming=True,
        max_tokens=4096,
    )

    prompt = get_agent_prompt()
    agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)
    memory = get_session_memory(session_id)

    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        max_iterations=settings.AGENT_MAX_ITERATIONS,
        verbose=False,
        return_intermediate_steps=False,
        handle_parsing_errors=True,
    )

    logger.info("agent_created", session_id=session_id, tools=[t.name for t in TOOLS])
    return executor
