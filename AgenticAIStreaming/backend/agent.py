"""
LangGraph agent implementation using Python LangChain
"""

from typing import AsyncGenerator, List, Dict, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

load_dotenv()


# Define the state for the agent using TypedDict
class AgentState(TypedDict):
    messages: List[BaseMessage]


# Initialize the OpenAI model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    streaming=True,
)


# Define the agent node
async def agent_node(state: AgentState) -> Dict[str, Any]:
    """Process messages through the agent"""
    messages = state.get("messages", [])

    # Add system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(
            content="You are a helpful AI assistant. Provide clear and concise answers."
        )
        messages = [system_msg] + messages

    # Get response from model
    response = await model.ainvoke(messages)

    return {"messages": [response]}


# Build the graph
def create_agent_graph():
    """Create and compile the LangGraph agent"""
    workflow = StateGraph(AgentState)

    # Add the agent node
    workflow.add_node("agent", agent_node)

    # Set entry point
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()


# Singleton graph instance
_agent_graph = None


def get_agent_graph():
    """Get or create the agent graph singleton"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
    return _agent_graph


# Streaming function - simplified direct streaming
async def stream_agent_response(
    user_message: str, conversation_history: List[Dict[str, str]] = None
) -> AsyncGenerator[str, None]:
    """
    Stream agent responses directly from the model

    Args:
        user_message: The user's message
        conversation_history: List of previous messages in format [{"role": "user|assistant", "content": "..."}]

    Yields:
        Response chunks as strings
    """
    if conversation_history is None:
        conversation_history = []

    # Convert conversation history to LangChain messages
    messages = []

    # Add system message first
    system_msg = SystemMessage(
        content="You are a helpful AI assistant. Provide clear and concise answers."
    )
    messages.append(system_msg)

    # Add conversation history
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # Add current user message
    messages.append(HumanMessage(content=user_message))

    # Stream response directly from model
    async for chunk in model.astream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content
