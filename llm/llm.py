# llm/llm.py

from typing import Any
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os


def get_llm(model: str) -> Any:
    """
    Returns an LLM instance based on the model name.
    
    Args:
        model (str): model name (e.g., "groq", "openai")
    
    Returns:
        LLM instance
    """

    if model.lower() == "groq":
        return ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    elif model.lower() == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )

    else:
        raise ValueError(f"Unsupported model: {model}")