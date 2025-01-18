from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    """Configuration settings for the application."""
    FLASK_PORT: int = 5000  # Port for Flask server
    OLLAMA_API_URL: str = "http://localhost:11434/api/generate"  # LLM API URL
    SYSTEM_PROMPT: str = """You are a helpful AI assistant. In regular mode, engage in conversation naturally, providing
     clear and accurate responses. In RAG mode, focus on answering questions based on provided context. Avoid assumptions
      about user identity or context unless explicitly stated."""
    CONTEXT_WINDOW: int = 2000  # Token limit for conversation history
    # add urls or local pdf
    RAG_URLS: List[str] = field(default_factory=lambda: [
        "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001813756/975b3e9b-268e-4798-a9e4-2a9a7c92dc10.pdf",
    ])  # Default documents for RAG mode