import requests
from typing import Dict, Optional, List
from config import Config

class LLMHandler:
    """Handles interaction with the LLM API."""
    def __init__(self, api_url: str, system_prompt: str):
        self.api_url = api_url
        self.system_prompt = system_prompt
        self.conversation_history = []  # Store conversation history
        self.current_mode = "chat"  # Default mode

    def generate_response(self, user_input: str, context: Optional[str] = None) -> str:
        """Send a query to the LLM and get a response."""
        messages = self._prepare_messages(user_input, context)
        response = requests.post(
            self.api_url,
            json={
                "model": "gemma:2b",  # Specify the LLM model
                "prompt": self._format_prompt(messages),
                "stream": False
            }
        )
        if response.status_code == 200:
            result = response.json()
            assistant_message = result['response']
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        else:
            return f"Error: {response.status_code}"

    def _prepare_messages(self, user_input: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        """Prepare messages for the LLM with history and context."""
        messages = [{"role": "system", "content": self.system_prompt}]
        history_tokens = 0
        for msg in reversed(self.conversation_history):
            msg_tokens = len(msg["content"].split())
            if history_tokens + msg_tokens > Config.CONTEXT_WINDOW:
                break
            messages.insert(1, msg)
            history_tokens += msg_tokens
        if context:
            messages.append({"role": "system", "content": f"Use this context:{context}"})
        messages.append({"role": "user", "content": user_input})
        return messages

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single prompt string."""
        formatted = ""
        for msg in messages:
            if msg["role"] == "system":
                formatted += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                formatted += f"Human: {msg['content']}\n\n"
            else:
                formatted += f"Assistant: {msg['content']}\n\n"
        return formatted

    def switch_mode(self, new_mode: str):
        """Switch between chat and RAG modes."""
        self.current_mode = new_mode