from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from src.config.prompt import GENERATE_SYSTEM_PROMPT, RAG_USER_TEMPLATE

load_dotenv()


class BaseLLMClient(ABC):
    """Abstract base class for LLM generation clients."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Generate an answer given query and context."""
        pass

    @abstractmethod
    def invoke_messages(self, messages: list[dict[str, str]]) -> str:
        """Invoke LLM directly with custom messages."""
        pass


def _to_langchain_messages(messages: list[dict[str, str]]) -> list:
    """Convert simple role/content dictionaries to LangChain messages."""
    langchain_messages = []

    for message in messages:
        if message["role"] == "system":
            langchain_messages.append(
                SystemMessage(content=message["content"])
            )
        elif message["role"] == "user":
            langchain_messages.append(
                HumanMessage(content=message["content"])
            )
        else:
            langchain_messages.append(
                HumanMessage(content=message["content"])
            )

    return langchain_messages


def _response_to_text(response: Any) -> str:
    """Normalize a LangChain response into plain text."""

    if hasattr(response, "content"):
        content = response.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "\n".join(str(item) for item in content)

        return str(content)

    return str(response)


class GroqLLMClient(BaseLLMClient):
    """LLM client implementation using Groq OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        llm: Any = None,
    ):
        model = model_name or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )

        super().__init__(model_name=model)

        self.api_key = api_key or os.getenv(
            "GROQ_API_KEY",
            "",
        )

        self.base_url = base_url or os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1",
        )

        self.temperature = temperature
        self.max_tokens = max_tokens

        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": GENERATE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": RAG_USER_TEMPLATE.format(
                    context=context,
                    query=query,
                ),
            },
        ]

        return self.invoke_messages(messages)

    def invoke_messages(
        self,
        messages: list[dict[str, str]],
        max_retries: int = 3,
    ) -> str:

        langchain_messages = _to_langchain_messages(messages)

        for attempt in range(max_retries + 1):

            try:
                response = self.llm.invoke(
                    langchain_messages
                )

                return _response_to_text(response)

            except Exception as e:

                err_str = str(e)

                if (
                    (
                        "429" in err_str
                        or "rate_limit" in err_str.lower()
                    )
                    and attempt < max_retries
                ):

                    wait_time = 5 * (attempt + 1)

                    logger.warning(
                        f"Rate limited on {self.model_name}. "
                        f"Retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )

                    time.sleep(wait_time)

                else:
                    raise

        return ""


class GeminiLLMClient(BaseLLMClient):
    """LLM client implementation for Google Gemini via OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/openai/"
        ),
        temperature: float = 0.0,
        max_tokens: int = 2048,
        llm: Any = None,
    ):

        model = model_name or os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash",
        )

        super().__init__(model_name=model)

        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY", "")
        )

        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": GENERATE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": RAG_USER_TEMPLATE.format(
                    context=context,
                    query=query,
                ),
            },
        ]

        return self.invoke_messages(messages)

    def invoke_messages(
        self,
        messages: list[dict[str, str]],
        max_retries: int = 3,
    ) -> str:

        langchain_messages = _to_langchain_messages(messages)

        for attempt in range(max_retries + 1):

            try:
                response = self.llm.invoke(
                    langchain_messages
                )

                return _response_to_text(response)

            except Exception as e:

                err_str = str(e)

                if (
                    (
                        "429" in err_str
                        or "rate_limit" in err_str.lower()
                    )
                    and attempt < max_retries
                ):

                    wait_time = 5 * (attempt + 1)

                    logger.warning(
                        f"Rate limited on Gemini "
                        f"{self.model_name}. "
                        f"Retrying in {wait_time}s..."
                    )

                    time.sleep(wait_time)

                else:
                    raise

        return ""


class OpenRouterLLMClient(BaseLLMClient):
    """
    LLM client implementation using OpenRouter's
    OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        llm: Any = None,
    ):

        model = model_name or os.getenv(
            "OPENROUTER_MODEL",
            "openai/gpt-oss-120b",
        )

        super().__init__(model_name=model)

        self.api_key = api_key or os.getenv(
            "OPENROUTER_API_KEY",
            "",
        )

        self.base_url = base_url or os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        )

        self.temperature = temperature
        self.max_tokens = max_tokens

        if llm is not None:
            self.llm = llm

        else:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                default_headers={
                    "HTTP-Referer": os.getenv(
                        "OPENROUTER_SITE_URL",
                        "http://localhost:3000",
                    ),
                    "X-Title": os.getenv(
                        "OPENROUTER_APP_NAME",
                        "Hybrid RAG Evaluation",
                    ),
                },
            )

    def generate(
        self,
        query: str,
        context: str,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": GENERATE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": RAG_USER_TEMPLATE.format(
                    context=context,
                    query=query,
                ),
            },
        ]

        return self.invoke_messages(messages)

    def invoke_messages(
        self,
        messages: list[dict[str, str]],
        max_retries: int = 3,
    ) -> str:

        langchain_messages = _to_langchain_messages(messages)

        for attempt in range(max_retries + 1):

            try:
                response = self.llm.invoke(
                    langchain_messages
                )

                return _response_to_text(response)

            except Exception as e:

                err_str = str(e)

                if (
                    (
                        "429" in err_str
                        or "rate_limit" in err_str.lower()
                    )
                    and attempt < max_retries
                ):

                    wait_time = 5 * (attempt + 1)

                    logger.warning(
                        f"Rate limited on OpenRouter "
                        f"{self.model_name}. "
                        f"Retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )

                    time.sleep(wait_time)

                else:
                    raise

        return ""


def get_eval_llm(
    name: Literal[
        "model_1",
        "model_2",
        "gemini",
        "openrouter",
    ] | str = "model_1",
    api_key: str | None = None,
) -> BaseLLMClient:
    """
    Factory for evaluation LLMs and the configured LLM judge.
    """

    # ---------------------------------------------------------
    # Model 1
    # ---------------------------------------------------------

    if name == "model_1":

        model_name = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        )

        return GroqLLMClient(
            model_name=model_name,
            api_key=api_key,
        )

    # ---------------------------------------------------------
    # Model 2
    # ---------------------------------------------------------

    if name == "model_2":

        model_name = os.getenv(
            "GROQ_MODEL_2",
            "qwen/qwen3.8-27b",
        )

        return GroqLLMClient(
            model_name=model_name,
            api_key=api_key,
        )

    # ---------------------------------------------------------
    # OpenRouter Judge
    # ---------------------------------------------------------

    if (
        name == "openrouter"
        or name.startswith("openrouter-")
    ):

        if name.startswith("openrouter-"):

            model_name = name[
                len("openrouter-"):
            ]

        else:

            model_name = os.getenv(
                "OPENROUTER_MODEL",
                "openai/gpt-oss-120b",
            )

        return OpenRouterLLMClient(
            model_name=model_name,
            api_key=api_key,
        )

    # ---------------------------------------------------------
    # Gemini
    # ---------------------------------------------------------

    if (
        name == "gemini"
        or name.startswith("gemini-")
    ):

        model_name = (
            name
            if name.startswith("gemini-")
            else os.getenv(
                "GEMINI_MODEL",
                "gemini-3.8-flash",
            )
        )

        return GeminiLLMClient(
            model_name=model_name,
            api_key=api_key,
        )

    # ---------------------------------------------------------
    # OpenAI-compatible GPT branch
    # ---------------------------------------------------------

    if (
        name == "gpt-4o-mini"
        or name.startswith("gpt-")
    ):

        openai_key = (
            api_key
            or os.getenv(
                "OPENAI_API_KEY",
                "",
            )
        )

        return GroqLLMClient(
            model_name=name,
            api_key=openai_key,
            base_url="https://api.openai.com/v1",
        )

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------

    return GroqLLMClient(
        model_name=name,
        api_key=api_key,
    )