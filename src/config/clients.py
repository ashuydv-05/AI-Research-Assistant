from contextvars import ContextVar
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from src.config.settings import settings

load_dotenv()

# Per-request Groq key from the UI. Do not write this into process-wide os.environ.
request_groq_api_key: ContextVar[str | None] = ContextVar(
    "request_groq_api_key", default=None
)

request_tavily_api_key: ContextVar[str | None] = ContextVar(
    "request_tavily_api_key", default=None
)


def extract_message_text(response) -> str:
    """Qwen/GPT-OSS often put the visible answer in reasoning fields, leaving .content empty."""
    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        joined = "\n".join(parts).strip()
        if joined:
            return joined
    additional = getattr(response, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning"):
        value = additional.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_llm_client(
    *, max_tokens: int | None = None, temperature: float = 0.7
) -> ChatOpenAI:
    resolved_max_tokens = max_tokens or int(os.getenv("GROQ_MAX_TOKENS", "8192"))
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    api_key = request_groq_api_key.get() or os.getenv("GROQ_API_KEY", "")
    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        ),
        model=model,
        temperature=temperature,
        max_tokens=resolved_max_tokens,
    )


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", settings.qdrant.url)
    api_key = os.getenv("QDRANT_API_KEY", settings.qdrant.api_key)
    return QdrantClient(
        url=url,
        api_key=api_key if api_key else None,
        check_compatibility=False,
    )


def get_es_client() -> Elasticsearch | None:
    es_url = os.getenv("ES_URL", settings.elasticsearch.url)
    es_api_key = os.getenv("ES_API_KEY", settings.elasticsearch.api_key)

    if not es_url:
        return None

    try:
        return Elasticsearch(
            es_url,
            api_key=es_api_key,
            request_timeout=settings.elasticsearch.request_timeout,
            max_retries=0,
        )
    except Exception:
        return None
