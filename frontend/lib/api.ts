import type { ChatRequest, ChatResponse } from '@/types/chat';

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (process.env.NODE_ENV === "development"
    ? "http://localhost:8000/api"
    : "https://arxiv-rag-backend.onrender.com/api");

export async function sendMessage(
  request: ChatRequest
): Promise<ChatResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (typeof window !== "undefined") {
    const groqKey = sessionStorage.getItem("groq_api_key");
    const tavilyKey = sessionStorage.getItem("tavily_api_key");

    //Adding API KEY to HTTP headers
    if (groqKey) headers["x-groq-api-key"] = groqKey;
    if (tavilyKey) headers["x-tavily-api-key"] = tavilyKey;
  }

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let errorMessage = `Request failed with status ${response.status}`;

    try {
      const errorData = await response.json();
      if (errorData?.detail) {
        errorMessage = errorData.detail;
      }
    } catch {
      // Keep default error message
    }

    throw new Error(errorMessage);
  }

  return response.json();
}
