'use client';

import { useCallback, useEffect, useState } from 'react';

export interface ApiKeyStatus {
  groq: boolean;
  openrouter: boolean;
  tavily: boolean;
}

const EMPTY_STATUS: ApiKeyStatus = {
  groq: false,
  openrouter: false,
  tavily: false,
};

function readApiKeyStatus(): ApiKeyStatus {
  if (typeof window === 'undefined') return EMPTY_STATUS;
  return {
    groq: Boolean(sessionStorage.getItem('groq_api_key')),
    openrouter: Boolean(sessionStorage.getItem('openrouter_api_key')),
    tavily: Boolean(sessionStorage.getItem('tavily_api_key')),
  };
}

export function useApiKeyStatus() {
  const [status, setStatus] = useState<ApiKeyStatus>(EMPTY_STATUS);
  const refresh = useCallback(() => setStatus(readApiKeyStatus()), []);

  useEffect(() => {
    refresh();
    window.addEventListener('apiKeyUpdated', refresh);
    window.addEventListener('storage', refresh);
    return () => {
      window.removeEventListener('apiKeyUpdated', refresh);
      window.removeEventListener('storage', refresh);
    };
  }, [refresh]);

  return {
    status,
    chatReady: status.groq && status.tavily,
    evaluationReady: status.groq && status.openrouter && status.tavily,
    refresh,
  };
}
