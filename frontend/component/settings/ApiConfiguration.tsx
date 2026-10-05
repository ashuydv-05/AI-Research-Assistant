'use client';

import { FormEvent, useState } from 'react';
import { Check, KeyRound, Trash2 } from 'lucide-react';
import { useApiKeyStatus } from '@/hooks/useApiKeyStatus';

type KeyName = 'groq_api_key' | 'openrouter_api_key' | 'tavily_api_key';

export function ApiConfiguration() {
  const [groqKey, setGroqKey] = useState('');
  const [openRouterKey, setOpenRouterKey] = useState('');
  const [tavilyKey, setTavilyKey] = useState('');
  const { status: configured, chatReady, evaluationReady, refresh } = useApiKeyStatus();
  const [saved, setSaved] = useState(false);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (groqKey.trim()) sessionStorage.setItem('groq_api_key', groqKey.trim());
    if (openRouterKey.trim()) sessionStorage.setItem('openrouter_api_key', openRouterKey.trim());
    if (tavilyKey.trim()) sessionStorage.setItem('tavily_api_key', tavilyKey.trim());
    setGroqKey('');
    setOpenRouterKey('');
    setTavilyKey('');
    window.dispatchEvent(new Event('apiKeyUpdated'));
    refresh();
    setSaved(true);
    window.setTimeout(() => setSaved(false), 1800);
  };

  const clearKey = (key: KeyName) => {
    sessionStorage.removeItem(key);
    window.dispatchEvent(new Event('apiKeyUpdated'));
    refresh();
  };

  return (
    <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_320px] lg:items-start">
      <form onSubmit={handleSubmit} className="space-y-5">
        <div className="grid gap-3 sm:grid-cols-2">
          <Readiness label="Chat" ready={chatReady} detail="Groq + Tavily required" />
          <Readiness label="Evaluation" ready={evaluationReady} detail="Groq + OpenRouter + Tavily required" />
        </div>
        <KeyField
          id="groq-key"
          label="Groq API Key"
          value={groqKey}
          configured={configured.groq}
          requirement="Required for Chat and Evaluation"
          onChange={setGroqKey}
          onClear={() => clearKey('groq_api_key')}
        />
        <KeyField
          id="openrouter-key"
          label="OpenRouter API Key"
          value={openRouterKey}
          configured={configured.openrouter}
          requirement="Required only for Evaluation"
          onChange={setOpenRouterKey}
          onClear={() => clearKey('openrouter_api_key')}
        />
        <KeyField
          id="tavily-key"
          label="Tavily API Key"
          value={tavilyKey}
          configured={configured.tavily}
          requirement="Required for Chat web fallback and Evaluation"
          onChange={setTavilyKey}
          onClear={() => clearKey('tavily_api_key')}
        />

        <button
          type="submit"
          disabled={!groqKey.trim() && !openRouterKey.trim() && !tavilyKey.trim()}
          className="inline-flex min-h-11 items-center gap-2 rounded-md bg-indigo-600 px-4 text-sm font-bold text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {saved ? <Check size={17} /> : <KeyRound size={17} />}
          {saved ? 'Configuration saved' : 'Save configuration'}
        </button>
      </form>

      <div className="border-l-0 border-slate-200 lg:border-l lg:pl-8">
        <div className="space-y-6 text-sm leading-6 text-slate-600">
          <div>
            <h3 className="font-bold text-slate-950">Why Groq?</h3>
            <p className="mt-1">Groq is used for fast LLM inference and answer generation in the research assistant.</p>
          </div>
          <div>
            <h3 className="font-bold text-slate-950">Why OpenRouter?</h3>
            <p className="mt-1">OpenRouter provides the LLM-as-Judge used to score generated answers during Evaluation.</p>
          </div>
          <div>
            <h3 className="font-bold text-slate-950">Why Tavily?</h3>
            <p className="mt-1">Tavily supplies web sources only when the Hybrid RAG validation step fails.</p>
          </div>
          <p className="border-t border-slate-200 pt-5 text-xs text-slate-500">
            Each user must provide their own keys. Keys remain masked for this browser session and are sent only to the project API.
          </p>
        </div>
      </div>
    </div>
  );
}

function KeyField({
  id,
  label,
  value,
  configured,
  requirement,
  onChange,
  onClear,
}: {
  id: string;
  label: string;
  value: string;
  configured: boolean;
  requirement: string;
  onChange: (value: string) => void;
  onClear: () => void;
}) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-3">
        <label htmlFor={id} className="text-sm font-bold text-slate-900">{label}</label>
        {configured && (
          <span className="inline-flex items-center gap-1 text-xs font-semibold text-emerald-700">
            <Check size={13} /> Saved
          </span>
        )}
      </div>
      <p className="mb-2 text-xs text-slate-500">{requirement}</p>
      <div className="flex gap-2">
        <input
          id={id}
          type="password"
          autoComplete="off"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={configured ? 'Enter a new key to replace the browser key' : 'Enter API key'}
          className="min-h-11 min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-900 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
        />
        {configured && (
          <button
            type="button"
            onClick={onClear}
            title={`Remove ${label}`}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-md border border-slate-300 text-slate-500 transition hover:border-rose-300 hover:bg-rose-50 hover:text-rose-700"
          >
            <Trash2 size={17} />
          </button>
        )}
      </div>
    </div>
  );
}

function Readiness({ label, ready, detail }: { label: string; ready: boolean; detail: string }) {
  return (
    <div className={`rounded-md border px-3 py-2 ${ready ? 'border-emerald-200 bg-emerald-50' : 'border-amber-200 bg-amber-50'}`}>
      <p className={`text-xs font-bold ${ready ? 'text-emerald-800' : 'text-amber-900'}`}>{label}: {ready ? 'Ready' : 'Keys missing'}</p>
      <p className="mt-0.5 text-xs text-slate-600">{detail}</p>
    </div>
  );
}
