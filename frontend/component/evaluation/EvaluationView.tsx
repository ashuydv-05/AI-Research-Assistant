'use client';

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import {
  Play,
  RotateCw,
  Trophy,
  BarChart3,
  Layers,
  Cpu,
  CheckCircle2,
  FileText,
  AlertCircle,
  Sparkles,
  ArrowLeft,
  Terminal,
  Bot,
  Activity,
  KeyRound,
} from 'lucide-react';
import { useApiKeyStatus } from '@/hooks/useApiKeyStatus';

interface ConfigSummary {
  retrieval: string;
  llm: string;
  total_samples: number;
  avg_correctness: number;
  avg_faithfulness: number;
  avg_relevance: number;
  avg_overall: number;
  avg_precision_at_k: number | null;
  avg_recall_at_k: number | null;
  avg_mrr: number | null;
  avg_latency_ms: number;
}

interface EvaluationSummary {
  status?: string;
  message?: string;
  valid?: boolean;
  warning?: string;
  timestamp?: string;
  dataset_path?: string;
  judge_model?: string;
  best_configuration?: string;
  best_reason?: string;
  comparison_matrix?: {
    [retrieval: string]: {
      [llm: string]: number;
    };
  };
  configurations?: {
    [key: string]: ConfigSummary;
  };
}

interface ConfigInfo {
  model_1: { key: string; name: string; provider: string; family: string };
  model_2: { key: string; name: string; provider: string; family: string };
  judge: { name: string; provider: string; description: string };
  retrievers: {
    vector: { name: string; description: string };
    hybrid: { name: string; description: string };
  };
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  (process.env.NODE_ENV === 'development'
    ? 'http://localhost:8000/api'
    : 'https://arxiv-rag-backend.onrender.com/api');

interface EvaluationViewProps {
  onBackToChat?: () => void;
}

interface StreamLog {
  id: string;
  timestamp: string;
  message: string;
  type: 'info' | 'step' | 'success' | 'start' | 'error';
}

interface EvaluationStreamEvent {
  type: 'combination_start' | 'step' | 'sample_complete' | 'complete' | 'error';
  message: string;
  timestamp?: string;
  best_configuration?: string;
  best_reason?: string;
  comparison_matrix?: EvaluationSummary['comparison_matrix'];
  configurations?: EvaluationSummary['configurations'];
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

export function EvaluationView({ onBackToChat }: EvaluationViewProps) {
  const { status, evaluationReady } = useApiKeyStatus();
  const [summary, setSummary] = useState<EvaluationSummary | null>(null);
  const [configInfo, setConfigInfo] = useState<ConfigInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [maxQuestions, setMaxQuestions] = useState<number>(5);

  // Streaming State
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamProgress, setStreamProgress] = useState<{ current: number; total: number; percent: number }>({
    current: 0,
    total: 0,
    percent: 0,
  });
  const [currentAction, setCurrentAction] = useState<string>('');
  const [logs, setLogs] = useState<StreamLog[]>([]);
  const logTerminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logTerminalRef.current) {
      logTerminalRef.current.scrollTop = logTerminalRef.current.scrollHeight;
    }
  }, [logs]);

  const fetchConfigInfo = async () => {
    try {
      const res = await fetch(`${API_BASE}/evaluation/config-info`);
      if (res.ok) {
        const data = await res.json();
        setConfigInfo(data);
      }
    } catch {
      // Use fallback defaults
      setConfigInfo({
        model_1: { key: 'model_1', name: 'qwen/qwen3.8-27b', provider: 'Groq', family: 'Alibaba Qwen' },
        model_2: { key: 'model_2', name: 'openai/gpt-oss-20b', provider: 'Groq', family: 'OpenAI Family' },
        judge: { name: 'openai/gpt-oss-120b', provider: 'OpenRouter', description: 'LLM Judge' },
        retrievers: {
          vector: { name: 'Vector Retrieval', description: 'Qdrant dense vector search' },
          hybrid: { name: 'Hybrid Retrieval', description: 'Qdrant + Elasticsearch BM25 + RRF' },
        },
      });
    }
  };

  const fetchSummary = async () => {
    setFetching(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/evaluation/summary`);
      if (res.ok) {
        const data = await res.json();
        setSummary(data);
      } else {
        setError('Could not fetch evaluation summary.');
      }
    } catch (err: unknown) {
      setError(errorMessage(err, 'Error connecting to API.'));
    } finally {
      setFetching(false);
    }
  };

  useEffect(() => {
    fetchConfigInfo();
    fetchSummary();
  }, []);

  const handleRunStreamingEvaluation = async () => {
    if (!evaluationReady) {
      setError('Add your Groq, OpenRouter, and Tavily API keys in Settings before running Evaluation.');
      return;
    }

    const groqKey = sessionStorage.getItem('groq_api_key');
    const openRouterKey = sessionStorage.getItem('openrouter_api_key');
    const tavilyKey = sessionStorage.getItem('tavily_api_key');

    if (!groqKey || !openRouterKey || !tavilyKey) {
      setError('Add your Groq, OpenRouter, and Tavily API keys in Settings before running Evaluation.');
      return;
    }

    setLoading(true);
    setIsStreaming(true);
    setError(null);
    setLogs([]);
    setStreamProgress({ current: 0, total: maxQuestions * 4, percent: 0 });
    setCurrentAction('Initializing 2×2 benchmark runner...');

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'x-groq-api-key': groqKey,
      'x-openrouter-api-key': openRouterKey,
      'x-tavily-api-key': tavilyKey,
    };

    try {
      const response = await fetch(`${API_BASE}/evaluation/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          dataset_path: 'data/evaluation/evaluation_dataset.json',
          top_k: 5,
          retrieval: 'both',
          llm: 'both',
          max_questions: maxQuestions > 0 ? maxQuestions : undefined,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Evaluation failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const eventData = JSON.parse(line.replace('data: ', '').trim()) as EvaluationStreamEvent;
              handleStreamEvent(eventData);
            } catch (e) {
              console.error('Failed to parse SSE line:', line, e);
            }
          }
        }
      }
    } catch (err: unknown) {
      setError(errorMessage(err, 'Failed to stream evaluation.'));
      setIsStreaming(false);
      setLoading(false);
    }
  };

  const handleStreamEvent = (event: EvaluationStreamEvent) => {
    const timeStr = new Date().toLocaleTimeString();

    if (event.type === 'combination_start') {
      setCurrentAction(event.message);
      setLogs((prev) => [
        ...prev,
        { id: Math.random().toString(), timestamp: timeStr, message: event.message, type: 'start' },
      ]);
    } else if (event.type === 'step') {
      setCurrentAction(event.message);
      setLogs((prev) => [
        ...prev,
        { id: Math.random().toString(), timestamp: timeStr, message: event.message, type: 'step' },
      ]);
    } else if (event.type === 'sample_complete') {
      setStreamProgress((prev) => {
        const nextCurrent = prev.current + 1;
        const total = maxQuestions * 4;
        return {
          current: nextCurrent,
          total,
          percent: Math.min(Math.round((nextCurrent / total) * 100), 100),
        };
      });
      setLogs((prev) => [
        ...prev,
        { id: Math.random().toString(), timestamp: timeStr, message: event.message, type: 'success' },
      ]);
    } else if (event.type === 'complete') {
      setIsStreaming(false);
      setLoading(false);
      setCurrentAction('Benchmark Complete!');
      setLogs((prev) => [
        ...prev,
        { id: Math.random().toString(), timestamp: timeStr, message: event.message, type: 'success' },
      ]);
      setSummary({
        timestamp: event.timestamp,
        best_configuration: event.best_configuration,
        best_reason: event.best_reason,
        comparison_matrix: event.comparison_matrix,
        configurations: event.configurations,
      });
    } else if (event.type === 'error') {
      setError(event.message);
      setIsStreaming(false);
      setLoading(false);
      setLogs((prev) => [
        ...prev,
        { id: Math.random().toString(), timestamp: timeStr, message: `Error: ${event.message}`, type: 'error' },
      ]);
    }
  };

  const matrix = summary?.comparison_matrix || {};
  const configs = summary?.configurations || {};
  const bestConfigKey = summary?.valid === false ? undefined : summary?.best_configuration;

  const model1Name = configInfo?.model_1.name || 'qwen/qwen3.8-27b';
  const model2Name = configInfo?.model_2.name || 'openai/gpt-oss-20b';
  const judgeName = configInfo?.judge.name || (summary?.valid !== false ? summary?.judge_model : undefined) || 'openai/gpt-oss-120b';

  const getReadableConfigName = (key: string) => {
    return key
      .replace('model_1', model1Name)
      .replace('model_2', model2Name)
      .replace('vector', 'Vector')
      .replace('hybrid', 'Hybrid');
  };

  return (
    <div className="flex-1 overflow-y-auto bg-[#f8f9fc] p-4 sm:p-6 md:p-10">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header Bar */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200/80 pb-6">
          <div className="min-w-0 space-y-2">
            <div className="flex flex-wrap items-center gap-3">
              {onBackToChat && (
                <button
                  onClick={onBackToChat}
                  className="flex items-center gap-1.5 text-xs font-semibold text-[#5542f6] hover:text-[#4332e6] bg-[#eeebff] px-3.5 py-1.5 rounded-md border border-indigo-100 transition-colors shadow-xs cursor-pointer"
                >
                  <ArrowLeft className="w-3.5 h-3.5" /> Back to Chat
                </button>
              )}
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                2 Retrievers × 2 LLMs
              </span>
            </div>
            <h1 className="flex items-start gap-2 text-2xl font-extrabold tracking-normal text-slate-900 sm:text-3xl">
              <BarChart3 className="mt-1 h-7 w-7 shrink-0 text-[#5542f6] sm:h-8 sm:w-8" />
              Automated RAG Evaluation Matrix
            </h1>
            <p className="text-slate-500 text-sm">
              Real-time empirical benchmark measuring Correctness, Faithfulness, Relevance, and Retrieval Quality (MRR, Recall@K)
            </p>
          </div>

          <div className="flex w-full flex-wrap items-center gap-3 md:w-auto md:justify-end">
            <div className="flex w-full items-center gap-2 rounded-md border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-600 shadow-xs sm:w-auto">
              <span>Sample size:</span>
              <select
                aria-label="Evaluation sample size"
                value={maxQuestions}
                disabled={loading}
                onChange={(e) => setMaxQuestions(Number(e.target.value))}
                className="min-w-0 flex-1 rounded-md border border-slate-200 bg-slate-50 px-2 py-1 font-medium text-slate-800 focus:outline-hidden sm:flex-none"
              >
                <option value={3}>3 Questions (Fast)</option>
                <option value={5}>5 Questions</option>
                <option value={10}>10 Questions</option>
                <option value={20}>20 Questions (Full)</option>
              </select>
            </div>

            <button
              onClick={fetchSummary}
              disabled={fetching || loading}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-md bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 text-sm font-medium shadow-xs transition disabled:opacity-50 cursor-pointer"
            >
              <RotateCw className={`w-4 h-4 text-slate-500 ${fetching ? 'animate-spin' : ''}`} />
              Refresh
            </button>

            <button
              onClick={handleRunStreamingEvaluation}
              disabled={loading || !evaluationReady}
              title={evaluationReady ? 'Run evaluation' : 'Configure Groq, OpenRouter, and Tavily API keys'}
              className="flex items-center gap-2 px-5 py-2.5 rounded-md bg-[#5542f6] hover:bg-[#4332e6] text-white text-sm font-semibold shadow-md shadow-[#5542f6]/20 transition disabled:opacity-50 cursor-pointer"
            >
              {loading ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" /> Evaluating...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" /> Run 2×2 Evaluation
                </>
              )}
            </button>
          </div>
        </div>

        {!evaluationReady && (
          <div className="flex flex-col gap-3 rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-amber-700" />
              <div>
                <p className="font-bold">Evaluation keys required</p>
                <p className="mt-0.5 text-amber-800">
                  Add {[!status.groq && 'Groq', !status.openrouter && 'OpenRouter', !status.tavily && 'Tavily'].filter(Boolean).join(', ')} to run the LLM-as-Judge benchmark.
                </p>
              </div>
            </div>
            <Link href="/settings" className="inline-flex min-h-9 items-center justify-center gap-2 rounded-md bg-amber-900 px-3 text-xs font-bold text-white hover:bg-amber-950">
              <KeyRound size={15} /> Open Settings
            </Link>
          </div>
        )}

        {/* Model Setup Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* LLM 1 Card */}
          <div className="bg-white border border-slate-200/90 rounded-lg p-4 shadow-xs flex items-start gap-3">
            <div className="w-9 h-9 rounded-md bg-[#eeebff] flex items-center justify-center text-[#5542f6] shrink-0 mt-0.5">
              <Bot size={18} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">LLM 1 (Primary)</span>
                <span className="text-[10px] bg-slate-100 text-slate-600 px-1.5 py-0.2 rounded font-mono">Groq</span>
              </div>
              <p className="font-mono font-bold text-slate-900 text-sm mt-0.5">{model1Name}</p>
              <p className="text-[11px] text-slate-500 mt-0.5">Alibaba Qwen reasoning model</p>
            </div>
          </div>

          {/* LLM 2 Card */}
          <div className="bg-white border border-slate-200/90 rounded-lg p-4 shadow-xs flex items-start gap-3">
            <div className="w-9 h-9 rounded-md bg-purple-50 flex items-center justify-center text-purple-600 shrink-0 mt-0.5">
              <Bot size={18} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">LLM 2 (Comparative)</span>
                <span className="text-[10px] bg-slate-100 text-slate-600 px-1.5 py-0.2 rounded font-mono">Groq</span>
              </div>
              <p className="font-mono font-bold text-slate-900 text-sm mt-0.5">{model2Name}</p>
              <p className="text-[11px] text-slate-500 mt-0.5">OpenAI family OSS baseline model</p>
            </div>
          </div>

          {/* Judge Model Card */}
          <div className="bg-white border border-slate-200/90 rounded-lg p-4 shadow-xs flex items-start gap-3">
            <div className="w-9 h-9 rounded-md bg-emerald-50 flex items-center justify-center text-emerald-600 shrink-0 mt-0.5">
              <Sparkles size={18} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Judge Evaluator</span>
                <span className="text-[10px] bg-emerald-100 text-emerald-800 px-1.5 py-0.2 rounded font-semibold">LLM Judge</span>
              </div>
              <p className="font-mono font-bold text-slate-900 text-sm mt-0.5">{judgeName}</p>
              <p className="text-[11px] text-slate-500 mt-0.5">Automated scoring of correctness & faithfulness</p>
            </div>
          </div>
        </div>

        {/* Live Streaming Progress Section */}
        {isStreaming && (
          <div className="bg-slate-900 text-slate-100 rounded-lg p-6 shadow-xl space-y-4 border border-slate-800 animate-fade-in">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2.5">
                <Activity className="w-5 h-5 text-[#818cf8] animate-pulse" />
                <span className="font-bold text-sm text-white">Live Benchmark Stream</span>
                <span className="text-xs font-mono bg-indigo-950 text-indigo-300 px-2 py-0.5 rounded-full border border-indigo-800">
                  {streamProgress.percent}%
                </span>
              </div>
              <span className="text-xs text-slate-400 font-mono truncate">{currentAction}</span>
            </div>

            {/* Progress Bar */}
            <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
              <div
                className="h-full rounded-full bg-indigo-500 transition-all duration-300"
                style={{ width: `${streamProgress.percent}%` }}
              />
            </div>

            {/* Streaming Logs Terminal */}
            <div
              ref={logTerminalRef}
              className="bg-slate-950/80 rounded-lg p-4 font-mono text-xs max-h-56 overflow-y-auto space-y-1.5 border border-slate-800/80 shadow-inner"
            >
              {logs.map((log) => (
                <div key={log.id} className="flex items-start gap-2 leading-relaxed">
                  <span className="text-slate-500 text-[10px] shrink-0">{log.timestamp}</span>
                  <span
                    className={
                      log.type === 'start'
                        ? 'text-[#818cf8] font-bold'
                        : log.type === 'success'
                        ? 'text-emerald-400 font-semibold'
                        : log.type === 'error'
                        ? 'text-rose-400 font-bold'
                        : 'text-slate-300'
                    }
                  >
                    {log.message}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-center gap-3 p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-sm">
            <AlertCircle className="w-5 h-5 shrink-0 text-rose-500" />
            <span>{error}</span>
          </div>
        )}

        {summary?.warning && (
          <div className="flex items-start gap-3 rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
            <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
            <span>{summary.warning}</span>
          </div>
        )}

        {/* Best Configuration Banner */}
        {summary?.valid !== false && summary?.best_configuration && (
          <div className="bg-white border-2 border-emerald-500/30 rounded-lg p-6 shadow-xs relative overflow-hidden">
            <div className="flex items-start gap-4">
              <div className="p-3.5 bg-emerald-50 rounded-lg border border-emerald-200 text-emerald-600">
                <Trophy className="w-7 h-7" />
              </div>
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-xs uppercase tracking-wider font-bold text-emerald-700">
                    Recommended Production Configuration
                  </span>
                  <span className="bg-emerald-600 text-white text-[10px] uppercase tracking-wider font-extrabold px-2 py-0.5 rounded-full">
                    Winner
                  </span>
                </div>
                <h2 className="text-xl font-bold text-slate-900 capitalize">
                  {getReadableConfigName(summary.best_configuration)}
                </h2>
                <p className="text-slate-600 text-sm leading-relaxed max-w-4xl">
                  {summary.best_reason}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* 2x2 Matrix Overview Cards */}
        <div className="space-y-4">
          <h2 className="text-base font-bold text-slate-800 flex items-center gap-2">
            <Layers className="w-5 h-5 text-[#5542f6]" />
            2 × 2 Comparison Matrix (Composite Overall Score)
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Vector Retrieval Column */}
            <div className="bg-white border border-slate-200/90 rounded-lg p-6 space-y-4 shadow-xs">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <span className="font-bold text-slate-800 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-sky-500" /> Vector Retrieval (Dense)
                </span>
                <span className="text-xs text-slate-400">Semantic cosine similarity</span>
              </div>

              <div className="space-y-3">
                {[
                  { key: 'model_1', name: `LLM 1 (${model1Name})` },
                  { key: 'model_2', name: `LLM 2 (${model2Name})` },
                ].map(({ key, name }) => {
                  const score = matrix['vector']?.[key] ?? 0;
                  const isBest = bestConfigKey === `vector+${key}`;
                  return (
                    <div
                      key={key}
                      className={`p-4 rounded-lg border transition-all ${
                        isBest
                          ? 'bg-emerald-50/60 border-emerald-300 shadow-xs'
                          : 'bg-slate-50 border-slate-200'
                      }`}
                    >
                      <div className="flex justify-between items-center text-sm mb-2">
                        <span className="font-bold text-slate-700 text-xs truncate max-w-[280px]">
                          {name}
                        </span>
                        <span className="font-extrabold text-slate-900 text-base">{score.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            isBest ? 'bg-emerald-500' : 'bg-[#5542f6]'
                          }`}
                          style={{ width: `${Math.min(score, 100)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Hybrid Retrieval Column */}
            <div className="bg-white border border-slate-200/90 rounded-lg p-6 space-y-4 shadow-xs">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <span className="font-bold text-slate-800 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-[#5542f6]" /> Hybrid Retrieval (Dense + BM25 + RRF)
                </span>
                <span className="text-xs text-slate-400">Reciprocal Rank Fusion</span>
              </div>

              <div className="space-y-3">
                {[
                  { key: 'model_1', name: `LLM 1 (${model1Name})` },
                  { key: 'model_2', name: `LLM 2 (${model2Name})` },
                ].map(({ key, name }) => {
                  const score = matrix['hybrid']?.[key] ?? 0;
                  const isBest = bestConfigKey === `hybrid+${key}`;
                  return (
                    <div
                      key={key}
                      className={`p-4 rounded-lg border transition-all ${
                        isBest
                          ? 'bg-emerald-50/60 border-emerald-300 shadow-xs'
                          : 'bg-slate-50 border-slate-200'
                      }`}
                    >
                      <div className="flex justify-between items-center text-sm mb-2">
                        <span className="font-bold text-slate-700 text-xs truncate max-w-[280px]">
                          {name}
                        </span>
                        <span className="font-extrabold text-slate-900 text-base">{score.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            isBest ? 'bg-emerald-500' : 'bg-[#5542f6]'
                          }`}
                          style={{ width: `${Math.min(score, 100)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Detailed Metrics Table */}
        <div className="bg-white border border-slate-200/90 rounded-lg p-6 space-y-4 shadow-xs">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
            <h2 className="text-base font-bold text-slate-800 flex items-center gap-2">
              <FileText className="w-5 h-5 text-[#5542f6]" />
              Detailed Multi-Metric Comparison Table
            </h2>
            <span className="text-xs text-slate-500 flex items-center gap-1 font-medium">
              <Sparkles size={13} className="text-[#5542f6]" /> Judge: <span className="font-semibold text-slate-700 font-mono">{judgeName}</span>
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-xs font-bold uppercase tracking-wider text-slate-500 bg-slate-50/50">
                  <th className="py-3 px-4 rounded-l-xl">Configuration</th>
                  <th className="py-3 px-3">Correctness</th>
                  <th className="py-3 px-3">Faithfulness</th>
                  <th className="py-3 px-3">Relevance</th>
                  <th className="py-3 px-3">Overall</th>
                  <th className="py-3 px-3">Precision@5</th>
                  <th className="py-3 px-3">Recall@5</th>
                  <th className="py-3 px-3">MRR</th>
                  <th className="py-3 px-4 rounded-r-xl">Avg Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {Object.keys(configs).length === 0 ? (
                  <tr>
                    <td colSpan={9} className="text-center py-8 text-slate-400">
                      No evaluation results available yet. Click <strong>"Run 2×2 Evaluation"</strong> above to benchmark.
                    </td>
                  </tr>
                ) : (
                  Object.entries(configs).map(([key, config]) => {
                    const isWinner = key === bestConfigKey;
                    return (
                      <tr
                        key={key}
                        className={`hover:bg-slate-50/80 transition-colors ${
                          isWinner ? 'bg-emerald-50/30 font-semibold' : ''
                        }`}
                      >
                        <td className="py-3.5 px-4 flex items-center gap-2">
                          {isWinner && <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />}
                          <span className="font-bold text-slate-800 font-mono text-xs">{getReadableConfigName(key)}</span>
                        </td>
                        <td className="py-3.5 px-3">{config.avg_correctness.toFixed(1)}%</td>
                        <td className="py-3.5 px-3">{config.avg_faithfulness.toFixed(1)}%</td>
                        <td className="py-3.5 px-3">{config.avg_relevance.toFixed(1)}%</td>
                        <td className="py-3.5 px-3 font-extrabold text-[#5542f6]">{config.avg_overall.toFixed(1)}%</td>
                        <td className="py-3.5 px-3">{config.avg_precision_at_k !== null ? config.avg_precision_at_k.toFixed(2) : '-'}</td>
                        <td className="py-3.5 px-3">{config.avg_recall_at_k !== null ? config.avg_recall_at_k.toFixed(2) : '-'}</td>
                        <td className="py-3.5 px-3 font-bold text-slate-700">{config.avg_mrr !== null ? config.avg_mrr.toFixed(2) : '-'}</td>
                        <td className="py-3.5 px-4 text-slate-500 font-mono text-xs">{(config.avg_latency_ms / 1000).toFixed(2)}s</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {summary?.dataset_path && (
            <div className="pt-2 text-xs text-slate-400">
              Dataset: <span className="font-mono text-slate-600">{summary.dataset_path}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
