'use client';

import Link from 'next/link';
import { AlertTriangle, KeyRound, Plus } from 'lucide-react';
import { useChat } from '@/hooks/useChat';
import { useApiKeyStatus } from '@/hooks/useApiKeyStatus';
import { MessageList } from './MessageList';
import { ChatInput } from './ChatInput';

export function ChatContainer() {
  const { status, chatReady } = useApiKeyStatus();
  const {
    messages,
    input,
    setInput,
    isLoading,
    isStreaming,
    sendMessage,
    stopStreaming,
    clearMessages,
  } = useChat({
    onError: (error) => console.error('Chat request failed:', error.message),
  });

  return (
    <main className="flex min-h-0 flex-1 flex-col overflow-hidden bg-white">
      <div className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200 px-4 sm:px-6">
        <div>
          <h1 className="text-sm font-extrabold text-slate-950 sm:text-base">Research Assistant</h1>
          <p className="hidden text-xs text-slate-500 sm:block">Ask a question. Retrieval and validation happen automatically.</p>
        </div>
        <button
          onClick={clearMessages}
          className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-300 px-3 text-xs font-bold text-slate-700 transition hover:bg-slate-50"
        >
          <Plus size={15} /> New chat
        </button>
      </div>

      {!chatReady && (
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-3 sm:px-6">
          <div className="mx-auto flex max-w-3xl items-start justify-between gap-4">
            <div className="flex items-start gap-3 text-sm text-amber-950">
              <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-700" />
              <div>
                <p className="font-bold">API configuration required</p>
                <p className="mt-0.5 text-amber-800">
                  Add {[!status.groq && 'Groq', !status.tavily && 'Tavily'].filter(Boolean).join(' and ')} before starting Chat.
                </p>
              </div>
            </div>
            <Link href="/settings" className="inline-flex min-h-9 shrink-0 items-center gap-2 rounded-md bg-amber-900 px-3 text-xs font-bold text-white hover:bg-amber-950">
              <KeyRound size={15} /> Settings
            </Link>
          </div>
        </div>
      )}

      <MessageList messages={messages} />
      <ChatInput
        input={input}
        setInput={setInput}
        onSubmit={sendMessage}
        onStop={stopStreaming}
        isLoading={isLoading}
        isStreaming={isStreaming}
        disabled={!chatReady}
        onExternalInput={setInput}
      />
    </main>
  );
}
