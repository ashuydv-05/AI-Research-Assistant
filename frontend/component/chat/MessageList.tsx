'use client';

import { useEffect, useRef } from 'react';
import { BookOpen, FileText, Search } from 'lucide-react';
import { Message } from '@/types/chat';
import { MessageItem } from './MessageItem';

export function MessageList({ messages }: { messages: Message[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center overflow-y-auto bg-slate-50 p-6 md:p-12">
        <div className="mx-auto w-full max-w-3xl text-center">
          <h2 className="text-3xl font-extrabold tracking-normal md:text-4xl">
            <span className="text-indigo-600">Ask the</span>{' '}
            <span className="text-slate-950">research corpus</span>
          </h2>
          <p className="mx-auto mb-9 mt-3 max-w-xl text-sm leading-6 text-slate-600 md:text-base">
            Ask a research question. The assistant retrieves, verifies, and clearly identifies where every answer came from.
          </p>

          <div className="grid grid-cols-1 gap-3 text-left md:grid-cols-3">
            <PromptCard icon={<Search size={19} />} label="Find papers about transformer models" query="Find papers about transformer models" />
            <PromptCard icon={<BookOpen size={19} />} label="How does RAG reduce hallucination?" query="How does RAG reduce hallucination?" />
            <PromptCard icon={<FileText size={19} />} label="Explain attention mechanisms" query="Explain the attention mechanism in transformers" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto bg-white">
      {messages.map((message) => <MessageItem key={message.id} message={message} />)}
      <div ref={bottomRef} />
    </div>
  );
}

function PromptCard({ icon, label, query }: { icon: React.ReactNode; label: string; query: string }) {
  return (
    <button
      onClick={() => window.dispatchEvent(new CustomEvent('setInput', { detail: query }))}
      className="flex min-h-[112px] flex-col justify-between rounded-lg border border-slate-200 bg-white p-4 text-left transition hover:border-indigo-400 hover:shadow-sm"
    >
      <span className="text-indigo-600">{icon}</span>
      <span className="mt-4 text-sm font-bold leading-5 text-slate-900">{label}</span>
    </button>
  );
}
