'use client';

import React, { useRef, useEffect, KeyboardEvent } from 'react';
import { Send, Square } from 'lucide-react';

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  isLoading: boolean;
  isStreaming: boolean;
  disabled?: boolean;
  onExternalInput?: (value: string) => void;
}

export function ChatInput({
  input,
  setInput,
  onSubmit,
  onStop,
  isLoading,
  isStreaming,
  disabled = false,
  onExternalInput,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [input]);

  useEffect(() => {
    if (!onExternalInput) return;

    const handleSetInput = (e: Event) => {
      const customEvent = e as CustomEvent<string>;
      setInput(customEvent.detail);
      textareaRef.current?.focus();
    };

    window.addEventListener('setInput', handleSetInput);
    return () => window.removeEventListener('setInput', handleSetInput);
  }, [setInput, onExternalInput]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() && !isLoading && !isStreaming && !disabled) {
        onSubmit();
      }
    }
  };

  const handleSubmit = () => {
    if (input.trim() && !isLoading && !isStreaming && !disabled) {
      onSubmit();
    }
  };

  const isActive = isLoading || isStreaming;

  return (
    <div className="bg-white border-t border-slate-100 pb-2">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-3">
        <div
          className={`relative flex items-center bg-white border border-slate-200/90 rounded-[28px] shadow-[0_2px_12px_rgba(0,0,0,0.03)] focus-within:border-[#5542f6]/60 focus-within:ring-4 focus-within:ring-[#eeebff]/60 transition-all duration-200 ${
            input.trim() || isActive ? 'shadow-md border-[#5542f6]/40' : ''
          }`}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={disabled ? 'Configure Groq and Tavily keys to start chatting' : 'Ask a research question...'}
            rows={1}
            className="flex-1 bg-transparent resize-none outline-none pl-5 pr-14 py-3.5 text-slate-800 placeholder-slate-400 max-h-[200px] text-sm leading-relaxed min-h-[48px]"
            disabled={isActive || disabled}
          />

          <div className="absolute right-2">
            <div className="relative inline-flex items-center">
              {isActive && (
                <span
                  className="absolute rounded-full border-2 border-[#5542f6]/30 border-t-[#5542f6] animate-spin pointer-events-none"
                  style={{ inset: '-3px' }}
                />
              )}
              <button
                onClick={isActive ? onStop : handleSubmit}
                disabled={!isActive && (!input.trim() || disabled)}
                className={`relative w-9 h-9 flex items-center justify-center rounded-full transition-all duration-200 ${
                  isActive
                    ? 'bg-[#5542f6] text-white cursor-pointer shadow-md'
                    : !input.trim() || disabled
                    ? 'bg-slate-100 text-slate-300 cursor-not-allowed'
                    : 'bg-[#5542f6] text-white hover:bg-[#4332e6] hover:scale-105 shadow-md shadow-[#5542f6]/20 cursor-pointer'
                }`}
                title={isActive ? 'Stop' : disabled ? 'Configure required API keys' : 'Send message'}
              >
                {isActive ? (
                  <Square size={13} fill="currentColor" />
                ) : (
                  <Send size={15} className="translate-x-[1px] translate-y-[-0.5px]" />
                )}
              </button>
            </div>
          </div>
        </div>

        <div className="text-center mt-2.5 text-[11px] text-slate-400 select-none">
          Press <kbd className="px-1.5 py-0.5 bg-slate-100 rounded text-slate-600 font-mono text-[10px]">Enter</kbd> to send,{' '}
          <kbd className="px-1.5 py-0.5 bg-slate-100 rounded text-slate-600 font-mono text-[10px]">Shift + Enter</kbd> for new line
        </div>
      </div>
    </div>
  );
}
