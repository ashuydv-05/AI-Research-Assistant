'use client';

import { useState, useCallback, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { ChatResponse, Message } from '@/types/chat';
import { sendMessage } from '@/lib/api';

const STREAMING_DELAY_MS = 3;

interface UseChatOptions {
  sessionId?: string;
  onError?: (error: Error) => void;
}

interface UseChatReturn {
  messages: Message[];
  input: string;
  setInput: (value: string) => void;
  isLoading: boolean;
  isStreaming: boolean;
  sessionId: string;
  sendMessage: () => Promise<void>;
  stopStreaming: () => void;
  clearMessages: () => void;
}

export function useChat(options: UseChatOptions = {}): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  const [sessionId, setSessionId] = useState<string>(
    options.sessionId || uuidv4()
  );

  const abortControllerRef = useRef<AbortController | null>(null);
  const streamingIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );

  const cleanupStreaming = useCallback(() => {
    if (streamingIntervalRef.current) {
      clearTimeout(streamingIntervalRef.current);
      streamingIntervalRef.current = null;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setIsStreaming(false);
    setIsLoading(false);
  }, []);

  const addMessage = useCallback((message: Message) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const updateMessage = useCallback(
    (id: string, updates: Partial<Message>) => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === id ? { ...msg, ...updates } : msg
        )
      );
    },
    []
  );

  const simulateStreaming = useCallback(
    (
      fullContent: string,
      assistantMessageId: string,
      response: ChatResponse
    ) => {
      setIsStreaming(true);

      let currentIndex = 0;
      const chars = fullContent.split('');

      const streamNext = () => {
        if (currentIndex >= chars.length) {
          updateMessage(assistantMessageId, {
            isStreaming: false,
            content: fullContent,
            sources: response.sources,
            sourceMode: response.source_mode,
            sourceLabel: response.source_label,
            retrievalVerified: response.retrieval_verified,
            retrievalStatus: response.retrieval_status,
            fallbackUsed: response.fallback_used,
            fallbackReason: response.fallback_reason,
          });

          setIsStreaming(false);
          streamingIntervalRef.current = null;
          return;
        }

        const currentContent = chars
          .slice(0, currentIndex + 1)
          .join('');

        updateMessage(assistantMessageId, {
          content: currentContent,
          sources: response.sources,
          sourceMode: response.source_mode,
          sourceLabel: response.source_label,
          retrievalVerified: response.retrieval_verified,
          retrievalStatus: response.retrieval_status,
          fallbackUsed: response.fallback_used,
          fallbackReason: response.fallback_reason,
        });

        currentIndex++;

        streamingIntervalRef.current = setTimeout(
          streamNext,
          STREAMING_DELAY_MS
        );
      };

      streamingIntervalRef.current = setTimeout(
        streamNext,
        STREAMING_DELAY_MS
      );
    },
    [updateMessage]
  );

  const handleSendMessage = useCallback(async () => {
    const trimmedInput = input.trim();

    if (!trimmedInput || isLoading) {
      return;
    }

    const userMessage: Message = {
      id: uuidv4(),
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
    };

    addMessage(userMessage);
    setInput('');
    setIsLoading(true);

    const assistantMessageId = uuidv4();

    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      isStreaming: true,
      timestamp: new Date(),
    };

    addMessage(assistantMessage);

    abortControllerRef.current = new AbortController();

    try {
      const response = await sendMessage({
        message: userMessage.content,
        session_id: sessionId,
      });

      simulateStreaming(
        response.answer,
        assistantMessageId,
        response
      );

      if (
        response.session_id &&
        response.session_id !== sessionId
      ) {
        setSessionId(response.session_id);
      }
    } catch (error) {
      let errorMessage =
        'An unexpected error occurred. Please try again.';

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Request was cancelled.';
        } else if (error.message.includes('Failed to fetch')) {
          errorMessage =
            'Cannot connect to backend server. Please verify your connection or wait for the server to wake up.';
        } else {
          errorMessage = error.message;
        }
      }

      updateMessage(assistantMessageId, {
        content: `Error: ${errorMessage}`,
        isStreaming: false,
        isError: true,
        sourceMode: 'failed',
        sourceLabel: 'No Verified Source',
      });

      options.onError?.(
        error instanceof Error
          ? error
          : new Error(errorMessage)
      );

      setIsStreaming(false);
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  }, [
    input,
    isLoading,
    sessionId,
    addMessage,
    updateMessage,
    simulateStreaming,
    options,
  ]);

  const stopStreaming = useCallback(() => {
    cleanupStreaming();

    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];

      if (
        lastMessage &&
        lastMessage.role === 'assistant'
      ) {
        return prev.map((message) =>
          message.id === lastMessage.id
            ? {
                ...message,
                isStreaming: false,
              }
            : message
        );
      }

      return prev;
    });
  }, [cleanupStreaming]);

  const clearMessages = useCallback(() => {
    cleanupStreaming();

    setMessages([]);
    setSessionId(uuidv4());
    setInput('');
  }, [cleanupStreaming]);

  return {
    messages,
    input,
    setInput,
    isLoading,
    isStreaming,
    sessionId,
    sendMessage: handleSendMessage,
    stopStreaming,
    clearMessages,
  };
}
