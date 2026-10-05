'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  ArrowRight,
  BadgeCheck,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Copy,
  ExternalLink,
  GitBranch,
  Info,
  AlertTriangle,
  XCircle,
} from 'lucide-react';
import { useState, ComponentPropsWithoutRef } from 'react';
import { Message, Source } from '@/types/chat';

interface MessageItemProps {
  message: Message;
}

export function MessageItem({ message }: MessageItemProps) {
  const isUser = message.role === 'user';

  return (
    <div className="py-4 px-4 message-enter">
      <div className="max-w-3xl mx-auto">
        <div className={`${isUser ? 'flex justify-end' : ''}`}>
          {isUser ? (
            <div className="max-w-[80%] rounded-lg rounded-tr-sm border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm font-medium leading-relaxed text-slate-800 shadow-sm break-words">
              {message.content}
            </div>
          ) : (
            <>
              <div
                className={`text-slate-800 text-sm leading-relaxed ${
                  message.isStreaming && message.content
                    ? 'streaming-cursor'
                    : ''
                }`}
              >
                {message.isStreaming && !message.content ? (
                  <div className="not-prose flex items-center gap-1.5 py-2">
                    <div className="w-2 h-2 bg-[#5542f6] rounded-full animate-bounce [animation-delay:-0.3s]" />
                    <div className="w-2 h-2 bg-[#5542f6] rounded-full animate-bounce [animation-delay:-0.15s]" />
                    <div className="w-2 h-2 bg-[#5542f6] rounded-full animate-bounce" />
                  </div>
                ) : (
                  <MarkdownRenderer content={message.content} />
                )}
              </div>

              {!message.isStreaming && message.sourceMode && (
                <>
                  <SourceBadge message={message} />
                  <RetrievalJourney message={message} />
                </>
              )}

              {message.sources && message.sources.length > 0 && (
                <SourcesSection sources={message.sources} />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function SourceBadge({ message }: { message: Message }) {
  const mode = message.sourceMode || 'failed';
  const config = {
    hybrid: {
      icon: <BadgeCheck size={18} />,
      label: message.sourceLabel || 'Fully Verified Hybrid RAG',
      detail: 'Dense + BM25 + RRF - Validation passed',
      classes: 'border-emerald-200 bg-emerald-50 text-emerald-900',
    },
    web: {
      icon: <ExternalLink size={18} />,
      label: message.sourceLabel || 'Web Search',
      detail: 'Hybrid retrieval could not provide sufficient context.',
      classes: 'border-sky-200 bg-sky-50 text-sky-900',
    },
    direct: {
      icon: <Info size={18} />,
      label: message.sourceLabel || 'Direct LLM',
      detail: 'No document retrieval was used.',
      classes: 'border-slate-200 bg-slate-50 text-slate-900',
    },
    failed: {
      icon: <AlertTriangle size={18} />,
      label: message.sourceLabel || 'No Verified Source',
      detail: 'No sufficient reliable context was found.',
      classes: 'border-amber-200 bg-amber-50 text-amber-950',
    },
  }[mode];

  return (
    <div className={`mt-5 rounded-lg border p-3 ${config.classes}`}>
      <div className="flex items-center gap-2 text-sm font-extrabold">
        {config.icon}
        <span>{config.label}</span>
      </div>
      <p className="mt-1 pl-6 text-xs opacity-75">{config.detail}</p>
    </div>
  );
}

function RetrievalJourney({ message }: { message: Message }) {
  const [expanded, setExpanded] = useState(false);
  const status = message.retrievalStatus;
  if (!status) return null;

  const steps = message.sourceMode === 'direct'
    ? [
        { label: 'Planner', detail: 'General query detected', state: 'success' as const },
        { label: 'Direct LLM', detail: 'No retrieval used', state: 'next' as const },
      ]
    : [
        { label: 'Planner', detail: 'Research query detected', state: 'success' as const },
        { label: 'Dense Retrieval', detail: 'Qdrant', state: status.dense },
        { label: 'BM25 Retrieval', detail: 'Elasticsearch', state: status.bm25 },
        { label: 'RRF', detail: 'Results combined', state: status.rrf },
        { label: 'Validation', detail: status.validation === 'passed' ? 'Context verified' : 'Insufficient context', state: status.validation },
        ...(message.fallbackUsed
          ? [{ label: 'Tavily', detail: status.tavily === 'success' ? 'Web sources retrieved' : 'No reliable web context', state: status.tavily }]
          : []),
        { label: 'Answer', detail: message.sourceMode === 'failed' ? 'No verified answer generated' : 'Response generated', state: message.sourceMode === 'failed' ? 'failed' : 'success' },
      ];

  return (
    <div className="mt-3 border-b border-slate-200 pb-4">
      <button
        onClick={() => setExpanded((value) => !value)}
        className="flex items-center gap-2 text-xs font-bold text-slate-600 transition hover:text-indigo-700"
      >
        <GitBranch size={15} />
        View Retrieval Journey
        {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      {expanded && (
        <ol className="mt-3 space-y-2 rounded-lg border border-slate-200 bg-slate-50 p-3">
          {steps.map((step) => {
            const failed = step.state === 'failed';
            const skipped = step.state === 'not_run';
            return (
              <li key={step.label} className="flex items-start gap-2 text-xs">
                {failed ? <XCircle size={15} className="mt-0.5 shrink-0 text-rose-600" /> : skipped ? <ArrowRight size={15} className="mt-0.5 shrink-0 text-slate-400" /> : <CheckCircle2 size={15} className="mt-0.5 shrink-0 text-emerald-600" />}
                <span><strong className="text-slate-800">{step.label}</strong><span className="ml-2 text-slate-500">{step.detail}</span></span>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

function MarkdownRenderer({ content }: { content: string }) {
  const [copiedBlock, setCopiedBlock] = useState<string | null>(null);

  const handleCopy = async (text: string, blockId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedBlock(blockId);

      setTimeout(() => {
        setCopiedBlock(null);
      }, 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => (
          <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>
        ),

        strong: ({ children }) => (
          <strong className="font-semibold text-gray-900">
            {children}
          </strong>
        ),

        em: ({ children }) => (
          <em className="italic">{children}</em>
        ),

        h1: ({ children }) => (
          <h1 className="text-2xl font-bold mt-6 mb-3 text-gray-900">
            {children}
          </h1>
        ),

        h2: ({ children }) => (
          <h2 className="text-xl font-semibold mt-5 mb-2 text-gray-900">
            {children}
          </h2>
        ),

        h3: ({ children }) => (
          <h3 className="text-lg font-semibold mt-4 mb-2 text-gray-900">
            {children}
          </h3>
        ),

        h4: ({ children }) => (
          <h4 className="text-base font-semibold mt-3 mb-1 text-gray-900">
            {children}
          </h4>
        ),

        ul: ({ children }) => (
          <ul className="list-disc list-outside pl-5 mb-3 space-y-1">
            {children}
          </ul>
        ),

        ol: ({ children }) => (
          <ol className="list-decimal list-outside pl-5 mb-3 space-y-1">
            {children}
          </ol>
        ),

        li: ({ children }) => (
          <li className="leading-relaxed pl-1">{children}</li>
        ),

        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-gray-200 pl-4 text-gray-500 my-3">
            {children}
          </blockquote>
        ),

        hr: () => (
          <hr className="border-gray-200 my-4" />
        ),

        table: ({ children }) => (
          <div className="overflow-x-auto mb-3">
            <table className="w-full border-collapse text-sm">
              {children}
            </table>
          </div>
        ),

        thead: ({ children }) => (
          <thead className="bg-gray-50">
            {children}
          </thead>
        ),

        th: ({ children }) => (
          <th className="border border-gray-200 px-3 py-2 text-left font-semibold text-gray-700">
            {children}
          </th>
        ),

        td: ({ children }) => (
          <td className="border border-gray-200 px-3 py-2 text-gray-700">
            {children}
          </td>
        ),

        /*
         * IMPORTANT:
         * Do not use `node` here.
         *
         * react-markdown's current TypeScript types do not expose
         * `node` through ComponentPropsWithoutRef<'code'>.
         *
         * The previous version caused:
         * "Property 'node' does not exist..."
         */
        code({
          inline,
          className,
          children,
          ...props
        }: ComponentPropsWithoutRef<'code'> & {
          inline?: boolean;
        }) {
          const match = /language-(\w+)/.exec(className || '');

          const codeString = String(children).replace(/\n$/, '');

          const blockId = `code-${crypto.randomUUID()}`;

          if (!inline && match) {
            return (
              <div className="relative group rounded-lg overflow-hidden my-4">
                <div className="flex items-center justify-between bg-gray-100 px-4 py-2 border-b border-gray-200">
                  <span className="text-xs text-gray-500 font-medium uppercase">
                    {match[1]}
                  </span>

                  <button
                    onClick={() =>
                      handleCopy(codeString, blockId)
                    }
                    className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 transition-colors"
                  >
                    {copiedBlock === blockId ? (
                      <>
                        <Check size={14} />
                        <span>Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy size={14} />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                </div>

                <SyntaxHighlighter
                  style={oneLight}
                  language={match[1]}
                  PreTag="div"
                  customStyle={{
                    margin: 0,
                    padding: '1rem',
                    fontSize: '0.875rem',
                    lineHeight: '1.7',
                  }}
                  wrapLines={true}
                  wrapLongLines={true}
                >
                  {codeString}
                </SyntaxHighlighter>
              </div>
            );
          }

          return (
            <code
              className="bg-gray-100 text-red-500 px-1.5 py-0.5 rounded text-sm font-mono"
              {...props}
            >
              {children}
            </code>
          );
        },

        pre({ children }) {
          return <>{children}</>;
        },

        a({ href, children, ...props }) {
          return (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#5542f6] font-medium hover:underline"
              {...props}
            >
              {children}
            </a>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function SourcesSection({
  sources,
}: {
  sources: Source[];
}) {
  if (sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-4">
      <h3 className="text-xs font-extrabold uppercase tracking-[0.12em] text-slate-500">Sources ({sources.length})</h3>
      <div className="mt-3 grid gap-2">
        {sources.map((source, index) => (
          <SourceItem key={source.id ?? index} source={source} index={index} />
        ))}
      </div>
    </div>
  );
}

function SourceItem({
  source,
  index,
}: {
  source: Source;
  index: number;
}) {
  const metadata = source.metadata || {};
  const paperId = source.paper_id || source.arxiv_id || metadata.paper_id || metadata.arxiv_id;
  const year = source.year || metadata.year;
  const url = source.pdf_url || metadata.pdf_url || metadata.url;
  const section = source.section || metadata.section;

  return (
    <div className="flex items-start gap-3 rounded-lg border border-slate-200 bg-white p-3 transition hover:border-indigo-300">
      <div className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md bg-indigo-50">
        <span className="text-xs font-bold text-indigo-700">
          {index + 1}
        </span>
      </div>

      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-gray-800 truncate">
          {source.title}
        </div>

        <div className="flex items-center gap-2 text-xs text-gray-500 mt-1 flex-wrap">
          {paperId && (
            <span className="rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 font-mono text-indigo-700">
              {String(paperId)}
            </span>
          )}

          {year && (
            <span>{String(year)}</span>
          )}

          {section && <span>{String(section)}</span>}
        </div>

        {Array.isArray(source.authors) &&
          source.authors.length > 0 && (
            <div className="text-xs text-gray-400 mt-1 truncate">
              {source.authors.slice(0, 3).join(', ')}
              {source.authors.length > 3 && ' et al.'}
            </div>
          )}
      </div>

      {typeof url === 'string' && url && (
        <a
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-shrink-0 p-1 text-slate-400 transition-colors hover:text-indigo-700"
          title="Open source"
        >
          <ExternalLink size={16} />
        </a>
      )}
    </div>
  );
}
