export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
  isError?: boolean;
  timestamp: Date;
  sourceMode?: SourceMode;
  sourceLabel?: string;
  retrievalVerified?: boolean;
  retrievalStatus?: RetrievalStatus;
  fallbackUsed?: boolean;
  fallbackReason?: string | null;
}

export type SourceMode = 'hybrid' | 'web' | 'direct' | 'failed';

export type RetrievalStageStatus =
  | 'not_run'
  | 'success'
  | 'failed'
  | 'passed';

export interface RetrievalStatus {
  dense: RetrievalStageStatus;
  bm25: RetrievalStageStatus;
  rrf: RetrievalStageStatus;
  validation: RetrievalStageStatus;
  tavily: RetrievalStageStatus;
}

export interface SourceMetadata {
  paper_id?: string;
  arxiv_id?: string;
  year?: number;
  section?: string;
  content_type?: string;
  page_info?: number;
  url?: string;
  authors?: string[];
  pdf_url?: string;
  [key: string]: unknown;
}

export interface Source {
  id?: number;
  title: string;
  score?: number;
  year?: number;
  arxiv_id?: string;
  paper_id?: string;
  section?: string;
  categories?: string[];
  authors?: string[];
  pdf_url?: string;
  content?: string;
  source?: 'dense' | 'bm25' | 'web' | string;
  metadata?: SourceMetadata;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  stream?: boolean;
  groq_api_key?: string;
  tavily_api_key?: string;
}

export interface ChatResponse {
  answer: string;
  session_id: string;
  reasoning_steps: ReasoningStep[];
  sources: Source[];
  execution_time: number;
  node_timings: Record<string, number>;
  source_mode: SourceMode;
  source_label: string;
  retrieval_verified: boolean;
  retrieval_status: RetrievalStatus;
  fallback_used: boolean;
  fallback_reason: string | null;
}

export interface ReasoningStep {
  thought: string;
  action?: string;
  action_input?: Record<string, unknown>;
  observation?: string;
}

export interface StreamChunk {
  type: 'token' | 'sources' | 'done' | 'error';
  data: string | Record<string, unknown>;
}
