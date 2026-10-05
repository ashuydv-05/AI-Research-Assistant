import Link from 'next/link';
import {
  ArrowDown,
  ArrowRight,
  BookOpen,
  Check,
  AlertCircle,
  Database,
  ExternalLink,
  FileQuestion,
  Info,
  MessageSquareText,
  SearchCheck,
} from 'lucide-react';
import { AppHeader } from '@/component/layout/AppHeader';
import { ApiConfiguration } from '@/component/settings/ApiConfiguration';

export function LandingPage() {
  return (
    <div className="min-h-screen bg-white text-slate-950">
      <AppHeader />
      <main>
        <section className="border-b border-slate-200 bg-slate-950 text-white">
          <div className="mx-auto grid min-h-[560px] max-w-7xl items-center gap-12 px-4 py-16 sm:px-6 md:min-h-[620px] lg:grid-cols-[minmax(0,1fr)_440px] lg:px-8">
            <div className="max-w-3xl">
              <p className="mb-5 text-sm font-bold uppercase tracking-[0.18em] text-emerald-400">arXiv research, verified</p>
              <h1 className="text-5xl font-extrabold tracking-normal sm:text-6xl lg:text-7xl">HYBRID RAG</h1>
              <p className="mt-5 max-w-2xl text-xl font-semibold leading-8 text-slate-200 sm:text-2xl">
                AI Research Assistant for Verified Retrieval over arXiv Papers
              </p>
              <p className="mt-5 max-w-2xl text-base leading-7 text-slate-400">
                A research assistant that combines document retrieval, validation, and web fallback to provide answers grounded in research literature.
              </p>
              <div className="mt-9 flex flex-col gap-3 sm:flex-row sm:flex-wrap">
                <Link href="/chat" className="inline-flex min-h-12 items-center justify-center gap-2 rounded-md bg-emerald-400 px-5 text-sm font-extrabold text-slate-950 transition hover:bg-emerald-300">
                  Ask a research question <ArrowRight size={18} />
                </Link>
                <Link href="/evaluation" className="inline-flex min-h-12 items-center justify-center gap-2 rounded-md border border-slate-600 px-5 text-sm font-bold text-white transition hover:border-slate-400 hover:bg-slate-900">
                  View evaluation
                </Link>
              </div>
            </div>

            <div className="hidden border-l border-slate-700 pl-10 lg:block" aria-hidden="true">
              <div className="space-y-6 font-mono text-sm text-slate-300">
                <SignalLine label="Query understood" tone="text-sky-300" />
                <SignalLine label="Research retrieved" tone="text-amber-300" />
                <SignalLine label="Context validated" tone="text-emerald-300" />
                <SignalLine label="Source disclosed" tone="text-white" />
              </div>
            </div>
          </div>
        </section>

        <section className="border-b border-slate-200 bg-slate-50 py-14">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="mb-8 max-w-2xl">
              <p className="text-sm font-bold uppercase tracking-[0.14em] text-indigo-700">Research data</p>
              <h2 className="mt-2 text-3xl font-extrabold tracking-normal">Built on the corpus in this repository</h2>
            </div>
            <div className="grid gap-4 sm:grid-cols-3">
              <Stat icon={<BookOpen />} value="107" label="Research papers" />
              <Stat icon={<FileQuestion />} value="20" label="Evaluation questions" />
              <Stat icon={<Database />} value="arXiv" label="Data source" />
            </div>
          </div>
        </section>

        <section className="border-b border-slate-200 py-16 sm:py-20">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="mx-auto mb-12 max-w-2xl text-center">
              <p className="text-sm font-bold uppercase tracking-[0.14em] text-indigo-700">How answers are made</p>
              <h2 className="mt-2 text-3xl font-extrabold tracking-normal">Ask once. The system chooses the path.</h2>
              <p className="mt-3 text-slate-600">Research questions use the complete hybrid pipeline first. Web search is used only when the research context does not pass validation.</p>
            </div>
            <WorkflowVisual />
          </div>
        </section>

        <section className="border-b border-slate-200 bg-slate-50 py-16">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="grid gap-10 lg:grid-cols-[320px_minmax(0,1fr)]">
              <div>
                <p className="text-sm font-bold uppercase tracking-[0.14em] text-indigo-700">Source transparency</p>
                <h2 className="mt-2 text-3xl font-extrabold tracking-normal">Every answer shows where it came from.</h2>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <Outcome icon={<Check />} title="Fully Verified Hybrid RAG" text="Dense + BM25 + RRF, with validation passed." tone="emerald" />
                <Outcome icon={<ExternalLink />} title="Web Search" text="Hybrid context was insufficient, so Tavily was used." tone="sky" />
                <Outcome icon={<Info />} title="Direct LLM" text="A general question answered without retrieval." tone="slate" />
                <Outcome icon={<AlertCircle />} title="No Verified Source" text="Neither research retrieval nor web search found reliable context." tone="amber" />
              </div>
            </div>
          </div>
        </section>

        <section className="border-b border-slate-200 py-16" id="configuration">
          <div className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
            <div className="mb-9 max-w-2xl">
              <p className="text-sm font-bold uppercase tracking-[0.14em] text-indigo-700">API configuration</p>
              <h2 className="mt-2 text-3xl font-extrabold tracking-normal">Connect generation, evaluation, and web fallback</h2>
            </div>
            <ApiConfiguration />
          </div>
        </section>

        <section className="py-16">
          <div className="mx-auto max-w-5xl px-4 text-center sm:px-6">
            <h2 className="text-3xl font-extrabold tracking-normal">Ready to explore the research?</h2>
            <p className="mt-3 text-slate-600">Ask a question or inspect how the retrieval system performs on its evaluation dataset.</p>
            <div className="mt-7 flex flex-wrap justify-center gap-3">
              <Link href="/chat" className="inline-flex min-h-11 items-center gap-2 rounded-md bg-indigo-600 px-5 text-sm font-bold text-white hover:bg-indigo-700"><MessageSquareText size={17} /> Chat</Link>
              <Link href="/evaluation" className="inline-flex min-h-11 items-center gap-2 rounded-md border border-slate-300 px-5 text-sm font-bold text-slate-800 hover:bg-slate-50"><SearchCheck size={17} /> Evaluation</Link>
            </div>
          </div>
        </section>
      </main>

      <Link href="/chat" className="fixed bottom-4 right-4 z-30 inline-flex min-h-11 items-center gap-2 rounded-md bg-slate-950 px-4 text-sm font-bold text-white shadow-lg transition hover:bg-indigo-700 sm:bottom-6 sm:right-6">
        <MessageSquareText size={17} /> <span className="hidden sm:inline">Ask a Research Question</span><span className="sm:hidden">Ask</span> <ArrowRight size={16} />
      </Link>
    </div>
  );
}

function SignalLine({ label, tone }: { label: string; tone: string }) {
  return <div className={`workflow-reveal flex items-center gap-3 ${tone}`}><span className="h-2 w-2 bg-current" /><span>{label}</span><span className="h-px flex-1 bg-current opacity-30" /></div>;
}

function Stat({ icon, value, label }: { icon: React.ReactNode; value: string; label: string }) {
  return <div className="rounded-lg border border-slate-200 bg-white p-5"><div className="mb-5 text-indigo-600">{icon}</div><div className="text-3xl font-extrabold">{value}</div><div className="mt-1 text-sm font-medium text-slate-500">{label}</div></div>;
}

function FlowNode({ children, tone = 'default' }: { children: React.ReactNode; tone?: 'default' | 'dark' | 'success' | 'warning' }) {
  const colors = { default: 'border-slate-300 bg-white text-slate-900', dark: 'border-slate-900 bg-slate-900 text-white', success: 'border-emerald-300 bg-emerald-50 text-emerald-900', warning: 'border-amber-300 bg-amber-50 text-amber-950' };
  return <div className={`workflow-reveal flex min-h-12 items-center justify-center rounded-md border px-3 py-2 text-center text-xs font-bold sm:text-sm ${colors[tone]}`}>{children}</div>;
}

function DownArrow() { return <ArrowDown className="mx-auto my-2 text-slate-400" size={18} />; }

function WorkflowVisual() {
  return (
    <div className="mx-auto max-w-5xl overflow-hidden rounded-lg border border-slate-200 bg-slate-50 p-4 sm:p-8">
      <div className="mx-auto max-w-xs"><FlowNode tone="dark">User Query</FlowNode><DownArrow /><FlowNode>Planner</FlowNode></div>
      <div className="mx-auto mt-3 grid max-w-3xl grid-cols-2 gap-4 sm:gap-10">
        <div className="text-center"><div className="mb-2 text-xs font-bold uppercase text-slate-500">General</div><FlowNode>Direct LLM</FlowNode><DownArrow /><FlowNode tone="success">Answer</FlowNode></div>
        <div className="text-center"><div className="mb-2 text-xs font-bold uppercase text-indigo-700">Research</div><FlowNode>Hybrid RAG</FlowNode><DownArrow /><FlowNode>Dense + BM25</FlowNode><DownArrow /><FlowNode>RRF</FlowNode><DownArrow /><FlowNode>Validation</FlowNode></div>
      </div>
      <div className="ml-auto mt-3 grid w-full grid-cols-2 gap-2 sm:w-1/2 sm:gap-5 sm:pl-5">
        <div className="text-center"><span className="text-xs font-bold text-emerald-700">PASS</span><DownArrow /><FlowNode tone="success">RAG Answer</FlowNode></div>
        <div className="text-center"><span className="text-xs font-bold text-amber-700">FAIL</span><DownArrow /><FlowNode tone="warning">Tavily</FlowNode><DownArrow /><FlowNode tone="success">Web Answer</FlowNode></div>
      </div>
    </div>
  );
}

function Outcome({ icon, title, text, tone }: { icon: React.ReactNode; title: string; text: string; tone: 'emerald' | 'sky' | 'slate' | 'amber' }) {
  const colors = { emerald: 'border-emerald-200 text-emerald-700', sky: 'border-sky-200 text-sky-700', slate: 'border-slate-200 text-slate-700', amber: 'border-amber-200 text-amber-800' };
  return <div className={`rounded-lg border bg-white p-4 ${colors[tone]}`}><div className="flex items-center gap-2 font-bold">{icon}<span>{title}</span></div><p className="mt-2 text-sm leading-6 text-slate-600">{text}</p></div>;
}
