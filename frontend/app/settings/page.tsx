import { AppHeader } from '@/component/layout/AppHeader';
import { ApiConfiguration } from '@/component/settings/ApiConfiguration';

export default function SettingsPage() {
  return (
    <div className="min-h-screen bg-slate-50">
      <AppHeader />
      <main className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="mb-10 max-w-2xl">
          <p className="text-sm font-bold uppercase tracking-[0.14em] text-indigo-700">Settings</p>
          <h1 className="mt-2 text-3xl font-extrabold tracking-normal text-slate-950">API Configuration</h1>
          <p className="mt-3 text-slate-600">Configure answer generation, evaluation, and web fallback services.</p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-5 sm:p-8">
          <ApiConfiguration />
        </div>
      </main>
    </div>
  );
}
