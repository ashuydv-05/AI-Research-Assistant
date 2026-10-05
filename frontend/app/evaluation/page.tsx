'use client';

import { EvaluationView } from '@/component/evaluation/EvaluationView';
import { useRouter } from 'next/navigation';
import { AppHeader } from '@/component/layout/AppHeader';

export default function EvaluationPage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-[#f8f9fc] flex flex-col">
      <AppHeader />
      <EvaluationView onBackToChat={() => router.push('/chat')} />
    </div>
  );
}
