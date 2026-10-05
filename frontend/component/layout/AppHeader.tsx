'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { FlaskConical, Home, KeyRound, MessageSquareText } from 'lucide-react';

const links = [
  { href: '/', label: 'Home', icon: Home },
  { href: '/chat', label: 'Chat', icon: MessageSquareText },
  { href: '/evaluation', label: 'Evaluation', icon: FlaskConical },
  { href: '/settings', label: 'Settings', icon: KeyRound },
];

export function AppHeader() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/95 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-2 px-4 sm:gap-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex shrink-0 items-center gap-2 text-slate-950">
          <span className="flex h-8 w-8 items-center justify-center rounded-md bg-indigo-600 text-sm font-extrabold text-white">
            H
          </span>
          <span className="hidden text-sm font-extrabold tracking-normal sm:inline sm:text-base">HYBRID RAG</span>
        </Link>

        <nav aria-label="Primary navigation" className="flex min-w-0 items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => {
            const active = href === '/' ? pathname === href : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? 'page' : undefined}
                title={label}
                className={`flex min-h-10 min-w-10 items-center justify-center gap-1.5 rounded-md px-2 text-xs font-semibold transition-colors sm:px-3 sm:text-sm ${
                  active
                    ? 'bg-slate-900 text-white'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-950'
                }`}
              >
                <Icon size={15} aria-hidden="true" />
                <span className="hidden md:inline">{label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
