import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'HYBRID RAG | arXiv Research Assistant',
  description: 'Verified AI research assistance over an arXiv paper corpus.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans antialiased bg-white text-slate-900 min-h-screen">
        {children}
      </body>
    </html>
  );
}
