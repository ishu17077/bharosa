import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'Bharosa (भरोसा) — Voice Risk Co-Pilot',
  description:
    'Real-time voice confidence analysis and synthetic voice detection for banking contact centers. Inline acoustic telephony guard with tri-signal detection engine.',
  keywords: [
    'voice cloning detection',
    'deepfake detection',
    'banking security',
    'voice forensics',
  ],
};

export default function RootLayout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-[#060a14] text-white">
        {children}
      </body>
    </html>
  );
}
