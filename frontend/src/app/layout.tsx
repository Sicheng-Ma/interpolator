import type { Metadata } from 'next'
import './globals.css'
import Navigation from '@/components/Navigation'

export const metadata: Metadata = {
  title: '5D Regressor - Neural Network Interpolator',
  description: 'A machine learning system for 5-dimensional dataset interpolation',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
        <Navigation />
        <main className="container mx-auto px-4 py-8 max-w-4xl">
          {children}
        </main>
      </body>
    </html>
  )
}