'use client'

import { useState } from 'react'
import { predict, PredictResponse } from '@/lib/api'

export default function PredictPage() {
  const [features, setFeatures] = useState<number[]>([0, 0, 0, 0, 0])
  const [isPredicting, setIsPredicting] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFeatureChange = (index: number, value: string) => {
    const newFeatures = [...features]
    newFeatures[index] = parseFloat(value) || 0
    setFeatures(newFeatures)
  }

  const handlePredict = async () => {
    setIsPredicting(true)
    setError(null)
    setResult(null)

    try {
      const response = await predict({ features })
      setResult(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setIsPredicting(false)
    }
  }

  const handleReset = () => {
    setFeatures([0, 0, 0, 0, 0])
    setResult(null)
    setError(null)
  }

  const handleRandom = () => {
    const randomFeatures = Array.from({ length: 5 }, () => 
      parseFloat((Math.random() * 2 - 1).toFixed(3))
    )
    setFeatures(randomFeatures)
    setResult(null)
    setError(null)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800 dark:text-white mb-2">
          Make Predictions
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          Enter 5 feature values to get a prediction from your trained model.
        </p>
      </div>

      {/* Input Form */}
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-sm border border-slate-200 dark:border-slate-700 space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-800 dark:text-white">
            Input Features
          </h2>
          <div className="flex gap-2">
            <button
              onClick={handleRandom}
              className="text-sm px-3 py-1 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
            >
              🎲 Random
            </button>
            <button
              onClick={handleReset}
              className="text-sm px-3 py-1 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
            >
              ↺ Reset
            </button>
          </div>
        </div>

        <div className="grid grid-cols-5 gap-4">
          {features.map((value, index) => (
            <div key={index}>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2 text-center">
                x<sub>{index + 1}</sub>
              </label>
              <input
                type="number"
                value={value}
                onChange={(e) => handleFeatureChange(index, e.target.value)}
                step="0.1"
                className="w-full px-3 py-3 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-800 dark:text-white text-center text-lg font-mono focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          ))}
        </div>

        {/* Visual representation */}
        <div className="flex items-center justify-center gap-2 text-slate-600 dark:text-slate-400 py-4">
          <span className="font-mono text-lg">f(</span>
          {features.map((v, i) => (
            <span key={i} className="font-mono">
              <span className="text-blue-600 dark:text-blue-400">{v.toFixed(2)}</span>
              {i < 4 && <span>, </span>}
            </span>
          ))}
          <span className="font-mono text-lg">) = ?</span>
        </div>
      </div>

      {/* Predict Button */}
      <button
        onClick={handlePredict}
        disabled={isPredicting}
        className={`w-full py-4 px-4 rounded-lg font-medium text-lg transition-colors ${
          isPredicting
            ? 'bg-slate-300 dark:bg-slate-700 text-slate-500 cursor-not-allowed'
            : 'bg-purple-600 hover:bg-purple-700 text-white'
        }`}
      >
        {isPredicting ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Predicting...
          </span>
        ) : (
          '🎯 Predict'
        )}
      </button>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-600 dark:text-red-400 flex items-center gap-2">
            <span>❌</span>
            {error}
          </p>
          {error.includes('No trained model') && (
            <p className="text-sm text-red-500 dark:text-red-400 mt-2">
              Please <a href="/upload" className="underline">upload a dataset</a> and <a href="/train" className="underline">train a model</a> first.
            </p>
          )}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="bg-gradient-to-br from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border border-purple-200 dark:border-purple-800 rounded-xl p-8 text-center space-y-4">
          <p className="text-slate-600 dark:text-slate-400">Prediction Result</p>
          
          <div className="text-5xl font-bold text-purple-600 dark:text-purple-400 font-mono">
            {result.prediction.toFixed(4)}
          </div>

          <div className="flex items-center justify-center gap-2 text-slate-500 dark:text-slate-400 text-sm">
            <span>Model:</span>
            <span className="font-mono bg-slate-200 dark:bg-slate-700 px-2 py-1 rounded">
              {result.model_id}
            </span>
          </div>

          {/* Formula display */}
          <div className="pt-4 border-t border-purple-200 dark:border-purple-700">
            <p className="font-mono text-slate-600 dark:text-slate-400">
              f({features.map(v => v.toFixed(2)).join(', ')}) = {' '}
              <span className="text-purple-600 dark:text-purple-400 font-bold">
                {result.prediction.toFixed(4)}
              </span>
            </p>
          </div>
        </div>
      )}
    </div>
  )
}