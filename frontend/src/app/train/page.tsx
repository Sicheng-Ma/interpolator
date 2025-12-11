'use client'

import { useState } from 'react'
import { trainModel, TrainRequest, TrainResponse } from '@/lib/api'

export default function TrainPage() {
  const [config, setConfig] = useState<TrainRequest>({
    hidden_layers: [64, 32, 16],
    learning_rate: 0.005,
    max_epochs: 100,
    batch_size: 256,
    patience: 20,
    test_size: 0.2,
    val_size: 0.1,
  })
  
  const [hiddenLayersStr, setHiddenLayersStr] = useState('64, 32, 16')
  const [isTraining, setIsTraining] = useState(false)
  const [result, setResult] = useState<TrainResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleHiddenLayersChange = (value: string) => {
    setHiddenLayersStr(value)
    try {
      const layers = value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
      if (layers.length > 0) {
        setConfig(prev => ({ ...prev, hidden_layers: layers }))
      }
    } catch {
      // Invalid input, keep previous value
    }
  }

  const handleTrain = async () => {
    setIsTraining(true)
    setError(null)
    setResult(null)

    try {
      const response = await trainModel(config)
      setResult(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed')
    } finally {
      setIsTraining(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800 dark:text-white mb-2">
          Train Model
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          Configure hyperparameters and train a neural network on your uploaded dataset.
        </p>
      </div>

      {/* Configuration Form */}
      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-sm border border-slate-200 dark:border-slate-700 space-y-6">
        <h2 className="text-lg font-semibold text-slate-800 dark:text-white">
          Hyperparameters
        </h2>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Hidden Layers */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Hidden Layers
            </label>
            <input
              type="text"
              value={hiddenLayersStr}
              onChange={(e) => handleHiddenLayersChange(e.target.value)}
              placeholder="64, 32, 16"
              className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-800 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Comma-separated neuron counts per layer
            </p>
          </div>

          {/* Learning Rate */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Learning Rate
            </label>
            <input
              type="number"
              value={config.learning_rate}
              onChange={(e) => setConfig(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
              step="0.001"
              min="0.0001"
              max="0.1"
              className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-800 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Max Epochs */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Max Epochs
            </label>
            <input
              type="number"
              value={config.max_epochs}
              onChange={(e) => setConfig(prev => ({ ...prev, max_epochs: parseInt(e.target.value) }))}
              min="10"
              max="1000"
              className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-800 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Batch Size */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Batch Size
            </label>
            <input
              type="number"
              value={config.batch_size}
              onChange={(e) => setConfig(prev => ({ ...prev, batch_size: parseInt(e.target.value) }))}
              min="16"
              max="1024"
              className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-800 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Patience */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Early Stopping Patience
            </label>
            <input
              type="number"
              value={config.patience}
              onChange={(e) => setConfig(prev => ({ ...prev, patience: parseInt(e.target.value) }))}
              min="5"
              max="100"
              className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-800 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Test Size */}
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Test Size
            </label>
            <input
              type="number"
              value={config.test_size}
              onChange={(e) => setConfig(prev => ({ ...prev, test_size: parseFloat(e.target.value) }))}
              step="0.05"
              min="0.1"
              max="0.4"
              className="w-full px-4 py-2 rounded-lg border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 text-slate-800 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>
      </div>

      {/* Train Button */}
      <button
        onClick={handleTrain}
        disabled={isTraining}
        className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
          isTraining
            ? 'bg-slate-300 dark:bg-slate-700 text-slate-500 cursor-not-allowed'
            : 'bg-green-600 hover:bg-green-700 text-white'
        }`}
      >
        {isTraining ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Training... (this may take a few seconds)
          </span>
        ) : (
          '🚀 Start Training'
        )}
      </button>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-600 dark:text-red-400 flex items-center gap-2">
            <span>❌</span>
            {error}
          </p>
        </div>
      )}

      {/* Success Result */}
      {result && (
        <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6 space-y-4">
          <p className="text-green-600 dark:text-green-400 font-medium flex items-center gap-2">
            <span>✅</span>
            {result.message}
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Model ID" value={result.model_id} mono />
            <MetricCard label="Training Time" value={`${result.training_time_seconds}s`} />
            <MetricCard label="R² Score" value={result.test_metrics.r2.toFixed(4)} highlight />
            <MetricCard label="RMSE" value={result.test_metrics.rmse.toFixed(4)} />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <MetricCard label="Train Samples" value={result.train_samples.toLocaleString()} />
            <MetricCard label="Val Samples" value={result.val_samples.toLocaleString()} />
            <MetricCard label="Test Samples" value={result.test_samples.toLocaleString()} />
          </div>

          <p className="text-sm text-slate-600 dark:text-slate-400">
            Your model is trained! Head to the <a href="/predict" className="text-blue-600 dark:text-blue-400 underline">Predict</a> page to make predictions.
          </p>
        </div>
      )}
    </div>
  )
}

function MetricCard({ 
  label, 
  value, 
  mono = false, 
  highlight = false 
}: { 
  label: string
  value: string
  mono?: boolean
  highlight?: boolean
}) {
  return (
    <div className={`bg-white dark:bg-slate-800 rounded-lg p-4 ${highlight ? 'ring-2 ring-green-500' : ''}`}>
      <p className="text-sm text-slate-500 dark:text-slate-400">{label}</p>
      <p className={`font-medium text-slate-800 dark:text-white ${mono ? 'font-mono' : ''} ${highlight ? 'text-green-600 dark:text-green-400' : ''}`}>
        {value}
      </p>
    </div>
  )
}