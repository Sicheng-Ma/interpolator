'use client'

import { useState, useCallback } from 'react'
import { uploadDataset, UploadResponse } from '@/lib/api'

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [result, setResult] = useState<UploadResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && (droppedFile.name.endsWith('.pkl') || droppedFile.name.endsWith('.pickle'))) {
      setFile(droppedFile)
      setError(null)
      setResult(null)
    } else {
      setError('Please upload a .pkl file')
    }
  }, [])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      if (selectedFile.name.endsWith('.pkl') || selectedFile.name.endsWith('.pickle')) {
        setFile(selectedFile)
        setError(null)
        setResult(null)
      } else {
        setError('Please upload a .pkl file')
      }
    }
  }, [])

  const handleUpload = async () => {
    if (!file) return

    setIsUploading(true)
    setError(null)
    setResult(null)

    try {
      const response = await uploadDataset(file)
      setResult(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800 dark:text-white mb-2">
          Upload Dataset
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          Upload your 5D dataset in .pkl format. The file should contain X (features) and y (target) arrays.
        </p>
      </div>

      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-colors ${
          isDragging
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
            : 'border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500'
        }`}
      >
        <input
          type="file"
          accept=".pkl,.pickle"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <div className="space-y-4">
          <div className="text-5xl">📁</div>
          <div>
            <p className="text-lg font-medium text-slate-700 dark:text-slate-200">
              {file ? file.name : 'Drop your .pkl file here'}
            </p>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
              or click to browse
            </p>
          </div>
        </div>
      </div>

      {/* File Info */}
      {file && (
        <div className="bg-slate-100 dark:bg-slate-800 rounded-lg p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">📄</span>
            <div>
              <p className="font-medium text-slate-800 dark:text-white">{file.name}</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                {(file.size / 1024).toFixed(1)} KB
              </p>
            </div>
          </div>
          <button
            onClick={() => {
              setFile(null)
              setResult(null)
            }}
            className="text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
          >
            ✕
          </button>
        </div>
      )}

      {/* Upload Button */}
      <button
        onClick={handleUpload}
        disabled={!file || isUploading}
        className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
          !file || isUploading
            ? 'bg-slate-300 dark:bg-slate-700 text-slate-500 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700 text-white'
        }`}
      >
        {isUploading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Uploading...
          </span>
        ) : (
          'Upload Dataset'
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
          
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-white dark:bg-slate-800 rounded-lg p-4">
              <p className="text-sm text-slate-500 dark:text-slate-400">Dataset ID</p>
              <p className="font-mono font-medium text-slate-800 dark:text-white">{result.dataset_id}</p>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-lg p-4">
              <p className="text-sm text-slate-500 dark:text-slate-400">Samples</p>
              <p className="font-medium text-slate-800 dark:text-white">{result.n_samples.toLocaleString()}</p>
            </div>
          </div>

          <p className="text-sm text-slate-600 dark:text-slate-400">
            Your dataset is ready! Head to the <a href="/train" className="text-blue-600 dark:text-blue-400 underline">Train</a> page to train a model.
          </p>
        </div>
      )}
    </div>
  )
}