// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface HealthResponse {
  status: string
  timestamp: string
  version: string
}

export interface UploadResponse {
  message: string
  dataset_id: string
  n_samples: number
  n_features: number
}

export interface TrainRequest {
  dataset_id?: string
  hidden_layers?: number[]
  learning_rate?: number
  max_epochs?: number
  batch_size?: number
  patience?: number
  test_size?: number
  val_size?: number
}

export interface TrainResponse {
  message: string
  model_id: string
  train_samples: number
  val_samples: number
  test_samples: number
  test_metrics: {
    mse: number
    rmse: number
    mae: number
    r2: number
  }
  training_time_seconds: number
}

export interface PredictRequest {
  features: number[]
  model_id?: string
}

export interface PredictResponse {
  prediction: number
  model_id: string
}

export interface DatasetInfo {
  dataset_id: string
  n_samples: number
  n_features: number
  uploaded_at: string
}

export interface ModelInfo {
  model_id: string
  hidden_layers: number[]
  test_r2: number
  test_rmse: number
  trained_at: string
}

// API functions
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) throw new Error('API is not healthy')
  return response.json()
}

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Upload failed')
  }
  
  return response.json()
}

export async function trainModel(config: TrainRequest): Promise<TrainResponse> {
  const response = await fetch(`${API_BASE_URL}/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(config),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Training failed')
  }
  
  return response.json()
}

export async function predict(request: PredictRequest): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Prediction failed')
  }
  
  return response.json()
}

export async function listDatasets(): Promise<DatasetInfo[]> {
  const response = await fetch(`${API_BASE_URL}/datasets`)
  if (!response.ok) throw new Error('Failed to fetch datasets')
  return response.json()
}

export async function listModels(): Promise<ModelInfo[]> {
  const response = await fetch(`${API_BASE_URL}/models`)
  if (!response.ok) throw new Error('Failed to fetch models')
  return response.json()
}