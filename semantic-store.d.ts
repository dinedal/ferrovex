export type Metadata = Record<string, unknown>

export interface StoreOptions {
  dbPath?: string
  tableName?: string
  runtime?: 'hf' | 'onnx'
  modelArchitecture?: string
  modelId?: string
  revision?: string
  onnxModelId?: string
  onnxPathInRepo?: string
}

export interface QueryParams {
  limit?: number
  filter?: string
}

export interface QueryResult {
  text: string
  metadata?: Metadata
  distance?: number
  score?: number
}

export declare class SemanticStore {
  constructor(options?: StoreOptions)
  insert(text: string, metadata?: Metadata): Promise<void>
  embed(text: string): Promise<number[]>
  embed(texts: string[]): Promise<number[][]>
  embedBatch(texts: string[]): Promise<number[][]>
  query(text: string, params?: QueryParams): Promise<QueryResult[]>
}
