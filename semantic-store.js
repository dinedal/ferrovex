const native = require('./index.js')

function parseMetadata(raw) {
  if (raw == null) {
    return undefined
  }
  return JSON.parse(raw)
}

class SemanticStore {
  constructor(options = {}) {
    this._native = new native.NativeSemanticStore({
      dbPath: options.dbPath,
      tableName: options.tableName,
      runtime: options.runtime,
      modelArchitecture: options.modelArchitecture,
      modelId: options.modelId,
      revision: options.revision,
      onnxModelId: options.onnxModelId,
      onnxPathInRepo: options.onnxPathInRepo
    })
  }

  async insert(text, metadata) {
    const metadataJson = metadata == null ? undefined : JSON.stringify(metadata)
    return this._native.insert(text, metadataJson)
  }

  async embed(input) {
    if (Array.isArray(input)) {
      return this.embedBatch(input)
    }
    if (typeof input !== 'string') {
      throw new TypeError('embed input must be a string or string[]')
    }
    return this._native.embed(input)
  }

  async embedBatch(texts) {
    if (!Array.isArray(texts)) {
      throw new TypeError('embedBatch expects string[]')
    }
    return this._native.embedBatch(texts)
  }

  async query(text, params = {}) {
    const rows = await this._native.query(text, {
      limit: params.limit,
      filter: params.filter
    })

    return rows.map((row) => ({
      text: row.text,
      metadata: parseMetadata(row.metadataJson),
      distance: row.distance,
      score: row.score
    }))
  }
}

module.exports = {
  SemanticStore
}
