const assert = require('node:assert/strict')
const fs = require('node:fs/promises')
const os = require('node:os')
const path = require('node:path')

const { SemanticStore } = require('../semantic-store.js')

const DEFAULT_RUNTIME = process.env.FERROVEX_SMOKE_RUNTIME || 'hf'
const DEFAULT_MODEL_ID =
  process.env.FERROVEX_SMOKE_MODEL_ID || 'sentence-transformers/paraphrase-MiniLM-L3-v2'

async function run() {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ferrovex-smoke-'))
  const dbPath = path.join(tmpDir, 'lancedb')

  const store = new SemanticStore({
    dbPath,
    runtime: DEFAULT_RUNTIME,
    modelId: DEFAULT_MODEL_ID
  })

  const docs = [
    {
      text: 'Rust and N-API are a good fit for high performance Node.js extensions.',
      metadata: { id: 'doc-1', suite: 'smoke', source: 'example' }
    },
    {
      text: 'LanceDB stores vectors on disk and supports nearest-neighbor queries.',
      metadata: { id: 'doc-2', suite: 'smoke', source: 'example' }
    },
    {
      text: 'TypeScript bindings make native modules easier to adopt.',
      metadata: { id: 'doc-3', suite: 'smoke', source: 'example' }
    }
  ]

  try {
    const singleEmbedding = await store.embed('Rust native module embedding smoke test')
    assert(Array.isArray(singleEmbedding) && singleEmbedding.length > 0, 'expected single embedding vector')

    const batchEmbeddings = await store.embedBatch([
      'first batch embedding',
      'second batch embedding'
    ])
    assert(batchEmbeddings.length === 2, 'expected two batch embeddings')
    assert(
      batchEmbeddings.every((row) => Array.isArray(row) && row.length === singleEmbedding.length),
      'expected embedding dimension parity between single and batch calls'
    )

    for (const doc of docs) {
      await store.insert(doc.text, doc.metadata)
    }

    const matches = await store.query('fast rust node bindings', { limit: 3 })
    assert(matches.length > 0, 'expected at least one query match')
    assert(
      matches.some((m) => m.metadata && m.metadata.suite === 'smoke'),
      'expected smoke metadata in query matches'
    )
    assert(typeof matches[0].text === 'string' && matches[0].text.length > 0, 'expected match text')
    assert(typeof matches[0].score === 'number', 'expected query match score')

    console.log(
      `[smoke] inserted=${docs.length} matched=${matches.length} runtime=${DEFAULT_RUNTIME} embedDim=${singleEmbedding.length}`
    )
    console.log(
      `[smoke] top="${matches[0].text}" score=${matches[0].score} distance=${matches[0].distance}`
    )
  } finally {
    await fs.rm(tmpDir, { recursive: true, force: true })
  }
}

run().catch((err) => {
  console.error('[smoke] failed')
  console.error(err)
  process.exit(1)
})
