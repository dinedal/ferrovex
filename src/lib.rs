use std::sync::Arc;

use anyhow::{anyhow, Context, Result as AnyResult};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, RecordBatch,
    RecordBatchIterator, StringArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use embed_anything::{embed_query, embeddings::embed::Embedder};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{Connection, Table};
use napi_derive::napi;
use serde_json::Value;
use tempfile::TempDir;
use tokio::sync::{Mutex, OnceCell};

const DEFAULT_TABLE_NAME: &str = "embeddings";
const DEFAULT_HF_MODEL_ID: &str = "jinaai/jina-embeddings-v2-small-en";
const DEFAULT_ONNX_ARCHITECTURE: &str = "bert";

#[derive(Clone, Debug, Default)]
pub struct StoreOptions {
    pub db_path: Option<String>,
    pub table_name: Option<String>,
    pub runtime: Option<String>,
    pub model_architecture: Option<String>,
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub onnx_model_id: Option<String>,
    pub onnx_path_in_repo: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct QueryParams {
    pub limit: Option<usize>,
    pub filter: Option<String>,
}

#[derive(Clone, Debug)]
pub struct QueryMatch {
    pub text: String,
    pub metadata: Option<Value>,
    pub distance: Option<f64>,
    pub score: Option<f64>,
}

#[derive(Clone, Debug)]
struct StoreConfig {
    db_path: Option<String>,
    table_name: String,
    runtime: String,
    model_architecture: Option<String>,
    model_id: Option<String>,
    revision: Option<String>,
    onnx_model_id: Option<String>,
    onnx_path_in_repo: Option<String>,
}

impl From<StoreOptions> for StoreConfig {
    fn from(value: StoreOptions) -> Self {
        Self {
            db_path: value.db_path,
            table_name: value
                .table_name
                .unwrap_or_else(|| DEFAULT_TABLE_NAME.to_owned()),
            runtime: value
                .runtime
                .unwrap_or_else(|| "hf".to_owned())
                .to_ascii_lowercase(),
            model_architecture: value.model_architecture,
            model_id: value.model_id,
            revision: value.revision,
            onnx_model_id: value.onnx_model_id,
            onnx_path_in_repo: value.onnx_path_in_repo,
        }
    }
}

pub struct SemanticStore {
    inner: Mutex<StoreInner>,
}

impl SemanticStore {
    pub async fn new(options: StoreOptions) -> AnyResult<Self> {
        let config = StoreConfig::from(options);
        let inner = StoreInner::from_config(config).await?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    pub async fn insert(&self, text: &str, metadata: Option<Value>) -> AnyResult<()> {
        let mut inner = self.inner.lock().await;
        inner.insert(text, metadata).await
    }

    pub async fn query(&self, text: &str, params: QueryParams) -> AnyResult<Vec<QueryMatch>> {
        let mut inner = self.inner.lock().await;
        inner.query(text, params).await
    }

    pub async fn embed(&self, text: &str) -> AnyResult<Vec<f32>> {
        let inner = self.inner.lock().await;
        inner.embed_text(text).await
    }

    pub async fn embed_batch(&self, texts: &[String]) -> AnyResult<Vec<Vec<f32>>> {
        let inner = self.inner.lock().await;
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        inner.embed_texts(&text_refs).await
    }
}

struct StoreInner {
    db: Connection,
    table: Option<Table>,
    table_name: String,
    vector_dim: Option<usize>,
    embedder: Embedder,
    _temp_dir: Option<TempDir>,
}

impl StoreInner {
    async fn from_config(config: StoreConfig) -> AnyResult<Self> {
        let embedder = build_embedder(&config)?;
        let (db_path, temp_dir) = match config.db_path.clone() {
            Some(path) => (path, None),
            None => {
                let dir =
                    tempfile::tempdir().context("failed to create temporary LanceDB directory")?;
                (dir.path().to_string_lossy().to_string(), Some(dir))
            }
        };

        let db = lancedb::connect(&db_path)
            .execute()
            .await
            .with_context(|| format!("failed to connect to LanceDB at `{db_path}`"))?;

        let table_names = db
            .table_names()
            .execute()
            .await
            .context("failed to list LanceDB tables")?;

        let table = if table_names.iter().any(|name| name == &config.table_name) {
            Some(
                db.open_table(config.table_name.clone())
                    .execute()
                    .await
                    .with_context(|| format!("failed to open table `{}`", config.table_name))?,
            )
        } else {
            None
        };

        let vector_dim = match &table {
            Some(existing) => extract_vector_dim(existing).await?,
            None => None,
        };

        Ok(Self {
            db,
            table,
            table_name: config.table_name,
            vector_dim,
            embedder,
            _temp_dir: temp_dir,
        })
    }

    async fn insert(&mut self, text: &str, metadata: Option<Value>) -> AnyResult<()> {
        let embedding = self.embed_text(text).await?;
        if embedding.is_empty() {
            return Err(anyhow!("embed_anything returned an empty embedding"));
        }
        if let Some(expected_dim) = self.vector_dim {
            if expected_dim != embedding.len() {
                return Err(anyhow!(
                    "embedding dimension mismatch: table expects {}, model returned {}",
                    expected_dim,
                    embedding.len()
                ));
            }
        } else {
            self.vector_dim = Some(embedding.len());
        }

        let metadata_json = metadata
            .map(|value| serde_json::to_string(&value))
            .transpose()
            .context("failed to serialize metadata JSON")?;

        let schema = schema_for_dim(embedding.len())?;
        let batch = build_single_row_batch(schema.clone(), text, metadata_json, &embedding)?;
        let reader = RecordBatchIterator::new(
            std::iter::once(Ok::<RecordBatch, ArrowError>(batch)),
            schema,
        );

        if let Some(table) = &self.table {
            table
                .add(Box::new(reader))
                .execute()
                .await
                .context("failed to insert row into LanceDB table")?;
            return Ok(());
        }

        let created = self
            .db
            .create_table(self.table_name.clone(), Box::new(reader))
            .execute()
            .await
            .with_context(|| format!("failed to create LanceDB table `{}`", self.table_name))?;
        self.table = Some(created);
        Ok(())
    }

    async fn query(&mut self, text: &str, params: QueryParams) -> AnyResult<Vec<QueryMatch>> {
        if self.table.is_none() {
            return Err(anyhow!(
                "cannot query before data exists: no rows have been inserted yet"
            ));
        }

        let query_embedding = self.embed_text(text).await?;
        if let Some(expected_dim) = self.vector_dim {
            if expected_dim != query_embedding.len() {
                return Err(anyhow!(
                    "embedding dimension mismatch: table expects {}, model returned {}",
                    expected_dim,
                    query_embedding.len()
                ));
            }
        }
        let table = self.table.as_ref().expect("table existence checked");

        let QueryParams { limit, filter } = params;
        let mut vector_query = table
            .query()
            .select(Select::columns(&["text", "metadata_json", "_distance"]))
            .nearest_to(query_embedding.as_slice())
            .context("failed to build nearest-neighbor query")?;

        if let Some(value) = limit {
            vector_query = vector_query.limit(value);
        }
        if let Some(predicate) = filter {
            vector_query = vector_query.only_if(predicate);
        }

        let stream = vector_query
            .execute()
            .await
            .context("failed to execute vector query")?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .context("failed to stream vector query results")?;

        parse_query_rows(&batches)
    }

    async fn embed_text(&self, text: &str) -> AnyResult<Vec<f32>> {
        let mut rows = self.embed_texts(&[text]).await?;
        rows.pop()
            .ok_or_else(|| anyhow!("embed_anything did not return an embedding"))
    }

    async fn embed_texts(&self, texts: &[&str]) -> AnyResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let rows = embed_query(texts, &self.embedder, None)
            .await
            .context("embed_anything query embedding failed")?;

        rows.into_iter()
            .map(|row| {
                row.embedding
                    .to_dense()
                    .context("embed_anything returned a sparse embedding")
            })
            .collect()
    }
}

fn build_embedder(config: &StoreConfig) -> AnyResult<Embedder> {
    match config.runtime.as_str() {
        "hf" => {
            let model_id = config.model_id.as_deref().unwrap_or(DEFAULT_HF_MODEL_ID);
            build_hf_embedder_with_cpu_fallback(model_id, config.revision.as_deref())
                .with_context(|| format!("failed to load HF embedding model `{model_id}`"))
        }
        "onnx" => {
            let architecture = config
                .model_architecture
                .as_deref()
                .unwrap_or(DEFAULT_ONNX_ARCHITECTURE);
            let model_id = config
                .onnx_model_id
                .as_deref()
                .or(config.model_id.as_deref())
                .ok_or_else(|| anyhow!("ONNX runtime requires `onnx_model_id` or `model_id`"))?;
            Embedder::from_pretrained_onnx(
                architecture,
                None,
                config.revision.as_deref(),
                Some(model_id),
                None,
                config.onnx_path_in_repo.as_deref(),
            )
            .with_context(|| format!("failed to load ONNX embedding model `{model_id}`"))
        }
        other => Err(anyhow!(
            "unsupported runtime `{other}` (expected `hf` or `onnx`)"
        )),
    }
}

fn build_hf_embedder_with_cpu_fallback(
    model_id: &str,
    revision: Option<&str>,
) -> AnyResult<Embedder> {
    configure_hf_runtime_for_macos();
    Embedder::from_pretrained_hf(model_id, revision, None, None).map_err(Into::into)
}

#[cfg(target_os = "macos")]
fn configure_hf_runtime_for_macos() {
    if std::env::var_os("CANDLE_FORCE_CPU").is_some() {
        return;
    }
    if metal::Device::all().is_empty() {
        std::env::set_var("CANDLE_FORCE_CPU", "1");
    }
}

#[cfg(not(target_os = "macos"))]
fn configure_hf_runtime_for_macos() {}

async fn extract_vector_dim(table: &Table) -> AnyResult<Option<usize>> {
    let schema = table
        .schema()
        .await
        .context("failed to read LanceDB table schema")?;

    let vector_field = match schema.field_with_name("vector") {
        Ok(field) => field,
        Err(_) => return Ok(None),
    };

    match vector_field.data_type() {
        DataType::FixedSizeList(_, size) => usize::try_from(*size)
            .map(Some)
            .map_err(|_| anyhow!("table vector dimension was negative")),
        other => Err(anyhow!(
            "table `vector` column has unsupported type: {other:?}"
        )),
    }
}

fn schema_for_dim(dim: usize) -> AnyResult<SchemaRef> {
    let dim_i32 = i32::try_from(dim).context("embedding dimension exceeded i32")?;
    Ok(Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("metadata_json", DataType::Utf8, true),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim_i32,
            ),
            false,
        ),
    ])))
}

fn build_single_row_batch(
    schema: SchemaRef,
    text: &str,
    metadata_json: Option<String>,
    embedding: &[f32],
) -> AnyResult<RecordBatch> {
    let dim_i32 = i32::try_from(embedding.len()).context("embedding dimension exceeded i32")?;

    let text_col: ArrayRef = Arc::new(StringArray::from(vec![Some(text)]));
    let metadata_col: ArrayRef = Arc::new(StringArray::from(vec![metadata_json.as_deref()]));
    let vector_values: ArrayRef = Arc::new(Float32Array::from(embedding.to_vec()));
    let vector_col: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim_i32,
            vector_values,
            None,
        )
        .context("failed to build vector column for insert")?,
    );

    RecordBatch::try_new(schema, vec![text_col, metadata_col, vector_col])
        .context("failed to create Arrow record batch")
}

fn parse_query_rows(batches: &[RecordBatch]) -> AnyResult<Vec<QueryMatch>> {
    let mut results = Vec::new();

    for batch in batches {
        let text_col = batch
            .column_by_name("text")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| anyhow!("vector query results missing `text` column"))?;

        let metadata_col = batch
            .column_by_name("metadata_json")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>());

        for row in 0..batch.num_rows() {
            let text = text_col.value(row).to_owned();
            let metadata = match metadata_col {
                Some(col) if !col.is_null(row) => Some(
                    serde_json::from_str::<Value>(col.value(row))
                        .with_context(|| format!("failed to parse metadata JSON for row {row}"))?,
                ),
                _ => None,
            };
            let distance = extract_distance(batch, row);
            let score = distance.map(distance_to_similarity_score);
            results.push(QueryMatch {
                text,
                metadata,
                distance,
                score,
            });
        }
    }

    Ok(results)
}

fn extract_distance(batch: &RecordBatch, row: usize) -> Option<f64> {
    let distance_col = batch
        .column_by_name("_distance")
        .or_else(|| batch.column_by_name("distance"))?;

    if let Some(col) = distance_col.as_any().downcast_ref::<Float32Array>() {
        if col.is_null(row) {
            return None;
        }
        return Some(f64::from(col.value(row)));
    }

    if let Some(col) = distance_col.as_any().downcast_ref::<Float64Array>() {
        if col.is_null(row) {
            return None;
        }
        return Some(col.value(row));
    }

    None
}

fn distance_to_similarity_score(distance: f64) -> f64 {
    1.0 / (1.0 + distance.max(0.0))
}

#[napi(object)]
#[derive(Default)]
pub struct NativeStoreOptions {
    pub db_path: Option<String>,
    pub table_name: Option<String>,
    pub runtime: Option<String>,
    pub model_architecture: Option<String>,
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub onnx_model_id: Option<String>,
    pub onnx_path_in_repo: Option<String>,
}

impl From<NativeStoreOptions> for StoreOptions {
    fn from(value: NativeStoreOptions) -> Self {
        Self {
            db_path: value.db_path,
            table_name: value.table_name,
            runtime: value.runtime,
            model_architecture: value.model_architecture,
            model_id: value.model_id,
            revision: value.revision,
            onnx_model_id: value.onnx_model_id,
            onnx_path_in_repo: value.onnx_path_in_repo,
        }
    }
}

#[napi(object)]
#[derive(Default)]
pub struct NativeQueryParams {
    pub limit: Option<u32>,
    pub filter: Option<String>,
}

impl From<NativeQueryParams> for QueryParams {
    fn from(value: NativeQueryParams) -> Self {
        Self {
            limit: value.limit.map(|v| v as usize),
            filter: value.filter,
        }
    }
}

#[napi(object)]
pub struct NativeQueryResult {
    pub text: String,
    pub metadata_json: Option<String>,
    pub distance: Option<f64>,
    pub score: Option<f64>,
}

#[napi]
pub struct NativeSemanticStore {
    options: StoreOptions,
    inner: OnceCell<SemanticStore>,
}

#[napi]
impl NativeSemanticStore {
    #[napi(constructor)]
    pub fn new(options: Option<NativeStoreOptions>) -> Self {
        Self {
            options: options.map(StoreOptions::from).unwrap_or_default(),
            inner: OnceCell::new(),
        }
    }

    #[napi]
    pub async fn insert(&self, text: String, metadata_json: Option<String>) -> napi::Result<()> {
        let store = self.get_store().await?;
        let metadata = metadata_json
            .as_deref()
            .map(serde_json::from_str::<Value>)
            .transpose()
            .map_err(|err| napi::Error::from_reason(err.to_string()))?;

        store.insert(&text, metadata).await.map_err(anyhow_to_napi)
    }

    #[napi]
    pub async fn embed(&self, text: String) -> napi::Result<Vec<f32>> {
        let store = self.get_store().await?;
        store.embed(&text).await.map_err(anyhow_to_napi)
    }

    #[napi(js_name = "embedBatch")]
    pub async fn embed_batch(&self, texts: Vec<String>) -> napi::Result<Vec<Vec<f32>>> {
        let store = self.get_store().await?;
        store.embed_batch(&texts).await.map_err(anyhow_to_napi)
    }

    #[napi]
    pub async fn query(
        &self,
        text: String,
        params: Option<NativeQueryParams>,
    ) -> napi::Result<Vec<NativeQueryResult>> {
        let store = self.get_store().await?;
        let rows = store
            .query(&text, params.map(QueryParams::from).unwrap_or_default())
            .await
            .map_err(anyhow_to_napi)?;

        rows.into_iter()
            .map(|row| {
                let metadata_json = row
                    .metadata
                    .map(|value| serde_json::to_string(&value))
                    .transpose()
                    .map_err(|err| napi::Error::from_reason(err.to_string()))?;

                Ok(NativeQueryResult {
                    text: row.text,
                    metadata_json,
                    distance: row.distance,
                    score: row.score,
                })
            })
            .collect()
    }

    async fn get_store(&self) -> napi::Result<&SemanticStore> {
        let options = self.options.clone();
        self.inner
            .get_or_try_init(|| async move { SemanticStore::new(options).await })
            .await
            .map_err(anyhow_to_napi)
    }
}

fn anyhow_to_napi(err: anyhow::Error) -> napi::Error {
    napi::Error::from_reason(format!("{err:#}"))
}
