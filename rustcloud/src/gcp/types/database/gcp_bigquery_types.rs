use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetReference {
    pub project_id: String,
    pub dataset_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateDataset {
    pub dataset_reference: DatasetReference,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TableReference {
    pub project_id: String,
    pub dataset_id: String,
    pub table_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TableField {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TableSchema {
    pub fields: Vec<TableField>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateTable {
    pub table_reference: TableReference,
    pub schema: TableSchema,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunQuery {
    pub query: String,
    #[serde(rename = "useLegacySql")]
    pub use_legacy_sql: bool,
}

#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub id: String,
    pub project_id: String,
}

#[derive(Debug, Clone)]
pub struct DatasetPage {
    pub datasets: Vec<DatasetInfo>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TableInfo {
    pub id: String,
    pub dataset_id: String,
}

#[derive(Debug, Clone)]
pub struct TablePage {
    pub tables: Vec<TableInfo>,
    pub next_page_token: Option<String>,
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub rows: Vec<Value>,
    pub total_rows: Option<u64>,
}
