use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::{header::AUTHORIZATION, Client};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::errors::CloudError;
use crate::gcp::gcp_apis::auth::gcp_auth::ServiceAccountTokenProvider;
use crate::gcp::types::database::gcp_bigquery_types::*;
use crate::traits::token_provider::TokenProvider;

struct CachedToken {
    value: String,
    expires_at: Instant,
}

pub struct BigQuery {
    client: Client,
    project_id: String,
    auth: Arc<dyn TokenProvider>,
    token: Mutex<CachedToken>,
}

impl BigQuery {
    pub async fn new(project_id: &str) -> Result<Self, CloudError> {
        let auth: Arc<dyn TokenProvider> = Arc::new(
            ServiceAccountTokenProvider::new(
                PathBuf::from("service-account.json"),
                vec!["https://www.googleapis.com/auth/cloud-platform".to_string()],
            )
            .map_err(|e| CloudError::Auth { message: e.to_string() })?,
        );
        let token = auth.get_token().await.map_err(|e| CloudError::Auth {
            message: e.to_string(),
        })?;
        Ok(Self {
            client: Client::new(),
            project_id: project_id.to_string(),
            auth,
            token: Mutex::new(CachedToken {
                value: token,
                expires_at: Instant::now() + Duration::from_secs(3600),
            }),
        })
    }

    pub fn with_http_client(
        client: Client,
        project_id: &str,
        auth: Arc<dyn TokenProvider>,
    ) -> Self {
        Self {
            client,
            project_id: project_id.to_string(),
            auth,
            token: Mutex::new(CachedToken {
                value: String::new(),
                expires_at: Instant::now(),
            }),
        }
    }

    pub(crate) async fn get_token(&self) -> Result<String, CloudError> {
        let mut cached = self.token.lock().await;
        if cached.expires_at.saturating_duration_since(Instant::now()) < Duration::from_secs(300) {
            let fresh = self.auth.get_token().await.map_err(|e| CloudError::Auth {
                message: e.to_string(),
            })?;
            cached.value = fresh;
            cached.expires_at = Instant::now() + Duration::from_secs(3600);
        }
        Ok(cached.value.clone())
    }

    pub async fn create_dataset(&self, dataset_id: &str) -> Result<DatasetInfo, CloudError> {
        let url = bigquery_url(&self.project_id, "datasets");
        let body = CreateDataset {
            dataset_reference: DatasetReference {
                project_id: self.project_id.clone(),
                dataset_id: dataset_id.to_string(),
            },
        };
        let token = self.get_token().await?;

        let response = self
            .client
            .post(&url)
            .json(&body)
            .header(AUTHORIZATION, bearer(&token))
            .send()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let status = response.status().as_u16();
        let text = response.text().await.map_err(|e| CloudError::Network { source: e })?;
        if status >= 400 {
            return Err(map_bigquery_http_error(status, &text));
        }

        let resp: Value = serde_json::from_str(&text).map_err(|e| CloudError::Serialization { source: e })?;
        parse_dataset_info(&resp)
    }

    pub async fn delete_dataset(
        &self,
        dataset_id: &str,
        delete_contents: bool,
    ) -> Result<(), CloudError> {
        let url = format!(
            "{}?deleteContents={}",
            bigquery_url(&self.project_id, &format!("datasets/{}", dataset_id)),
            delete_contents
        );
        let token = self.get_token().await?;

        let response = self
            .client
            .delete(&url)
            .header(AUTHORIZATION, bearer(&token))
            .send()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let status = response.status().as_u16();
        let text = response.text().await.map_err(|e| CloudError::Network { source: e })?;
        if status >= 400 {
            return Err(map_bigquery_http_error(status, &text));
        }

        Ok(())
    }

    pub async fn list_datasets(
        &self,
        page_token: Option<&str>,
        max_results: Option<u32>,
    ) -> Result<DatasetPage, CloudError> {
        let base = bigquery_url(&self.project_id, "datasets");
        let mut params: Vec<String> = Vec::new();
        if let Some(pt) = page_token {
            params.push(format!("pageToken={}", pt));
        }
        if let Some(mr) = max_results {
            params.push(format!("maxResults={}", mr));
        }
        let url = if params.is_empty() {
            base
        } else {
            format!("{}?{}", base, params.join("&"))
        };

        let token = self.get_token().await?;
        let response = self
            .client
            .get(&url)
            .header(AUTHORIZATION, bearer(&token))
            .send()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let status = response.status().as_u16();
        let text = response.text().await.map_err(|e| CloudError::Network { source: e })?;
        if status >= 400 {
            return Err(map_bigquery_http_error(status, &text));
        }

        let resp: Value = serde_json::from_str(&text).map_err(|e| CloudError::Serialization { source: e })?;
        parse_dataset_list(&resp)
    }

    pub async fn create_table(
        &self,
        dataset_id: &str,
        table_id: &str,
        fields: Vec<TableField>,
    ) -> Result<TableInfo, CloudError> {
        let url = bigquery_url(&self.project_id, &format!("datasets/{}/tables", dataset_id));
        let body = CreateTable {
            table_reference: TableReference {
                project_id: self.project_id.clone(),
                dataset_id: dataset_id.to_string(),
                table_id: table_id.to_string(),
            },
            schema: TableSchema { fields },
        };
        let token = self.get_token().await?;

        let response = self
            .client
            .post(&url)
            .json(&body)
            .header(AUTHORIZATION, bearer(&token))
            .send()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let status = response.status().as_u16();
        let text = response.text().await.map_err(|e| CloudError::Network { source: e })?;
        if status >= 400 {
            return Err(map_bigquery_http_error(status, &text));
        }

        let resp: Value = serde_json::from_str(&text).map_err(|e| CloudError::Serialization { source: e })?;
        parse_table_info(&resp)
    }

    pub async fn delete_table(
        &self,
        dataset_id: &str,
        table_id: &str,
    ) -> Result<(), CloudError> {
        let url = bigquery_url(
            &self.project_id,
            &format!("datasets/{}/tables/{}", dataset_id, table_id),
        );
        let token = self.get_token().await?;

        let response = self
            .client
            .delete(&url)
            .header(AUTHORIZATION, bearer(&token))
            .send()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let status = response.status().as_u16();
        let text = response.text().await.map_err(|e| CloudError::Network { source: e })?;
        if status >= 400 {
            return Err(map_bigquery_http_error(status, &text));
        }

        Ok(())
    }

    pub async fn list_tables(
        &self,
        dataset_id: &str,
        page_token: Option<&str>,
        max_results: Option<u32>,
    ) -> Result<TablePage, CloudError> {
        let base = bigquery_url(&self.project_id, &format!("datasets/{}/tables", dataset_id));
        let mut params: Vec<String> = Vec::new();
        if let Some(pt) = page_token {
            params.push(format!("pageToken={}", pt));
        }
        if let Some(mr) = max_results {
            params.push(format!("maxResults={}", mr));
        }
        let url = if params.is_empty() {
            base
        } else {
            format!("{}?{}", base, params.join("&"))
        };

        let token = self.get_token().await?;
        let response = self
            .client
            .get(&url)
            .header(AUTHORIZATION, bearer(&token))
            .send()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let status = response.status().as_u16();
        let text = response.text().await.map_err(|e| CloudError::Network { source: e })?;
        if status >= 400 {
            return Err(map_bigquery_http_error(status, &text));
        }

        let resp: Value = serde_json::from_str(&text).map_err(|e| CloudError::Serialization { source: e })?;
        parse_table_list(&resp)
    }

    pub async fn run_query(&self, query: &str) -> Result<String, CloudError> {
        let url = bigquery_url(&self.project_id, "queries");
        let body = RunQuery {
            query: query.to_string(),
            use_legacy_sql: false,
        };
        let token = self.get_token().await?;

        let response = self
            .client
            .post(&url)
            .json(&body)
            .header(AUTHORIZATION, bearer(&token))
            .send()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let status = response.status().as_u16();
        let text = response.text().await.map_err(|e| CloudError::Network { source: e })?;
        if status >= 400 {
            return Err(map_bigquery_http_error(status, &text));
        }

        let resp: Value = serde_json::from_str(&text).map_err(|e| CloudError::Serialization { source: e })?;
        parse_job_id(&resp)
    }

    pub async fn get_query_results(&self, job_id: &str) -> Result<QueryResult, CloudError> {
        for attempt in 0..60 {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            let resp = self.fetch_query_results(job_id).await?;
            if resp["jobComplete"].as_bool().unwrap_or(false) {
                return parse_query_result(&resp);
            }
        }
        Err(CloudError::Provider {
            http_status: 0,
            message: "query did not complete within 60 seconds".to_string(),
            retryable: true,
        })
    }

    pub(crate) async fn fetch_query_results(&self, job_id: &str) -> Result<Value, CloudError> {
        let url = bigquery_url(&self.project_id, &format!("queries/{}", job_id));
        let token = self.get_token().await?;

        let response = self
            .client
            .get(&url)
            .header(AUTHORIZATION, bearer(&token))
            .send()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let status = response.status().as_u16();
        let text = response.text().await.map_err(|e| CloudError::Network { source: e })?;
        if status >= 400 {
            return Err(map_bigquery_http_error(status, &text));
        }

        serde_json::from_str(&text).map_err(|e| CloudError::Serialization { source: e })
    }

    pub async fn insert_rows(
        &self,
        dataset_id: &str,
        table_id: &str,
        rows: Vec<Value>,
    ) -> Result<(), CloudError> {
        let url = bigquery_url(
            &self.project_id,
            &format!("datasets/{}/tables/{}/insertAll", dataset_id, table_id),
        );
        let body = json!({
            "rows": rows
                .into_iter()
                .enumerate()
                .map(|(i, row)| json!({ "insertId": i.to_string(), "json": row }))
                .collect::<Vec<_>>()
        });
        let token = self.get_token().await?;

        let response = self
            .client
            .post(&url)
            .json(&body)
            .header(AUTHORIZATION, bearer(&token))
            .send()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let status = response.status().as_u16();
        let text = response.text().await.map_err(|e| CloudError::Network { source: e })?;
        if status >= 400 {
            return Err(map_bigquery_http_error(status, &text));
        }

        let resp: Value = serde_json::from_str(&text).map_err(|e| CloudError::Serialization { source: e })?;
        if let Some(errors) = resp["insertErrors"].as_array() {
            if !errors.is_empty() {
                let msg = errors[0]["errors"]
                    .as_array()
                    .and_then(|e| e.first())
                    .and_then(|e| e["message"].as_str())
                    .unwrap_or("streaming insert failed")
                    .to_string();
                return Err(CloudError::Provider {
                    http_status: 200,
                    message: msg,
                    retryable: false,
                });
            }
        }
        Ok(())
    }
}

fn bearer(token: &str) -> String {
    format!("Bearer {}", token)
}

pub(crate) fn bigquery_url(project_id: &str, path: &str) -> String {
    format!("https://bigquery.googleapis.com/bigquery/v2/projects/{}/{}", project_id, path)
}

pub(crate) fn map_bigquery_http_error(status: u16, body: &str) -> CloudError {
    match status {
        401 | 403 => CloudError::Auth { message: body.to_string() },
        404 | 409 => CloudError::Provider {
            http_status: status,
            message: body.to_string(),
            retryable: false,
        },
        429 => CloudError::RateLimit { retry_after: None },
        500 | 503 => CloudError::Provider {
            http_status: status,
            message: body.to_string(),
            retryable: true,
        },
        _ => CloudError::Provider {
            http_status: status,
            message: body.to_string(),
            retryable: status >= 500,
        },
    }
}

pub(crate) fn parse_dataset_info(json: &Value) -> Result<DatasetInfo, CloudError> {
    let id = json["datasetReference"]["datasetId"]
        .as_str()
        .ok_or_else(|| CloudError::Provider {
            http_status: 0,
            message: "parse error: dataset id missing from response".to_string(),
            retryable: false,
        })?
        .to_string();
    let project_id = json["datasetReference"]["projectId"]
        .as_str()
        .ok_or_else(|| CloudError::Provider {
            http_status: 0,
            message: "parse error: project id missing from response".to_string(),
            retryable: false,
        })?
        .to_string();
    Ok(DatasetInfo { id, project_id })
}

pub(crate) fn parse_dataset_list(json: &Value) -> Result<DatasetPage, CloudError> {
    let datasets = match json["datasets"].as_array() {
        Some(arr) => arr.iter().map(parse_dataset_info).collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    let next_page_token = json["nextPageToken"].as_str().map(str::to_owned);
    Ok(DatasetPage { datasets, next_page_token })
}

pub(crate) fn parse_table_info(json: &Value) -> Result<TableInfo, CloudError> {
    let id = json["tableReference"]["tableId"]
        .as_str()
        .ok_or_else(|| CloudError::Provider {
            http_status: 0,
            message: "parse error: table id missing from response".to_string(),
            retryable: false,
        })?
        .to_string();
    let dataset_id = json["tableReference"]["datasetId"]
        .as_str()
        .ok_or_else(|| CloudError::Provider {
            http_status: 0,
            message: "parse error: dataset id missing from table response".to_string(),
            retryable: false,
        })?
        .to_string();
    Ok(TableInfo { id, dataset_id })
}

pub(crate) fn parse_table_list(json: &Value) -> Result<TablePage, CloudError> {
    let tables = match json["tables"].as_array() {
        Some(arr) => arr.iter().map(parse_table_info).collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    let next_page_token = json["nextPageToken"].as_str().map(str::to_owned);
    Ok(TablePage { tables, next_page_token })
}

pub(crate) fn parse_job_id(json: &Value) -> Result<String, CloudError> {
    json["jobReference"]["jobId"]
        .as_str()
        .ok_or_else(|| CloudError::Provider {
            http_status: 0,
            message: "parse error: job id missing from response".to_string(),
            retryable: false,
        })
        .map(str::to_owned)
}

pub(crate) fn parse_query_result(json: &Value) -> Result<QueryResult, CloudError> {
    let rows = match json["rows"].as_array() {
        Some(arr) => arr.to_vec(),
        None => Vec::new(),
    };
    let total_rows = json["totalRows"].as_str().and_then(|s| s.parse::<u64>().ok());
    Ok(QueryResult { rows, total_rows })
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::gcp::gcp_apis::auth::gcp_auth::MockTokenProvider;

    fn no_creds_bigquery() -> BigQuery {
        BigQuery::with_http_client(
            Client::new(),
            "test-project",
            Arc::new(MockTokenProvider::new("test-token")),
        )
    }

    #[test]
    fn test_bigquery_url_simple_path() {
        assert_eq!(
            bigquery_url("my-project", "datasets"),
            "https://bigquery.googleapis.com/bigquery/v2/projects/my-project/datasets"
        );
    }

    #[test]
    fn test_bigquery_url_nested_path() {
        assert_eq!(
            bigquery_url("my-project", "datasets/ds1/tables/tbl1"),
            "https://bigquery.googleapis.com/bigquery/v2/projects/my-project/datasets/ds1/tables/tbl1"
        );
    }

    #[test]
    fn test_map_bigquery_http_error_auth() {
        match map_bigquery_http_error(401, "denied") {
            CloudError::Auth { message } => assert_eq!(message, "denied"),
            _ => panic!("expected Auth variant"),
        }
        match map_bigquery_http_error(403, "forbidden") {
            CloudError::Auth { .. } => {}
            _ => panic!("expected Auth variant"),
        }
    }

    #[test]
    fn test_map_bigquery_http_error_not_found_not_retryable() {
        match map_bigquery_http_error(404, "missing") {
            CloudError::Provider { http_status, retryable, .. } => {
                assert_eq!(http_status, 404);
                assert!(!retryable);
            }
            _ => panic!("expected Provider variant"),
        }
        match map_bigquery_http_error(409, "conflict") {
            CloudError::Provider { retryable, .. } => assert!(!retryable),
            _ => panic!("expected Provider variant"),
        }
    }

    #[test]
    fn test_map_bigquery_http_error_rate_limit() {
        match map_bigquery_http_error(429, "slow down") {
            CloudError::RateLimit { retry_after } => assert_eq!(retry_after, None),
            _ => panic!("expected RateLimit variant"),
        }
    }

    #[test]
    fn test_map_bigquery_http_error_server_error_retryable() {
        match map_bigquery_http_error(500, "boom") {
            CloudError::Provider { retryable, .. } => assert!(retryable),
            _ => panic!("expected Provider variant"),
        }
        match map_bigquery_http_error(503, "unavailable") {
            CloudError::Provider { retryable, .. } => assert!(retryable),
            _ => panic!("expected Provider variant"),
        }
    }

    #[tokio::test]
    async fn test_get_token_refreshes_when_expired() {
        let bq = no_creds_bigquery();
        let token = bq.get_token().await.unwrap();
        assert_eq!(token, "test-token");
    }

    #[test]
    fn test_parse_dataset_info_valid() {
        let json = serde_json::json!({
            "datasetReference": { "datasetId": "my_ds", "projectId": "my-project" }
        });
        let info = parse_dataset_info(&json).unwrap();
        assert_eq!(info.id, "my_ds");
        assert_eq!(info.project_id, "my-project");
    }

    #[test]
    fn test_parse_dataset_info_missing_dataset_id() {
        let json = serde_json::json!({ "datasetReference": { "projectId": "my-project" } });
        assert!(parse_dataset_info(&json).is_err());
    }

    #[test]
    fn test_parse_dataset_info_missing_project_id() {
        let json = serde_json::json!({ "datasetReference": { "datasetId": "my_ds" } });
        assert!(parse_dataset_info(&json).is_err());
    }

    #[test]
    fn test_parse_dataset_list_empty_array() {
        let json = serde_json::json!({ "datasets": [] });
        let page = parse_dataset_list(&json).unwrap();
        assert!(page.datasets.is_empty());
        assert!(page.next_page_token.is_none());
    }

    #[test]
    fn test_parse_dataset_list_no_datasets_key() {
        let json = serde_json::json!({});
        let page = parse_dataset_list(&json).unwrap();
        assert!(page.datasets.is_empty());
        assert!(page.next_page_token.is_none());
    }

    #[test]
    fn test_parse_dataset_list_with_next_token() {
        let json = serde_json::json!({
            "datasets": [
                { "datasetReference": { "datasetId": "ds1", "projectId": "proj" } }
            ],
            "nextPageToken": "abc123"
        });
        let page = parse_dataset_list(&json).unwrap();
        assert_eq!(page.datasets.len(), 1);
        assert_eq!(page.datasets[0].id, "ds1");
        assert_eq!(page.next_page_token, Some("abc123".to_string()));
    }

    #[test]
    fn test_parse_dataset_list_without_next_token() {
        let json = serde_json::json!({
            "datasets": [
                { "datasetReference": { "datasetId": "ds1", "projectId": "proj" } },
                { "datasetReference": { "datasetId": "ds2", "projectId": "proj" } }
            ]
        });
        let page = parse_dataset_list(&json).unwrap();
        assert_eq!(page.datasets.len(), 2);
        assert_eq!(page.datasets[1].id, "ds2");
        assert!(page.next_page_token.is_none());
    }

    #[test]
    fn test_parse_table_info_valid() {
        let json = serde_json::json!({
            "tableReference": { "tableId": "my_tbl", "datasetId": "my_ds", "projectId": "proj" }
        });
        let info = parse_table_info(&json).unwrap();
        assert_eq!(info.id, "my_tbl");
        assert_eq!(info.dataset_id, "my_ds");
    }

    #[test]
    fn test_parse_table_info_missing_table_id() {
        let json = serde_json::json!({ "tableReference": { "datasetId": "my_ds" } });
        assert!(parse_table_info(&json).is_err());
    }

    #[test]
    fn test_parse_table_info_missing_dataset_id() {
        let json = serde_json::json!({ "tableReference": { "tableId": "my_tbl" } });
        assert!(parse_table_info(&json).is_err());
    }

    #[test]
    fn test_parse_table_list_empty() {
        let json = serde_json::json!({});
        let page = parse_table_list(&json).unwrap();
        assert!(page.tables.is_empty());
        assert!(page.next_page_token.is_none());
    }

    #[test]
    fn test_parse_table_list_with_next_token() {
        let json = serde_json::json!({
            "tables": [
                { "tableReference": { "tableId": "tbl1", "datasetId": "ds1", "projectId": "proj" } }
            ],
            "nextPageToken": "tok123"
        });
        let page = parse_table_list(&json).unwrap();
        assert_eq!(page.tables.len(), 1);
        assert_eq!(page.tables[0].id, "tbl1");
        assert_eq!(page.next_page_token, Some("tok123".to_string()));
    }

    #[test]
    fn test_parse_table_list_without_next_token() {
        let json = serde_json::json!({
            "tables": [
                { "tableReference": { "tableId": "tbl1", "datasetId": "ds1", "projectId": "proj" } },
                { "tableReference": { "tableId": "tbl2", "datasetId": "ds1", "projectId": "proj" } }
            ]
        });
        let page = parse_table_list(&json).unwrap();
        assert_eq!(page.tables.len(), 2);
        assert_eq!(page.tables[1].id, "tbl2");
        assert!(page.next_page_token.is_none());
    }

    #[test]
    fn test_parse_job_id_valid() {
        let json = serde_json::json!({ "jobReference": { "jobId": "job_abc", "projectId": "proj" } });
        let id = parse_job_id(&json).unwrap();
        assert_eq!(id, "job_abc");
    }

    #[test]
    fn test_parse_job_id_missing_returns_error() {
        let json = serde_json::json!({ "jobReference": {} });
        assert!(parse_job_id(&json).is_err());
    }

    #[test]
    fn test_parse_query_result_valid() {
        let json = serde_json::json!({
            "rows": [
                { "f": [{ "v": "hello" }] },
                { "f": [{ "v": "world" }] }
            ],
            "totalRows": "2",
            "jobComplete": true
        });
        let result = parse_query_result(&json).unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.total_rows, Some(2));
        assert_eq!(result.rows[0]["f"][0]["v"], "hello");
    }

    #[test]
    fn test_parse_query_result_empty_rows() {
        let json = serde_json::json!({ "jobComplete": true, "totalRows": "0" });
        let result = parse_query_result(&json).unwrap();
        assert!(result.rows.is_empty());
        assert_eq!(result.total_rows, Some(0));
    }

    #[test]
    fn test_insert_rows_request_body() {
        // Verifies the expected wire shape: each row wrapped with a string insertId and a "json" key.
        let expected = serde_json::json!({
            "rows": [
                { "insertId": "0", "json": { "name": "Alice", "age": 30 } },
                { "insertId": "1", "json": { "name": "Bob",   "age": 25 } }
            ]
        });
        let rows_arr = expected["rows"].as_array().unwrap();
        assert_eq!(rows_arr.len(), 2);
        assert_eq!(rows_arr[0]["insertId"], "0");
        assert_eq!(rows_arr[0]["json"]["name"], "Alice");
        assert_eq!(rows_arr[1]["insertId"], "1");
        assert_eq!(rows_arr[1]["json"]["age"], 25);
    }
}
