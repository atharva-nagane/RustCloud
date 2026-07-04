
# rustcloud - GCP BigQuery

## Configure GCP credentials

Place your service account key file at `service-account.json` in the root of your project:

```sh
gcloud iam service-accounts keys create service-account.json \
  --iam-account=my-sa@my-project.iam.gserviceaccount.com
```

The service account must have the `roles/bigquery.dataEditor` and `roles/bigquery.jobUser` IAM roles on the project.

## Initialize the client

```rust
use rustcloud::gcp::gcp_apis::database::gcp_bigquery::BigQuery;

#[tokio::main]
async fn main() {
    let client = BigQuery::new("my-gcp-project")
        .await
        .expect("failed to authenticate with BigQuery");
}
```

The client reads `service-account.json` on startup and caches the OAuth2 token. The token is refreshed automatically when fewer than five minutes remain before expiry.

## Create a dataset

```rust
use rustcloud::gcp::gcp_apis::database::gcp_bigquery::BigQuery;

#[tokio::main]
async fn main() {
    let client = BigQuery::new("my-gcp-project").await.unwrap();

    let info = client.create_dataset("my_dataset").await.unwrap();
    println!("created dataset: {}", info.id);
}
```

## Create a table with schema

```rust
use rustcloud::gcp::gcp_apis::database::gcp_bigquery::BigQuery;
use rustcloud::gcp::types::database::gcp_bigquery_types::TableField;

#[tokio::main]
async fn main() {
    let client = BigQuery::new("my-gcp-project").await.unwrap();

    let fields = vec![
        TableField {
            name: "user_id".to_string(),
            field_type: "INTEGER".to_string(),
            mode: Some("REQUIRED".to_string()),
        },
        TableField {
            name: "email".to_string(),
            field_type: "STRING".to_string(),
            mode: Some("NULLABLE".to_string()),
        },
        TableField {
            name: "tags".to_string(),
            field_type: "STRING".to_string(),
            mode: Some("REPEATED".to_string()),
        },
    ];

    let info = client
        .create_table("my_dataset", "users", fields)
        .await
        .unwrap();

    println!("created table: {} in dataset: {}", info.id, info.dataset_id);
}
```

Valid values for `mode` are `NULLABLE`, `REQUIRED`, and `REPEATED`. Omitting `mode` (passing `None`) leaves the field nullable, which is the BigQuery default.

## List datasets with pagination

```rust
use rustcloud::gcp::gcp_apis::database::gcp_bigquery::BigQuery;

#[tokio::main]
async fn main() {
    let client = BigQuery::new("my-gcp-project").await.unwrap();

    let mut page_token: Option<String> = None;

    loop {
        let page = client
            .list_datasets(page_token.as_deref(), Some(50))
            .await
            .unwrap();

        for ds in &page.datasets {
            println!("{}", ds.id);
        }

        page_token = page.next_page_token;
        if page_token.is_none() {
            break;
        }
    }
}
```

Pass `None` for both arguments to retrieve datasets with no page size limit. The `"datasets"` key is absent from the BigQuery response when a project has no datasets; the client returns an empty page rather than an error.

## List tables with pagination

```rust
use rustcloud::gcp::gcp_apis::database::gcp_bigquery::BigQuery;

#[tokio::main]
async fn main() {
    let client = BigQuery::new("my-gcp-project").await.unwrap();

    let page = client
        .list_tables("my_dataset", None, Some(100))
        .await
        .unwrap();

    for tbl in &page.tables {
        println!("{}.{}", tbl.dataset_id, tbl.id);
    }

    if let Some(tok) = page.next_page_token {
        println!("more tables available, next page token: {}", tok);
    }
}
```

## Streaming row insert

```rust
use rustcloud::gcp::gcp_apis::database::gcp_bigquery::BigQuery;

#[tokio::main]
async fn main() {
    let client = BigQuery::new("my-gcp-project").await.unwrap();

    let rows = vec![
        serde_json::json!({ "user_id": 1, "email": "alice@example.com" }),
        serde_json::json!({ "user_id": 2, "email": "bob@example.com" }),
    ];

    client
        .insert_rows("my_dataset", "users", rows)
        .await
        .unwrap();
}
```

Each row is a plain JSON object whose keys match the table schema. The method assigns sequential `insertId` values automatically. If BigQuery reports any `insertErrors` in the response, the call returns an error with the first error message.

## Run a query and poll for results

```rust
use rustcloud::gcp::gcp_apis::database::gcp_bigquery::BigQuery;

#[tokio::main]
async fn main() {
    let client = BigQuery::new("my-gcp-project").await.unwrap();

    let job_id = client
        .run_query("SELECT user_id, email FROM my_dataset.users LIMIT 10")
        .await
        .unwrap();

    let result = client.get_query_results(&job_id).await.unwrap();

    println!("total rows: {:?}", result.total_rows);
    for row in &result.rows {
        // Each row has shape {"f": [{"v": "..."}]}; fields appear in schema order.
        if let Some(fields) = row["f"].as_array() {
            let values: Vec<_> = fields.iter().map(|f| &f["v"]).collect();
            println!("{:?}", values);
        }
    }
}
```

`run_query` submits the job and returns the job ID immediately. `get_query_results` polls the Jobs API up to 60 seconds, checking once per second after the first attempt. If the query does not complete within that window, it returns a retryable `CloudError::Provider`.

## Delete a table

```rust
use rustcloud::gcp::gcp_apis::database::gcp_bigquery::BigQuery;

#[tokio::main]
async fn main() {
    let client = BigQuery::new("my-gcp-project").await.unwrap();

    client.delete_table("my_dataset", "users").await.unwrap();
}
```

## Delete a dataset

```rust
use rustcloud::gcp::gcp_apis::database::gcp_bigquery::BigQuery;

#[tokio::main]
async fn main() {
    let client = BigQuery::new("my-gcp-project").await.unwrap();

    client.delete_dataset("my_dataset", true).await.unwrap();
}
```

Pass `false` for the second argument to fail if the dataset is non-empty, which prevents accidental data loss.
