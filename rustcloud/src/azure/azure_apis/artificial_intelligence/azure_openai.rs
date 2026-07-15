use futures::channel::mpsc;
use futures::SinkExt;

use crate::errors::CloudError;
use crate::traits::llm_provider::LlmStream;
use crate::types::llm::{
    FinishReason, LlmRequest, LlmResponse, LlmStreamEvent, ModelRef, UsageStats,
};

pub struct AzureOpenAiProvider {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    api_version: String,
    embed_deployment: String,
}

impl AzureOpenAiProvider {
    pub fn new() -> Result<Self, CloudError> {
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").map_err(|_| CloudError::Auth {
            message: "AZURE_OPENAI_ENDPOINT not set".to_string(),
        })?;
        let api_key = std::env::var("AZURE_OPENAI_API_KEY").map_err(|_| CloudError::Auth {
            message: "AZURE_OPENAI_API_KEY not set".to_string(),
        })?;
        let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
            .unwrap_or_else(|_| "2024-10-21".to_owned());
        let embed_deployment = std::env::var("AZURE_OPENAI_EMBED_DEPLOYMENT").unwrap_or_default();

        Ok(Self {
            client: reqwest::Client::new(),
            endpoint,
            api_key,
            api_version,
            embed_deployment,
        })
    }

    pub fn with_http_client(
        client: reqwest::Client,
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        api_version: impl Into<String>,
        embed_deployment: impl Into<String>,
    ) -> Self {
        Self {
            client,
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            api_version: api_version.into(),
            embed_deployment: embed_deployment.into(),
        }
    }

    pub(crate) fn azure_endpoint(&self, deployment: &str, method: &str) -> String {
        format!(
            "{}/openai/deployments/{}/{}?api-version={}",
            self.endpoint.trim_end_matches('/'), deployment, method, self.api_version
        )
    }

    pub(crate) fn request(&self, url: &str) -> reqwest::RequestBuilder {
        self.client.post(url).header("api-key", self.api_key.as_str())
    }

    pub async fn generate(&self, req: LlmRequest) -> Result<LlmResponse, CloudError> {
        let deployment = extract_deployment_name(&req.model)?;
        let url = self.azure_endpoint(&deployment, "chat/completions");
        let body = build_chat_request(&req, &deployment);

        let response = self
            .request(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let status = response.status().as_u16();
        if status >= 400 {
            let text = response.text().await.unwrap_or_default();
            return Err(map_azure_http_error(status, &text));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let resp_json: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| CloudError::Serialization { source: e })?;

        parse_chat_response(&resp_json)
    }

    pub async fn stream(&self, req: LlmRequest) -> Result<LlmStream, CloudError> {
        let deployment = extract_deployment_name(&req.model)?;
        let url = self.azure_endpoint(&deployment, "chat/completions");
        let mut body = build_chat_request(&req, &deployment);
        body["stream"] = serde_json::json!(true);
        body["stream_options"] = serde_json::json!({ "include_usage": true });

        let response = self
            .request(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| CloudError::Network { source: e })?;

        let status = response.status().as_u16();
        if status >= 400 {
            let text = response.text().await.unwrap_or_default();
            return Err(map_azure_http_error(status, &text));
        }

        let (tx, rx) = mpsc::channel::<LlmStreamEvent>(32);
        tokio::spawn(pump_stream(response, tx));

        Ok(Box::pin(rx))
    }
}

pub(crate) fn extract_deployment_name(model: &ModelRef) -> Result<String, CloudError> {
    match model {
        ModelRef::Deployment(name) => Ok(name.clone()),
        ModelRef::Provider(id) => Ok(id.clone()),
        ModelRef::Logical { .. } => Err(CloudError::Unsupported {
            feature: "Azure OpenAI model resolution from ModelRef::Logical",
        }),
    }
}

pub(crate) fn is_reasoning_model(deployment: &str) -> bool {
    let d = deployment.to_ascii_lowercase();
    d.starts_with("o1") || d.starts_with("o3")
}

pub(crate) fn build_chat_request(req: &LlmRequest, deployment: &str) -> serde_json::Value {
    let mut messages = Vec::with_capacity(req.messages.len() + 1);
    if let Some(system) = &req.system_prompt {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    for msg in &req.messages {
        messages.push(serde_json::json!({ "role": msg.role, "content": msg.content }));
    }

    let mut body = serde_json::json!({ "messages": messages });
    let reasoning = is_reasoning_model(deployment);

    if let Some(max_tokens) = req.max_tokens {
        let key = if reasoning { "max_completion_tokens" } else { "max_tokens" };
        body[key] = serde_json::json!(max_tokens);
    }
    if !reasoning {
        if let Some(temperature) = req.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
    }

    body
}

pub(crate) fn map_azure_http_error(status: u16, body: &str) -> CloudError {
    match status {
        401 | 403 => CloudError::Auth { message: body.to_string() },
        429 => CloudError::RateLimit { retry_after: None },
        400 => CloudError::Provider {
            http_status: 400,
            message: body.to_string(),
            retryable: false,
        },
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

pub(crate) fn map_finish_reason(s: &str) -> FinishReason {
    match s {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCall,
        other => FinishReason::Other(other.to_string()),
    }
}

pub(crate) fn parse_chat_response(json: &serde_json::Value) -> Result<LlmResponse, CloudError> {
    let choice = json["choices"].get(0).ok_or_else(|| CloudError::Provider {
        http_status: 0,
        message: "parse error: response contained no choices".to_string(),
        retryable: false,
    })?;

    let text = choice["message"]["content"].as_str().unwrap_or("").to_string();

    let finish_reason = choice["finish_reason"]
        .as_str()
        .map(map_finish_reason)
        .unwrap_or(FinishReason::Other("unknown".to_string()));

    let usage = match (
        json["usage"]["prompt_tokens"].as_u64(),
        json["usage"]["completion_tokens"].as_u64(),
    ) {
        (Some(p), Some(c)) => Some(UsageStats {
            prompt_tokens: p as u32,
            completion_tokens: c as u32,
        }),
        _ => None,
    };

    Ok(LlmResponse { text, finish_reason, usage })
}

pub(crate) fn parse_sse_line(line: &str) -> Option<serde_json::Value> {
    let data = line.strip_prefix("data: ")?;
    if data == "[DONE]" {
        return None;
    }
    serde_json::from_str(data).ok()
}

pub(crate) fn drain_complete_lines(buffer: &mut Vec<u8>) -> Vec<serde_json::Value> {
    let mut parsed = Vec::new();
    while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
        let line_bytes: Vec<u8> = buffer.drain(..=pos).collect();
        let line = String::from_utf8_lossy(&line_bytes);
        let line = line.trim_end_matches('\n').trim_end_matches('\r');
        if let Some(json) = parse_sse_line(line) {
            parsed.push(json);
        }
    }
    parsed
}

pub(crate) fn sse_chunk_to_events(json: &serde_json::Value) -> Vec<LlmStreamEvent> {
    let mut events = Vec::new();

    if let Some(message) = json["error"]["message"].as_str() {
        events.push(LlmStreamEvent::Error(CloudError::Provider {
            http_status: 200,
            message: message.to_string(),
            retryable: false,
        }));
        return events;
    }

    let choice = json["choices"].get(0);

    if let Some(text) = choice.and_then(|c| c["delta"]["content"].as_str()) {
        if !text.is_empty() {
            events.push(LlmStreamEvent::DeltaText(text.to_string()));
        }
    }

    if let (Some(p), Some(c)) = (
        json["usage"]["prompt_tokens"].as_u64(),
        json["usage"]["completion_tokens"].as_u64(),
    ) {
        events.push(LlmStreamEvent::Usage(UsageStats {
            prompt_tokens: p as u32,
            completion_tokens: c as u32,
        }));
    }

    if let Some(reason) = choice.and_then(|c| c["finish_reason"].as_str()) {
        events.push(LlmStreamEvent::Done(map_finish_reason(reason)));
    }

    events
}

async fn emit_chunk_events(
    events: Vec<LlmStreamEvent>,
    pending_done: &mut Option<LlmStreamEvent>,
    tx: &mut mpsc::Sender<LlmStreamEvent>,
) -> bool {
    let has_usage = events.iter().any(|e| matches!(e, LlmStreamEvent::Usage(_)));

    if !has_usage {
        if let Some(done) = pending_done.take() {
            if tx.send(done).await.is_err() {
                return false;
            }
        }
    }

    for event in events {
        match event {
            LlmStreamEvent::Done(_) => *pending_done = Some(event),
            other => {
                if tx.send(other).await.is_err() {
                    return false;
                }
            }
        }
    }

    if has_usage {
        if let Some(done) = pending_done.take() {
            if tx.send(done).await.is_err() {
                return false;
            }
        }
    }

    true
}

pub(crate) async fn pump_stream(mut response: reqwest::Response, mut tx: mpsc::Sender<LlmStreamEvent>) {
    let mut buffer: Vec<u8> = Vec::new();
    let mut pending_done: Option<LlmStreamEvent> = None;

    loop {
        match response.chunk().await {
            Ok(Some(bytes)) => {
                buffer.extend_from_slice(&bytes);
                for json in drain_complete_lines(&mut buffer) {
                    let events = sse_chunk_to_events(&json);
                    if !emit_chunk_events(events, &mut pending_done, &mut tx).await {
                        return;
                    }
                }
            }
            Ok(None) => {
                if !buffer.is_empty() {
                    buffer.push(b'\n');
                    for json in drain_complete_lines(&mut buffer) {
                        let events = sse_chunk_to_events(&json);
                        if !emit_chunk_events(events, &mut pending_done, &mut tx).await {
                            return;
                        }
                    }
                }
                if let Some(done) = pending_done.take() {
                    let _ = tx.send(done).await;
                }
                break;
            }
            Err(e) => {
                let _ = tx.send(LlmStreamEvent::Error(CloudError::Network { source: e })).await;
                break;
            }
        }
    }
}
