use futures::StreamExt;

use crate::azure::azure_apis::artificial_intelligence::azure_openai::{
    AzureOpenAiProvider,
    build_chat_request,
    drain_complete_lines,
    extract_deployment_name,
    is_reasoning_model,
    map_azure_http_error,
    map_finish_reason,
    parse_chat_response,
    parse_sse_line,
    pump_stream,
    sse_chunk_to_events,
};
use crate::errors::CloudError;
use crate::types::llm::{FinishReason, LlmRequest, LlmStreamEvent, Message, ModelRef};

fn no_creds_provider() -> AzureOpenAiProvider {
    AzureOpenAiProvider::with_http_client(
        reqwest::Client::new(),
        "https://my-resource.openai.azure.com",
        "fake-key",
        "2024-10-21",
        "",
    )
}

fn make_request(model: ModelRef, messages: Vec<Message>) -> LlmRequest {
    LlmRequest {
        model,
        messages,
        max_tokens: None,
        temperature: None,
        system_prompt: None,
    }
}

// --- azure_endpoint ---

#[test]
fn test_azure_endpoint_format() {
    let provider = no_creds_provider();
    let url = provider.azure_endpoint("gpt-4o", "chat/completions");
    assert_eq!(
        url,
        "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
    );
}

#[test]
fn test_azure_endpoint_trims_trailing_slash() {
    let provider = AzureOpenAiProvider::with_http_client(
        reqwest::Client::new(),
        "https://my-resource.openai.azure.com/",
        "fake-key",
        "2024-10-21",
        "",
    );
    let url = provider.azure_endpoint("gpt-4o", "chat/completions");
    assert_eq!(
        url,
        "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
    );
}

// --- extract_deployment_name ---

#[test]
fn test_extract_deployment_name_from_deployment() {
    let result = extract_deployment_name(&ModelRef::Deployment("gpt-4o-mini".to_string()));
    assert_eq!(result.unwrap(), "gpt-4o-mini");
}

#[test]
fn test_extract_deployment_name_provider_alias() {
    let result = extract_deployment_name(&ModelRef::Provider("gpt-4o".to_string()));
    assert_eq!(result.unwrap(), "gpt-4o");
}

#[test]
fn test_extract_deployment_name_rejects_logical() {
    let err = extract_deployment_name(&ModelRef::Logical {
        family: "gpt".to_string(),
        tier: Some("4o".to_string()),
    })
    .unwrap_err();
    assert!(
        matches!(err, CloudError::Unsupported { .. }),
        "expected Unsupported, got {:?}",
        err
    );
}

// --- build_chat_request ---

#[test]
fn test_build_chat_request_includes_system_message() {
    let mut req = make_request(
        ModelRef::Deployment("gpt-4o".to_string()),
        vec![Message { role: "user".to_string(), content: "hi".to_string() }],
    );
    req.system_prompt = Some("You are a helpful assistant.".to_string());
    let body = build_chat_request(&req, "gpt-4o");
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "You are a helpful assistant.");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][1]["content"], "hi");
}

#[test]
fn test_build_chat_request_omits_unset_params() {
    let req = make_request(
        ModelRef::Deployment("gpt-4o".to_string()),
        vec![Message { role: "user".to_string(), content: "hi".to_string() }],
    );
    let body = build_chat_request(&req, "gpt-4o");
    assert!(body["max_tokens"].is_null());
    assert!(body["temperature"].is_null());
    assert_eq!(body["messages"][0]["role"], "user");
}

#[test]
fn test_is_reasoning_model_o1_o3() {
    assert!(is_reasoning_model("o1-mini"));
    assert!(is_reasoning_model("o3"));
    assert!(is_reasoning_model("O1-PREVIEW"));
    assert!(!is_reasoning_model("gpt-4o"));
    assert!(!is_reasoning_model("gpt-35-turbo"));
}

#[test]
fn test_build_chat_request_reasoning_model_uses_max_completion_tokens() {
    let mut req = make_request(
        ModelRef::Deployment("o1-mini".to_string()),
        vec![Message { role: "user".to_string(), content: "hi".to_string() }],
    );
    req.max_tokens = Some(256);
    let body = build_chat_request(&req, "o1-mini");
    assert!(body["max_tokens"].is_null());
    assert_eq!(body["max_completion_tokens"], 256);
}

#[test]
fn test_build_chat_request_reasoning_model_omits_temperature() {
    let mut req = make_request(
        ModelRef::Deployment("o1-mini".to_string()),
        vec![Message { role: "user".to_string(), content: "hi".to_string() }],
    );
    req.temperature = Some(0.7);
    let body = build_chat_request(&req, "o1-mini");
    assert!(body["temperature"].is_null());
}

// --- map_finish_reason ---

#[test]
fn test_map_finish_reason_stop() {
    assert!(matches!(map_finish_reason("stop"), FinishReason::Stop));
}

#[test]
fn test_map_finish_reason_length() {
    assert!(matches!(map_finish_reason("length"), FinishReason::Length));
}

#[test]
fn test_map_finish_reason_tool_calls() {
    assert!(matches!(map_finish_reason("tool_calls"), FinishReason::ToolCall));
}

#[test]
fn test_map_finish_reason_other() {
    let r = map_finish_reason("content_filter");
    assert!(matches!(r, FinishReason::Other(s) if s == "content_filter"));
}

// --- parse_chat_response ---

#[test]
fn test_parse_chat_response_valid() {
    let json = serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": "Hello!" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 2 }
    });
    let resp = parse_chat_response(&json).unwrap();
    assert_eq!(resp.text, "Hello!");
    assert!(matches!(resp.finish_reason, FinishReason::Stop));
    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 2);
}

#[test]
fn test_parse_chat_response_missing_choices() {
    let json = serde_json::json!({ "choices": [] });
    let err = parse_chat_response(&json).unwrap_err();
    assert!(
        matches!(err, CloudError::Provider { http_status: 0, .. }),
        "expected parse-error Provider variant, got {:?}",
        err
    );
}

#[test]
fn test_parse_chat_response_no_usage() {
    let json = serde_json::json!({
        "choices": [{
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }]
    });
    let resp = parse_chat_response(&json).unwrap();
    assert!(resp.usage.is_none());
}

// --- map_azure_http_error ---

#[test]
fn test_map_azure_http_error_401_is_auth() {
    assert!(matches!(map_azure_http_error(401, "unauthorized"), CloudError::Auth { .. }));
}

#[test]
fn test_map_azure_http_error_403_is_auth() {
    assert!(matches!(map_azure_http_error(403, "forbidden"), CloudError::Auth { .. }));
}

#[test]
fn test_map_azure_http_error_429_is_rate_limit() {
    assert!(matches!(map_azure_http_error(429, "quota exceeded"), CloudError::RateLimit { .. }));
}

#[test]
fn test_map_azure_http_error_400_is_not_retryable() {
    assert!(matches!(
        map_azure_http_error(400, "bad request"),
        CloudError::Provider { retryable: false, .. }
    ));
}

#[test]
fn test_map_azure_http_error_500_is_retryable() {
    assert!(matches!(
        map_azure_http_error(500, "internal error"),
        CloudError::Provider { retryable: true, .. }
    ));
}

#[test]
fn test_map_azure_http_error_503_is_retryable() {
    assert!(matches!(
        map_azure_http_error(503, "unavailable"),
        CloudError::Provider { retryable: true, .. }
    ));
}

#[test]
fn test_map_azure_http_error_404_wildcard_not_retryable() {
    assert!(matches!(
        map_azure_http_error(404, "not found"),
        CloudError::Provider { retryable: false, .. }
    ));
}

// --- parse_sse_line ---

#[test]
fn test_parse_sse_line_data_prefix() {
    let json = parse_sse_line(r#"data: {"choices": []}"#);
    assert!(json.is_some());
    assert_eq!(json.unwrap()["choices"], serde_json::json!([]));
}

#[test]
fn test_parse_sse_line_done_terminator() {
    assert!(parse_sse_line("data: [DONE]").is_none());
}

#[test]
fn test_parse_sse_line_non_data() {
    assert!(parse_sse_line(": keep-alive").is_none());
}

#[test]
fn test_parse_sse_line_invalid_json() {
    assert!(parse_sse_line("data: not-json").is_none());
}

#[test]
fn test_parse_sse_line_blank_line() {
    assert!(parse_sse_line("").is_none());
}

// --- sse_chunk_to_events ---

#[test]
fn test_sse_chunk_delta_text() {
    let json = serde_json::json!({
        "choices": [{ "delta": { "content": "Hello" }, "finish_reason": null }]
    });
    let events = sse_chunk_to_events(&json);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], LlmStreamEvent::DeltaText(t) if t == "Hello"));
}

#[test]
fn test_sse_chunk_finish_reason() {
    let json = serde_json::json!({
        "choices": [{ "delta": {}, "finish_reason": "stop" }]
    });
    let events = sse_chunk_to_events(&json);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], LlmStreamEvent::Done(FinishReason::Stop)));
}

#[test]
fn test_sse_chunk_usage_only_empty_choices() {
    let json = serde_json::json!({
        "choices": [],
        "usage": { "prompt_tokens": 5, "completion_tokens": 10 }
    });
    let events = sse_chunk_to_events(&json);
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], LlmStreamEvent::Usage(u) if u.prompt_tokens == 5 && u.completion_tokens == 10));
}

#[test]
fn test_sse_chunk_finish_emits_usage_then_done() {
    let json = serde_json::json!({
        "choices": [{ "delta": {}, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": 5, "completion_tokens": 10 }
    });
    let events = sse_chunk_to_events(&json);
    assert_eq!(events.len(), 2);
    assert!(matches!(&events[0], LlmStreamEvent::Usage(_)));
    assert!(matches!(&events[1], LlmStreamEvent::Done(FinishReason::Stop)));
}

#[test]
fn test_sse_chunk_error_object() {
    let json = serde_json::json!({
        "error": { "message": "content filtered", "code": "content_filter" }
    });
    let events = sse_chunk_to_events(&json);
    assert_eq!(events.len(), 1);
    assert!(matches!(
        &events[0],
        LlmStreamEvent::Error(CloudError::Provider { http_status: 200, .. })
    ));
}

// --- drain_complete_lines ---

#[test]
fn test_drain_complete_lines_handles_multibyte_utf8_split_across_calls() {
    let line = "data: {\"choices\":[{\"delta\":{\"content\":\"caf\u{e9}\"},\"finish_reason\":null}]}\n";
    let bytes = line.as_bytes();
    // split mid two-byte UTF-8 char
    let split_at = bytes.iter().position(|&b| b == 0xC3).unwrap() + 1;

    let mut buffer: Vec<u8> = Vec::new();
    buffer.extend_from_slice(&bytes[..split_at]);
    assert!(drain_complete_lines(&mut buffer).is_empty(), "no complete line yet");

    buffer.extend_from_slice(&bytes[split_at..]);
    let parsed = drain_complete_lines(&mut buffer);
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0]["choices"][0]["delta"]["content"], "café");
}

// --- pump_stream edge cases (no live credentials) ---

#[tokio::test]
async fn test_stream_connection_drop() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        // unread request bytes on close cause an RST, not a truncated body
        let mut discard = [0u8; 1024];
        let _ = socket.read(&mut discard).await;
        let _ = socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort")
            .await;
    });

    let response = reqwest::Client::new()
        .get(format!("http://{}/", addr))
        .send()
        .await
        .unwrap();

    let (tx, mut rx) = futures::channel::mpsc::channel::<LlmStreamEvent>(32);
    pump_stream(response, tx).await;

    match rx.next().await {
        Some(LlmStreamEvent::Error(CloudError::Network { .. })) => {}
        other => panic!("expected a Network error event, got {:?}", other),
    }
    assert!(rx.next().await.is_none());
}

#[tokio::test]
async fn test_stream_flushes_trailing_line_without_terminator() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // no trailing newline before the clean close
    let body = r#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#;
    let body_len = body.len();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut discard = [0u8; 1024];
        let _ = socket.read(&mut discard).await;
        let response = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}", body_len, body);
        let _ = socket.write_all(response.as_bytes()).await;
    });

    let response = reqwest::Client::new()
        .get(format!("http://{}/", addr))
        .send()
        .await
        .unwrap();

    let (tx, mut rx) = futures::channel::mpsc::channel::<LlmStreamEvent>(32);
    pump_stream(response, tx).await;

    match rx.next().await {
        Some(LlmStreamEvent::DeltaText(t)) => assert_eq!(t, "hi"),
        other => panic!("expected DeltaText(\"hi\"), got {:?}", other),
    }
}

#[tokio::test]
async fn test_stream_orders_usage_before_done_across_separate_chunks() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // real Azure/OpenAI wire behavior: finish_reason and usage arrive as two
    // separate chunks, not combined in one JSON payload
    let finish_line = r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
    let usage_line = r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":10}}"#;
    let body = format!("{}\n{}\n", finish_line, usage_line);
    let body_len = body.len();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut discard = [0u8; 1024];
        let _ = socket.read(&mut discard).await;
        let response = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}", body_len, body);
        let _ = socket.write_all(response.as_bytes()).await;
    });

    let response = reqwest::Client::new()
        .get(format!("http://{}/", addr))
        .send()
        .await
        .unwrap();

    let (tx, mut rx) = futures::channel::mpsc::channel::<LlmStreamEvent>(32);
    pump_stream(response, tx).await;

    let mut events = Vec::new();
    while let Some(event) = rx.next().await {
        events.push(event);
    }

    assert_eq!(events.len(), 2);
    assert!(matches!(&events[0], LlmStreamEvent::Usage(u) if u.prompt_tokens == 5 && u.completion_tokens == 10));
    assert!(matches!(&events[1], LlmStreamEvent::Done(FinishReason::Stop)));
}

// --- integration tests (require live Azure OpenAI credentials, run with --ignored) ---

#[tokio::test]
#[ignore]
async fn test_generate_live() {
    let provider = AzureOpenAiProvider::new().expect("failed to create provider");
    let req = LlmRequest {
        model: ModelRef::Deployment("gpt-4o".to_string()),
        messages: vec![Message { role: "user".to_string(), content: "What is 2 + 2?".to_string() }],
        max_tokens: Some(64),
        temperature: Some(0.0),
        system_prompt: None,
    };
    let resp = provider.generate(req).await.expect("generate failed");
    assert!(!resp.text.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_stream_live() {
    let provider = AzureOpenAiProvider::new().expect("failed to create provider");
    let req = LlmRequest {
        model: ModelRef::Deployment("gpt-4o".to_string()),
        messages: vec![Message { role: "user".to_string(), content: "Count from 1 to 5.".to_string() }],
        max_tokens: Some(64),
        temperature: Some(0.0),
        system_prompt: None,
    };
    let mut stream = provider.stream(req).await.expect("stream failed");
    let mut got_text = false;
    while let Some(event) = stream.next().await {
        match event {
            LlmStreamEvent::DeltaText(_) => got_text = true,
            LlmStreamEvent::Error(e) => panic!("stream error: {:?}", e),
            _ => {}
        }
    }
    assert!(got_text);
}
