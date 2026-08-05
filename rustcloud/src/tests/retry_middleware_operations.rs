use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::{stream, StreamExt};

use crate::errors::CloudError;
use crate::genai::client::UnifiedLlmClient;
use crate::genai::retry::RetryMiddleware;
use crate::genai::routing::RoutingStrategy;
use crate::traits::llm_provider::{LlmProvider, LlmStream};
use crate::types::llm::{
    EmbedResponse, FinishReason, LlmRequest, LlmResponse, LlmStreamEvent, Message, ModelRef,
    ToolCallResponse, ToolDefinition,
};

#[derive(Debug, Clone, Copy)]
enum MockBehavior {
    Ok,
    RateLimit(Option<u64>),
    Auth,
    RetryableProvider(u16),
    NonRetryableProvider,
    Unsupported(&'static str),
    Network,
    StreamErrorMidway,
}

fn mock_network_error() -> reqwest::Error {
    reqwest::Client::new().get("not a url").build().unwrap_err()
}

struct SequencedMock {
    behaviors: Mutex<VecDeque<MockBehavior>>,
    calls: Arc<AtomicUsize>,
}

impl SequencedMock {
    fn new(behaviors: Vec<MockBehavior>, calls: Arc<AtomicUsize>) -> Self {
        Self {
            behaviors: Mutex::new(behaviors.into()),
            calls,
        }
    }

    fn pop_behavior(&self) -> MockBehavior {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.behaviors
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(MockBehavior::Ok)
    }

    fn failure(behavior: MockBehavior) -> Option<CloudError> {
        match behavior {
            MockBehavior::Ok | MockBehavior::StreamErrorMidway => None,
            MockBehavior::RateLimit(retry_after) => Some(CloudError::RateLimit { retry_after }),
            MockBehavior::Auth => Some(CloudError::Auth {
                message: "mock auth failure".to_string(),
            }),
            MockBehavior::RetryableProvider(http_status) => Some(CloudError::Provider {
                http_status,
                message: "mock retryable failure".to_string(),
                retryable: true,
            }),
            MockBehavior::NonRetryableProvider => Some(CloudError::Provider {
                http_status: 400,
                message: "mock non-retryable failure".to_string(),
                retryable: false,
            }),
            MockBehavior::Unsupported(feature) => Some(CloudError::Unsupported { feature }),
            MockBehavior::Network => Some(CloudError::Network {
                source: mock_network_error(),
            }),
        }
    }

    fn next_result<T>(&self, ok: T) -> Result<T, CloudError> {
        match Self::failure(self.pop_behavior()) {
            Some(err) => Err(err),
            None => Ok(ok),
        }
    }
}

#[async_trait]
impl LlmProvider for SequencedMock {
    async fn generate(&self, _req: LlmRequest) -> Result<LlmResponse, CloudError> {
        self.next_result(LlmResponse {
            text: "mock response".to_string(),
            finish_reason: FinishReason::Stop,
            usage: None,
        })
    }

    async fn stream(&self, _req: LlmRequest) -> Result<LlmStream, CloudError> {
        let behavior = self.pop_behavior();
        if let Some(err) = Self::failure(behavior) {
            return Err(err);
        }
        let event = match behavior {
            MockBehavior::StreamErrorMidway => {
                LlmStreamEvent::Error(CloudError::RateLimit { retry_after: None })
            }
            _ => LlmStreamEvent::Done(FinishReason::Stop),
        };
        Ok(Box::pin(stream::once(async move { event })))
    }

    async fn embed(&self, _texts: Vec<String>) -> Result<EmbedResponse, CloudError> {
        self.next_result(EmbedResponse {
            embeddings: vec![vec![0.0]],
        })
    }

    async fn generate_with_tools(
        &self,
        _req: LlmRequest,
        _tools: Vec<ToolDefinition>,
    ) -> Result<ToolCallResponse, CloudError> {
        self.next_result(ToolCallResponse::Text(LlmResponse {
            text: "mock response".to_string(),
            finish_reason: FinishReason::Stop,
            usage: None,
        }))
    }
}

fn make_request() -> LlmRequest {
    LlmRequest {
        model: ModelRef::Provider("test-model".to_string()),
        messages: vec![Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        }],
        max_tokens: None,
        temperature: None,
        system_prompt: None,
    }
}

#[test]
fn test_retry_delay_exponential_growth() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![], calls);
    let middleware = RetryMiddleware::with_config(mock, 3, 100, false);

    assert_eq!(middleware.retry_delay(0), Duration::from_millis(100));
    assert_eq!(middleware.retry_delay(1), Duration::from_millis(200));
    assert_eq!(middleware.retry_delay(2), Duration::from_millis(400));
}

#[test]
fn test_retry_delay_jitter_within_bounds() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![], calls);
    let middleware = RetryMiddleware::with_config(mock, 3, 100, true);

    for attempt in 0..5 {
        let base = 100u64 * 2u64.pow(attempt);
        let delay = middleware.retry_delay(attempt).as_millis() as u64;
        assert!(
            delay >= base && delay < base * 2,
            "attempt {attempt}: got {delay}, expected [{base}, {})",
            base * 2
        );
    }
}

#[test]
fn test_retry_delay_saturates_on_large_attempt() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![], calls);
    let middleware = RetryMiddleware::with_config(mock, 3, 100, false);

    assert_eq!(middleware.retry_delay(100), Duration::from_millis(u64::MAX));
}

#[test]
fn test_delay_for_honors_retry_after() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![], calls);
    let middleware = RetryMiddleware::with_config(mock, 3, 100, false);

    let err = CloudError::RateLimit {
        retry_after: Some(5),
    };
    assert_eq!(middleware.delay_for(&err, 0), Duration::from_secs(5));

    let err = CloudError::RateLimit { retry_after: None };
    assert_eq!(middleware.delay_for(&err, 1), Duration::from_millis(200));
}

#[test]
fn test_delay_for_ignores_jitter_for_retry_after() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![], calls);
    let middleware = RetryMiddleware::with_config(mock, 3, 100, true);

    let err = CloudError::RateLimit {
        retry_after: Some(5),
    };
    assert_eq!(middleware.delay_for(&err, 0), Duration::from_secs(5));
    assert_eq!(middleware.delay_for(&err, 3), Duration::from_secs(5));
}

#[tokio::test(start_paused = true)]
async fn test_generate_retries_transient_then_succeeds() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::RetryableProvider(503)], calls.clone());
    let middleware = RetryMiddleware::wrap(mock);

    let result = middleware.generate(make_request()).await;

    assert!(result.is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn test_retry_auth_error_no_retry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::Auth], calls.clone());
    let middleware = RetryMiddleware::wrap(mock);

    let result = middleware.generate(make_request()).await;

    assert!(matches!(result, Err(CloudError::Auth { .. })));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn test_generate_non_retryable_provider_no_retry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::NonRetryableProvider], calls.clone());
    let middleware = RetryMiddleware::wrap(mock);

    let result = middleware.generate(make_request()).await;

    assert!(matches!(
        result,
        Err(CloudError::Provider {
            retryable: false,
            ..
        })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn test_generate_exhausts_retries_returns_last_error() {
    let calls = Arc::new(AtomicUsize::new(0));
    let behaviors = vec![
        MockBehavior::RetryableProvider(502),
        MockBehavior::RetryableProvider(503),
        MockBehavior::RetryableProvider(504),
    ];
    let mock = SequencedMock::new(behaviors, calls.clone());
    let middleware = RetryMiddleware::with_config(mock, 2, 10, false);

    let result = middleware.generate(make_request()).await;

    let Err(CloudError::Provider {
        http_status,
        retryable,
        ..
    }) = result
    else {
        panic!("expected a Provider error");
    };
    assert!(retryable);
    assert_eq!(http_status, 504);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test(start_paused = true)]
async fn test_generate_zero_retries_single_attempt() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::RetryableProvider(503)], calls.clone());
    let middleware = RetryMiddleware::with_config(mock, 0, 10, false);

    let result = middleware.generate(make_request()).await;

    assert!(result.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn test_generate_rate_limit_waits_server_hint() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::RateLimit(Some(2))], calls.clone());
    let middleware = RetryMiddleware::wrap(mock);

    let start = tokio::time::Instant::now();
    let result = middleware.generate(make_request()).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    assert!(elapsed >= Duration::from_secs(2));
    assert!(elapsed < Duration::from_secs(3));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn test_generate_retries_network_error() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::Network], calls.clone());
    let middleware = RetryMiddleware::wrap(mock);

    let result = middleware.generate(make_request()).await;

    assert!(result.is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn test_stream_retries_initial_transient_error() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::RetryableProvider(503)], calls.clone());
    let middleware = RetryMiddleware::wrap(mock);

    let result = middleware.stream(make_request()).await;

    assert!(result.is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn test_stream_auth_error_no_retry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::Auth], calls.clone());
    let middleware = RetryMiddleware::wrap(mock);

    let result = middleware.stream(make_request()).await;

    assert!(matches!(result, Err(CloudError::Auth { .. })));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn test_stream_ok_passes_events_through_unchanged() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![], calls);
    let middleware = RetryMiddleware::wrap(mock);

    let mut stream = middleware.stream(make_request()).await.unwrap();
    let event = stream.next().await.expect("stream should yield one event");

    assert!(matches!(event, LlmStreamEvent::Done(FinishReason::Stop)));
    assert!(stream.next().await.is_none());
}

#[tokio::test(start_paused = true)]
async fn test_stream_ignores_mid_stream_error() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::StreamErrorMidway], calls.clone());
    let middleware = RetryMiddleware::wrap(mock);

    let mut stream = middleware.stream(make_request()).await.unwrap();
    let event = stream.next().await.expect("stream should yield one event");

    assert!(matches!(event, LlmStreamEvent::Error(_)));
    // the initial .stream() call succeeded, so no retry against a mid-stream error
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn test_stream_zero_retries_single_attempt() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::RetryableProvider(503)], calls.clone());
    let middleware = RetryMiddleware::with_config(mock, 0, 10, false);

    let result = middleware.stream(make_request()).await;

    assert!(result.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn test_stream_exhausts_retries() {
    let calls = Arc::new(AtomicUsize::new(0));
    let behaviors = vec![
        MockBehavior::RetryableProvider(502),
        MockBehavior::RetryableProvider(503),
    ];
    let mock = SequencedMock::new(behaviors, calls.clone());
    let middleware = RetryMiddleware::with_config(mock, 1, 10, false);

    let result = middleware.stream(make_request()).await;

    let Err(CloudError::Provider {
        http_status,
        retryable,
        ..
    }) = result
    else {
        panic!("expected a Provider error");
    };
    assert!(retryable);
    assert_eq!(http_status, 503);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn test_embed_retries_transient_then_succeeds() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::RetryableProvider(503)], calls.clone());
    let middleware = RetryMiddleware::wrap(mock);

    let result = middleware.embed(vec!["hello".to_string()]).await;

    assert!(result.is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn test_embed_exhausts_retries() {
    let calls = Arc::new(AtomicUsize::new(0));
    let behaviors = vec![
        MockBehavior::RetryableProvider(502),
        MockBehavior::RetryableProvider(503),
    ];
    let mock = SequencedMock::new(behaviors, calls.clone());
    let middleware = RetryMiddleware::with_config(mock, 1, 10, false);

    let result = middleware.embed(vec!["hello".to_string()]).await;

    let Err(CloudError::Provider {
        http_status,
        retryable,
        ..
    }) = result
    else {
        panic!("expected a Provider error");
    };
    assert!(retryable);
    assert_eq!(http_status, 503);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn test_embed_zero_retries_single_attempt() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::RetryableProvider(503)], calls.clone());
    let middleware = RetryMiddleware::with_config(mock, 0, 10, false);

    let result = middleware.embed(vec!["hello".to_string()]).await;

    assert!(result.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn test_tools_retries_transient_then_succeeds() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::RetryableProvider(503)], calls.clone());
    let middleware = RetryMiddleware::wrap(mock);

    let result = middleware.generate_with_tools(make_request(), vec![]).await;

    assert!(result.is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn test_tools_unsupported_no_retry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(
        vec![MockBehavior::Unsupported(
            "generate_with_tools requires at least one tool",
        )],
        calls.clone(),
    );
    let middleware = RetryMiddleware::wrap(mock);

    let result = middleware.generate_with_tools(make_request(), vec![]).await;

    assert!(matches!(result, Err(CloudError::Unsupported { .. })));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn test_tools_zero_retries_single_attempt() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::RetryableProvider(503)], calls.clone());
    let middleware = RetryMiddleware::with_config(mock, 0, 10, false);

    let result = middleware.generate_with_tools(make_request(), vec![]).await;

    assert!(result.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn test_tools_exhausts_retries() {
    let calls = Arc::new(AtomicUsize::new(0));
    let behaviors = vec![
        MockBehavior::RetryableProvider(502),
        MockBehavior::RetryableProvider(503),
    ];
    let mock = SequencedMock::new(behaviors, calls.clone());
    let middleware = RetryMiddleware::with_config(mock, 1, 10, false);

    let result = middleware.generate_with_tools(make_request(), vec![]).await;

    let Err(CloudError::Provider {
        http_status,
        retryable,
        ..
    }) = result
    else {
        panic!("expected a Provider error");
    };
    assert!(retryable);
    assert_eq!(http_status, 503);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn test_retry_wraps_unified_client() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mock = SequencedMock::new(vec![MockBehavior::RetryableProvider(503)], calls.clone());
    let client = UnifiedLlmClient::builder()
        .register("only", mock)
        .routing(RoutingStrategy::Explicit)
        .build()
        .unwrap();
    let middleware = RetryMiddleware::wrap(client);

    let result = middleware.generate(make_request()).await;

    assert!(result.is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test(start_paused = true)]
async fn test_fallback_client_retries_each_provider_before_moving_on() {
    let calls_a = Arc::new(AtomicUsize::new(0));
    let calls_b = Arc::new(AtomicUsize::new(0));

    let mock_a = SequencedMock::new(
        vec![
            MockBehavior::RetryableProvider(503),
            MockBehavior::RetryableProvider(503),
        ],
        calls_a.clone(),
    );
    let mock_b = SequencedMock::new(vec![], calls_b.clone());

    let provider_a = RetryMiddleware::with_config(mock_a, 1, 10, false);
    let provider_b = RetryMiddleware::with_config(mock_b, 1, 10, false);

    let client = UnifiedLlmClient::builder()
        .register("a", provider_a)
        .register("b", provider_b)
        .routing(RoutingStrategy::Fallback)
        .build()
        .unwrap();

    let result = client.generate(make_request()).await;

    assert!(result.is_ok());
    // a's own retry budget (1 retry = 2 attempts) is exhausted before fallback tries b
    assert_eq!(calls_a.load(Ordering::SeqCst), 2);
    assert_eq!(calls_b.load(Ordering::SeqCst), 1);
}

#[tokio::test]
#[ignore]
async fn test_retry_wrapped_bedrock_live() {
    use crate::aws::aws_apis::artificial_intelligence::aws_bedrock::BedrockProvider;

    let middleware = RetryMiddleware::wrap(BedrockProvider::new().await);
    let req = LlmRequest {
        model: ModelRef::Provider("anthropic.claude-3-5-haiku-20241022-v1:0".to_string()),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Reply with the single word OK and nothing else.".to_string(),
        }],
        max_tokens: Some(10),
        temperature: Some(0.0),
        system_prompt: None,
    };

    let result = middleware.generate(req).await.unwrap();

    assert!(!result.text.is_empty());
}
