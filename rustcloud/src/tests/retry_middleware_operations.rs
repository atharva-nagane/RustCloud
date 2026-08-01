use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;

use crate::errors::CloudError;
use crate::genai::retry::RetryMiddleware;
use crate::traits::llm_provider::{LlmProvider, LlmStream};
use crate::types::llm::{
    EmbedResponse, FinishReason, LlmRequest, LlmResponse, Message, ModelRef, ToolCallResponse,
    ToolDefinition,
};

enum MockBehavior {
    Ok,
    RateLimit(Option<u64>),
    Auth,
    RetryableProvider(u16),
    NonRetryableProvider,
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

    fn next_result<T>(&self, ok: T) -> Result<T, CloudError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let behavior = self
            .behaviors
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(MockBehavior::Ok);
        match behavior {
            MockBehavior::Ok => Ok(ok),
            MockBehavior::RateLimit(retry_after) => Err(CloudError::RateLimit { retry_after }),
            MockBehavior::Auth => Err(CloudError::Auth {
                message: "mock auth failure".to_string(),
            }),
            MockBehavior::RetryableProvider(http_status) => Err(CloudError::Provider {
                http_status,
                message: "mock retryable failure".to_string(),
                retryable: true,
            }),
            MockBehavior::NonRetryableProvider => Err(CloudError::Provider {
                http_status: 400,
                message: "mock non-retryable failure".to_string(),
                retryable: false,
            }),
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
        unimplemented!()
    }

    async fn embed(&self, _texts: Vec<String>) -> Result<EmbedResponse, CloudError> {
        unimplemented!()
    }

    async fn generate_with_tools(
        &self,
        _req: LlmRequest,
        _tools: Vec<ToolDefinition>,
    ) -> Result<ToolCallResponse, CloudError> {
        unimplemented!()
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
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
