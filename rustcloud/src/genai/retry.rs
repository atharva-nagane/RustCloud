use std::time::Duration;

use async_trait::async_trait;
use rand::random;
use tokio::time::sleep;

use crate::errors::CloudError;
use crate::genai::routing::is_transient;
use crate::traits::llm_provider::{LlmProvider, LlmStream};
use crate::types::llm::{EmbedResponse, LlmRequest, LlmResponse, ToolCallResponse, ToolDefinition};

pub struct RetryMiddleware<P: LlmProvider> {
    inner: P,
    max_retries: u32,
    base_delay_ms: u64,
    jitter: bool,
}

impl<P: LlmProvider> RetryMiddleware<P> {
    pub fn wrap(provider: P) -> Self {
        Self::with_config(provider, 3, 100, true)
    }

    pub fn with_config(provider: P, max_retries: u32, base_delay_ms: u64, jitter: bool) -> Self {
        Self {
            inner: provider,
            max_retries,
            base_delay_ms,
            jitter,
        }
    }

    pub(crate) fn retry_delay(&self, attempt: u32) -> Duration {
        let delay = self
            .base_delay_ms
            .saturating_mul(2u64.saturating_pow(attempt));
        let delay = if self.jitter && delay > 0 {
            delay.saturating_add(random::<u64>() % delay)
        } else {
            delay
        };
        Duration::from_millis(delay)
    }

    pub(crate) fn delay_for(&self, error: &CloudError, attempt: u32) -> Duration {
        match error {
            CloudError::RateLimit {
                retry_after: Some(secs),
            } => Duration::from_secs(*secs),
            _ => self.retry_delay(attempt),
        }
    }
}

#[async_trait]
impl<P: LlmProvider> LlmProvider for RetryMiddleware<P> {
    async fn generate(&self, req: LlmRequest) -> Result<LlmResponse, CloudError> {
        let mut attempt = 0;
        loop {
            match self.inner.generate(req.clone()).await {
                Ok(response) => return Ok(response),
                Err(err) if attempt < self.max_retries && is_transient(&err) => {
                    sleep(self.delay_for(&err, attempt)).await;
                    attempt += 1;
                }
                Err(err) => return Err(err),
            }
        }
    }

    // retries only the initial call — once a stream is handed back it's already partially
    // consumed, so a mid-stream error surfaces as an event instead of triggering a hidden replay
    async fn stream(&self, req: LlmRequest) -> Result<LlmStream, CloudError> {
        let mut attempt = 0;
        loop {
            match self.inner.stream(req.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(err) if attempt < self.max_retries && is_transient(&err) => {
                    sleep(self.delay_for(&err, attempt)).await;
                    attempt += 1;
                }
                Err(err) => return Err(err),
            }
        }
    }

    async fn embed(&self, texts: Vec<String>) -> Result<EmbedResponse, CloudError> {
        let mut attempt = 0;
        loop {
            match self.inner.embed(texts.clone()).await {
                Ok(response) => return Ok(response),
                Err(err) if attempt < self.max_retries && is_transient(&err) => {
                    sleep(self.delay_for(&err, attempt)).await;
                    attempt += 1;
                }
                Err(err) => return Err(err),
            }
        }
    }

    async fn generate_with_tools(
        &self,
        req: LlmRequest,
        tools: Vec<ToolDefinition>,
    ) -> Result<ToolCallResponse, CloudError> {
        let mut attempt = 0;
        loop {
            match self
                .inner
                .generate_with_tools(req.clone(), tools.clone())
                .await
            {
                Ok(response) => return Ok(response),
                Err(err) if attempt < self.max_retries && is_transient(&err) => {
                    sleep(self.delay_for(&err, attempt)).await;
                    attempt += 1;
                }
                Err(err) => return Err(err),
            }
        }
    }
}
