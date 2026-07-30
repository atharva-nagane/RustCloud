use std::collections::HashMap;
use std::future::Future;

use async_trait::async_trait;

use crate::errors::CloudError;
use crate::genai::routing::{is_transient, route_key_for_model, RoutingStrategy};
use crate::traits::llm_provider::{LlmProvider, LlmStream};
use crate::types::llm::{
    EmbedResponse, LlmRequest, LlmResponse, ModelRef, ToolCallResponse, ToolDefinition,
};

pub struct UnifiedLlmClient {
    providers: Vec<(String, Box<dyn LlmProvider>)>,
    index: HashMap<String, usize>,
    default_provider: String,
    routing: RoutingStrategy,
}

impl UnifiedLlmClient {
    pub fn builder() -> UnifiedLlmClientBuilder {
        UnifiedLlmClientBuilder::new()
    }

    pub(crate) fn provider_by_key(&self, key: &str) -> Result<&dyn LlmProvider, CloudError> {
        let index = *self.index.get(key).ok_or_else(|| CloudError::Provider {
            http_status: 0,
            message: format!("no provider registered under '{}'", key),
            retryable: false,
        })?;
        Ok(self.providers[index].1.as_ref())
    }

    pub(crate) fn resolve(&self, model: &ModelRef) -> Result<&dyn LlmProvider, CloudError> {
        match self.routing {
            RoutingStrategy::ModelBased => match route_key_for_model(model) {
                Some(key) => self.provider_by_key(key),
                None => self.provider_by_key(&self.default_provider),
            },
            RoutingStrategy::Explicit => self.provider_by_key(&self.default_provider),
            RoutingStrategy::Fallback => Err(CloudError::Provider {
                http_status: 0,
                message: "fallback routing does not resolve to a single provider; call fallback_with instead"
                    .to_string(),
                retryable: false,
            }),
        }
    }

    pub(crate) async fn fallback_with<'a, T, F, Fut>(&'a self, call: F) -> Result<T, CloudError>
    where
        F: Fn(&'a dyn LlmProvider) -> Fut,
        Fut: Future<Output = Result<T, CloudError>>,
    {
        let mut last_error = None;
        for (_, provider) in &self.providers {
            match call(provider.as_ref()).await {
                Ok(value) => return Ok(value),
                Err(err) if is_transient(&err) => last_error = Some(err),
                Err(err) => return Err(err),
            }
        }
        Err(last_error.expect("build() guarantees at least one registered provider"))
    }

    pub(crate) async fn fallback_generate(&self, req: LlmRequest) -> Result<LlmResponse, CloudError> {
        self.fallback_with(|p| p.generate(req.clone())).await
    }

    // fallback only covers the initial call, not mid-stream errors — the caller already owns the stream by then
    pub(crate) async fn fallback_stream(&self, req: LlmRequest) -> Result<LlmStream, CloudError> {
        self.fallback_with(|p| p.stream(req.clone())).await
    }

    pub(crate) async fn fallback_embed(&self, texts: Vec<String>) -> Result<EmbedResponse, CloudError> {
        self.fallback_with(|p| p.embed(texts.clone())).await
    }

    pub(crate) async fn fallback_generate_with_tools(
        &self,
        req: LlmRequest,
        tools: Vec<ToolDefinition>,
    ) -> Result<ToolCallResponse, CloudError> {
        self.fallback_with(|p| p.generate_with_tools(req.clone(), tools.clone()))
            .await
    }
}

pub struct UnifiedLlmClientBuilder {
    providers: Vec<(String, Box<dyn LlmProvider>)>,
    default_provider: Option<String>,
    routing: RoutingStrategy,
}

impl UnifiedLlmClientBuilder {
    fn new() -> Self {
        Self {
            providers: Vec::new(),
            default_provider: None,
            routing: RoutingStrategy::Explicit,
        }
    }

    pub fn register(mut self, name: impl Into<String>, provider: impl LlmProvider + 'static) -> Self {
        self.providers.push((name.into(), Box::new(provider)));
        self
    }

    pub fn default_provider(mut self, name: impl Into<String>) -> Self {
        self.default_provider = Some(name.into());
        self
    }

    pub fn routing(mut self, strategy: RoutingStrategy) -> Self {
        self.routing = strategy;
        self
    }

    pub fn build(self) -> Result<UnifiedLlmClient, CloudError> {
        if self.providers.is_empty() {
            return Err(CloudError::Provider {
                http_status: 0,
                message: "no providers registered".to_string(),
                retryable: false,
            });
        }

        let mut index = HashMap::with_capacity(self.providers.len());
        for (position, (name, _)) in self.providers.iter().enumerate() {
            if index.insert(name.clone(), position).is_some() {
                return Err(CloudError::Provider {
                    http_status: 0,
                    message: format!("provider '{}' registered more than once", name),
                    retryable: false,
                });
            }
        }

        let default_provider = match self.default_provider {
            Some(name) if index.contains_key(&name) => name,
            Some(name) => {
                return Err(CloudError::Provider {
                    http_status: 0,
                    message: format!("default provider '{}' was never registered", name),
                    retryable: false,
                });
            }
            None => self.providers[0].0.clone(),
        };

        Ok(UnifiedLlmClient {
            providers: self.providers,
            index,
            default_provider,
            routing: self.routing,
        })
    }
}

#[async_trait]
impl LlmProvider for UnifiedLlmClient {
    async fn generate(&self, req: LlmRequest) -> Result<LlmResponse, CloudError> {
        if self.routing == RoutingStrategy::Fallback {
            return self.fallback_generate(req).await;
        }
        self.resolve(&req.model)?.generate(req).await
    }

    async fn stream(&self, req: LlmRequest) -> Result<LlmStream, CloudError> {
        if self.routing == RoutingStrategy::Fallback {
            return self.fallback_stream(req).await;
        }
        self.resolve(&req.model)?.stream(req).await
    }

    async fn embed(&self, texts: Vec<String>) -> Result<EmbedResponse, CloudError> {
        if self.routing == RoutingStrategy::Fallback {
            return self.fallback_embed(texts).await;
        }
        self.provider_by_key(&self.default_provider)?
            .embed(texts)
            .await
    }

    async fn generate_with_tools(
        &self,
        req: LlmRequest,
        tools: Vec<ToolDefinition>,
    ) -> Result<ToolCallResponse, CloudError> {
        if self.routing == RoutingStrategy::Fallback {
            return self.fallback_generate_with_tools(req, tools).await;
        }
        self.resolve(&req.model)?
            .generate_with_tools(req, tools)
            .await
    }
}
