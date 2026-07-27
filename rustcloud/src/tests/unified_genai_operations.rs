use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::{stream, StreamExt};

use crate::errors::CloudError;
use crate::genai::client::UnifiedLlmClient;
use crate::genai::routing::{route_key_for_model, RoutingStrategy};
use crate::traits::llm_provider::{LlmProvider, LlmStream};
use crate::types::llm::{
    EmbedResponse, FinishReason, LlmRequest, LlmResponse, LlmStreamEvent, Message, ModelRef,
    ToolCallResponse, ToolDefinition,
};

#[derive(Debug, Clone, Copy)]
enum MockBehavior {
    Ok,
    RateLimited,
    AuthError,
    RetryableProviderError,
    NonRetryableProviderError,
}

struct MockProvider {
    name: String,
    behavior: MockBehavior,
    calls: Arc<Mutex<Vec<String>>>,
}

impl MockProvider {
    fn new(name: &str, behavior: MockBehavior, calls: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            name: name.to_string(),
            behavior,
            calls,
        }
    }

    fn record(&self) {
        self.calls.lock().unwrap().push(self.name.clone());
    }

    fn failure(&self) -> Option<CloudError> {
        match self.behavior {
            MockBehavior::Ok => None,
            MockBehavior::RateLimited => Some(CloudError::RateLimit {
                retry_after: Some(1),
            }),
            MockBehavior::AuthError => Some(CloudError::Auth {
                message: "mock auth failure".to_string(),
            }),
            MockBehavior::RetryableProviderError => Some(CloudError::Provider {
                http_status: 503,
                message: "mock retryable failure".to_string(),
                retryable: true,
            }),
            MockBehavior::NonRetryableProviderError => Some(CloudError::Provider {
                http_status: 400,
                message: "mock non-retryable failure".to_string(),
                retryable: false,
            }),
        }
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    async fn generate(&self, _req: LlmRequest) -> Result<LlmResponse, CloudError> {
        self.record();
        if let Some(err) = self.failure() {
            return Err(err);
        }
        Ok(LlmResponse {
            text: format!("{} response", self.name),
            finish_reason: FinishReason::Stop,
            usage: None,
        })
    }

    async fn stream(&self, _req: LlmRequest) -> Result<LlmStream, CloudError> {
        self.record();
        if let Some(err) = self.failure() {
            return Err(err);
        }
        let event = LlmStreamEvent::Done(FinishReason::Stop);
        Ok(Box::pin(stream::once(async move { event })))
    }

    async fn embed(&self, texts: Vec<String>) -> Result<EmbedResponse, CloudError> {
        self.record();
        if let Some(err) = self.failure() {
            return Err(err);
        }
        Ok(EmbedResponse {
            embeddings: texts.iter().map(|_| vec![0.0]).collect(),
        })
    }

    async fn generate_with_tools(
        &self,
        _req: LlmRequest,
        _tools: Vec<ToolDefinition>,
    ) -> Result<ToolCallResponse, CloudError> {
        self.record();
        if let Some(err) = self.failure() {
            return Err(err);
        }
        Ok(ToolCallResponse::Text(LlmResponse {
            text: format!("{} response", self.name),
            finish_reason: FinishReason::Stop,
            usage: None,
        }))
    }
}

fn make_request(model: ModelRef) -> LlmRequest {
    LlmRequest {
        model,
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
fn test_builder_rejects_empty_providers() {
    let result = UnifiedLlmClient::builder().build();
    let Err(CloudError::Provider {
        retryable, message, ..
    }) = result
    else {
        panic!("expected a Provider error");
    };
    assert!(!retryable);
    assert_eq!(message, "no providers registered");
}

#[test]
fn test_builder_rejects_duplicate_name() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let result = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls.clone()))
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls))
        .build();
    let Err(CloudError::Provider {
        retryable, message, ..
    }) = result
    else {
        panic!("expected a Provider error");
    };
    assert!(!retryable);
    assert_eq!(message, "provider 'aws' registered more than once");
}

#[test]
fn test_builder_rejects_unknown_default() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let result = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls))
        .default_provider("azure")
        .build();
    let Err(CloudError::Provider {
        retryable, message, ..
    }) = result
    else {
        panic!("expected a Provider error");
    };
    assert!(!retryable);
    assert_eq!(message, "default provider 'azure' was never registered");
}

#[tokio::test]
async fn test_builder_defaults_to_first_registered() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls.clone()))
        .register("gcp", MockProvider::new("gcp", MockBehavior::Ok, calls.clone()))
        .build()
        .unwrap();

    client
        .generate(make_request(ModelRef::Provider("test-model".to_string())))
        .await
        .unwrap();

    assert_eq!(calls.lock().unwrap().as_slice(), ["aws"]);
}

#[tokio::test]
async fn test_explicit_routes_generate_to_default() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls.clone()))
        .register("gcp", MockProvider::new("gcp", MockBehavior::Ok, calls.clone()))
        .default_provider("gcp")
        .routing(RoutingStrategy::Explicit)
        .build()
        .unwrap();

    let response = client
        .generate(make_request(ModelRef::Provider("test-model".to_string())))
        .await
        .unwrap();

    assert_eq!(response.text, "gcp response");
    assert_eq!(calls.lock().unwrap().as_slice(), ["gcp"]);
}

#[tokio::test]
async fn test_explicit_routes_all_four_methods() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls.clone()))
        .build()
        .unwrap();

    client
        .generate(make_request(ModelRef::Provider("test-model".to_string())))
        .await
        .unwrap();
    client
        .stream(make_request(ModelRef::Provider("test-model".to_string())))
        .await
        .unwrap();
    client.embed(vec!["text".to_string()]).await.unwrap();
    client
        .generate_with_tools(
            make_request(ModelRef::Provider("test-model".to_string())),
            vec![ToolDefinition {
                name: "tool".to_string(),
                description: "desc".to_string(),
                parameters: serde_json::json!({}),
            }],
        )
        .await
        .unwrap();

    assert_eq!(calls.lock().unwrap().as_slice(), ["aws", "aws", "aws", "aws"]);
}

#[test]
fn test_route_key_bedrock_prefixes() {
    for id in [
        "anthropic.claude-3",
        "amazon.titan-text",
        "meta.llama3",
        "mistral.mixtral-8x7b",
    ] {
        assert_eq!(
            route_key_for_model(&ModelRef::Provider(id.to_string())),
            Some("aws")
        );
    }
}

#[test]
fn test_route_key_vertex_prefixes() {
    for id in ["gemini-1.5-pro", "text-embedding-004"] {
        assert_eq!(
            route_key_for_model(&ModelRef::Provider(id.to_string())),
            Some("gcp")
        );
    }
}

#[test]
fn test_route_key_azure_prefixes() {
    for id in ["gpt-4o", "o1-preview", "o3-mini"] {
        assert_eq!(
            route_key_for_model(&ModelRef::Provider(id.to_string())),
            Some("azure")
        );
    }
}

#[test]
fn test_route_key_deployment_is_azure() {
    assert_eq!(
        route_key_for_model(&ModelRef::Deployment("my-custom-deployment".to_string())),
        Some("azure")
    );
}

#[test]
fn test_route_key_bare_model_is_none() {
    assert_eq!(
        route_key_for_model(&ModelRef::Provider("claude-3-opus".to_string())),
        None
    );
}

#[test]
fn test_route_key_logical_family_prefix() {
    let gemini = ModelRef::Logical {
        family: "gemini".to_string(),
        tier: None,
    };
    assert_eq!(route_key_for_model(&gemini), Some("gcp"));

    let claude = ModelRef::Logical {
        family: "claude".to_string(),
        tier: None,
    };
    assert_eq!(route_key_for_model(&claude), None);
}

#[tokio::test]
async fn test_model_based_unknown_prefix_uses_default() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls.clone()))
        .register("gcp", MockProvider::new("gcp", MockBehavior::Ok, calls.clone()))
        .default_provider("aws")
        .routing(RoutingStrategy::ModelBased)
        .build()
        .unwrap();

    client
        .generate(make_request(ModelRef::Provider("claude-3-opus".to_string())))
        .await
        .unwrap();

    assert_eq!(calls.lock().unwrap().as_slice(), ["aws"]);
}

#[tokio::test]
async fn test_model_based_unregistered_target_errors() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls))
        .routing(RoutingStrategy::ModelBased)
        .build()
        .unwrap();

    let result = client
        .generate(make_request(ModelRef::Provider("gpt-4o".to_string())))
        .await;

    let Err(CloudError::Provider {
        retryable, message, ..
    }) = result
    else {
        panic!("expected a Provider error");
    };
    assert!(!retryable);
    assert_eq!(message, "no provider registered under 'azure'");
}

#[tokio::test]
async fn test_model_based_embed_uses_default() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls.clone()))
        .register("azure", MockProvider::new("azure", MockBehavior::Ok, calls.clone()))
        .default_provider("aws")
        .routing(RoutingStrategy::ModelBased)
        .build()
        .unwrap();

    client.embed(vec!["hello".to_string()]).await.unwrap();

    assert_eq!(calls.lock().unwrap().as_slice(), ["aws"]);
}

#[tokio::test]
async fn test_model_based_routes_known_prefix_to_registered_target() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls.clone()))
        .register("azure", MockProvider::new("azure", MockBehavior::Ok, calls.clone()))
        .default_provider("aws")
        .routing(RoutingStrategy::ModelBased)
        .build()
        .unwrap();

    let response = client
        .generate(make_request(ModelRef::Provider("gpt-4o".to_string())))
        .await
        .unwrap();

    assert_eq!(response.text, "azure response");
    assert_eq!(calls.lock().unwrap().as_slice(), ["azure"]);
}

#[tokio::test]
async fn test_model_based_routes_stream_to_target() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls.clone()))
        .register("gcp", MockProvider::new("gcp", MockBehavior::Ok, calls.clone()))
        .default_provider("aws")
        .routing(RoutingStrategy::ModelBased)
        .build()
        .unwrap();

    client
        .stream(make_request(ModelRef::Provider("gemini-1.5-pro".to_string())))
        .await
        .unwrap();

    assert_eq!(calls.lock().unwrap().as_slice(), ["gcp"]);
}

#[tokio::test]
async fn test_model_based_routes_tools_to_target() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls.clone()))
        .register("azure", MockProvider::new("azure", MockBehavior::Ok, calls.clone()))
        .default_provider("aws")
        .routing(RoutingStrategy::ModelBased)
        .build()
        .unwrap();

    let tools = vec![ToolDefinition {
        name: "tool".to_string(),
        description: "desc".to_string(),
        parameters: serde_json::json!({}),
    }];
    client
        .generate_with_tools(make_request(ModelRef::Provider("o1-preview".to_string())), tools)
        .await
        .unwrap();

    assert_eq!(calls.lock().unwrap().as_slice(), ["azure"]);
}

#[test]
fn test_route_key_case_insensitive() {
    for id in ["GPT-4o", "O1-Preview", "Anthropic.Claude-3", "Gemini-1.5-Pro"] {
        assert!(route_key_for_model(&ModelRef::Provider(id.to_string())).is_some());
    }
}

#[tokio::test]
async fn test_fallback_currently_resolves_to_default() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls.clone()))
        .register("azure", MockProvider::new("azure", MockBehavior::Ok, calls.clone()))
        .default_provider("azure")
        .routing(RoutingStrategy::Fallback)
        .build()
        .unwrap();

    let response = client
        .generate(make_request(ModelRef::Provider("gpt-4o".to_string())))
        .await
        .unwrap();

    assert_eq!(response.text, "azure response");
    assert_eq!(calls.lock().unwrap().as_slice(), ["azure"]);
}

#[tokio::test]
async fn test_stream_passes_through_underlying_events() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let client = UnifiedLlmClient::builder()
        .register("aws", MockProvider::new("aws", MockBehavior::Ok, calls))
        .build()
        .unwrap();

    let mut stream = client
        .stream(make_request(ModelRef::Provider("test-model".to_string())))
        .await
        .unwrap();

    let event = stream.next().await.expect("stream should yield one event");
    assert!(matches!(event, LlmStreamEvent::Done(FinishReason::Stop)));
    assert!(stream.next().await.is_none());
}
