
# rustcloud - Unified GenAI Client

## Overview

`UnifiedLlmClient` registers several `LlmProvider` implementations under one client and routes each call to one of them, by one of three strategies: `Explicit`, `ModelBased`, or `Fallback`. It implements `LlmProvider` itself, so it can be passed anywhere a single provider is expected — including wrapped in `RetryMiddleware`.

## Build a client with the builder

```rust
use rustcloud::aws::aws_apis::artificial_intelligence::aws_bedrock::BedrockProvider;
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::gcp::gcp_apis::artificial_intelligence::gcp_vertex_ai::VertexAiProvider;
use rustcloud::genai::client::UnifiedLlmClient;
use rustcloud::genai::routing::RoutingStrategy;

#[tokio::main]
async fn main() {
    let aws = BedrockProvider::new().await;
    let gcp = VertexAiProvider::new("my-gcp-project", "us-central1")
        .await
        .expect("failed to authenticate with Vertex AI");
    let azure = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");

    let client = UnifiedLlmClient::builder()
        .register("aws", aws)
        .register("gcp", gcp)
        .register("azure", azure)
        .default_provider("aws")
        .routing(RoutingStrategy::ModelBased)
        .build()
        .expect("client configuration is invalid");
}
```

`build()` checks, in order: at least one provider is registered, no two providers share a name, and `default_provider` (if set) names a provider that was actually registered. Had `.default_provider("aws")` been left out above, it would still resolve to `"aws"` — the first provider registered.

## Explicit routing

Every call goes to `default_provider`, regardless of what `model` the request names.

```rust
use rustcloud::aws::aws_apis::artificial_intelligence::aws_bedrock::BedrockProvider;
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::genai::client::UnifiedLlmClient;
use rustcloud::genai::routing::RoutingStrategy;
use rustcloud::traits::llm_provider::LlmProvider;
use rustcloud::types::llm::{LlmRequest, Message, ModelRef};

#[tokio::main]
async fn main() {
    let aws = BedrockProvider::new().await;
    let azure = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");

    let client = UnifiedLlmClient::builder()
        .register("aws", aws)
        .register("azure", azure)
        .default_provider("azure")
        .routing(RoutingStrategy::Explicit)
        .build()
        .expect("client configuration is invalid");

    let req = LlmRequest {
        model: ModelRef::Provider("anthropic.claude-3-5-haiku-20241022-v1:0".to_string()),
        messages: vec![Message { role: "user".to_string(), content: "hi".to_string() }],
        max_tokens: Some(64),
        temperature: Some(0.0),
        system_prompt: None,
    };

    // still goes to azure, despite the anthropic model id
    let response = client.generate(req).await.unwrap();
    println!("{}", response.text);
}
```

Use `Explicit` when the caller picks a provider once, up front, and every request should honor that choice regardless of which model it names.

## Model-based routing

`ModelBased` inspects `req.model` and picks a provider by prefix:

| Model prefix | Routes to |
| --- | --- |
| `anthropic.`, `amazon.`, `meta.`, `mistral.` | `"aws"` |
| `gemini`, `text-embedding-` | `"gcp"` |
| `gpt-`, `o1`, `o3` | `"azure"` |
| `ModelRef::Deployment(_)` (any name) | `"azure"` |

Matching is case-insensitive and by prefix (`starts_with`), so `"Gemini-1.5-Pro"` and `"gemini-1.5-pro"` route the same way. `ModelRef::Deployment` always routes to `"azure"` without checking the table — deployment names are Azure's own scheme, not a cross-cloud model-id convention.

Register providers under the exact keys the table produces — `"aws"`, `"gcp"`, `"azure"` — or model-based routing has nothing to look up.

```rust
use rustcloud::aws::aws_apis::artificial_intelligence::aws_bedrock::BedrockProvider;
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::gcp::gcp_apis::artificial_intelligence::gcp_vertex_ai::VertexAiProvider;
use rustcloud::genai::client::UnifiedLlmClient;
use rustcloud::genai::routing::RoutingStrategy;
use rustcloud::traits::llm_provider::LlmProvider;
use rustcloud::types::llm::{LlmRequest, Message, ModelRef};

#[tokio::main]
async fn main() {
    let aws = BedrockProvider::new().await;
    let gcp = VertexAiProvider::new("my-gcp-project", "us-central1")
        .await
        .expect("failed to authenticate with Vertex AI");
    let azure = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");

    let client = UnifiedLlmClient::builder()
        .register("aws", aws)
        .register("gcp", gcp)
        .register("azure", azure)
        .default_provider("azure")
        .routing(RoutingStrategy::ModelBased)
        .build()
        .expect("client configuration is invalid");

    let req = LlmRequest {
        model: ModelRef::Provider("gemini-1.5-pro".to_string()),
        messages: vec![Message { role: "user".to_string(), content: "hi".to_string() }],
        max_tokens: Some(64),
        temperature: Some(0.0),
        system_prompt: None,
    };

    // routes to gcp — "gemini-1.5-pro" matches the gemini prefix
    let response = client.generate(req).await.unwrap();
    println!("{}", response.text);
}
```

A model id that matches no prefix falls back to `default_provider` — that's not an error. A model id that matches a prefix naming a provider you never registered *is* an error: `provider_by_key` returns `CloudError::Provider` with a message like `"no provider registered under 'gcp'"`. `embed()` never consults this table, since it takes a list of strings with no `ModelRef` attached — it always goes to `default_provider`.

## Fallback routing

`Fallback` tries providers in registration order until one succeeds.

```rust
use rustcloud::aws::aws_apis::artificial_intelligence::aws_bedrock::BedrockProvider;
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::gcp::gcp_apis::artificial_intelligence::gcp_vertex_ai::VertexAiProvider;
use rustcloud::genai::client::UnifiedLlmClient;
use rustcloud::genai::routing::RoutingStrategy;
use rustcloud::traits::llm_provider::LlmProvider;
use rustcloud::types::llm::{LlmRequest, Message, ModelRef};

#[tokio::main]
async fn main() {
    let aws = BedrockProvider::new().await;
    let gcp = VertexAiProvider::new("my-gcp-project", "us-central1")
        .await
        .expect("failed to authenticate with Vertex AI");
    let azure = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");

    let client = UnifiedLlmClient::builder()
        .register("azure", azure)
        .register("aws", aws)
        .register("gcp", gcp)
        .routing(RoutingStrategy::Fallback)
        .build()
        .expect("client configuration is invalid");

    let req = LlmRequest {
        model: ModelRef::Deployment("gpt-4o".to_string()),
        messages: vec![Message { role: "user".to_string(), content: "hi".to_string() }],
        max_tokens: Some(64),
        temperature: Some(0.0),
        system_prompt: None,
    };

    // tries azure first; if azure fails with a transient error, tries aws, then gcp
    let response = client.generate(req).await.unwrap();
    println!("{}", response.text);
}
```

A transient failure — `CloudError::RateLimit`, `CloudError::Network`, or `CloudError::Provider { retryable: true, .. }` — moves on to the next provider. A hard failure — `Auth`, `Unsupported`, `Serialization`, or `Provider { retryable: false, .. }` — propagates immediately, without trying the rest. If every provider fails transiently, the error returned is the last one seen, not the first. `stream()` only applies this to the call that returns the stream; once a stream is handed back, it's already partially consumed, so a later error inside it surfaces as an `LlmStreamEvent::Error` rather than triggering a retry against the next provider.

## Composing with RetryMiddleware

`RetryMiddleware<P>` wraps anything implementing `LlmProvider` — including a concrete provider or a whole `UnifiedLlmClient` — since `UnifiedLlmClient` implements the same trait. That gives three ways to combine retry with routing, depending on where the retry budget should apply.

### Retry a single provider

```rust
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::genai::retry::RetryMiddleware;
use rustcloud::traits::llm_provider::LlmProvider;
use rustcloud::types::llm::{LlmRequest, Message, ModelRef};

#[tokio::main]
async fn main() {
    let azure = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");
    let provider = RetryMiddleware::wrap(azure);

    let req = LlmRequest {
        model: ModelRef::Deployment("gpt-4o".to_string()),
        messages: vec![Message { role: "user".to_string(), content: "hi".to_string() }],
        max_tokens: Some(64),
        temperature: Some(0.0),
        system_prompt: None,
    };

    let response = provider.generate(req).await.unwrap();
    println!("{}", response.text);
}
```

No routing involved — this is retry on top of one provider, nothing more.

### Retry the whole routed client

```rust
use rustcloud::aws::aws_apis::artificial_intelligence::aws_bedrock::BedrockProvider;
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::gcp::gcp_apis::artificial_intelligence::gcp_vertex_ai::VertexAiProvider;
use rustcloud::genai::client::UnifiedLlmClient;
use rustcloud::genai::retry::RetryMiddleware;
use rustcloud::genai::routing::RoutingStrategy;
use rustcloud::traits::llm_provider::LlmProvider;
use rustcloud::types::llm::{LlmRequest, Message, ModelRef};

#[tokio::main]
async fn main() {
    let aws = BedrockProvider::new().await;
    let gcp = VertexAiProvider::new("my-gcp-project", "us-central1")
        .await
        .expect("failed to authenticate with Vertex AI");
    let azure = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");

    let client = UnifiedLlmClient::builder()
        .register("azure", azure)
        .register("aws", aws)
        .register("gcp", gcp)
        .routing(RoutingStrategy::Fallback)
        .build()
        .expect("client configuration is invalid");

    let provider = RetryMiddleware::wrap(client);

    let req = LlmRequest {
        model: ModelRef::Deployment("gpt-4o".to_string()),
        messages: vec![Message { role: "user".to_string(), content: "hi".to_string() }],
        max_tokens: Some(64),
        temperature: Some(0.0),
        system_prompt: None,
    };

    let response = provider.generate(req).await.unwrap();
    println!("{}", response.text);
}
```

Here the retry budget covers one pass through the *entire* fallback chain. If every provider fails transiently once, `RetryMiddleware` waits out a backoff and runs the whole chain again from `"azure"`, rather than retrying any single provider in place.

### Retry each provider before fallback moves on

```rust
use rustcloud::aws::aws_apis::artificial_intelligence::aws_bedrock::BedrockProvider;
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::gcp::gcp_apis::artificial_intelligence::gcp_vertex_ai::VertexAiProvider;
use rustcloud::genai::client::UnifiedLlmClient;
use rustcloud::genai::retry::RetryMiddleware;
use rustcloud::genai::routing::RoutingStrategy;
use rustcloud::traits::llm_provider::LlmProvider;
use rustcloud::types::llm::{LlmRequest, Message, ModelRef};

#[tokio::main]
async fn main() {
    let aws = RetryMiddleware::wrap(BedrockProvider::new().await);
    let gcp = RetryMiddleware::wrap(
        VertexAiProvider::new("my-gcp-project", "us-central1")
            .await
            .expect("failed to authenticate with Vertex AI"),
    );
    let azure = RetryMiddleware::wrap(
        AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials"),
    );

    let client = UnifiedLlmClient::builder()
        .register("azure", azure)
        .register("aws", aws)
        .register("gcp", gcp)
        .routing(RoutingStrategy::Fallback)
        .build()
        .expect("client configuration is invalid");

    let req = LlmRequest {
        model: ModelRef::Deployment("gpt-4o".to_string()),
        messages: vec![Message { role: "user".to_string(), content: "hi".to_string() }],
        max_tokens: Some(64),
        temperature: Some(0.0),
        system_prompt: None,
    };

    let response = client.generate(req).await.unwrap();
    println!("{}", response.text);
}
```

Here each provider gets its own retry budget. A transient failure on `"azure"` is retried against `"azure"` itself first; fallback only moves on to `"aws"` once `"azure"`'s own retries are exhausted. This trades a longer worst-case latency for giving each provider a fair chance before writing it off for the request.

## Error semantics

Retry and fallback share one classification, `is_transient`, so they never disagree about what's worth retrying:

- **Transient** — `RateLimit`, `Network`, `Provider { retryable: true, .. }`. Retried locally (`RetryMiddleware`) or against the next provider (`Fallback`).
- **Propagates immediately** — `Auth`, `Unsupported`, `Provider { retryable: false, .. }`, and `Serialization`. A serialization error means a response failed to parse; retrying elsewhere would risk masking a real parsing bug instead of surfacing it.

When a `RateLimit` error carries a server-provided `retry_after`, `RetryMiddleware` waits that long instead of computing its own exponential delay — an explicit hint from the provider beats a heuristic guess. Without one, the delay doubles per attempt from `base_delay_ms`, with optional jitter, and both are configurable via `RetryMiddleware::with_config`.
