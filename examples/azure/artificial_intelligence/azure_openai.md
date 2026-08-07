
# rustcloud - Azure OpenAI

## Configure credentials

`AzureOpenAiProvider` authenticates with a static API key, not OAuth2. Set these environment variables:

```sh
export AZURE_OPENAI_ENDPOINT="https://my-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_API_VERSION="2024-10-21"       # optional, defaults to 2024-10-21
export AZURE_OPENAI_EMBED_DEPLOYMENT="text-embedding-3-small"  # only needed for embed()
```

## Initialize the provider

```rust
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;

fn main() {
    let provider = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");
}
```

## Generate text

```rust
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::traits::llm_provider::LlmProvider;
use rustcloud::types::llm::{LlmRequest, Message, ModelRef};

#[tokio::main]
async fn main() {
    let provider = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");

    let req = LlmRequest {
        model: ModelRef::Deployment("gpt-4o".to_string()),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Explain what a Rust lifetime is.".to_string(),
        }],
        max_tokens: Some(256),
        temperature: Some(0.5),
        system_prompt: Some("You are a concise technical writer.".to_string()),
    };

    let response = provider.generate(req).await.unwrap();
    println!("{}", response.text);
}
```

Reasoning deployments (`o1*`, `o3*`) are detected automatically: `max_tokens` is sent as `max_completion_tokens` and `temperature` is omitted, since those models reject it.

## Stream a response

```rust
use futures::StreamExt;
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::traits::llm_provider::LlmProvider;
use rustcloud::types::llm::{LlmRequest, LlmStreamEvent, Message, ModelRef};

#[tokio::main]
async fn main() {
    let provider = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");

    let req = LlmRequest {
        model: ModelRef::Deployment("gpt-4o".to_string()),
        messages: vec![Message {
            role: "user".to_string(),
            content: "List five uses of Rust in production systems.".to_string(),
        }],
        max_tokens: Some(512),
        temperature: None,
        system_prompt: None,
    };

    let mut stream = provider.stream(req).await.unwrap();

    while let Some(event) = stream.next().await {
        match event {
            LlmStreamEvent::DeltaText(chunk) => print!("{}", chunk),
            LlmStreamEvent::Done(_) => println!(),
            LlmStreamEvent::Error(e) => eprintln!("stream error: {:?}", e),
            _ => {}
        }
    }
}
```

Azure sends `finish_reason` and the final `usage` block as two separate SSE chunks; the stream buffers the finish event until usage arrives (or the connection closes) so `Usage` is always emitted before `Done`.

## Embed text

```rust
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::traits::llm_provider::LlmProvider;

#[tokio::main]
async fn main() {
    let provider = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");

    let response = provider
        .embed(vec!["The quick brown fox".to_string()])
        .await
        .unwrap();

    println!("dimensions: {}", response.embeddings[0].len());
}
```

Embeddings use the deployment named by `AZURE_OPENAI_EMBED_DEPLOYMENT`, separate from the chat deployment. Passing an empty slice returns immediately without a network request.

## Call tools

```rust
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::traits::llm_provider::LlmProvider;
use rustcloud::types::llm::{LlmRequest, Message, ModelRef, ToolCallResponse, ToolDefinition};

#[tokio::main]
async fn main() {
    let provider = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");

    let tools = vec![ToolDefinition {
        name: "get_weather".to_string(),
        description: "Returns the current weather for a given city.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        }),
    }];

    let req = LlmRequest {
        model: ModelRef::Deployment("gpt-4o".to_string()),
        messages: vec![Message {
            role: "user".to_string(),
            content: "What is the weather in London?".to_string(),
        }],
        max_tokens: Some(256),
        temperature: Some(0.0),
        system_prompt: None,
    };

    match provider.generate_with_tools(req, tools).await.unwrap() {
        ToolCallResponse::ToolCall { name, arguments } => {
            println!("tool: {}, args: {}", name, arguments);
        }
        ToolCallResponse::Text(resp) => {
            println!("{}", resp.text);
        }
    }
}
```

`generate_with_tools` requires at least one tool; an empty list returns `CloudError::Unsupported` instead of making a request. When the model calls a tool, Azure sends its arguments as a JSON-encoded string, which is parsed back into a `serde_json::Value` before being returned.

## Common errors

```rust
use rustcloud::azure::azure_apis::artificial_intelligence::azure_openai::AzureOpenAiProvider;
use rustcloud::errors::CloudError;
use rustcloud::traits::llm_provider::LlmProvider;
use rustcloud::types::llm::{LlmRequest, Message, ModelRef};

#[tokio::main]
async fn main() {
    let provider = AzureOpenAiProvider::new().expect("failed to load Azure OpenAI credentials");

    let req = LlmRequest {
        model: ModelRef::Deployment("gpt-4o".to_string()),
        messages: vec![Message { role: "user".to_string(), content: "hi".to_string() }],
        max_tokens: Some(64),
        temperature: Some(0.0),
        system_prompt: None,
    };

    match provider.generate(req).await {
        Ok(resp) => println!("{}", resp.text),
        Err(CloudError::Auth { message }) => eprintln!("auth failed: {message}"),
        Err(CloudError::RateLimit { retry_after }) => {
            eprintln!("rate limited, retry after {:?}s", retry_after)
        }
        Err(CloudError::Provider { http_status, message, retryable }) => {
            eprintln!("provider error {http_status} (retryable: {retryable}): {message}")
        }
        Err(e) => eprintln!("request failed: {:?}", e),
    }
}
```

- **Invalid API key** — Azure returns 401/403 with a JSON error body. `AzureOpenAiProvider` maps this to `CloudError::Auth`, and the `message` field is the human-readable text Azure sent (e.g. "The api-key you provided is invalid."), not the raw JSON blob.
- **Rate limited (429)** — mapped to `CloudError::RateLimit { retry_after }`. When Azure includes a `Retry-After` header, `retry_after` is `Some(seconds)`; otherwise it's `None`.
- **Prompt-level content filter** — a prompt rejected by Azure's content filter comes back as HTTP 400 with `"code": "content_filter"` in the error body. This maps to `CloudError::Provider { http_status: 400, retryable: false, .. }`, since resubmitting the same prompt unmodified will fail the same way.
- **Response-level content filter** — this is not an error. Azure returns HTTP 200 with the message `content` set to `null` and `finish_reason: "content_filter"`. `generate` returns `Ok(LlmResponse { text: "".to_string(), finish_reason: FinishReason::Other("content_filter".to_string()), .. })`, and `stream` emits a matching `LlmStreamEvent::Done(FinishReason::Other("content_filter".to_string()))`. Check `finish_reason` if you need to tell a filtered response apart from a genuinely empty one.
- **Mid-stream error** — a `stream()` call can succeed initially and still fail partway through, if Azure sends a `data: {"error": {...}}` SSE chunk after generation has already started. This surfaces as `LlmStreamEvent::Error(CloudError::Provider { http_status: 200, .. })` from the stream itself, not as an `Err` from the initial `.await`. Check for the `Error` variant while iterating the stream, not just at the call site.

## ModelRef semantics

- `ModelRef::Deployment(name)` — used as-is as the Azure deployment name. This is the native way to address an Azure OpenAI resource.
- `ModelRef::Provider(id)` — also treated as a deployment name. Azure has no separate model registry, and users routinely name deployments after the underlying model (e.g. `gpt-4o`), so treating a provider id as a deployment alias lets model-based routing work without a second lookup.
- `ModelRef::Logical { .. }` — returns `CloudError::Unsupported`. Azure has no generic model-family resolution API, so there's no deployment to resolve a logical model reference against.
