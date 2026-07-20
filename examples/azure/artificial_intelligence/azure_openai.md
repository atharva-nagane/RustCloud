
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

## ModelRef semantics

- `ModelRef::Deployment(name)` — used as-is as the Azure deployment name. This is the native way to address an Azure OpenAI resource.
- `ModelRef::Provider(id)` — also treated as a deployment name. Azure has no separate model registry, and users routinely name deployments after the underlying model (e.g. `gpt-4o`), so treating a provider id as a deployment alias lets model-based routing work without a second lookup.
- `ModelRef::Logical { .. }` — returns `CloudError::Unsupported`. Azure has no generic model-family resolution API, so there's no deployment to resolve a logical model reference against.
