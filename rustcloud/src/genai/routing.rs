use crate::types::llm::ModelRef;

#[derive(Debug, Clone, PartialEq)]
pub enum RoutingStrategy {
    Explicit,
    ModelBased,
    Fallback,
}

fn prefix_route_key(id: &str) -> Option<&'static str> {
    const AWS_PREFIXES: &[&str] = &["anthropic.", "amazon.", "meta.", "mistral."];
    const GCP_PREFIXES: &[&str] = &["gemini", "text-embedding-"];
    const AZURE_PREFIXES: &[&str] = &["gpt-", "o1", "o3"];

    let id = id.to_ascii_lowercase();

    if AWS_PREFIXES.iter().any(|prefix| id.starts_with(prefix)) {
        Some("aws")
    } else if GCP_PREFIXES.iter().any(|prefix| id.starts_with(prefix)) {
        Some("gcp")
    } else if AZURE_PREFIXES.iter().any(|prefix| id.starts_with(prefix)) {
        Some("azure")
    } else {
        None
    }
}

pub(crate) fn route_key_for_model(model: &ModelRef) -> Option<&'static str> {
    match model {
        ModelRef::Deployment(_) => Some("azure"),
        ModelRef::Provider(id) => prefix_route_key(id),
        ModelRef::Logical { family, .. } => prefix_route_key(family),
    }
}
