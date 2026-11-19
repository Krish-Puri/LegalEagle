SYSTEM_PROMPT = """You are LegalEagle, an educational contract risk review assistant.
You do not provide legal advice. You identify practical business risks, quote the contract,
explain impact in plain English, and suggest negotiation notes.
Return only data that fits the supplied schema."""


def build_review_prompt(contract_excerpt: str, classifier_notes: str, retrieved_context: str, reviewer_memory: str) -> str:
    return f"""Review this contract excerpt for risky clauses.

Reviewer memory:
{reviewer_memory or "No reviewer memory yet."}

Classifier notes:
{classifier_notes}

Retrieved context:
{retrieved_context}

Contract excerpt:
{contract_excerpt}
"""
