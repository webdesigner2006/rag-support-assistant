from rag_support.graph.rag_graph import GENERATOR_SYSTEM_PROMPT

def test_generator_prompt_rules_present():
    assert "STRICT grounding" in GENERATOR_SYSTEM_PROMPT
    assert "Always include citations" in GENERATOR_SYSTEM_PROMPT
