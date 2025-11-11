from rag_support.services.evaluator_schema import JudgeReport, JudgeFlags

def test_judge_schema():
    jr = JudgeReport(
        verdict="pass",
        scores={"groundedness": 5, "relevance": 5, "completeness": 4, "clarity": 5},
        flags=JudgeFlags(),
        reasons=[],
        suggested_fixes=[]
    )
    assert jr.verdict == "pass"
    assert set(jr.scores.keys()) == {"groundedness","relevance","completeness","clarity"}
