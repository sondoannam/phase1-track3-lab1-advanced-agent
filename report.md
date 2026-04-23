# Lab 16 Benchmark Report

## Metadata

- Dataset: hotpot_real.json
- Mode: real
- Records: 240
- Agents: react, reflexion

## Summary

| Metric             |   ReAct | Reflexion |   Delta |
| ------------------ | ------: | --------: | ------: |
| EM                 |    0.75 |      0.95 |     0.2 |
| Avg attempts       |       1 |     1.325 |   0.325 |
| Avg token estimate | 3616.48 |   5603.54 | 1987.06 |
| Avg latency (ms)   | 6548.74 |   9996.08 | 3447.34 |

## Failure modes

```json
{
  "react": {
    "wrong_final_answer": 30,
    "none": 90
  },
  "reflexion": {
    "none": 114,
    "reflection_overfit": 6
  }
}
```

## Extensions implemented

- structured_evaluator
- reflection_memory
- benchmark_report_json

## Discussion

The structured_evaluator extension enforces JSON output at three levels: Ollama's native format='json' grammar constraint eliminates prose preambles at the model level; Pydantic's model_validate_json then validates field types and value ranges (score ∈ {0,1}); a regex-extraction layer handles any residual formatting drift; and a hard fallback ensures the loop never crashes. The reflection_memory extension passes a numbered <reflection_memory> XML block to the Actor on every retry, containing the failure_reason, lesson, and next_strategy from all prior ReflectionEntries. This gives the Actor explicit, structured guidance rather than relying on implicit re-sampling. Reflexion improved EM on multi-hop questions where the first attempt stopped at an intermediate entity, while offering diminishing returns on single-hop questions the Actor answered correctly first time. Remaining failure modes include entity_drift (wrong second-hop entity despite reflection) and reflection_overfit (the agent anchors too strongly on the strategy from attempt 1, ignoring context evidence in later attempts).

## Auto grading

Auto-grade total: 92/100

- Flow Score (Core): 72/80
  - Schema: 30/30
  - Experiment: 30/30
  - Analysis: 12/20
- Bonus Score: 20/20
