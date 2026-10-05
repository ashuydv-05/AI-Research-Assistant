# Evaluation System Flow

```mermaid
flowchart TD
    A["User opens Evaluation tab"] --> B["Fetch /evaluation/config-info<br/>and /evaluation/summary"]
    B --> C["Select sample size<br/>3, 5, 10 or 20 questions"]
    C --> D["POST /api/evaluation/stream<br/>top_k=5, retrieval=both, llm=both"]
    D --> E["API copies supplied evaluation keys<br/>into process environment variables"]
    E --> F["Start daemon worker thread<br/>and SSE event queue"]
    F --> G["Load evaluation_dataset.json<br/>20 questions, ground truths<br/>and relevant paper IDs"]
    G --> H["Limit questions to selected sample size"]

    H --> I["Create four configurations"]
    I --> J["Vector Retriever + Model 1"]
    I --> K["Vector Retriever + Model 2"]
    I --> L["Hybrid Retriever + Model 1"]
    I --> M["Hybrid Retriever + Model 2"]

    J --> N["For every question"]
    K --> N
    L --> N
    M --> N
    N --> O["Retrieve top 5 chunks"]
    O --> P["Convert results to serializable documents<br/>and format complete context"]
    P --> Q["Selected Groq model generates answer<br/>temperature=0, max_tokens=2048"]
    Q --> R["Measure retrieval plus generation latency"]
    R --> S["LLM judge receives question, ground truth,<br/>first 3,000 context characters and answer"]
    S --> T["Judge scores 0-100<br/>correctness, faithfulness and relevance"]
    T --> U["Overall = 0.4 correctness<br/>+ 0.3 faithfulness + 0.3 relevance"]
    U --> V["Compare retrieved paper IDs with annotations"]
    V --> W["Precision@5<br/>Recall@5 using unique relevant papers<br/>MRR from first relevant rank"]
    W --> X["Emit SSE sample_complete event"]

    X --> Y["Average every metric and latency<br/>for each configuration"]
    Y --> Z["Build 2 x 2 overall-score matrix"]
    Z --> AA["Select configuration with highest average overall score"]
    AA --> AB["Write latest results.json<br/>timestamped results file<br/>and summary.json"]
    AB --> AC["Emit SSE complete event"]
    AC --> AD["Frontend updates progress, logs,<br/>matrix, metrics and winner"]
```

Source files:

- `frontend/component/evaluation/EvaluationView.tsx`
- `src/api/route/evaluation.py`
- `src/evaluation/runner.py`
- `src/evaluation/evaluator.py`
- `src/evaluation/metrics.py`
- `src/evaluation/llm_clients.py`

