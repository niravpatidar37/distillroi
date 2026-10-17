| System | Accuracy (test, n=770) | Macro-F1 | $ / 1K requests | p50 / p95 latency @ c=1 | p50 / p95 @ c=4 |
|---|---|---|---|---|---|
| Teacher: gpt-oss-120b on Groq (1 query/request) | 81.2% | 0.801 | $0.111 (paid list price) | 443 / 566 ms | 417 / 874 ms |
| Student: q15_r16 on vLLM | 73.2% (90.2% of teacher) | 0.733 | $0.025 at 1M/day; $0.25 at 100K/day (idle included) | 78 / 114 ms | 105 / 158 ms |
| Routed: student -> teacher if conf < 0.82 | 80.9% (99.7% of teacher) | 0.795 | $0.056 at 1M/day (escalation 28.1%) | live end-to-end at c=2: 568 / 1708 ms (all requests) | see c=1 column |
