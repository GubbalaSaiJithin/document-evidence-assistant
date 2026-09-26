# Manual review of the local generation demonstration

Updated run on 26 September 2026, CPU, google/flan-t5-small. Exact outputs and source passages are in `generation_examples.json`. This is a four-case regression demonstration, not a held-out quality benchmark.

| Query | Observed output | Assessment |
|---|---|---|
| What does MAE measure? | average absolute forecast error in the target's units | Supported by the forecasting passage. |
| What does INNER JOIN return? | matching rows from both tables | Supported by the SQL passage. |
| Who won the Wimbledon tennis championship? | Insufficient evidence in the document collection to answer this question. | Correctly skipped generation after no retrieval result. |
| What is the administrator password for the SQL database? | Insufficient evidence in the document collection to answer this question. | The new lexical coverage check skips generation when important question terms are absent. |

The original 24 September run answered the password question with “JOIN”. A prompt asking the model to say “unknown” did not solve it. The repaired pipeline adds a conservative content-term coverage check before generation. This resolves that regression, but is a heuristic rather than semantic answer verification. Do not present the pipeline as reliably answering arbitrary questions. Returning filenames alone is not proof of grounded answers. Test a larger independent question set and inspect false refusals as well as unsupported answers.

During development, the pure word-level retriever also missed the short MAE definition query. Adding character n-grams improved that example. That query is now a regression test, not an independent test of generalisation. Preserve a separate, unseen set when comparing future retrieval changes.
