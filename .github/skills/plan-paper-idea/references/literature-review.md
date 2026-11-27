# Literature Review for Research Planning

## Search coverage

Search the research question, observed failure, proposed mechanism, and method using synonyms and adjacent terminology. Include seminal work, recent work, closest competing approaches, relevant benchmarks, and contradictory/negative findings. Follow references and forward citations of decisive papers. Use multiple independent indexes or publisher sources where available; deduplicate preprint/published versions.

Record queries, sources, search date, scope, and access gaps. Paper count is not a completion criterion; citation count alone does not establish relevance or correctness.

## Extract from decisive papers

| Field | Required substance |
|---|---|
| Identity and reading depth | Verified title/version/link; sections read; access gaps |
| Problem and setting | Task, assumptions, data/model regime, scope |
| Contribution and mechanism | What changes and why authors expect it to work |
| Evidence | Relevant results, controls, uncertainty, section/table/figure locations |
| Boundaries | Limitations, failed settings, confounders, contrary evidence |
| Closest-work comparison | Candidate claims already covered; remaining distinction |
| Reusable advice | Method, baseline, dataset, metric, diagnostic, or ablation; transfer rationale and conditions |

Read methods and experiments of closest work rather than extrapolating from abstracts. Check whether mechanism claims were tested or merely proposed. Source empirical claims to exact sections/tables where possible.

Separate author findings, author suggestions, and your derived recommendations. An experiment suggestion inferred from a limitation is not an established conclusion. Prefer primary sources; retrieve citation metadata from verified publisher/index records rather than memory. Mark unresolved citations and do not use them to establish central novelty.

## Synthesize across papers

- What is established, under which conditions?
- Where do closest approaches agree/conflict, and why might settings matter?
- Is the remaining opening unaddressed, under-evaluated, or already covered?
- What can be reused directly, what needs adaptation, and what justifies transfer?
- Which baselines or explanations could invalidate the proposed contribution?

Connect recommendations to stable paper IDs (`P01`, ...), then claim/experiment IDs. A useful form is “E02 adopts diagnostic D from P03 §4, adapted to setting S; it tests C01 against explanation A.” Fill it only with verified sources and actual project details.

## Review completion

Proceed when closest competitors are identified and read in depth; central motivation/positioning have verified sources; reusable practices include transfer conditions; contrary evidence is addressed; and additional searches/citation tracing mostly repeat known themes without exposing an unresolved central competitor.

Document that assessment and residual gaps, without claiming exhaustive coverage. For inaccessible decisive full text, seek an accessible author version or equivalent verified evidence. Otherwise keep the central comparison unresolved and the pre-experiment gate incomplete. Peripheral gaps may remain explicit paper TODOs.
