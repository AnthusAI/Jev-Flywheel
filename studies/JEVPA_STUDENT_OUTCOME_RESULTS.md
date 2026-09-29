# Results: student-outcome JEVPA selection stage

The study stopped after the shared initial selection stage. No branch or held-out test Jev request was made.

## What ran

The frozen binary cohort and manifest excluded the deterministic evaluation partition used by the earlier local feasibility screen. The base question ran on 500 discovery records. Four Kimi analyst slots inspected discovery-only base errors; two produced complete valid additions and two exhausted their response budget before producing valid JSON. The valid additions were repeated second-semester assessment pattern and a retake or carried-unit signal across semesters.

Jev answered the base plus both additions on the 500 discovery and 600 sealed selection records. All 1,600 item requests completed: 500 base requests (202,995 input and 16,656 output tokens) and 1,100 initial-batch requests (599,917 input and 130,778 output tokens).

## Selection result

Discovery-label logistic fits were evaluated on selection only.

| Candidate | Overall Brier | Accuracy | Academic-underperformance Brier | Financial-strain Brier |
| --- | ---: | ---: | ---: | ---: |
| Repeated assessment | 0.10962 | 0.8650 | **0.07880** | 0.19652 |
| Retake / carried-unit signal | **0.10894** | 0.8567 | 0.08446 | **0.17042** |

Each candidate uniquely won one pre-registered factor slice. But their selection prediction vectors had Pearson correlation **0.9747** and hard-class disagreement **0.0550**. The protocol required correlation below 0.95 or disagreement of at least 0.10 before a merge. Neither condition passed.

The prescribed result is therefore **no qualifying JEVPA merge**. We stopped before branch proposals and before unsealing or requesting the held-out test.

## Interpretation and limitation

This is another mechanism non-activation, despite a dataset whose raw academic and financial fields showed diverse local statistical signals. The discovery analyst nevertheless proposed two closely related academic-assessment questions. Favorable data structure alone did not make residual-driven prompt generation find the distinct factors.

This run is exploratory rather than a fully conforming confirmation: the study runner used a direct Kimi proposal call and a local discovery-fit logistic screen, rather than the repository's full Tactus/scorecard fitting path. The frozen split, discovery-only analyst evidence, sealed selection scoring, diversity gate, and no-test stopping decision were followed. It must not be treated as a third clean real-data efficacy test of JEVPA.
