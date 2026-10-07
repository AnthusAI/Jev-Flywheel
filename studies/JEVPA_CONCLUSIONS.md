# JEVPA conclusions after real-data investigations

## What the evidence supports

The project has a working implementation of Pareto candidate retention,
compatibility checking, question-set merging, discovery-only fitting, and
held-out stopping controls. The synthetic pilot shows that this machinery can
retain and combine pre-existing slice specialists.

The project has **not** demonstrated that the metacognitive JEVPA loop improves
a real decision task. The real investigations reached the same operational
failure in different domains: the analyst proposed closely related questions,
so no candidate pair met the pre-registered diversity gate for a meaningful
merge.

## Real-data record

| Source | Candidate outcome | JEVPA result |
| --- | --- | --- |
| Civil Comments | Near-duplicate language cues | No qualifying merge |
| Bias in Bios, journalist/professor | Near-duplicate academic-biography cues | No qualifying merge |
| UCI student outcomes, exploratory | Two academic-assessment cues, despite available financial fields | No qualifying merge |

The diversity gate was useful: it stopped the project from treating minor
selection-slice differences among nearly identical heads as evidence for a
Pareto composition.

## Interpretation

The basic Jev-Flywheel claim and the JEVPA claim are different. The basic loop
asks whether disagreement evidence can reveal **one missing criterion** and
produce one useful new question. The recorded synthetic flywheel succeeded in
that setting because the labels followed a large, coherent hidden factor
(subject domain) absent from the base sentiment question.

JEVPA additionally requires several independently useful questions, distinct
selection specialists, diverse predictions, and a beneficial composition. In
the investigated real datasets, residual-driven proposal generation repeatedly
converged on the most obvious evidence channel. A favorable underlying feature
schema did not make the analyst discover a separate channel.

## Decision

Stop treating JEVPA as a promising general optimization method. Retain the
diversity gate as a safeguard in the prototype, but do not make Pareto search a
default flywheel stage. Future work should return to the ordinary flywheel and
test its narrower, more plausible claim on real datasets: whether one steering
round can name and add a useful missing criterion. A future Pareto study would
need a deliberately diverse candidate generator and should be described as a
test of composition, not free-form metacognitive discovery.
