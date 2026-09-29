# JEVPA dataset screen: student outcome prediction

This note records a **pre-study feasibility screen**. It selects a possible
third real-data source for a diversity-gated JEVPA study; it does not report a
Jev run or establish a JEVPA result.

## Candidate selected for pre-registration

The candidate is UCI's [Predict Students' Dropout and Academic
Success](https://archive.ics.uci.edu/dataset/697/predict%20students%20dropout%20and%20academic%20success)
dataset. It contains 4,424 student records with enrollment, demographic,
socioeconomic, academic-performance, and macroeconomic fields. The published
task has three outcomes: dropout, enrolled, and graduate. A proposed binary
study would exclude `Enrolled` before every split and predict `Dropout` versus
`Graduate` from information available after the second semester. This timing
and exclusion must be stated in the pre-registration; the task is a research
exercise using a historical, anonymized public dataset, not an intervention
tool.

This is a better structural candidate than the two completed real-data studies:

| Requirement learned from the null studies | Student-outcome evidence |
| --- | --- |
| Different real decision factors | Academic progress, financial persistence, and enrollment background are separately recorded. |
| A base task with meaningful room to improve | A local academic-only probe is accurate but imperfect. |
| Plausibly complementary predictors | Financial and enrollment models make materially different predictions from academic-performance models. |
| Enough rows for sealed evaluation | 3,630 dropout/graduate rows remain after excluding `Enrolled`. |
| Public, inspectable source | UCI distributes the CSV and documentation under CC BY 4.0. |

## Local, no-Jev probe

We downloaded the UCI CSV and filtered it to `Dropout` and `Graduate`.
With seed `20260928`, a random 60% fit / 40% evaluation split supplied a
regularized logistic regression screen. Categorical enrollment fields were
one-hot encoded; all scaling, fitting, and scoring occurred locally. These
numbers are only a channel-diversity screen, not a model benchmark and not an
outcome for the eventual study.

| Field group | Brier | Accuracy | AUC |
| --- | ---: | ---: | ---: |
| Academic performance: first- and second-semester curricular-unit outcomes | 0.0849 | 0.8815 | 0.9331 |
| Financial persistence: debtor, tuition-fee status, scholarship | 0.1753 | 0.7410 | 0.7569 |
| Enrollment background: course, application, qualifications, admission, age and status | 0.1930 | 0.7114 | 0.7521 |
| Academic performance + financial persistence | 0.0726 | 0.9084 | 0.9429 |
| All three groups | 0.0706 | 0.9098 | 0.9490 |

The performance and financial-only predictions had Pearson correlation `0.4896`
and disagreed on hard class for `27.27%` of evaluation records. Academic and
financial models each had lower squared error on a nonzero portion of records
(80.17% and 19.83%, respectively). This differs sharply from the completed
Bias-in-Bios JEVPA screen, whose candidate predictions had correlations above
`0.999` and disagreement at most `0.5%`.

## Why it is still only a candidate

The local probe uses field groups chosen with knowledge of the schema and an
ordinary statistical model. It does **not** tell us whether Jev will answer
candidate questions faithfully, whether a reflective analyst will propose a
financial question rather than another academic variation, or whether a
selection-stage Pareto merge will beat greedy on a sealed test split. Those are
the actual JEVPA questions.

Before any analyst or Jev call, the next study must freeze corpus checksum,
binary inclusion rule, source-disjoint discovery/selection/test IDs, outcome
timing, two sealed factor slices, questions, candidate-generation budget, and
the same diversity and practical-effect gates used in the prior protocol.
The high-impact nature of student outcomes also means the report should keep
the task explicitly descriptive and avoid presenting any result as a deployment
recommendation.
