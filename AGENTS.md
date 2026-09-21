# Working in Jev-Flywheel

A minimal, Plexus-compatible demonstration of Jev plus a learned decision head and a
human-in-the-loop feedback flywheel. Keep it small: it is a calling card and an on-ramp,
and its value is that a stranger can read all of it.

## Rules that matter here

- **Specs first, beside the module.** `foo.py` has `foo_test.py` next to it; cross-module
  behavior lives in `tests/`. Write the spec, watch it fail, then write the code. Specs are
  named as sentences ("a label with no propensity is dropped, not guessed").
- **No network and no keys in specs.** Jev is a fake client that counts calls. The steering
  analyst is scripted (`mock_replies`). Live runs are for `scripts/`, never for `make test`.
- **Never print, log or commit a secret.** `TYPESAFE_API_KEY` and AWS credentials come from
  the environment or a gitignored `.env`, loaded with python-dotenv. Do not `source` a `.env`
  in a shell: a malformed line echoes its value into the terminal.
- **Scripted Tactus rounds must enable the mock manager before the agent is parsed.** Agents
  are built at parse time; without a `MockManager` the agent is real and the call is paid.
  `test_a_scripted_round_never_reaches_a_real_model` guards this.
- **The held-out test split is the scoreboard.** Nothing may label it, select from it, or show
  it to the steering agent. Reported accuracy is always computed there.
- **The agent proposes edits; code applies them; the fit sets numbers.** Do not add a way for
  a language model to write weights, calibration or provenance.
- **Fit only on trusted labels, with full feature coverage, weighted by selection
  propensity, and calibrate only on out-of-fold predictions.** Each has a spec for the
  failure it prevents.

## Vocabulary (kept identical to Plexus)

Item, FeedbackItem (`initial_answer_value`, `final_answer_value`, `edit_comment_value`),
label sources, score result, element, feature, decision, scorecard. Label normalization is
copied from Plexus's Evaluation and must not drift, or metrics silently go to zero.

## Commands

```bash
make install    # venv with the steer, charts and dev extras
make test       # the specs
make demo       # offline replay of the recorded run, redraws images/results.png
```
