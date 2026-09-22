.DEFAULT_GOAL := help
.PHONY: help install demo laya student finetune bios flipopt race race2 age pairs test diagrams

help:
	@echo "Jev Flywheel: three things you can run. Nothing here needs a Jev key."
	@echo ""
	@echo "  make install   set up a virtualenv (once)"
	@echo "  make demo      1. the flywheel with Jev, replayed offline from a recording (~10 s)"
	@echo "  make laya      2. the same recording answered by a local Laya model (downloads ~843 MB)"
	@echo "  make student   3. distil the result into a small local BERT classifier (downloads ~1.2 GB in all)"
	@echo "  make finetune  4. does gradient fine-tuning of Laya itself beat the fitted head? (one seed, quick)"
	@echo "  make bios      5. does the engine read gender? (surgeon/physician bios, J0/L0, offline)"
	@echo "  make flipopt   optimising the head against the flip (baseline/L3/L4/L5/L6, offline)"
	@echo "  make pairs     does the gender result hold on other decisions? (3 more pairs, offline)"
	@echo "  make test      run the test suite"
	@echo ""
	@echo "What each one does, what it needs and what to expect: README.md, section 'Try it'."

install:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[steer,charts,dev]'

# Stage 1. The whole flywheel with Jev, replayed offline from the committed recording: no keys,
# no network, no model and no person. Prints the scorecard lineage and writes the figure.
demo:
	@echo "Replaying a recorded run: 140 labels from a scripted labeler, every refit, and one"
	@echo "steering round in which an AI analyst proposed a new question. Nothing is sent anywhere;"
	@echo "the answers Jev gave when this was recorded are read from fixtures/."
	@echo ""
	.venv/bin/flywheel -w var/demo replay fixtures/recordings/simulated-labeler --chart images/results.png
	@echo ""
	@echo "How to read that: each row is a version of the scorecard, all scored on the same 600"
	@echo "held-out items. 'fit' rows reweight the existing questions; the 'steer' row is the"
	@echo "analyst's new question. Accuracy is how often the verdict matches the labeler; ECE is"
	@echo "how far stated confidence is from how often it is right (lower is better)."
	@echo "The figure was redrawn at images/results.png. Next: README.md, 'The result'."

# Stage 2. The same recording, replayed against a local Laya model instead of Jev's answers.
# Needs Apple silicon. Downloads the ~843 MB checkpoint on first use.
laya:
	.venv/bin/pip install -e '.[laya]'
	@echo ""
	@echo "Replaying the same 140 labels twice, once with Jev's recorded answers and once with a"
	@echo "local Laya model answering every question on this machine. Free, no key, no network"
	@echo "after the one-time model download."
	@echo ""
	.venv/bin/python scripts/laya_paired.py --out var/laya_paired.jsonl
	@echo ""
	@echo "Rows are versions of the scorecard for each engine. Compare the last row of each: the"
	@echo "layer helps the weaker local model too, but does not close the gap. Written to"
	@echo "var/laya_paired.jsonl. Next: README.md, 'The same layer on a local model'."

# Stage 3. Use the calibrated head as a teacher and train a small local text classifier on it.
# Downloads PyTorch (~80 MB) and DistilBERT (~270 MB) on top of Laya (~843 MB), and asks Laya a topic question about every item
# once (a few minutes) for the teacher's features. One seed; the README's table uses three.
student:
	.venv/bin/pip install -e '.[laya,student]'
	@echo ""
	@echo "Step 1 of 2: building the teacher's inputs (asks a local Laya about 8,801 items)."
	.venv/bin/python scripts/learning_curve.py --scratch var/student --prepare
	@echo ""
	@echo "Step 2 of 2: fine-tuning DistilBERT on the teacher's labels, three variants."
	.venv/bin/python scripts/distill_student.py --scratch var/student --seeds 1 --out var/student/distill.jsonl
	@echo ""
	@echo "Each line: a student's accuracy against the human labels on 3,521 held-out items,"
	@echo "and how often it agrees with the teacher. Next: README.md, 'Moving off the hosted model'."

# Stage 4. Fine-tune Laya's own weights on the same 140 labels the flywheel's head is fit on,
# and compare: full fine-tune, head-only fine-tune, and a DistilBERT baseline. One seed, arms
# A/B/D only, to keep this quick; `studies/PREREGISTERED.md` and `studies/finetune_laya.jsonl`
# have the full pre-registered study (multiple seeds, and the Arm C learning curve).
finetune:
	.venv/bin/pip install -e '.[laya,student]'
	@echo ""
	@echo "Fine-tuning Laya (full and head-only) and DistilBERT on the same 140 human labels the"
	@echo "flywheel's fitted head uses, one seed each. This mutates a fresh copy of Laya's weights"
	@echo "in memory only -- nothing is written back to the checkpoint."
	@echo ""
	.venv/bin/python scripts/finetune_laya.py --arms A B D --seeds 1 --folds 3 \
		--out var/finetune_laya_quick.jsonl
	@echo ""
	@echo "Each line: an arm's accuracy on paper600 and the full 3,521 held-out items, next to the"
	@echo "chosen learning rate and its cross-validation scores. Compare paper600 accuracy against"
	@echo "the flywheel's own 0.802 (README, 'The same layer on a local model')."

# Stage 5. Does the engine read gender? The engine-alone arms (J0, L0) of the Bias-in-Bios
# study, scored from committed answer fixtures: no keys, no network, no model, no spend. J1,
# J2, L1 and L2 have all run (fixtures/bios/recordings/<arm>-seed<N>/) and replay offline, one
# recording at a time, via `flywheel replay <dir> --fixtures fixtures/bios` -- not looped here
# because each recording's chart/report is meant to be inspected on its own. LF needs the GPU
# (laya-mlx) and has no recording to replay from.
bios:
	@echo "Scoring Jev's and Laya's own answers (no labels, no fitted head) on 2,000 held-out"
	@echo "surgeon/physician bios and their gender-swapped, name-redacted counterfactual twins."
	@echo "Offline: answers are read from fixtures/bios/answers.jsonl.gz and answers-laya.jsonl.gz."
	@echo ""
	.venv/bin/python scripts/run_bios_arms.py --arm J0 --out var/bios_gender.jsonl
	.venv/bin/python scripts/run_bios_arms.py --arm L0 --out var/bios_gender.jsonl
	@echo ""
	@echo "counterfactual_flip_rate is a LOWER BOUND on gender sensitivity (redaction removes"
	@echo "first names; titles and other gendered nouns not on the name list can remain)."
	@echo ""
	@echo "J1/J2/L1/L2 replay with no keys and no GPU from their recordings, one seed at a time:"
	@echo "  flywheel replay fixtures/bios/recordings/J1-seed1 --fixtures fixtures/bios"
	@echo "  (J2-seed<N>, L1-seed<N>, L2-seed<N> for N in 1 2 3, same pattern)"
	@echo "LF needs the GPU (laya-mlx) and is not replayable from a recording."
	@echo "Next: studies/PREREGISTERED.md, 'does the engine read gender, and can the layer refuse to?'"

# Stage 5b. Optimising the head against the flip: baseline (twin averaging) and L3/L4/L5/L6 at
# the registered 2% gate, replayed offline -- baseline replays each seed's actual L1 recording
# (jev_flywheel.recording.replay); L3/L4/L5 need only the single holistic feature's answers,
# already in the committed fixtures/bios/answers-laya.jsonl.gz (test split and its
# counterfactual twins alike). Writes to var/, not studies/, so a replay never touches the
# authoritative rows. Where feasible only: the 5%/10% exploratory gate extension
# (studies/bios_flipopt_gate_sensitivity.jsonl) and any seed where a new element got promoted
# (L6's own steering round, or L5 at the 5%/10% gates) need that element's answers on the 2,000
# held-out bios and twins, which are never persisted past the run that answered them -- those
# are not replayable here and need the GPU (laya-mlx) again, same as LF.
flipopt:
	@echo "Replaying baseline/L3/L4/L5/L6 at the registered 2% gate (studies/PREREGISTERED.md,"
	@echo "'optimising the head against the flip') from the committed Laya answer fixtures."
	@echo ""
	rm -f var/bios_flipopt.jsonl
	.venv/bin/python scripts/run_bios_flipopt.py --arm baseline --seeds 1 2 3 --offline --out var/bios_flipopt.jsonl
	.venv/bin/python scripts/run_bios_flipopt.py --arm L3 --seeds 1 2 3 --offline --out var/bios_flipopt.jsonl
	.venv/bin/python scripts/run_bios_flipopt.py --arm L4 --seeds 1 2 3 --offline --out var/bios_flipopt.jsonl
	.venv/bin/python scripts/run_bios_flipopt.py --arm L5 --seeds 1 2 3 --offline --out var/bios_flipopt.jsonl
	.venv/bin/python scripts/run_bios_flipopt.py --arm L6 --seeds 1 2 3 --offline --out var/bios_flipopt.jsonl
	@echo ""
	@echo "Not replayed here (need the GPU: a promoted element's held-out answers are never"
	@echo "cached past the run that made them): the 5%%/10%% exploratory gate extension, and"
	@echo "L6 seeds where the analyst's proposal was promoted rather than rejected."
	@echo "Next: studies/PREREGISTERED.md, 'optimising the head against the flip'"

# Does the engine read race from a name? The Bertrand & Mullainathan (2004) name-swap
# counterfactual (studies/PREREGISTERED.md, "does the engine read race from a name?"), scored
# from committed fixtures: no keys, no network, no model, no spend.
race:
	@echo "Scoring Jev's and Laya's own answers on 1,571 held-out bios under three named"
	@echo "versions each (two white names, one Black name). Offline: answers are read from"
	@echo "fixtures/bios/answers-race.jsonl.gz and answers-race-laya.jsonl.gz."
	@echo ""
	rm -f var/bios_race.jsonl
	.venv/bin/python scripts/run_bios_race.py --engine jev --out var/bios_race.jsonl
	.venv/bin/python scripts/run_bios_race.py --engine laya --out var/bios_race.jsonl
	@echo ""
	@echo "race_flip is a name-only effect measured against its own control floor (white-A vs"
	@echo "white-B); read the excess and ratio, not the race rate alone."
	@echo "Next: studies/PREREGISTERED.md, 'does the engine read race from a name?'"

# Race from a full name, second attempt: a recurring cue (pronoun, every [name] placeholder,
# every PERSON-span surname) instead of the first attempt's one-token pronoun swap, four groups
# (white, black, hispanic, asian) instead of one, and a continuous outcome (the signed mean
# shift in P(surgeon)) instead of a flip rate alone. Offline: answers are read from
# fixtures/bios/answers-race2-laya.jsonl.gz and answers-race2.jsonl.gz.
race2:
	@echo "Scoring Laya on all eligible bios and on the 500-bio Jev subsample, and Jev on that"
	@echo "same 500-bio subsample, under four full-name groups (4 names each)."
	@echo ""
	rm -f var/bios_race2.jsonl
	.venv/bin/python scripts/run_bios_race2.py --engine laya --sample all --out var/bios_race2.jsonl
	.venv/bin/python scripts/run_bios_race2.py --engine laya --sample 500 --out var/bios_race2.jsonl
	.venv/bin/python scripts/run_bios_race2.py --engine jev --sample 500 --out var/bios_race2.jsonl
	@echo ""
	@echo "shift is a name-only effect measured against its own floor (white names 1-2 vs 3-4);"
	@echo "read the shift and its interval against the floor's, not the shift alone."
	@echo "Next: studies/PREREGISTERED.md, 'race from a full name, second attempt'"

# Does the engine read age? An age (34, 35, 61, or 62) is inserted at the bio's first subject
# pronoun, holding the bio's own stated experience fixed (studies/PREREGISTERED.md, "does the
# engine read age?"). Offline: answers are read from fixtures/bios/answers-age.jsonl.gz and
# answers-age-laya.jsonl.gz.
age:
	@echo "Scoring Jev's and Laya's own answers on 1,231 eligible held-out bios under four aged"
	@echo "versions each (34, 35, 61, 62)."
	@echo ""
	rm -f var/bios_age.jsonl
	.venv/bin/python scripts/run_bios_age.py --engine jev --out var/bios_age.jsonl
	.venv/bin/python scripts/run_bios_age.py --engine laya --out var/bios_age.jsonl
	@echo ""
	@echo "age_flip/age_shift (34 vs 61) are read against their own floors (34 vs 35, 61 vs 62);"
	@echo "read the shift and its interval against the floor's, not the age effect alone."
	@echo "Next: studies/PREREGISTERED.md, 'does the engine read age?'"

# Does the gender result hold on other decisions? Three more occupation pairs (nurse/physician,
# paralegal/attorney, teacher/professor), same method as the primary surgeon/physician study, no
# fitted head, no labels spent (studies/PREREGISTERED.md, "does the gender result hold on other
# decisions?"). Offline: answers are read from the committed fixtures/bios_pairs/<pair>/ fixtures.
pairs:
	@echo "Scoring Jev's and Laya's own answers on three more occupation pairs (1,000 bios per"
	@echo "label each, all held out) and their gender-swapped, name-redacted counterfactual twins."
	@echo ""
	rm -f var/bios_pairs.jsonl
	.venv/bin/python scripts/run_bios_pairs.py --pair nurse_physician --engine jev --out var/bios_pairs.jsonl
	.venv/bin/python scripts/run_bios_pairs.py --pair nurse_physician --engine laya --out var/bios_pairs.jsonl
	.venv/bin/python scripts/run_bios_pairs.py --pair paralegal_attorney --engine jev --out var/bios_pairs.jsonl
	.venv/bin/python scripts/run_bios_pairs.py --pair paralegal_attorney --engine laya --out var/bios_pairs.jsonl
	.venv/bin/python scripts/run_bios_pairs.py --pair teacher_professor --engine jev --out var/bios_pairs.jsonl
	.venv/bin/python scripts/run_bios_pairs.py --pair teacher_professor --engine laya --out var/bios_pairs.jsonl
	.venv/bin/python scripts/run_bios_pairs.py --copy-surgeon-physician --out var/bios_pairs.jsonl
	@echo ""
	@echo "Four pairs in one table, ordered by the gap in women's share between the two labels;"
	@echo "counterfactual_flip_rate is a LOWER BOUND on gender sensitivity, as in the primary study."
	@echo "Next: studies/PREREGISTERED.md, 'does the gender result hold on other decisions?'"

# The learning loop on the pair that matters: paralegal/attorney, the worst case the four-pair
# test found (studies/PREREGISTERED.md, final section). J0/L0 (the engines alone) are scored
# offline from committed fixtures, no keys, no network, no spend. J1/J2/L1/L2 (140 labels, one
# steering round, the invariance gate on J2/L2) and the twin-averaging baseline are recorded to
# fixtures/bios_attorney/recordings/<arm>-seed<N>/ when they are run
# (scripts/run_bios_attorney_loop.py) and can be replayed from there with no keys via
# `flywheel replay <recording-dir> --fixtures fixtures/bios_attorney`, one recording at a time.
attorney:
	@echo "Scoring Jev's and Laya's own answers (no labels, no fitted head) on the paralegal/"
	@echo "attorney held-out bios and their amended-rule gender-swapped, name-redacted twins."
	@echo ""
	.venv/bin/python scripts/run_bios_attorney_loop.py --arm J0
	.venv/bin/python scripts/run_bios_attorney_loop.py --arm L0
	@echo ""
	@echo "J1/J2/L1/L2 and the twin-averaging baseline need a steering round (Bedrock analyst for"
	@echo "J1/J2/L1/L2; Laya locally for L1/L2) and are not part of this offline target. Replay a"
	@echo "committed recording with: flywheel replay fixtures/bios_attorney/recordings/<arm>-seed<N> \\"
	@echo "  --fixtures fixtures/bios_attorney"
	@echo "Next: studies/PREREGISTERED.md, 'the learning loop on the pair that matters'"

# The learning loop on nurse vs physician (studies/PREREGISTERED.md, final section). Stopped by
# the author after J0/L0/J1/L1/J2; see that section's Deviations. J0/L0 (the engines alone) are
# scored offline from committed fixtures, no keys, no network, no spend. J1/J2/L1 (140 labels,
# one steering round, the invariance gate on J2) are recorded to
# fixtures/bios_nurse/recordings/<arm>-seed<N>/ and can be replayed from there with no keys via
# `flywheel replay <recording-dir> --fixtures fixtures/bios_nurse`, one recording at a time.
nurse:
	@echo "Scoring Jev's and Laya's own answers (no labels, no fitted head) on the nurse/"
	@echo "physician held-out bios and their amended-rule gender-swapped, name-redacted twins."
	@echo ""
	.venv/bin/python scripts/run_bios_nurse_loop.py --arm J0
	.venv/bin/python scripts/run_bios_nurse_loop.py --arm L0
	@echo ""
	@echo "J1/J2/L1 need a steering round (Jev for J1/J2; Laya locally for L1) and are not part"
	@echo "of this offline target. L2 and the twin-averaging baseline were not run (stopped by"
	@echo "the author). Replay a committed recording with:"
	@echo "  flywheel replay fixtures/bios_nurse/recordings/<arm>-seed<N> --fixtures fixtures/bios_nurse"
	@echo "Next: studies/PREREGISTERED.md, 'the learning loop on nurse vs physician'"

test:
	.venv/bin/python -m pytest -q

# Re-render the diagrams. Needs d2 (https://d2lang.com); the SVGs are committed, so this is
# only for changing one.
diagrams:
	./scripts/render_diagrams.sh
