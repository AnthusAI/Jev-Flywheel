.DEFAULT_GOAL := help
.PHONY: help install demo laya student finetune bios race race2 test diagrams

help:
	@echo "Jev Flywheel: three things you can run. Nothing here needs a Jev key."
	@echo ""
	@echo "  make install   set up a virtualenv (once)"
	@echo "  make demo      1. the flywheel with Jev, replayed offline from a recording (~10 s)"
	@echo "  make laya      2. the same recording answered by a local Laya model (downloads ~843 MB)"
	@echo "  make student   3. distil the result into a small local BERT classifier (downloads ~1.2 GB in all)"
	@echo "  make finetune  4. does gradient fine-tuning of Laya itself beat the fitted head? (one seed, quick)"
	@echo "  make bios      5. does the engine read gender? (surgeon/physician bios, J0/L0, offline)"
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
# study, scored from committed answer fixtures: no keys, no network, no model, no spend. The
# J1/J2/L1/L2/LF arms in `studies/PREREGISTERED.md` need a working Bedrock analyst (blocked in
# the environment this study ran in -- see the pre-registration's Outcome section) and are not
# part of this target.
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
	@echo "Next: studies/PREREGISTERED.md, 'does the engine read gender, and can the layer refuse to?'"

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

test:
	.venv/bin/python -m pytest -q

# Re-render the diagrams. Needs d2 (https://d2lang.com); the SVGs are committed, so this is
# only for changing one.
diagrams:
	./scripts/render_diagrams.sh
