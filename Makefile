.DEFAULT_GOAL := help
.PHONY: help install demo laya student test diagrams

help:
	@echo "Jev Flywheel: three things you can run. Nothing here needs a Jev key."
	@echo ""
	@echo "  make install   set up a virtualenv (once)"
	@echo "  make demo      1. the flywheel with Jev, replayed offline from a recording (~10 s)"
	@echo "  make laya      2. the same recording answered by a local Laya model (downloads ~843 MB)"
	@echo "  make student   3. distil the result into a small local BERT classifier (downloads ~1.2 GB in all)"
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

test:
	.venv/bin/python -m pytest -q

# Re-render the diagrams. Needs d2 (https://d2lang.com); the SVGs are committed, so this is
# only for changing one.
diagrams:
	./scripts/render_diagrams.sh
