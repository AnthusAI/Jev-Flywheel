.PHONY: install demo test diagrams

install:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[steer,charts,dev]'

# The whole flywheel, replayed offline from the committed recording: no keys, no network,
# no model and no person. Prints the scorecard lineage and writes the figure.
demo:
	.venv/bin/flywheel -w var/demo replay fixtures/recordings/simulated-labeler --chart images/results.png

test:
	.venv/bin/python -m pytest -q

# Re-render the diagrams. Needs d2 (https://d2lang.com); the SVGs are committed, so this is
# only for changing one.
diagrams:
	./scripts/render_diagrams.sh
