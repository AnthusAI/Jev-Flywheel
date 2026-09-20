.PHONY: install demo test chart

install:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[steer,charts,dev]'

# The whole flywheel, replayed offline from the committed recording: no keys, no network,
# no model and no person. Prints the scorecard lineage and writes the figure.
demo:
	.venv/bin/flywheel -w var/demo replay fixtures/recordings/simulated-labeler --chart images/flywheel.png

test:
	.venv/bin/python -m pytest -q
