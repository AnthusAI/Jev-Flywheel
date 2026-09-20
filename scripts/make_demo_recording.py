#!/usr/bin/env python
"""Make the committed demo recording from a live run.

Uses real Kimi K3 on Bedrock as the analyst and real Jev for the top-ups, with a *simulated*
labeler (the corpus's reference labels and templated comments). Costs a few cents. Run it
only to regenerate fixtures/recordings/; everyone else replays the committed one offline.

    python scripts/make_demo_recording.py --labels 140

Credentials: TYPESAFE_API_KEY (Jev) from the environment or a .env file, and AWS credentials
for Bedrock (`aws login` needs botocore[crt]). The key is never printed.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from jev_flywheel import recording
from jev_flywheel.cli import PACKAGED_FIXTURES
from jev_flywheel.jev import JevSession
from jev_flywheel.simulate import label_with_reference
from jev_flywheel.steer import ScriptedApprover, run_steering
from jev_flywheel.workspace import Workspace
import asyncio, random

PROVENANCE = (
    "**The labeler here is simulated, not a person.** It answers with the corpus's own "
    "reference label and explains each disagreement with a fixed template "
    "(\"This is really negative; the wording is weak and it misleads.\"). It cannot notice a "
    "factor nobody declared, so this recording demonstrates the machinery and the measurement, "
    "not the claim that human comments surface hidden factors. The analyst is Kimi K3 on AWS "
    "Bedrock; the answers are from Jev. Replace it with your own: run `flywheel label`, then "
    "`flywheel record`.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=int, default=140)
    parser.add_argument("--test-sample", type=int, default=600)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--env", type=Path, default=None, help="a .env file with TYPESAFE_API_KEY")
    parser.add_argument("--out", type=Path, default=Path("fixtures/recordings/simulated-labeler"))
    args = parser.parse_args()
    load_dotenv(args.env)          # python-dotenv, so the value is never echoed

    work = Path(tempfile.mkdtemp(prefix="flywheel-demo-"))
    workspace = Workspace.init(work / "var", PACKAGED_FIXTURES)
    score = workspace.scorecard().scores[0].name
    label_with_reference(workspace, score, args.labels, seed=args.seed)
    print(f"{workspace.n_labeled(score)} labels; scorecard v{workspace.version}")

    outcome = run_steering(workspace, score, allow_spend=True,
                           hitl_handler=ScriptedApprover(default=True))
    print("steering:", outcome.decision, outcome.detail.get("version"))

    # Score the steered scorecard on a fixed sample of the held-out split.
    sample = random.Random(0).sample(workspace.split("test"), args.test_sample)
    session = JevSession()
    report = asyncio.run(workspace.cache.fill(session, sample, workspace.scorecard().questions()))
    print(f"held-out top-up: {report.requested} requests, {report.failures} failed")

    if args.out.exists():
        shutil.rmtree(args.out)
    recording.record(workspace, score, args.out, PACKAGED_FIXTURES,
                     title="A simulated labeler, one steering round", provenance=PROVENANCE)
    print("recorded to", args.out)

if __name__ == "__main__":
    main()
