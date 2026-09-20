# A simulated labeler, one steering round

**The labeler here is simulated, not a person.** It answers with the corpus's own reference label and explains each disagreement with a fixed template ("This is really negative; the wording is weak and it misleads."). It cannot notice a factor nobody declared, so this recording demonstrates the machinery and the measurement, not the claim that human comments surface hidden factors. The analyst is Kimi K3 on AWS Bedrock; the answers are from Jev. Replace it with your own: run `flywheel label`, then `flywheel record`.

140 labels, 22 recorded steps. Replay with:

    flywheel replay fixtures/recordings/simulated-labeler
