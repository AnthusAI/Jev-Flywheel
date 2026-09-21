#!/usr/bin/env bash
# Render every diagrams/*.d2 to images/*.svg.
#
# The SVGs are committed, so reading or cloning this repo needs no diagram tooling at all.
# You only need d2 (https://d2lang.com) to change a diagram.
#
# --dark-theme embeds a prefers-color-scheme media query, so ONE file serves GitHub's light
# and dark modes. That only works if the .d2 source sets no explicit colours, which is why
# the sources style by shape and let the theme choose every colour.
set -euo pipefail

command -v d2 >/dev/null || { echo "d2 not found: see https://d2lang.com/tour/install"; exit 1; }

cd "$(dirname "$0")/.."
mkdir -p images
for src in diagrams/*.d2; do
  out="images/$(basename "${src%.d2}").svg"
  engine=elk; case "$(basename "$src")" in flywheel.d2|economics.d2) engine=dagre;; esac
  d2 --layout "${D2_LAYOUT:-$engine}" --theme 0 --dark-theme 200 --pad 30 "$src" "$out" >/dev/null
  printf '  %-34s -> %s\n' "$src" "$out"
done
echo "done. The SVGs are committed; d2 is only needed to change one."
