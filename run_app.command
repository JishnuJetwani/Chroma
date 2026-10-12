#!/bin/zsh
set -eu
cd "${0:A:h}"
checkpoint=""
for candidate in runs/multistyle_pilot/best.pt runs/local_pilot/best.pt runs/global_pilot/best.pt; do
  if [[ -f "$candidate" ]]; then checkpoint="$candidate"; break; fi
done
if [[ -z "$checkpoint" ]]; then
  print "Train a model before opening the grading app."
  exit 1
fi
exec uv run --locked --extra app photograder app --checkpoint "$checkpoint"
