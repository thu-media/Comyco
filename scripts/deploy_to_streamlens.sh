#!/bin/bash
# Install trained weights into a StreamLens checkout that already has the Comyco ABR (src/comyco.py).
#   scripts/deploy_to_streamlens.sh models/step05/nn_model_ep_950.npz [~/git/StreamLens]
set -euo pipefail
W="${1:?weights .npz}"
SL="${2:-$HOME/git/StreamLens}"
cp "$W" "$SL/src/comyco_weights.npz"
echo "installed $(basename "$W") -> $SL/src/comyco_weights.npz"
