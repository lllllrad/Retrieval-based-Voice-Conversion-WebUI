#!/usr/bin/env bash
# Convert every input file with every model in assets/weights at several pitch shifts.
#
# Usage: compare/run_batch.sh INPUT_DIR [PITCHES...]
#   INPUT_DIR  folder with source audio (wav/flac/mp3...)
#   PITCHES    semitone shifts to try (default: 0)
#              e.g. male speaker -> female models: 0 6 12
#
# Output: compare/out/<input-name>/<model>_p<pitch>.wav
# Env:    MODELS="kr_iu kr_rose"  limit to these models (default: all kr_*)
#         INDEX_RATE=0.75 PROTECT=0.33 F0=rmvpe
set -euo pipefail
cd "$(dirname "$0")/.."

in_dir=${1:?usage: compare/run_batch.sh INPUT_DIR [PITCHES...]}
shift
pitches=("${@:-0}")
[ ${#pitches[@]} -eq 0 ] && pitches=(0)
models=${MODELS:-$(cd assets/weights && ls kr_*.pth | sed 's/\.pth$//')}
index_rate=${INDEX_RATE:-0.75}
protect=${PROTECT:-0.33}
f0=${F0:-rmvpe}

shopt -s nullglob nocaseglob
inputs=("$in_dir"/*.{wav,flac,mp3,m4a,ogg})
[ ${#inputs[@]} -eq 0 ] && { echo "no audio files in $in_dir" >&2; exit 1; }

fails=0
for input in "${inputs[@]}"; do
  name=$(basename "${input%.*}")
  out_dir=compare/out/$name
  mkdir -p "$out_dir"
  for m in $models; do
    idx=assets/indices/$m.index
    if [ -f "$idx" ]; then idx_args=(--index "$idx" --index-rate "$index_rate")
    else idx_args=(--index-rate 0); fi
    for p in "${pitches[@]}"; do
      out=$out_dir/${m}_p${p}.wav
      [ -f "$out" ] && { echo "skip $out"; continue; }
      echo ">> $name | $m | pitch $p"
      if ! python -m infer.cli --model "$m.pth" --input "$input" --output "$out" \
          --pitch "$p" --f0-method "$f0" --protect "$protect" "${idx_args[@]}" \
          > "$out_dir/.${m}_p${p}.log" 2>&1; then
        echo "   FAILED (see $out_dir/.${m}_p${p}.log)"; fails=$((fails+1))
      fi
    done
  done
done
echo "done. failures: $fails"
