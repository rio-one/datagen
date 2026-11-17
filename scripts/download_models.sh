#!/usr/bin/env bash
# Failed/blocked downloads are recorded and skipped. No authentication or global configuration.
set -uo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$DATAGEN_ROOT/models" "$DATAGEN_ROOT/logs"
STATUS="$DATAGEN_ROOT/logs/model-download-status.tsv"
touch "$STATUS"
fetch() {
  local filename="$1" url="$2"
  if [[ -s "$DATAGEN_ROOT/models/$filename" ]]; then
    printf '%s\tPRESENT_not_rehashed\t%s\n' "$filename" "$url" >> "$STATUS"
    return
  fi
  if curl --fail --location --retry 0 --connect-timeout 10 --max-time 120 --speed-limit 102400 --speed-time 30 --output "$DATAGEN_ROOT/models/$filename.part" "$url"; then
    mv -- "$DATAGEN_ROOT/models/$filename.part" "$DATAGEN_ROOT/models/$filename"
    printf '%s\tDOWNLOADED\t%s\n' "$filename" "$url" >> "$STATUS"
  else
    printf '%s\tMANUAL_DOWNLOAD_REQUIRED_partial_file_not_used\t%s\n' "$filename" "$url" >> "$STATUS"
    printf '\nSkipped %s; see MANUAL_TODO.md\n' "$filename"
  fi
}
fetch sam2.1_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
fetch omnidata_dpt_depth_v2.ckpt https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt
fetch omnidata_dpt_normal_v2.ckpt https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt
