#!/usr/bin/env bash
# Download CGMacros v1.0.0 from PhysioNet (open access, CC BY-NC-SA 4.0) into data/raw/cgmacros/.
#
#   bash scripts/download_cgmacros.sh [destination]
#
# The data stays outside Git (data/raw is git-ignored) and is never redistributed by this
# repository. Read data/README.md for the licence terms and the required citations.
set -euo pipefail

DEST="${1:-data/raw/cgmacros}"
URL="https://physionet.org/files/cgmacros/1.0.0/"
ZIP="CGMacros_dateshifted365.zip"
mkdir -p "$DEST"

echo "CGMacros is licensed CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike)."
if command -v wget >/dev/null 2>&1; then
  wget -r -N -c -np -nH --cut-dirs=3 --reject "index.html*" -P "$DEST" "$URL"
else
  echo "wget not found; using curl for the data files ..."
  for f in "$ZIP" DataDictionary.pdf DataDictionary_Bio.csv DataDictionary_CGMacros-00X.csv SHA256SUMS.txt; do
    curl -fL --retry 3 -o "$DEST/$f" "${URL}${f}"
  done
fi

cd "$DEST"
if [ -f SHA256SUMS.txt ] && [ -f "$ZIP" ]; then
  expected=$(grep "$ZIP" SHA256SUMS.txt | awk '{print $1}')
  actual=$( (sha256sum "$ZIP" 2>/dev/null || shasum -a 256 "$ZIP") | awk '{print $1}')
  if [ "$expected" = "$actual" ]; then echo "Checksum OK"; else echo "WARNING: checksum mismatch - download again before analysis"; exit 1; fi
fi
if [ -f "$ZIP" ]; then unzip -o -q "$ZIP"; fi
echo "Done. Next: python scripts/run_pipeline.py"
