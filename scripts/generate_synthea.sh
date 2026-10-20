#!/usr/bin/env bash
# Generate the synthetic-track EHRs with Synthea, then build the compact profile table.
#
#   bash scripts/generate_synthea.sh
#
# Needs Java 17+ and internet access to GitHub. Raw FHIR output goes to
# data/synthetic/synthea_raw/ (git-ignored, large); the committed outputs are
# data/synthetic/synthetic_profiles.csv, data/synthetic/synthea_run.json and a few
# trimmed sample bundles in data/synthetic/fhir_sample/.
set -euo pipefail

SYNTHEA_VERSION="${SYNTHEA_VERSION:-v3.4.0}"
POPULATION="${SYNTHEA_POPULATION:-1000}"
SEED="${SYNTHEA_SEED:-42}"
REFERENCE_DATE="${SYNTHEA_REFERENCE_DATE:-20261001}"   # YYYYMMDD, fixed for reproducibility
WORK="${SYNTHEA_WORK:-data/synthetic/synthea_raw}"
JAR="$WORK/synthea-with-dependencies-${SYNTHEA_VERSION}.jar"
URL="https://github.com/synthetichealth/synthea/releases/download/${SYNTHEA_VERSION}/synthea-with-dependencies.jar"

mkdir -p "$WORK"
if [ ! -f "$JAR" ]; then
  echo "Downloading Synthea ${SYNTHEA_VERSION} ..."
  curl -fL --retry 3 -o "$JAR" "$URL"
fi

rm -rf "$WORK/output"
echo "Running Synthea: ${POPULATION} adults aged 25-75, seed ${SEED}, reference date ${REFERENCE_DATE} ..."
java -jar "$JAR" \
  -p "$POPULATION" -s "$SEED" -cs "$SEED" -r "$REFERENCE_DATE" -a 25-75 \
  --exporter.baseDirectory="$WORK/output" \
  --exporter.fhir.export=true \
  --exporter.csv.export=false \
  --exporter.hospital.fhir.export=false \
  --exporter.practitioner.fhir.export=false \
  --exporter.years_of_history=5 \
  > "$WORK/synthea.log" 2>&1

echo "Synthea finished; $(ls "$WORK/output/fhir" | wc -l) FHIR bundles written."

if [ "${SKIP_INGEST:-0}" != "1" ]; then
  python -m glucotwin.ingest_fhir \
    --fhir-dir "$WORK/output/fhir" \
    --out data/synthetic \
    --synthea-version "$SYNTHEA_VERSION" \
    --population "$POPULATION" --seed "$SEED" --reference-date "$REFERENCE_DATE"
fi
