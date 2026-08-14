#!/usr/bin/env bash
# STEP 2 (run on YOUR computer): create the bucket and upload your models.
source "$(dirname "$0")/config.sh"

LOCAL_ARTIFACTS="${LOCAL_ARTIFACTS:-./artifacts}"

if [[ ! -d "$LOCAL_ARTIFACTS" ]]; then
  echo "ERROR: '$LOCAL_ARTIFACTS' not found. Put your files there first (plan Step 3)."
  exit 1
fi

echo ">> Creating bucket $BUCKET (ok if it already exists)"
gcloud storage buckets create "$BUCKET" --location="$REGION" 2>/dev/null || true

echo ">> Uploading $LOCAL_ARTIFACTS -> $BUCKET  (this can take a while for the 3.8 GB snapshot)"
gcloud storage cp --recursive "$LOCAL_ARTIFACTS"/* "$BUCKET"/

echo ">> Done. Contents of the bucket:"
gcloud storage ls --recursive "$BUCKET"/ | sed 's/^/   /'
