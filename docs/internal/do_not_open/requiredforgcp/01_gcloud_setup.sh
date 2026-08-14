#!/usr/bin/env bash
# STEP 1 (run on YOUR computer): point gcloud at your project and enable APIs.
source "$(dirname "$0")/config.sh"

echo ">> Setting active project"
gcloud config set project "$PROJECT_ID"

echo ">> Enabling required APIs (compute + storage)"
gcloud services enable compute.googleapis.com storage.googleapis.com

cat <<'MSG'

============================================================
 NEXT, TWO THINGS YOU MUST DO IN THE WEB CONSOLE:
============================================================
 1) BUDGET ALERT (safety net):
    Billing -> Budgets & alerts -> Create budget -> e.g. $50,
    alerts at 50/90/100%.

 2) GPU QUOTA (this blocks everything until granted, do it now):
    IAM & Admin -> Quotas -> filter "GPUs (all regions)"
    -> Edit Quotas -> request 1 -> submit.
    Wait for the approval email (a few hours, sometimes a day).
============================================================
MSG
