#!/usr/bin/env bash
# STEP 3 (run on YOUR computer): create the GPU VM and copy the deploy files onto it.
source "$(dirname "$0")/config.sh"

MACHINE_TYPE="n1-standard-8"          # 8 vCPU / 30 GB RAM — enough for CPU models + Qdrant
GPU="type=nvidia-tesla-t4,count=1"    # cheapest GPU; plenty for one 4-bit 4B model
DISK_GB=150

# Deep Learning image = NVIDIA driver + Docker already installed (saves ~1h of setup).
# Google retires/renames these families over time, so DISCOVER the newest CUDA
# family that actually exists right now instead of hard-coding names that go stale.
IMAGE_PROJECT="deeplearning-platform-release"
IMAGE_FAMILY="${IMAGE_FAMILY:-$(gcloud compute images list --project="$IMAGE_PROJECT" \
  --filter="family ~ ^common-cu" --format="value(family)" 2>/dev/null | sort -Vur | head -n1)}"

if [[ -z "$IMAGE_FAMILY" ]]; then
  echo "ERROR: No common-cu* Deep Learning VM image family found in $IMAGE_PROJECT."
  echo "List them yourself and set IMAGE_FAMILY=... manually:"
  echo "  gcloud compute images list --project=$IMAGE_PROJECT --filter='family~common-cu' --format='value(family)'"
  exit 1
fi
echo ">> Using Deep Learning image family: $IMAGE_FAMILY"

# ---- OPTIONAL: cheaper Spot VM (~60-70% off, but can be shut down anytime) ----
# Uncomment these two lines for dev/demo runs you can afford to have interrupted:
# SPOT="--provisioning-model=SPOT --instance-termination-action=STOP"
SPOT=""

echo ">> Creating VM $VM_NAME in $ZONE ..."
gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="$GPU" \
  --maintenance-policy=TERMINATE \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${DISK_GB}GB" \
  --boot-disk-type=pd-balanced \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform \
  --tags=farmervision \
  $SPOT

echo ">> Waiting for SSH to come up ..."
until gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="echo ok" >/dev/null 2>&1; do
  sleep 8; echo "   ...still booting"
done

echo ">> Copying deploy files to the VM (~/farmervision)"
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="mkdir -p ~/farmervision"
gcloud compute scp --recurse "$DEPLOY_DIR"/* "$VM_NAME":~/farmervision/ --zone="$ZONE"
gcloud compute scp "$SCRIPT_DIR/.env" "$VM_NAME":~/farmervision/.env --zone="$ZONE"

cat <<MSG

============================================================
 VM is up. Now finish setup ON THE VM:

   gcloud compute ssh $VM_NAME --zone=$ZONE
   cd ~/farmervision && bash 04_vm_setup.sh

 Remember: STOP the VM when you're done (bash stop_vm.sh).
============================================================
MSG
