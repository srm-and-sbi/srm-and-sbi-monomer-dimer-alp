#!/usr/bin/env bash
# =============================================================================
# Bulk deletion on a parallel filesystem (GPFS / Lustre): frees a data tier that has
# hit its INODE quota. Deletes the CONTENTS of each given directory with P parallel
# streams and keeps the directory itself -- the machine profiles require
# data_bank_root / scratch_data_bank_root to exist, so a tier is emptied, never removed.
#
# Why this exists (2026-09-16, JUWELS $SCRATCH): "Disk quota exceeded" with 80 TB free
# meant the FILE-COUNT quota (4.26 M used, 4.0 M soft / 4.4 M hard), held by two retired
# TRAIN/TEST tiers. A single `find -delete` on the login node ran ~30 K files/min
# (metadata-bound; ~2 h for 3.75 M files); 16 parallel `rm -rf` streams over the
# top-level entries ran ~400 K files/min and finished in 11 min at a login-node load of
# ~20. Documented here so the method is not re-derived per campaign
# (Script_Bank/HPC/README.md, section 6).
#
# Dry-run is the default: prints the top-level entry count per directory and deletes
# nothing. DRYRUN=0 deletes. Detach the real run -- never wrap it in a timeout:
#   DRYRUN=0 nohup bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Bulk_Delete.sh DIR... \
#       > ~/bulk_delete_$(date +%F).log 2>&1 < /dev/null &
#
# Safety: every path must lie under a "Data_Bank" tree and outside the legacy read-only
# trees (RCL_Projects); the script refuses otherwise, before touching anything. It is
# idempotent -- re-running after an interruption finishes the job. Delete only the
# regenerable TRAIN/TEST scratch tier (videos, theta sets, READY_TRACT trajectories);
# EVAL, Posit, Labor and Experiment live on the permanent tier and are never targets.
#
# Usage:
#   [DRYRUN=1] [P=16] bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_HPC_Bulk_Delete.sh DIR [DIR ...]
# =============================================================================
set -euo pipefail
DRYRUN="${DRYRUN:-1}"
P="${P:-16}"
[ $# -ge 1 ] || { echo "usage: [DRYRUN=0] [P=16] $0 DIR [DIR ...]" >&2; exit 2; }
for D in "$@"; do
  case "$D" in */Data_Bank/*|*/Data_Bank) ;; *) echo "REFUSED (not under a Data_Bank tree): $D" >&2; exit 3 ;; esac
  case "$D" in */RCL_Projects/*) echo "REFUSED (legacy read-only tree): $D" >&2; exit 3 ;; esac
  [ -d "$D" ] || { echo "REFUSED (not a directory): $D" >&2; exit 3; }
done
echo "START $(date -u +%FT%TZ)  DRYRUN=$DRYRUN  streams=$P"
for D in "$@"; do
  n_top=$(find "$D" -mindepth 1 -maxdepth 1 | wc -l)
  if [ "$DRYRUN" != "0" ]; then
    echo "DRY-RUN: would delete the contents of $D  ($n_top top-level entries; directory kept)"
    continue
  fi
  echo "deleting the contents of $D  ($n_top top-level entries) with $P streams ... $(date -u +%FT%TZ)"
  # READY_TRACT (the trajectory folder, one directory per task) is expanded one level so its
  # hundreds of thousands of files spread across the streams instead of landing in one rm.
  { find "$D" -mindepth 1 -maxdepth 1 ! -name READY_TRACT -print0
    if [ -d "$D/READY_TRACT" ]; then find "$D/READY_TRACT" -mindepth 1 -maxdepth 1 -print0; fi; } \
    | xargs -0 -r -P "$P" -n 20 rm -rf
  if [ -d "$D/READY_TRACT" ]; then rmdir "$D/READY_TRACT" 2>/dev/null || true; fi
  echo "  done: remaining entries $(find "$D" -mindepth 1 | wc -l)  $(date -u +%FT%TZ)"
done
echo "END $(date -u +%FT%TZ)"
