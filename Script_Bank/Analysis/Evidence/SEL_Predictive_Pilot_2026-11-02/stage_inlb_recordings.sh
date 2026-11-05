#!/bin/bash
# Copy five MET-INLB recordings (60-cell catalog indexing) + the catalog note from the JUWELS project tier to the
# PC data bank's Experiment tier, verifying SHA-256 on both ends. Needs the lead's open JUWELS ControlMaster.
set -u
CELLS="${CELLS:-53 43 11 42 22}"
DST=/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Experiment/SPT_Data_MET_FAB_INLB_S-BSST712
HOST=juwels-cluster
ssh -o BatchMode=yes "$HOST" true || { echo "no live JUWELS master; open it first"; exit 75; }
SRC=$(ssh "$HOST" 'find /p/project1/chkf10/$USER -maxdepth 9 -name Experiment_INLB_Cell_11_20S_RAW.tif -path "*monomer-dimer-alp*" 2>/dev/null | head -1 | xargs -r dirname')
[ -n "$SRC" ] || { echo "InlB recordings not found on JUWELS"; exit 1; }
echo "source dir: $SRC"
files=""; for c in $CELLS; do files="$files Experiment_INLB_Cell_${c}_20S_RAW.tif"; done
files="$files Catalog_Note_Experiment.tsv"
ssh "$HOST" "cd '$SRC' && sha256sum $files" > "$DST/INLB_STAGED_FROM_JUWELS.sha256" || exit 1
for f in $files; do
  [ -e "$DST/$f" ] && { echo "exists, not overwritten: $f"; continue; }
  scp -q "$HOST:$SRC/$f" "$DST/$f" || exit 1
done
(cd "$DST" && sha256sum -c INLB_STAGED_FROM_JUWELS.sha256) && echo "INLB STAGING VERIFIED" || { echo "CHECKSUM MISMATCH"; exit 1; }
