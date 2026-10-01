#!/bin/bash
# =============================================================================
# Fleet sync: reconcile every remote machine's repo to this one, exactly.
# =============================================================================
# mars-fias is the reference. Every other machine holds the SAME repo content plus
# only its own machine-local files. This script is the single supported way to
# propagate the repo, because doing it by hand went wrong in three distinct ways,
# and a fourth failure showed that the script itself must never report an
# unverified sync as a success:
#
#   1. `rsync a.py b.py doc.md host:REPO/` silently FLATTENS every source into
#      REPO/, so package modules landed in the repo root instead of the package
#      directory -- and a stage then ran stale code while the root held the new copy.
#      Here the file list is always relative to the repo root, so a file can only
#      ever land where it belongs.
#   2. `rsync --files-from=...` CANNOT delete (--delete needs directory recursion,
#      which --files-from disables), so remotes accumulated every file the reference
#      had ever contained. This script recurses per directory WITH --delete, so a
#      removal on the reference propagates.
#   3. Nothing excluded the secrets file, so RECOVERY_CODES.md was copied to three
#      shared HPC filesystems. It is now excluded by name, permanently.
#   4. Live syncs to JUPITER on 2026-09-30 printed success, but the files they
#      changed did not reach storage intact. After the sync of 10:50 UTC the script
#      printed "verified: 0 outstanding difference(s)", and a search through the
#      login node that had performed the writes found the new content. About six
#      hours later the compute nodes imported __init__.py and parameterization.py
#      as empty files (their bytecode caches record a source size of 0) and the
#      jobs failed at import; the next day another login node read every file of
#      that sync as empty. That the data stayed in the writing node's cache is
#      inferred, not observed. The sync of about 11:11 UTC printed "(already
#      identical)" after its itemized lines, which under `rsync ... | grep ... ||
#      echo "(already identical)"` means rsync had failed; its error text was
#      discarded. Whether the 10:50 rsync failed is undetermined. The check after
#      each transfer compared only size and modification time, through the writing
#      node, and printed "verified: 0" even when its own rsync failed. Two defects
#      hid failures that day. Now every ssh, rsync and check status is tested
#      explicitly; a live transfer counts only after a verification that compares
#      file CONTENT by checksum; a check that cannot run counts as a failure, never
#      as a pass (the secrets check reports absence only after the same remote
#      command has entered the repository directory); and the script ends with a
#      per-machine summary and a non-zero exit status if any machine failed. By
#      default the verification reads through the same ssh session as the
#      transfer, so on JUPITER and JUWELS through the same login node, and a
#      checksum read there would most likely have matched on 2026-09-30; the
#      output says so. VERIFY_VIA_<name>=<user@host> runs it through another host
#      that sees the same filesystem (for example another login node, reached
#      through its own authenticated session); the script does not check that this
#      host is a different node.
#
# Dry-run is the default, matching the other submitters: DRYRUN=1 prints what would
# change and transfers nothing. Set DRYRUN=0 only after reading the printed plan.
#
# Usage:
#   bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_Fleet_Sync.sh [machine ...]
#   DRYRUN=0 bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_Fleet_Sync.sh jupiter
#   DRYRUN=0 VERIFY_VIA_jupiter=<user>@<another login node> \
#       bash Script_Bank/HPC/SRM_AND_SBI_MONOMER_DIMER_ALP_Fleet_Sync.sh jupiter
#   (no machine argument = every machine listed below)
# Exit status: 0 only if every selected machine was reached and, in a live run, its
# transfer succeeded, its content verified, and its secrets check ran and passed;
# 1 otherwise. Tests: tests/test_fleet_sync.py (local stand-ins via FLEET_FILE).
# -----------------------------------------------------------------------------
set -uo pipefail   # no -e: every status below is tested explicitly, so none is masked

DRYRUN="${DRYRUN:-1}"
CP="$HOME/.ssh/control-%h-%p-%r"
SSH_JSC="ssh -o ControlPath=$CP -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=15"
SSH_PLAIN="ssh -o BatchMode=yes -o ConnectTimeout=15"

# name | ssh command | host | absolute repo path on that machine
FLEET=(
  "jupiter|$SSH_JSC|ramirezsierra1@login.jupiter.fz-juelich.de|/e/project1/chkf10/ramirezsierra1/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-monomer-dimer-alp"
  "juwels|$SSH_JSC|ramirezsierra1@juwels-cluster.fz-juelich.de|/p/project1/chkf10/ramirezsierra1/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-monomer-dimer-alp"
  "goethe|$SSH_PLAIN|ramirez@goethe.hhlr-gu.de|/home/biochemsim/ramirez/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-monomer-dimer-alp"
  "rcl01|$SSH_PLAIN|rcl_fias@10.83.255.103|/home/rcl_fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-monomer-dimer-alp"
)

# Test hook: FLEET_FILE=<file> replaces the table above, one "name|ssh command|host|repo"
# line per machine (blank lines and lines starting with # are ignored).
if [ -n "${FLEET_FILE:-}" ]; then
    FLEET=()
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        FLEET+=( "$line" )
    done < "$FLEET_FILE"
fi

# NEVER leaves the reference machine. RECOVERY_CODES.md holds live recovery codes;
# it is gitignored so it never reaches GitHub, and excluded here so it never reaches
# a shared filesystem either.
SECRETS=( "RECOVERY_CODES.md" )

# Legitimately per-machine: each remote keeps its own and the sync must not touch,
# overwrite, or delete them.
MACHINE_LOCAL=( "machine_profiles.toml" "Script_Bank/HPC/hpc_local.env" )

# Not content: build artifacts, caches, editor scratch, and the data tree (which
# lives outside the repo on the reference but has appeared inside it on a remote).
NOT_CONTENT=( ".git" "__pycache__" "*.pyc" "*.egg-info" ".ipynb_checkpoints"
              ".virtual_documents" "Data_Bank" )

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot enter $REPO_ROOT" >&2; exit 1; }
[ -f pyproject.toml ] && [ -d srm_and_sbi_monomer_dimer_alp ] || {
    echo "FATAL: $REPO_ROOT is not the repo root." >&2; exit 1; }

EXCLUDES=()
for p in "${SECRETS[@]}" "${MACHINE_LOCAL[@]}" "${NOT_CONTENT[@]}"; do
    EXCLUDES+=( --exclude="$p" )
done

# Sync the repo as a recursive tree so --delete can remove what the reference no
# longer has. Excluded paths are protected from deletion too (--delete-excluded is
# deliberately NOT used), so machine-local files and the data tree survive.
RSYNC_OPTS=( -rlpt --delete --itemize-changes "${EXCLUDES[@]}" )
[ "$DRYRUN" = "0" ] || RSYNC_OPTS+=( --dry-run )
# The verification: a dry run that compares CONTENT (checksums), not size and time.
VERIFY_OPTS=( -rlpt --checksum --delete --dry-run --itemize-changes "${EXCLUDES[@]}" )

ITEMIZED='^([<>ch.*][fdLDS]|\*deleting)'   # any itemized line
CHANGES='^(\*deleting|[<>]f|cd\+)'         # the changes printed for the plan
CONTENT='^(\*deleting|[<>]f)'              # a file whose content (or presence) differs

WANT=("$@")
echo "======================================================================"
echo " Fleet sync from $REPO_ROOT"
echo " mode: $([ "$DRYRUN" = "0" ] && echo 'LIVE (transferring)' || echo 'DRY RUN (set DRYRUN=0 to apply)')"
echo " excluded: secrets(${#SECRETS[@]}) machine-local(${#MACHINE_LOCAL[@]}) non-content(${#NOT_CONTENT[@]})"
echo "======================================================================"

OK_MACHINES=()
FAILED_MACHINES=()
SAME_SESSION=()
fail() {           # record and print a failure of the current machine
    echo "     FAILED: $*"
    FAILED_MACHINES+=( "$NAME: $*" )
}
messages() {       # rsync's or ssh's own messages, verbatim: every line that is not an itemized change
    printf '%s\n' "$1" | grep -vE "$ITEMIZED" | grep -v '^[[:space:]]*$' | sed 's/^/     > /'
    return 0
}

for entry in "${FLEET[@]}"; do
    IFS='|' read -r NAME SSHC HOST REPO <<< "$entry"
    if [ ${#WANT[@]} -gt 0 ]; then
        printf '%s\n' "${WANT[@]}" | grep -qxF -- "$NAME" || continue
    fi
    case "$NAME" in
        ''|*[!A-Za-z0-9_]*) echo; echo "---- $NAME"
                            fail "the machine name is not usable in VERIFY_VIA_<name> (letters, digits and _ only)"; continue ;;
    esac
    echo
    echo "---- $NAME : $HOST:$REPO"

    reach=$(timeout 30 $SSHC "$HOST" "test -d '$REPO/srm_and_sbi_monomer_dimer_alp'" 2>&1); rc=$?
    if [ "$rc" -ne 0 ]; then
        messages "$reach"
        fail "unreachable or repo missing (status $rc); nothing transferred"
        continue
    fi

    out=$(rsync "${RSYNC_OPTS[@]}" -e "$SSHC" ./ "$HOST:$REPO/" 2>&1); rc=$?
    changes=$(printf '%s\n' "$out" | grep -E "$CHANGES")
    [ -z "$changes" ] || printf '%s\n' "$changes" | sed 's/^/     /'
    if [ "$rc" -ne 0 ]; then
        messages "$out"
        fail "rsync exited with status $rc$([ "$DRYRUN" = "0" ] && echo '; the remote may be partially updated')"
        continue
    fi
    [ -n "$changes" ] || echo "     (already identical by size and time)"

    if [ "$DRYRUN" = "0" ]; then
        via="VERIFY_VIA_${NAME}"
        VHOST="${!via:-$HOST}"
        vout=$(rsync "${VERIFY_OPTS[@]}" -e "$SSHC" ./ "$VHOST:$REPO/" 2>&1); rc=$?
        if [ "$rc" -ne 0 ]; then
            messages "$vout"
            fail "the verification through $VHOST could not run (rsync status $rc)"
            continue
        fi
        differs=$(printf '%s\n' "$vout" | grep -E "$CONTENT")
        if [ -n "$differs" ]; then
            printf '%s\n' "$differs" | sed 's/^/     differs: /'
            fail "$(printf '%s\n' "$differs" | wc -l) content difference(s) remain after the transfer (read through $VHOST)"
            continue
        fi
        echo "     verified: 0 content differences (checksums, read through $VHOST)"
        if [ "$VHOST" = "$HOST" ]; then
            echo "     note: read back through the session that wrote the files; set $via=<user@host> to read through another node"
            SAME_SESSION+=( "$NAME" )
        fi

        # Absence counts only when the same remote command first proved it can read the repo directory.
        sec=$(timeout 30 $SSHC "$HOST" "cd '$REPO' && test -d srm_and_sbi_monomer_dimer_alp || exit 3; if test -e RECOVERY_CODES.md || test -L RECOVERY_CODES.md; then echo RECOVERY_CODES.md:present; else echo RECOVERY_CODES.md:absent; fi" 2>&1); rc=$?
        if [ "$rc" -eq 0 ] && [[ $'\n'"$sec"$'\n' == *$'\nRECOVERY_CODES.md:present\n'* ]]; then
            fail "the secrets file RECOVERY_CODES.md is present on $NAME"; continue
        elif [ "$rc" -eq 0 ] && [[ $'\n'"$sec"$'\n' == *$'\nRECOVERY_CODES.md:absent\n'* ]]; then
            echo "     secrets check: RECOVERY_CODES.md absent  OK"
        else
            messages "$sec"
            fail "the secrets check could not run (status $rc)"; continue
        fi
    fi
    OK_MACHINES+=( "$NAME" )
done

if [ $(( ${#OK_MACHINES[@]} + ${#FAILED_MACHINES[@]} )) -gt 0 ]; then
    for w in "${WANT[@]}"; do
        known=""
        for entry in "${FLEET[@]}"; do [ "${entry%%|*}" = "$w" ] && known=1; done
        [ -n "$known" ] || FAILED_MACHINES+=( "$w: no such machine in the table" )
    done
fi
echo
echo "======================================================================"
[ "$DRYRUN" = "0" ] || echo " DRY RUN: nothing was transferred."
if [ ${#OK_MACHINES[@]} -gt 0 ]; then
    echo " $([ "$DRYRUN" = "0" ] && echo 'synced and verified' || echo 'planned without error'): ${OK_MACHINES[*]}"
fi
[ ${#SAME_SESSION[@]} -eq 0 ] || echo " read back only through the session that wrote the files (this does not detect data held only in that node's cache, as on 2026-09-30): ${SAME_SESSION[*]}"
for f in "${FAILED_MACHINES[@]}"; do echo " FAILED $f"; done
if [ ${#OK_MACHINES[@]} -eq 0 ] && [ ${#FAILED_MACHINES[@]} -eq 0 ]; then
    echo " FAILED: no machine matched: ${WANT[*]:-(none listed)}"
    echo "======================================================================"
    exit 1
fi
echo "======================================================================"
[ ${#FAILED_MACHINES[@]} -eq 0 ] || exit 1
exit 0
