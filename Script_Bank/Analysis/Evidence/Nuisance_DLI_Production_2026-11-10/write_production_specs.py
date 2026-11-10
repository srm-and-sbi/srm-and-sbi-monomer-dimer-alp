"""Write the production Nuisance_DLI specifications of the biology workflow (canonical artifact, no tag) into the Posit
directory of the active machine profile, from the reviewed proposal specifications (tag PROPOSALCW4): the components,
the comparison scope, the rank coordinates, the candidate archive with its checksum and the two override tables are
carried verbatim (asserted); only the header comment and the rationale's status sentences change. Refuses to overwrite.

    MACHINE_PROFILE=mars_pc PYTHONPATH=$PWD python Script_Bank/Analysis/Evidence/Nuisance_DLI_Production_2026-11-10/write_production_specs.py
"""
import re
import sys
from pathlib import Path

REVIEW = Path("/home/mars-fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/Special_Analyses/"
              "NUISANCE_DLI_PRODUCTION_PROPOSAL_2026-11-09/WEIGHTING_REVISION_2026-11-10")
POSIT = Path("/media/mars-fias/MARS_Data_Reserve_0/MARS_Data_Bank/Projects/RCL_Agent_Projects/srm-and-sbi-monomer-dimer-alp/Data_Bank/Posit")
HEADER = (
    "# Production Nuisance_DLI spec of the biology workflow (adopted by the lead on 2026-11-10 after review; the canonical\n"
    "# artifact, no tag): one SGM per cell and 2 s window over all ten windows, posterior_rank_median over mu_r, sigma_r, mu_pc,\n"
    "# sigma_pc with rank_scope construction (every representative against the other 599, normalized over all 600, no\n"
    "# reserved cell mass); bleaching and flicker overridden (reasons below). The earlier selections RANK4 and SEL and their\n"
    "# decision records (DETECTOR_WORKFLOW.md sec. 7.6) are unchanged historical records.\n")
OLD_HEAD = "Review proposal PROPOSALCW4 (2026-11-10; NOT adopted, built for review only): "
NEW_HEAD = "Production selection of the biology workflow (adopted by the lead on 2026-11-10 after review; canonical artifact, no tag): "
OLD_TAIL = ("It revises the scope of the adopted RANK4 selection (within-cell ranks with equal cell mass), which stays unchanged under "
            "its own tag. ")
NEW_TAIL = ("The earlier selection RANK4 (within-cell ranks with equal cell mass) stays unchanged under its own tag as a historical "
            "record. ")


def main():
    for cond in ("FAB", "INLB"):
        alias = f"SRM_AND_SBI_MONOMER_DIMER_ALP_DETECTOR_{cond}"
        src = REVIEW / cond / f"{alias}_2S_50FPS_PROPOSALCW4_Nuisance_DLI_Spec.toml"
        dst = POSIT / f"{alias}_2S_50FPS_Nuisance_DLI_Spec.toml"
        if dst.exists() or (POSIT / f"{alias}_2S_50FPS_Nuisance_DLI.npz").exists():
            raise SystemExit(f"refusing to overwrite {dst} or its artifact")
        text = src.read_text()
        body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#")).lstrip("\n")
        assert body.count(OLD_HEAD) == 1 and body.count(OLD_TAIL) == 1
        new = body.replace(OLD_HEAD, NEW_HEAD).replace(OLD_TAIL, NEW_TAIL)
        strip = lambda s: re.sub(r'^rationale = ".*"$', "rationale = <>", s, flags=re.M)      # noqa: E731
        assert strip(new) == strip(body)                     # everything except the rationale line is verbatim
        assert "PROPOSALCW4" not in new and "NOT adopted" not in new and 'rank_scope = "construction"' in new
        dst.write_text(HEADER + "\n" + new + "\n")
        print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
