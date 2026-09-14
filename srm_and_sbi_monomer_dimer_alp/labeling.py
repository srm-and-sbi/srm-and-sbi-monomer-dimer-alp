"""Static labeling stoichiometry: the degree of labeling (DOL) as an explicit input.

Every receptor subunit carries an integer dye count ``kappa`` drawn ONCE per recording from
the condition's labeling law and held fixed for the whole recording: dye conjugation
happened during sample preparation, long before acquisition. The draw is the single
primitive of the observation layer's static ingredient; everything below is arithmetic on
it, never an extra assumption:

- a subunit with ``kappa = 0`` has no emitter and never renders (it still diffuses and
  reacts in the RDS stage, which simulates the TRUE receptor population);
- a dimer renders the dyes of both subunits at one position, so its photons are the sum
  of independent per-dye processes -- no brightness multiplier anywhere;
- the visible fraction differs by species (monomer ``1 - P(0)``, dimer ``1 - P(0)^2``), so
  the visible population is dimer-enriched relative to the true composition, and the count
  parameters of the RDS stage are TRUE receptor abundances;
- the dye count travels with its subunit through fusion, fission, and conversion
  (``simulation_rds_support.extract_subunit_lineage``): a one-dye dimer that dissociates
  leaves one visible daughter and one permanently invisible one; a mobility switch (a
  type conversion) changes nothing about the labels.

The laws are fixed measured (or preparation-level) inputs and are never inferred: from the
video alone the labeling probability is nearly degenerate with the receptor counts
(``N_visible ~ q_eff * N_R``). Sensitivity to the law is exercised by re-imaging the same
trajectories under an alternative registered law, never by fitting it.

Condition laws for the MET recordings of Harwardt et al. (2017; BioStudies S-BSST712):

    INLB  ``Bernoulli(q)``, ``q = 0.5``. The InlB probe has one engineered attachment site,
          so a bound probe carries zero or one dye. The labeling probability of this
          preparation is a measured value reported by the collaborating laboratory (2026;
          not in the source publications), carried with a declared sensitivity band.
    FAB   ``Poisson(1.64)``. The Fab probe has several conjugatable residues; only the
          ensemble mean ``DOL = 1.64`` is measured (Harwardt et al., FEBS Open Bio 7, 1422,
          2017). Poisson is the working preparation-level model, not a measured
          distribution, so a matched-mean underdispersed (binomial) and overdispersed
          (negative binomial) alternative are registered for the sensitivity grid.

Derived visible fractions (arithmetic of the laws): INLB monomer 0.50 / dimer 0.75, with two
thirds of the visible INLB dimers carrying one dye; FAB monomer 0.806 / dimer 0.962.

A static probe-OCCUPANCY probability composes with the law: a subunit is occupied by a
probe with probability ``p_occ``, optionally per initial molecular species, and an
unoccupied subunit carries no dyes regardless of its draw. The occupancy is a DECLARED
PER-CONDITION INPUT of the visibility layer, not a sensitivity knob with an inert default:
the published protocol is uPAINT (both probes in the imaging medium at 0.25 nM, binding
during acquisition), which labels a sparse subset of receptors by design, so full occupancy
is not a defensible baseline. The values live on the condition settings of
``parameterization.py`` (``ConditionSetting``; MET-INLB 0.5 declared, MET-FAB 0.155 derived
from a declared Fab/InlB visibility ratio of 0.5 and the InlB anchor; both provisional until
the collaborators answer the questions of 2026-09-11); the DLI stage reads them through
``parameterization.occupancy_of`` and ``--occupancy`` overrides them for sensitivity runs.
The effective visibility per subunit is ``a = p_occ * P(kappa >= 1)`` (INLB 0.25, FAB 0.125),
and the share of visible dimers with BOTH subunits labeled is ``a / (2 - a)`` when the two subunits
are occupied independently -- what brightness can report about stoichiometry.

Probe kinetics (an assumption, stated). The dye count is static for the recording, so the
observation layer removes a dye only by photobleaching and never adds one: ligand binding
and unbinding within a recording are not modeled, and for MET-INLB the probe IS the ligand.
First-order, state-independent unbinding is statistically indistinguishable from the
modeled bleaching and is absorbed by the per-condition calibrated bleach parameter (the
detector calibrates it on the condition's own recordings). Not absorbed, and declared
rather than modeled: the appearance of a spot when free labeled ligand binds during the
recording, and any ligand affinity that differs between monomeric and dimeric receptors.
Partial ligand occupancy is the static occupancy multiplier above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .experiment_support import CONDITION_DISPLAY


LABELING_FAMILIES = ("bernoulli", "poisson", "binomial", "negative_binomial")


@dataclass(frozen=True)
class LabelingLaw:
    """Per-subunit dye-count law ``kappa ~ L_DOL`` on ``{0, 1, 2, ...}``.

    Attributes:
        family: one of ``LABELING_FAMILIES``.
        mean: ``E[kappa]`` per subunit (for ``bernoulli`` this is the labeling
            probability ``q``).
        shape: the family's second parameter, or None where the family has one
            parameter. ``binomial``: the number of conjugation sites ``n`` (a positive
            integer with ``n >= mean``; ``p = mean / n``). ``negative_binomial``: the
            variance-to-mean ratio (``> 1``; the Poisson limit is 1).
    """
    family: str
    mean: float
    shape: Optional[float] = None

    def __post_init__(self):
        if self.family not in LABELING_FAMILIES:
            raise ValueError(f"labeling family {self.family!r} is not one of {LABELING_FAMILIES}.")
        if not np.isfinite(self.mean) or self.mean < 0:
            raise ValueError(f"labeling mean must be finite and non-negative (got {self.mean}).")
        if self.family == "bernoulli":
            if self.mean > 1:
                raise ValueError(f"bernoulli labeling probability must lie in [0, 1] (got {self.mean}).")
            if self.shape is not None:
                raise ValueError("bernoulli takes no shape parameter.")
        elif self.family == "poisson":
            if self.shape is not None:
                raise ValueError("poisson takes no shape parameter.")
        elif self.family == "binomial":
            if self.shape is None or int(self.shape) != self.shape or self.shape < 1:
                raise ValueError(f"binomial needs an integer number of sites >= 1 as shape (got {self.shape}).")
            if self.mean > self.shape:
                raise ValueError(f"binomial mean {self.mean} exceeds its {int(self.shape)} sites.")
        elif self.family == "negative_binomial":
            if self.shape is None or not self.shape > 1:
                raise ValueError(f"negative_binomial needs a variance-to-mean ratio > 1 as shape (got {self.shape}).")

    # ---- moments --------------------------------------------------------------------
    @property
    def variance(self) -> float:
        if self.family == "bernoulli":
            return self.mean * (1.0 - self.mean)
        if self.family == "poisson":
            return self.mean
        if self.family == "binomial":
            return self.mean * (1.0 - self.mean / self.shape)
        return self.mean * self.shape                       # negative binomial: ratio * mean

    @property
    def probability_zero(self) -> float:
        """``P(kappa = 0)``: the per-subunit invisibility probability."""
        if self.family == "bernoulli":
            return 1.0 - self.mean
        if self.family == "poisson":
            return float(np.exp(-self.mean))
        if self.family == "binomial":
            return float((1.0 - self.mean / self.shape) ** self.shape)
        r, p = self._negative_binomial_parameters()
        return float(p ** r)

    @property
    def visible_probability(self) -> float:
        """``P(kappa >= 1)`` for one subunit (the visible fraction of monomers)."""
        return 1.0 - self.probability_zero

    def visible_fraction(self, n_subunits: int) -> float:
        """Probability that a particle of ``n_subunits`` subunits carries at least one dye."""
        return 1.0 - self.probability_zero ** n_subunits

    def _negative_binomial_parameters(self) -> Tuple[float, float]:
        # variance = mean + mean^2 / r  ->  r = mean / (ratio - 1);  numpy's p = r / (r + mean)
        r = self.mean / (self.shape - 1.0)
        return r, r / (r + self.mean)

    # ---- sampling -------------------------------------------------------------------
    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` independent dye counts, ``int64`` array of shape ``(n,)``."""
        n = int(n)
        if self.family == "bernoulli":
            return (rng.random(n) < self.mean).astype(np.int64)
        if self.family == "poisson":
            return rng.poisson(self.mean, size=n).astype(np.int64)
        if self.family == "binomial":
            return rng.binomial(int(self.shape), self.mean / self.shape, size=n).astype(np.int64)
        r, p = self._negative_binomial_parameters()
        return rng.negative_binomial(r, p, size=n).astype(np.int64)

    # ---- display --------------------------------------------------------------------
    def describe(self) -> str:
        if self.family == "bernoulli":
            return f"Bernoulli(q={self.mean:g})"
        if self.family == "poisson":
            return f"Poisson(mean={self.mean:g})"
        if self.family == "binomial":
            return f"Binomial(sites={int(self.shape)}, mean={self.mean:g})"
        return f"NegativeBinomial(mean={self.mean:g}, variance/mean={self.shape:g})"


# ---- Registry ----------------------------------------------------------------------------
# Baseline law per condition, plus the matched-mean dispersion alternatives the FAB sensitivity
# grid requires. Keys are condition-prefixed so a law can never be applied to the wrong probe
# by accident; ad-hoc laws for sensitivity runs use the ``family:mean[:shape]`` grammar of
# ``resolve_labeling_law`` and are named after their parameters.
LABELING_LAWS: Dict[str, LabelingLaw] = {
    "INLB_BERNOULLI": LabelingLaw("bernoulli", 0.5),
    "FAB_POISSON": LabelingLaw("poisson", 1.64),
    "FAB_BINOMIAL": LabelingLaw("binomial", 1.64, shape=4),                 # variance 0.97 (underdispersed)
    "FAB_NEGATIVE_BINOMIAL": LabelingLaw("negative_binomial", 1.64, shape=2.0),  # variance 3.28 (overdispersed)
}
BASELINE_LAW_OF_CONDITION: Dict[str, str] = {"FAB": "FAB_POISSON", "INLB": "INLB_BERNOULLI"}
LABELING_CONDITIONS: Tuple[str, ...] = tuple(BASELINE_LAW_OF_CONDITION)


def resolve_labeling_law(condition: str, spec: Optional[str] = None) -> Tuple[str, LabelingLaw]:
    """Resolve the labeling law for a condition: ``(name, law)``.

    Args:
        condition: stored condition token (``FAB`` or ``INLB``).
        spec: None for the condition's baseline law; a registry key (which must carry
            the condition's prefix); or an ad-hoc ``family:mean[:shape]`` triple for a
            sensitivity run, e.g. ``bernoulli:0.4`` or ``binomial:1.64:6``.
    """
    if condition not in BASELINE_LAW_OF_CONDITION:
        raise ValueError(f"condition {condition!r} has no labeling law; known: {LABELING_CONDITIONS}.")
    if spec is None or spec == "":
        name = BASELINE_LAW_OF_CONDITION[condition]
        return name, LABELING_LAWS[name]
    if spec in LABELING_LAWS:
        if not spec.startswith(condition + "_"):
            raise ValueError(f"labeling law {spec!r} is not a {condition} law "
                             f"({CONDITION_DISPLAY.get(condition, condition)}).")
        return spec, LABELING_LAWS[spec]
    parts = spec.split(":")
    if len(parts) not in (2, 3) or parts[0] not in LABELING_FAMILIES:
        raise ValueError(
            f"labeling law spec {spec!r} is neither a registry key {tuple(LABELING_LAWS)} nor "
            f"'family:mean[:shape]' with family in {LABELING_FAMILIES}.")
    mean = float(parts[1])
    shape = float(parts[2]) if len(parts) == 3 else None
    law = LabelingLaw(parts[0], mean, shape)
    name = f"{condition}_{parts[0].upper()}_{mean:g}" + (f"_{shape:g}" if shape is not None else "")
    return name, law


# ---- Occupancy ---------------------------------------------------------------------------
Occupancy = Union[float, Dict[str, float]]


def parse_occupancy(text: str) -> Occupancy:
    """Parse ``--occupancy``: a single probability (``"1.0"``) or per MOLECULAR species
    ``"A=1.0,B=0.8"`` (monomer A, dimer B; never a mobility mode). Every value must lie in
    [0, 1]."""
    text = text.strip()
    if "=" not in text:
        value = float(text)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"occupancy must lie in [0, 1] (got {value}).")
        return value
    result: Dict[str, float] = {}
    for item in text.split(","):
        key, _, val = item.partition("=")
        value = float(val)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"occupancy for species {key.strip()!r} must lie in [0, 1] (got {value}).")
        result[key.strip()] = value
    return result


def occupancy_per_subunit(occupancy: Occupancy, initial_species: Sequence[str]) -> np.ndarray:
    """Per-subunit probe-occupancy probabilities, shape ``(n_subunits,)``.

    ``initial_species`` names the species of each subunit's host at frame 0; a per-species
    occupancy applies by that initial species (static selection at labeling time).
    """
    initial_species = np.asarray(initial_species)
    if isinstance(occupancy, dict):
        missing = sorted(set(initial_species.tolist()) - set(occupancy))
        if missing:
            raise ValueError(f"occupancy has no entry for species {missing}.")
        return np.array([occupancy[s] for s in initial_species], dtype=float)
    return np.full(initial_species.shape[0], float(occupancy))


def occupancy_by_species(occupancy: Occupancy, species_names: Sequence[str]) -> Tuple[float, ...]:
    """The occupancy probability of each molecular species, in ``species_names`` order
    (a scalar occupancy repeats; a per-species mapping must cover every name)."""
    if isinstance(occupancy, dict):
        missing = [s for s in species_names if s not in occupancy]
        if missing:
            raise ValueError(f"occupancy has no entry for species {missing}.")
        return tuple(float(occupancy[s]) for s in species_names)
    return tuple(float(occupancy) for _ in species_names)


def draw_dye_counts(law: LabelingLaw, n_subunits: int, rng: np.random.Generator,
                    occupancy: Union[float, np.ndarray] = 1.0) -> np.ndarray:
    """Draw the static dye count of every subunit once: ``int64`` array ``(n_subunits,)``.

    ``occupancy`` is a scalar or a per-subunit array of probe-occupancy probabilities; a
    subunit that is not occupied by a probe carries no dyes. At the default 1.0 the
    result is the bare law.
    """
    n_subunits = int(n_subunits)
    kappa = law.draw(n_subunits, rng)
    p = np.broadcast_to(np.asarray(occupancy, dtype=float), (n_subunits,))
    if np.any(p < 1.0):
        occupied = rng.random(n_subunits) < p
        kappa = np.where(occupied, kappa, 0)
    return kappa.astype(np.int64)


# ---- Provenance summary ------------------------------------------------------------------
# One fixed-width row per rendered simulation, persisted beside the imaging draws as the
# Labeling_Set. The ``_0`` columns describe the initial frame (the true and visible initial
# composition); the first three are recording-wide. Counts are of PARTICLES for the dimer
# columns and of SUBUNITS otherwise.
LABELING_SET_COLUMNS: Tuple[str, ...] = (
    "n_subunits",            # true receptor count N_R (conserved)
    "n_dyes",                # emitters rendered (sum of kappa)
    "n_labeled_subunits",    # subunits with kappa >= 1
    "monomers_0",            # monomer particles at frame 0
    "monomers_visible_0",    # ... of which labeled
    "dimers_0",              # dimer particles (any mobility mode) at frame 0
    "dimers_visible_0",      # ... with at least one labeled subunit
    "dimers_two_labeled_0",  # ... with both subunits labeled
    "occupancy_monomer",     # probe-occupancy probability applied to monomer subunits (declared/derived/override)
    "occupancy_dimer",       # ... to dimer subunits
)


def labeling_summary(dye_counts: np.ndarray, host_index_0: np.ndarray, host_rank_0: np.ndarray,
                     monomer_ranks: Sequence[int],
                     occupancy_by_species_values: Tuple[float, float] = (float("nan"), float("nan"))) -> np.ndarray:
    """The ``LABELING_SET_COLUMNS`` row for one simulation (``float64`` array).

    Args:
        dye_counts: per-subunit dye counts ``(n_subunits,)``.
        host_index_0, host_rank_0: frame-0 rows of the subunit lineage.
        monomer_ranks: particle-type ranks whose particles are single subunits (all
            monomer modes; see ``simulation_rds_support.monomer_ranks``).
        occupancy_by_species_values: the (monomer, dimer) occupancy probabilities actually
            applied (``occupancy_by_species``); NaN when not recorded.
    """
    dye_counts = np.asarray(dye_counts)
    labeled = dye_counts >= 1
    is_monomer = np.isin(host_rank_0, np.asarray(list(monomer_ranks)))
    n_hosts = int(host_index_0.max()) + 1 if host_index_0.size else 0
    dimer_sub = ~is_monomer
    subunits_per_host = np.bincount(host_index_0[dimer_sub], minlength=n_hosts)
    labeled_per_host = np.bincount(host_index_0[dimer_sub], weights=labeled[dimer_sub].astype(float),
                                   minlength=n_hosts)
    dimer_hosts = subunits_per_host > 0
    return np.array([
        dye_counts.shape[0],
        int(dye_counts.sum()),
        int(labeled.sum()),
        int(is_monomer.sum()),
        int((is_monomer & labeled).sum()),
        int(dimer_hosts.sum()),
        int((labeled_per_host[dimer_hosts] >= 1).sum()),
        int((labeled_per_host[dimer_hosts] >= 2).sum()),
        float(occupancy_by_species_values[0]),
        float(occupancy_by_species_values[1]),
    ], dtype=np.float64)
