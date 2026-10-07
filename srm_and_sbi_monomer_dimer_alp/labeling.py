"""Static labeling stoichiometry: probe classes and the degree of labeling (DOL) as explicit inputs.

The simulated population is the RETAINED one (PROJECT_CONTEXT.md sec. 2, *Modeling assumptions
of the MET model*): every initial retained complex carries at least one probe, and a probe is
one molecule on one subunit (at most one Fab or one InlB per MET subunit, Harwardt et al. 2017).
Labeling therefore has two steps, both drawn ONCE per recording and held fixed, because probe
binding and dye conjugation happened before acquisition (fixed attachment, assumption 5):

1. PROBE ASSIGNMENT (``assign_probes``): which retained subunits carry a probe. Every monomer
   does. Under MET-INLB the dimer classes decide it by species: a ``B1`` host carries one
   retained subunit, bound; a ``B2`` host two, both bound; so every retained INLB subunit is
   bound. Under MET-FAB the dimer ``B`` carries two retained subunits and its class is assigned
   here: two Fab with the declared two-probe share s_2 = p / (2 - p) (independent binding), else
   one Fab on one subunit chosen at random; the other subunit is probe-free, dark, and inert,
   since nothing associates under FAB.
2. DYE COUNT: every bound probe draws an integer dye count ``kappa`` from the condition's law
   (``draw_dye_counts_bound``); a probe-free subunit carries ``kappa = 0``.

Everything below is arithmetic on these two draws, never an extra assumption:

- a subunit with ``kappa = 0`` has no emitter and never renders (a bound but dark probe still
  reacts: eligibility follows the probe, not the dye);
- a dimer renders the dyes of both subunits at one position, so its photons are the sum
  of independent per-dye processes -- no brightness multiplier anywhere; a one-probe dimer has
  no automatic brightness multiplier relative to a one-probe monomer;
- the probe and its dyes travel with their subunit through fusion, fission, and conversion
  (``simulation_rds_support.extract_subunit_lineage``): a one-Fab dimer that dissociates leaves
  one daughter with the Fab and all its dyes and one permanently dark daughter; a ``B1 -> A``
  conversion keeps the bound subunit's identity; a mobility switch changes nothing about the labels.

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

Derived visible fractions of the RETAINED population (arithmetic of the laws, ``q`` = P(dye >= 1)
of a bound probe: INLB 0.5, FAB 0.806): a retained monomer is visible with probability ``q``; a
one-probe dimer with ``q`` and, under INLB, never with two dyes; a two-probe dimer with
``1 - (1 - q)^2`` (INLB 0.75, FAB 0.962), and among visible two-InlB dimers one third carry two
dyes. Under FAB a bound probe carries a Poisson number of dyes, so brightness classes do not map
onto probe counts.

The probe OCCUPANCY ``p`` of the TRUE population is a DECLARED PER-CONDITION conversion, not a
labeling coin: it realizes the retained population from the inferred receptor total N_total at
the RDS stage (``parameterization.realize_initial_composition``) and sets the one-/two-probe
split of the initial dimers here (``two_probe_share_of``). The values live on the condition
settings of ``parameterization.py`` (``ConditionSetting``; MET-INLB 0.0476, the independent-site
equilibrium at the published 0.25 nM with K_D 5 nM; MET-FAB 0.0148, derived from a declared
Fab/InlB visibility ratio of 0.5 and the InlB anchor). Every renderer of simulated trajectories
-- the DLI stage of both workflows, the posterior-predictive video and the horizon audit --
resolves the run's labeling through ``resolve_labeling`` and labels each trajectory through
``label_trajectory``, so no renderer can apply a law, a probe rule or a species mapping other
than the training data's. ``--occupancy`` is an explicit SENSITIVITY OVERRIDE that replaces the
probe classes by independent per-subunit occupancy coins (``probe_rule = "coins"``), recorded as
such; it is never the default. Only the diagnostics that test the bare law on purpose (the
labeling audit's law and draw levels, the direct estimators' synthetic scenes) call
``draw_dye_counts`` directly, and each states its occupancy.

Probe kinetics (an assumption, stated). A probe stays attached for the recording and its dye
count is static, so the observation layer removes a dye only by photobleaching and never adds
one: ligand binding and unbinding within a recording are not modeled, and for MET-INLB the probe
IS the ligand. Photobleaching removes fluorescence and preserves the probe, hence a receptor's
association eligibility. First-order, state-independent unbinding is statistically
indistinguishable from the modeled bleaching and is absorbed by the per-condition calibrated
bleach parameter (the detector calibrates it on the condition's own recordings). Not absorbed,
and declared rather than modeled: the appearance of a spot when free labeled ligand binds
during the recording, and any ligand affinity that differs between monomeric and dimeric
receptors beyond the independent binding of the initial classes.
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


def draw_dye_counts_bound(law: LabelingLaw, probe_bound: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw the static dye count of every subunit given which subunits carry a probe: one draw of
    the law per subunit, zeroed where ``probe_bound`` is False. ``int64`` array ``(n_subunits,)``."""
    probe_bound = np.asarray(probe_bound, dtype=bool)
    kappa = law.draw(int(probe_bound.shape[0]), rng)
    return np.where(probe_bound, kappa, 0).astype(np.int64)


def assign_probes(plan: "LabelingPlan", host_index_0: np.ndarray, host_rank_0: np.ndarray,
                  species_of_rank: Dict[int, str], rng: np.random.Generator) -> np.ndarray:
    """Which retained subunits carry a probe at the recording start: ``bool`` array ``(n_subunits,)``.

    ``probe_rule = "classes"`` (the model): every monomer subunit is bound. A dimer host with ONE
    retained subunit (``B1`` under INLB) is bound; a host with two subunits is a two-probe dimer
    where the condition's dimer classes make it one (``B2`` under INLB), and otherwise (``B`` under
    FAB) a two-Fab dimer with probability ``plan.two_probe_share`` or a one-Fab dimer with the
    probe on one subunit chosen at random. ``probe_rule = "coins"`` (the ``--occupancy`` override):
    independent per-subunit coins at the occupancy of each subunit's initial molecular species.
    """
    host_index_0 = np.asarray(host_index_0)
    host_rank_0 = np.asarray(host_rank_0)
    n = int(host_rank_0.shape[0])
    initial_species = np.array([species_of_rank[int(rank)] for rank in host_rank_0])
    if plan.probe_rule == "coins":
        p = occupancy_per_subunit(plan.occupancy, initial_species)
        return rng.random(n) < p
    if plan.probe_rule != "classes":
        raise ValueError(f"unknown probe rule {plan.probe_rule!r}; expected 'classes' or 'coins'.")
    bound = np.ones(n, dtype=bool)
    is_dimer = initial_species == plan.species_names[1]
    if not is_dimer.any():
        return bound
    n_hosts = int(host_index_0.max()) + 1
    subunits_per_host = np.bincount(host_index_0[is_dimer], minlength=n_hosts)
    two_subunit_hosts = np.flatnonzero(subunits_per_host == 2)
    if two_subunit_hosts.size and not plan.ligand_classes:
        # FAB: the one-/two-Fab class of every basal dimer, drawn here; one-Fab -> one random subunit dark
        two_probe = rng.random(two_subunit_hosts.size) < float(plan.two_probe_share)
        one_probe_hosts = two_subunit_hosts[~two_probe]
        if one_probe_hosts.size:
            members = np.flatnonzero(is_dimer)
            order = members[np.argsort(host_index_0[members], kind="stable")]
            hosts_sorted = host_index_0[order]
            first = np.searchsorted(hosts_sorted, one_probe_hosts)          # index of each host's first subunit
            dark_offset = rng.integers(0, 2, size=one_probe_hosts.size)     # which of the two stays probe-free
            bound[order[first + dark_offset]] = False
    return bound


# ---- Provenance summary ------------------------------------------------------------------
# One fixed-width row per rendered simulation, persisted beside the imaging draws as the
# Labeling_Set. The ``_0`` columns describe the initial frame (the true and visible initial
# composition); the first three are recording-wide. Counts are of PARTICLES for the dimer
# columns and of SUBUNITS otherwise.
LABELING_SET_COLUMNS: Tuple[str, ...] = (
    "n_subunits",            # retained receptor subunit count N_R (conserved)
    "n_dyes",                # emitters rendered (sum of kappa)
    "n_labeled_subunits",    # subunits with kappa >= 1
    "monomers_0",            # monomer particles at frame 0
    "monomers_visible_0",    # ... of which labeled
    "dimers_0",              # dimer particles (any class, any mobility mode) at frame 0
    "dimers_visible_0",      # ... with at least one labeled subunit
    "dimers_two_labeled_0",  # ... with two labeled subunits
    "occupancy_monomer",     # probability that a retained monomer subunit carries a probe (1 under the model; the coin under an override)
    "occupancy_dimer",       # ... that a retained dimer subunit does ((1 + s_2) / 2 under FAB, 1 under INLB; the coin under an override)
    "dimers_one_probe_0",    # dimer particles carrying ONE probe at frame 0 (B1 under INLB; one-Fab under FAB)
    "dimers_two_probe_0",    # ... carrying two probes (B2 under INLB; two-Fab under FAB)
    "two_probe_share",       # declared two-probe share s_2 = p / (2 - p) of the initial dimers (NaN under an override)
)


def labeling_summary(dye_counts: np.ndarray, host_index_0: np.ndarray, host_rank_0: np.ndarray,
                     monomer_ranks: Sequence[int],
                     occupancy_by_species_values: Tuple[float, float] = (float("nan"), float("nan")),
                     probe_bound: Optional[np.ndarray] = None,
                     two_probe_share: float = float("nan")) -> np.ndarray:
    """The ``LABELING_SET_COLUMNS`` row for one simulation (``float64`` array).

    Args:
        dye_counts: per-subunit dye counts ``(n_subunits,)``.
        host_index_0, host_rank_0: frame-0 rows of the subunit lineage.
        monomer_ranks: particle-type ranks of the monomer types (all monomer modes; see
            ``simulation_rds_support.monomer_ranks``).
        occupancy_by_species_values: the (monomer, dimer) per-subunit probe probabilities
            actually applied; NaN when not recorded.
        probe_bound: which subunits carry a probe (``assign_probes``); the probe-class columns
            are NaN when it is not given.
        two_probe_share: the declared s_2 the dimer classes were drawn with (NaN under an override).
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
    if probe_bound is None:
        one_probe = two_probe = float("nan")
    else:
        bound = np.asarray(probe_bound, dtype=bool)
        bound_per_host = np.bincount(host_index_0[dimer_sub], weights=bound[dimer_sub].astype(float),
                                     minlength=n_hosts)
        one_probe = int((bound_per_host[dimer_hosts] == 1).sum())
        two_probe = int((bound_per_host[dimer_hosts] >= 2).sum())
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
        float(one_probe),
        float(two_probe),
        float(two_probe_share),
    ], dtype=np.float64)


# ---- The one labeling path of every renderer of simulated trajectories ---------------------
# The DLI stage of both workflows, the posterior-predictive video and the horizon audit resolve
# a run's labeling with `resolve_labeling` and label each trajectory with `label_trajectory`, so
# the renders a check compares against a recording carry the observation model of the training
# data: the condition's law, its declared (or derived) probe occupancy applied by initial
# molecular species, and the same record. An override (`--labeling-law`, `--occupancy`) enters
# through the same call and is recorded as such.

@dataclass(frozen=True)
class LabelingPlan:
    """The resolved static labeling of one run.

    Attributes:
        condition: stored condition token (``FAB`` or ``INLB``).
        law_name, law: the resolved labeling law (``resolve_labeling_law``).
        probe_rule: ``classes`` (the model: every retained monomer bound, dimer classes by
            species under INLB and drawn with ``two_probe_share`` under FAB) or ``coins`` (the
            ``--occupancy`` override: independent per-subunit coins at ``occupancy``).
        occupancy: the condition's probe occupancy p of the true population (``classes``), or
            the override as applied, one probability or a mapping by stoichiometric class.
        occupancy_source: ``equilibrium``, ``derived`` or ``declared`` (the condition
            setting), or ``override`` (``--occupancy``).
        ligand_classes: whether the condition's dimers carry their probe class as a species.
        two_probe_share: s_2 = p / (2 - p), the declared two-probe share of the initial dimers
            (NaN under an override).
        species_names: the stoichiometric class names, in configuration order (monomer, dimer).
        occupancy_pair: the probability that a retained subunit of each class carries a probe,
            as applied, in ``species_names`` order (the ``occupancy_monomer`` /
            ``occupancy_dimer`` columns): under ``classes`` 1 for monomers and, for dimer
            subunits, 1 under INLB and (1 + s_2) / 2 under FAB.
    """
    condition: str
    law_name: str
    law: LabelingLaw
    probe_rule: str
    occupancy: Occupancy
    occupancy_source: str
    ligand_classes: bool
    two_probe_share: Optional[float]
    species_names: Tuple[str, ...]
    occupancy_pair: Tuple[float, ...]

    @property
    def visible_per_subunit(self) -> Tuple[float, ...]:
        """Probability that a RETAINED subunit of each class is visible: bound probability x P(kappa >= 1)."""
        return tuple(p * self.law.visible_probability for p in self.occupancy_pair)

    def describe(self) -> str:
        a = ", ".join(f"{s} {v:.4f}" for s, v in zip(self.species_names, self.visible_per_subunit))
        if self.probe_rule == "classes":
            rule = (f"probe classes by species (B1 one, B2 two)" if self.ligand_classes
                    else f"probe classes drawn, two-probe share {self.two_probe_share:.4f}")
            return (f"{self.condition} {self.law_name} = {self.law.describe()}, retained population: "
                    f"{rule}, occupancy {self.occupancy:.4f} ({self.occupancy_source}); "
                    f"visible per retained subunit {a}")
        return (f"{self.condition} {self.law_name} = {self.law.describe()}, occupancy coins "
                f"{self.occupancy} ({self.occupancy_source}); visible per subunit {a}")

    def record(self) -> dict:
        """JSON-ready provenance of the plan, for the files a render writes."""
        occupancy = (dict(self.occupancy) if isinstance(self.occupancy, dict)
                     else float(self.occupancy))
        return {"condition": self.condition, "law_name": self.law_name,
                "law": {"family": self.law.family, "mean": self.law.mean, "shape": self.law.shape,
                        "description": self.law.describe()},
                "probe_rule": self.probe_rule, "ligand_classes": self.ligand_classes,
                "two_probe_share": (None if self.two_probe_share is None else float(self.two_probe_share)),
                "occupancy": occupancy, "occupancy_source": self.occupancy_source,
                "occupancy_by_species": dict(zip(self.species_names, self.occupancy_pair)),
                "visible_per_subunit": dict(zip(self.species_names, self.visible_per_subunit))}


def resolve_labeling(condition: str, law_spec: Optional[str] = None,
                     occupancy_spec: Optional[str] = None) -> LabelingPlan:
    """Resolve a run's static labeling, exactly as the DLI stage does.

    Args:
        condition: stored condition token (``FAB`` or ``INLB``).
        law_spec: ``--labeling-law``: None for the condition's baseline law, else a registry key
            or ``family:mean[:shape]`` (``resolve_labeling_law``).
        occupancy_spec: ``--occupancy``: None for the model's probe rule (``classes``, with the
            condition's occupancy from ``parameterization.ConditionSetting`` setting the
            two-probe share), else an override in the ``parse_occupancy`` grammar that replaces
            the classes by independent per-subunit coins, recorded as ``override``.
    """
    # Local import: parameterization imports this module lazily, for the laws' dye probabilities.
    from .parameterization import PARAMETERS, occupancy_of, occupancy_source_of, two_probe_share_of
    law_name, law = resolve_labeling_law(condition, law_spec)
    rds = PARAMETERS.simulation.rds
    species_names = tuple(rds.molecular_species_names)
    ligand_classes = bool(rds.condition_setting(condition).ligand_classes)
    if occupancy_spec is None:
        p = float(occupancy_of(condition))
        s2 = float(two_probe_share_of(condition))
        dimer_subunit_bound = 1.0 if ligand_classes else 0.5 * (1.0 + s2)
        return LabelingPlan(condition, law_name, law, "classes", p, occupancy_source_of(condition),
                            ligand_classes, s2, species_names, (1.0, dimer_subunit_bound))
    occupancy = parse_occupancy(str(occupancy_spec))
    return LabelingPlan(condition, law_name, law, "coins", occupancy, "override", ligand_classes,
                        None, species_names, occupancy_by_species(occupancy, species_names))


def labeling_rng(seed: Optional[int], task: int = 0, sim: int = 0) -> np.random.Generator:
    """The labeling stream of simulation ``sim`` of task ``task``, the DLI stage's convention:
    ``default_rng([seed, task, sim])`` when a seed is given, so a task index draws the same labeling
    under ``--task-id`` fan-out; None stays non-deterministic. A single render uses
    ``task = sim = 0``.

    numpy's ``SeedSequence`` fills missing entropy words with zeros, so ``[seed, 0, 0]`` yields the
    same stream as the bare ``seed``: for task 0, simulation 0 the labeling stream coincides with
    the placement and render streams, which are seeded with the bare seed. That is the production
    convention, retained here unchanged; the independence of those streams is not established. It
    matters only for seeded runs (production generation is seedless), and ReaDDy's dynamics stay
    OS-seeded, so a seed does not reproduce a trajectory."""
    return np.random.default_rng(None if seed is None else [seed, task, sim])


def label_subunits(plan: LabelingPlan, host_index_0: np.ndarray, host_rank_0: np.ndarray,
                   species_of_rank: Dict[int, str], monomer_rank_list: Sequence[int],
                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Assign the probes and draw the static dye counts of every subunit under ``plan``, and
    summarize them.

    The probe rule applies by each subunit's host at frame 0 (``species_of_rank`` maps a
    particle-type rank to its stoichiometric class; mobility modes are not a selection axis):
    ``assign_probes`` decides which retained subunits carry a probe, ``draw_dye_counts_bound``
    draws one dye count per bound probe.

    Returns:
        ``(dye_counts, row)``: the ``int64`` dye count per subunit and the
        ``LABELING_SET_COLUMNS`` row, with the applied rule recorded.
    """
    host_index_0 = np.asarray(host_index_0)
    host_rank_0 = np.asarray(host_rank_0)
    probe_bound = assign_probes(plan, host_index_0, host_rank_0, species_of_rank, rng)
    dye_counts = draw_dye_counts_bound(plan.law, probe_bound, rng)
    row = labeling_summary(dye_counts, host_index_0, host_rank_0, monomer_rank_list,
                           occupancy_by_species_values=plan.occupancy_pair, probe_bound=probe_bound,
                           two_probe_share=(float("nan") if plan.two_probe_share is None else plan.two_probe_share))
    return dye_counts, row


def label_trajectory(plan: LabelingPlan, tray, lineage, rng: np.random.Generator
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """``label_subunits`` for one simulated trajectory: ``tray`` is the ``readdy.Trajectory`` and
    ``lineage`` its ``SubunitLineage`` (``simulation_rds_support.extract_subunit_lineage``)."""
    # Local import: keeps this module free of ReaDDy at import time.
    from .simulation_rds_support import monomer_ranks, rank_to_species
    return label_subunits(plan, lineage.host_index[0], lineage.host_rank[0],
                          rank_to_species(tray), monomer_ranks(tray), rng)
