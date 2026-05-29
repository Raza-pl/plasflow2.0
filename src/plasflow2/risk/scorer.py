"""AMR risk scoring engine.

Scoring formula (capped at 10):

    Host taxonomy:   ESKAPE/ESKAPEE genus or Enterobacteriaceae = 3
                     WHO 2024 critical/high priority (non-ESKAPE) = 2
                     opportunistic pathogen (not above lists)     = 0
    Mobility:        conjugative=3, mobilizable=2, non-mobilizable=0
    ARG burden:      ≥5 genes or ≥3 classes=3 | 3-4 genes or 2 classes=2 | 1-2 genes=1
    Replicon:        broad-host-range (IncP/Q/W)=2 | narrow=1 | unknown=0
    Source context:  clinical=3 | wastewater/food=2 | environmental=1 | unspecified=0

Rationale for host taxonomy as the highest-weight factor:
    The same conjugative plasmid carrying a single ARG is an acute clinical
    threat in Klebsiella pneumoniae but a distant ecological reservoir in a
    soil Bacillus. Taxonomy computed from DIAMOND + GTDB LCA is already
    available for every contig, so this bonus costs nothing extra at runtime.

    Implements the MetaCompare 2.0 Human Health Resistome Risk framework
    (Rumi et al. 2024) and WHO 2024 Bacterial Priority Pathogens List.

ESKAPE pathogens (ESKAPEE variant, +3):
    Enterococcus faecium, Staphylococcus aureus, Klebsiella pneumoniae,
    Acinetobacter baumannii, Pseudomonas aeruginosa, Enterobacter spp.,
    Escherichia coli

Source context clarification:
    'clinical'     — direct patient sample (blood, urine, wound); plasmid is
                     already inside a human host, highest immediacy.
    'wastewater'   — active mixing zone between clinical and environmental
                     strains; well-documented ARG dissemination route.
    'food'         — animal/food production; direct human exposure pathway.
    'environmental'— soil, freshwater, marine; distal reservoir. Not zero risk
                     because environmental→clinical transfer is documented, but
                     the transmission chain is longer.
    'unspecified'  — unknown origin; no adjustment applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plasflow2.annotate.args import ARGHit
from plasflow2.annotate.mobility import MobilityResult
from plasflow2.annotate.taxonomy import TaxResult

# ---------------------------------------------------------------------------
# Replicon / context constants
# ---------------------------------------------------------------------------

# Broad-host-range replicon types (plan §Risk scoring formula)
BROAD_HOST_RANGE_REPLICONS = {"IncP", "IncQ", "IncW"}

# Source context values accepted by the CLI --context flag
VALID_CONTEXTS = {"clinical", "wastewater", "food", "environmental", "unspecified"}

# ---------------------------------------------------------------------------
# Pathogen host lookup tables
# ---------------------------------------------------------------------------

# Core ESKAPE / ESKAPEE genera — +2 to risk score
# Source: Rice et al. 2008 (ESKAPE), Yu et al. 2020 (ESKAPEE)
#         MetaCompare 2.0 (Rumi et al. 2024)
ESKAPE_GENERA: frozenset[str] = frozenset(
    {
        "Enterococcus",  # E — faecium (VRE); WHO Critical
        "Staphylococcus",  # S — aureus (MRSA); WHO Critical
        "Klebsiella",  # K — pneumoniae (ESBL / KPC); WHO Critical
        "Acinetobacter",  # A — baumannii (CRAB); WHO Critical
        "Pseudomonas",  # P — aeruginosa (CRPA); WHO Critical
        "Enterobacter",  # E — cloacae complex; WHO Critical (Enterobacteriaceae)
        "Escherichia",  # E — coli (ESBL / CR-Ec); WHO Critical
    }
)

# Broader Enterobacteriaceae family: captures E. coli / Klebsiella / Enterobacter
# even when taxonomy resolution stops at family level (MetaCompare 2.0 approach)
ESKAPE_FAMILIES: frozenset[str] = frozenset({"Enterobacteriaceae"})

# Additional WHO 2024 Bacterial Priority Pathogens — Critical & High tiers
# (not already covered by ESKAPE_GENERA) — +1 to risk score
# Source: WHO Bacterial Priority Pathogens List 2024 (9789240093461)
WHO_PRIORITY_GENERA: frozenset[str] = frozenset(
    {
        "Mycobacterium",  # Critical — M. tuberculosis (rifampicin-resistant)
        "Salmonella",  # High     — S. Typhi (fluoroquinolone-resistant)
        "Neisseria",  # High     — N. gonorrhoeae (3GC / FQ-resistant)
        "Campylobacter",  # High     — C. jejuni / coli (fluoroquinolone-resistant)
        "Helicobacter",  # High     — H. pylori (clarithromycin-resistant)
        "Shigella",  # Medium   — (fluoroquinolone-resistant)
        "Streptococcus",  # Medium   — S. pneumoniae (penicillin non-susceptible)
        "Haemophilus",  # Medium   — H. influenzae (ampicillin-resistant)
        "Clostridioides",  # Medium   — C. difficile
        "Vibrio",  # Notable  — V. cholerae
    }
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RiskScore:
    """AMR risk assessment for a single plasmid contig."""

    contig_id: str
    score: int  # 0–10 (capped)
    evidence: list[str] = field(default_factory=list)  # human-readable justifications

    # Subscores (for transparency / report detail)
    mobility_score: int = 0
    arg_score: int = 0
    replicon_score: int = 0
    context_score: int = 0
    host_score: int = 0  # ESKAPE / WHO priority pathogen host bonus

    # Pathogen host annotation (Option 2 — separate from score)
    eskape_host: bool = False  # True if taxonomy matches a recognized pathogen
    eskape_genus: str = ""  # Matched genus or family name, e.g. "Klebsiella"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_genus_family(tax: TaxResult) -> tuple[str, str]:
    """Extract genus and family from a TaxResult, stripping GTDB rank prefixes.

    GTDB lineage tokens use prefixes: g__ (genus), f__ (family), etc.
    Both the ``taxon`` field and the full ``lineage`` string are searched so
    that genus is resolved even when the LCA only reached family level.

    Args:
        tax: TaxResult from :func:`plasflow2.annotate.taxonomy.assign_taxonomy`.

    Returns:
        Tuple (genus, family) — either may be an empty string if not found.
    """
    genus = ""
    family = ""

    for token in tax.lineage.split(";"):
        token = token.strip()
        if token.startswith("g__"):
            genus = token[3:]
        elif token.startswith("f__"):
            family = token[3:]

    # Fall back to the taxon field if lineage didn't supply the rank
    if not genus and tax.taxon.startswith("g__"):
        genus = tax.taxon[3:]
    if not family and tax.taxon.startswith("f__"):
        family = tax.taxon[3:]

    return genus, family


def _detect_pathogen_host(taxonomy: TaxResult | None) -> tuple[bool, str, int]:
    """Check whether a contig's LCA taxonomy matches a priority pathogen.

    Implements the ESKAPEE + Enterobacteriaceae check from MetaCompare 2.0,
    extended with additional genera from the WHO 2024 Bacterial Priority
    Pathogens List.

    Args:
        taxonomy: TaxResult for the contig (may be None if taxonomy was skipped
                  or the contig was unclassified).

    Returns:
        Tuple (is_pathogen_host, matched_name, score_bonus):
            - ``is_pathogen_host``: True if any pathogen match was found.
            - ``matched_name``: The matched genus (or family) name, empty string
              if no match.
            - ``score_bonus``: 3 for ESKAPE core, 2 for WHO priority non-ESKAPE,
              0 if no match.
    """
    if taxonomy is None:
        return False, "", 0

    genus, family = _extract_genus_family(taxonomy)

    # ESKAPE core genus match (highest priority) → +3
    if genus in ESKAPE_GENERA:
        return True, genus, 3

    # Enterobacteriaceae family match (incomplete taxonomy → still ESKAPE level) → +3
    if family in ESKAPE_FAMILIES:
        return True, family, 3

    # WHO priority genus (non-ESKAPE) → +2
    if genus in WHO_PRIORITY_GENERA:
        return True, genus, 2

    return False, "", 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_plasmid(
    contig_id: str,
    mobility: MobilityResult | None,
    arg_hits: list[ARGHit],
    source_context: str = "unspecified",
    taxonomy: TaxResult | None = None,
) -> RiskScore:
    """Compute AMR risk score for one plasmid contig.

    Args:
        contig_id: Sequence identifier.
        mobility: MOB-suite result for this contig (None if not typed).
        arg_hits: CARD / SARG hits for this contig.
        source_context: Sample source; one of VALID_CONTEXTS.
        taxonomy: Optional LCA taxonomy from
            :func:`plasflow2.annotate.taxonomy.assign_taxonomy`.  Used to
            detect ESKAPE / WHO priority pathogen hosts and add a host
            bonus (+2 for ESKAPE, +1 for WHO priority) to the score.

    Returns:
        RiskScore with score (0–10), evidence list, and ESKAPE host fields.
    """
    evidence: list[str] = []

    # --- Mobility score ---
    mob_class = mobility.mobility_class if mobility else "non-mobilizable"
    if mob_class == "conjugative":
        mob_score = 3
        evidence.append("Conjugative plasmid (+3)")
    elif mob_class == "mobilizable":
        mob_score = 2
        evidence.append("Mobilizable plasmid (+2)")
    else:
        mob_score = 0

    # --- ARG burden score ---
    num_genes = len(arg_hits)
    num_classes = len({h.drug_class for h in arg_hits})
    if num_genes >= 5 or num_classes >= 3:
        arg_score = 3
        evidence.append(f"{num_genes} ARGs across {num_classes} drug classes (+3)")
    elif num_genes >= 3 or num_classes >= 2:
        arg_score = 2
        evidence.append(f"{num_genes} ARGs across {num_classes} drug classes (+2)")
    elif num_genes >= 1:
        arg_score = 1
        evidence.append(f"{num_genes} ARG(s) detected (+1)")
    else:
        arg_score = 0

    # --- Replicon breadth score ---
    rep_type = mobility.replicon_type if mobility else "unknown"
    rep_base = rep_type.split("/")[0].split(",")[0].strip()  # handle multi-replicon
    if rep_base in BROAD_HOST_RANGE_REPLICONS:
        rep_score = 2
        evidence.append(f"Broad-host-range replicon {rep_base} (+2)")
    elif rep_base not in {"unknown", "none", ""}:
        rep_score = 1
        evidence.append(f"Known replicon {rep_base} (+1)")
    else:
        rep_score = 0

    # --- Source context score ---
    ctx = source_context.lower()
    if ctx == "clinical":
        ctx_score = 3
        evidence.append("Source context: clinical (+3)")
    elif ctx in {"wastewater", "food"}:
        ctx_score = 2
        evidence.append(f"Source context: {ctx} (+2)")
    elif ctx == "environmental":
        ctx_score = 1
        evidence.append("Source context: environmental (+1)")
    else:
        ctx_score = 0

    # --- Pathogen host score (ESKAPE / WHO priority) ---
    is_pathogen, matched_genus, host_score = _detect_pathogen_host(taxonomy)
    if is_pathogen and host_score == 3:
        evidence.append(f"ESKAPE pathogen host: {matched_genus} (+3)")
    elif is_pathogen and host_score == 2:
        evidence.append(f"WHO priority pathogen host: {matched_genus} (+2)")

    total = min(mob_score + arg_score + rep_score + ctx_score + host_score, 10)

    return RiskScore(
        contig_id=contig_id,
        score=total,
        evidence=evidence,
        mobility_score=mob_score,
        arg_score=arg_score,
        replicon_score=rep_score,
        context_score=ctx_score,
        host_score=host_score,
        eskape_host=is_pathogen,
        eskape_genus=matched_genus,
    )
