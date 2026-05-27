"""AMR risk scoring engine.

Scoring formula (capped at 10):
    Mobility:        conjugative=3, mobilizable=2, non-mobilizable=0
    ARG burden:      ≥5 genes or ≥3 classes=3 | 3-4 genes or 2 classes=2 | 1-2 genes=1
    Replicon:        broad-host-range (IncP/Q/W)=2 | narrow=1 | unknown=0
    Source context:  clinical/wastewater=2 | environmental=1 | unspecified=0

Week 3 — Day 19 implementation target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plasflow2.annotate.args import ARGHit
from plasflow2.annotate.mobility import MobilityResult

# Broad-host-range replicon types (plan §Risk scoring formula)
BROAD_HOST_RANGE_REPLICONS = {"IncP", "IncQ", "IncW"}

# Source context values accepted by the CLI --context flag
VALID_CONTEXTS = {"clinical", "wastewater", "environmental", "unspecified"}


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


def score_plasmid(
    contig_id: str,
    mobility: MobilityResult | None,
    arg_hits: list[ARGHit],
    source_context: str = "unspecified",
) -> RiskScore:
    """Compute AMR risk score for one plasmid contig.

    Args:
        contig_id: Sequence identifier.
        mobility: MOB-suite result for this contig (None if not typed).
        arg_hits: CARD hits for this contig.
        source_context: Sample source; one of VALID_CONTEXTS.

    Returns:
        RiskScore with score (0–10) and evidence list.
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
    if ctx in {"clinical", "wastewater"}:
        ctx_score = 2
        evidence.append(f"Source context: {ctx} (+2)")
    elif ctx == "environmental":
        ctx_score = 1
        evidence.append("Source context: environmental (+1)")
    else:
        ctx_score = 0

    total = min(mob_score + arg_score + rep_score + ctx_score, 10)

    return RiskScore(
        contig_id=contig_id,
        score=total,
        evidence=evidence,
        mobility_score=mob_score,
        arg_score=arg_score,
        replicon_score=rep_score,
        context_score=ctx_score,
    )
