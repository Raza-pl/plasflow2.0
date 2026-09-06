"""Prediction-independent ARG annotation with CARD, SARG, and AMRFinderPlus."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from mobiorigin import __version__
from mobiorigin.fasta import IUPAC_DNA, FastaRecord, read_fasta
from mobiorigin.provenance import atomic_json, atomic_text, sha256_file
from mobiorigin.runtime import external_tool_environment, resolve_executable, validate_threads

MIN_IDENTITY = 80.0
MIN_QUERY_COVERAGE = 80.0
AMR_SOURCES = ("CARD", "AMRFINDERPLUS", "AMRPROT_DIAMOND", "SARG")
SOURCE_PRIORITY = {source: rank for rank, source in enumerate(AMR_SOURCES)}
AMBIGUITY_TO_N = str.maketrans({base: "N" for base in IUPAC_DNA - set("ACGT")})
CARD_HEADER = re.compile(r"gb\|(?P<protein>[^|]+)\|(?P<aro>ARO:\d+)\|(?P<gene>[^\s\[]+)")
SARG_HEADER = re.compile(r"SARG\|(?P<drug_class>[^|]+)\|(?P<family>[^|*]+)\*?\|(?P<accession>\S+)")


@dataclass(frozen=True)
class Orf:
    identifier: str
    sequence_id: str
    start: int
    end: int
    strand: int
    amino_acid_length: int


@dataclass(frozen=True)
class ArgHit:
    sequence_id: str
    orf_id: str
    orf_start: int
    orf_end: int
    orf_strand: int
    source: str
    gene_symbol: str
    gene_name: str
    accession: str
    amr_family: str
    drug_class: str
    resistance_mechanism: str
    method: str
    identity: float | None
    query_coverage: float | None
    evalue: float | None
    bitscore: float | None


HIT_COLUMNS = tuple(ArgHit.__dataclass_fields__)


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} file is missing: {path}")
    return path


def _executable(value: Path, label: str) -> Path:
    resolved = resolve_executable(value, label=f"{label} executable")
    assert resolved is not None
    return resolved


def predict_annotation_orfs(records: Sequence[FastaRecord], output: Path) -> dict[str, Orf]:
    """Call translation-table-11 ORFs while preserving coordinates."""
    import pyrodigal  # type: ignore[import]

    bins = pyrodigal.MetagenomicBins(
        item for item in pyrodigal.METAGENOMIC_BINS if item.training_info.translation_table == 11
    )
    if not bins:
        raise RuntimeError("Pyrodigal has no metagenomic translation-table-11 bins")
    finder = pyrodigal.GeneFinder(meta=True, metagenomic_bins=bins, mask=True, min_mask=1)
    orfs: dict[str, Orf] = {}
    with output.open("w", encoding="ascii") as handle:
        for record in records:
            sequence = record.sequence.translate(AMBIGUITY_TO_N)
            rank = 0
            for gene in finder.find_genes(sequence.encode("ascii")):
                protein = gene.translate().rstrip("*")
                if len(protein) < 30:
                    continue
                rank += 1
                identifier = f"{record.identifier}__orf_{rank}"
                if identifier in orfs:
                    raise ValueError(f"Generated duplicate ORF identifier: {identifier}")
                orfs[identifier] = Orf(
                    identifier=identifier,
                    sequence_id=record.identifier,
                    start=int(gene.begin),
                    end=int(gene.end),
                    strand=int(gene.strand),
                    amino_acid_length=len(protein),
                )
                handle.write(f">{identifier}\n{protein}\n")
    return orfs


def run_arg_diamond(
    *, diamond: Path, proteins: Path, database: Path, output: Path, threads: int
) -> None:
    completed = subprocess.run(
        [
            str(diamond),
            "blastp",
            "--query",
            str(proteins),
            "--db",
            str(database).removesuffix(".dmnd"),
            "--out",
            str(output),
            "--outfmt",
            "6",
            "qseqid",
            "sseqid",
            "pident",
            "qcovhsp",
            "evalue",
            "bitscore",
            "stitle",
            "--id",
            str(MIN_IDENTITY),
            "--query-cover",
            str(MIN_QUERY_COVERAGE),
            "--threads",
            str(threads),
            "--sensitive",
            "--max-target-seqs",
            "1",
            "--quiet",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"DIAMOND failed for {database.name}: {completed.stderr.strip()}")
    if not output.exists():
        output.write_text("", encoding="utf-8")


def _diamond_rows(path: Path) -> Iterable[tuple[str, str, float, float, float, float, str]]:
    # Third-party database descriptions occasionally contain legacy bytes
    # (for example, a Windows-1252 non-breaking space).  DIAMOND copies those
    # bytes into the free-text title column.  Preserve the evidence row and
    # replace only undecodable text instead of aborting the complete analysis.
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            parts = raw.rstrip("\n").split("\t", 6)
            if len(parts) != 7:
                raise ValueError(f"Malformed DIAMOND row {line_number} in {path.name}")
            try:
                yield (
                    parts[0],
                    parts[1],
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                    float(parts[5]),
                    parts[6],
                )
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric DIAMOND value on row {line_number} in {path.name}"
                ) from exc


def load_card_metadata(path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            accession = row.get("ARO Accession", "").strip()
            if not accession:
                continue
            metadata[accession] = {
                "gene": row.get("ARO Name", "unknown").strip() or "unknown",
                "family": row.get("AMR Gene Family", "unknown").strip() or "unknown",
                "drug_class": "; ".join(
                    item.strip() for item in row.get("Drug Class", "").split(";") if item.strip()
                )
                or "unknown",
                "mechanism": row.get("Resistance Mechanism", "unknown").strip() or "unknown",
            }
    if not metadata:
        raise ValueError(f"CARD metadata is empty or unsupported: {path}")
    return metadata


def _base_hit(
    orfs: Mapping[str, Orf],
    query: str,
    *,
    source: str,
    subject: str,
    identity: float,
    coverage: float,
    evalue: float,
    bitscore: float,
) -> tuple[Orf, dict[str, object]]:
    if query not in orfs:
        raise ValueError(f"Search output contains an unknown ORF identifier: {query}")
    orf = orfs[query]
    return orf, {
        "sequence_id": orf.sequence_id,
        "orf_id": query,
        "orf_start": orf.start,
        "orf_end": orf.end,
        "orf_strand": orf.strand,
        "source": source,
        "accession": subject,
        "identity": identity,
        "query_coverage": coverage,
        "evalue": evalue,
        "bitscore": bitscore,
    }


def parse_card_hits(
    path: Path, orfs: Mapping[str, Orf], metadata: Mapping[str, Mapping[str, str]]
) -> list[ArgHit]:
    hits: list[ArgHit] = []
    for query, subject, identity, coverage, evalue, bitscore, title in _diamond_rows(path):
        _, values = _base_hit(
            orfs,
            query,
            source="CARD",
            subject=subject,
            identity=identity,
            coverage=coverage,
            evalue=evalue,
            bitscore=bitscore,
        )
        match = CARD_HEADER.search(subject) or CARD_HEADER.search(title)
        accession = match.group("aro") if match else "unknown"
        fallback = match.group("gene") if match else subject
        item = metadata.get(accession, {})
        values.update(
            gene_symbol=str(item.get("gene", fallback)),
            gene_name=str(item.get("gene", fallback)),
            accession=accession,
            amr_family=str(item.get("family", "unknown")),
            drug_class=str(item.get("drug_class", "unknown")),
            resistance_mechanism=str(item.get("mechanism", "unknown")),
            method="DIAMOND_BLASTP",
        )
        hits.append(ArgHit(**values))  # type: ignore[arg-type]
    return hits


def parse_sarg_hits(path: Path, orfs: Mapping[str, Orf]) -> list[ArgHit]:
    hits: list[ArgHit] = []
    for query, subject, identity, coverage, evalue, bitscore, title in _diamond_rows(path):
        _, values = _base_hit(
            orfs,
            query,
            source="SARG",
            subject=subject,
            identity=identity,
            coverage=coverage,
            evalue=evalue,
            bitscore=bitscore,
        )
        match = SARG_HEADER.search(subject) or SARG_HEADER.search(title)
        if match:
            gene = match.group("family").strip()
            accession = match.group("accession").strip()
            drug_class = match.group("drug_class").strip()
        else:
            gene = subject.split("|")[-1]
            accession = subject
            drug_class = "unknown"
        values.update(
            gene_symbol=gene,
            gene_name=gene,
            accession=accession,
            amr_family=gene,
            drug_class=drug_class,
            resistance_mechanism="unknown",
            method="DIAMOND_BLASTP",
        )
        hits.append(ArgHit(**values))  # type: ignore[arg-type]
    return hits


AMRFinderEntry = dict[str, str]


def load_amrfinder_hierarchy(path: Path) -> dict[str, list[AMRFinderEntry]]:
    """Load modern hierarchical fam.tsv and inherit AMR/type metadata."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            {str(key).lstrip("#"): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]
    nodes = {row["node_id"]: row for row in rows if row.get("node_id")}
    if not nodes:
        raise ValueError(f"AMRFinderPlus hierarchy is empty or unsupported: {path}")
    cache: dict[tuple[str, str], str] = {}

    def inherited(node: str, field: str, seen: frozenset[str] = frozenset()) -> str:
        key = (node, field)
        if key in cache:
            return cache[key]
        if node in seen or node not in nodes:
            return ""
        row = nodes[node]
        value = row.get(field, "")
        if value and value != "-":
            cache[key] = value
            return value
        parent = row.get("parent_node_id", "")
        value = inherited(parent, field, seen | {node}) if parent and parent != node else ""
        cache[key] = value
        return value

    result: dict[str, list[AMRFinderEntry]] = defaultdict(list)
    for node, row in nodes.items():
        entry = {
            "element_type": (inherited(node, "type") or "UNKNOWN").upper(),
            "drug_class": (
                inherited(node, "class") or inherited(node, "subclass") or "unknown"
            ).lower(),
            "subclass": inherited(node, "subclass").lower(),
            "description": inherited(node, "family_name") or "unknown",
        }
        for alias in (row.get("gene_symbol", ""), node):
            alias = alias.strip().lower()
            if alias and alias != "-" and entry not in result[alias]:
                result[alias].append(entry)
    return dict(result)


def _amr_hierarchy_entry(
    hierarchy: Mapping[str, Sequence[AMRFinderEntry]],
    gene: str,
    family: str,
    header_class: str,
    header_subclass: str,
) -> AMRFinderEntry | None:
    entries: list[AMRFinderEntry] = []
    for query in (gene.strip().lower(), family.strip().lower()):
        entries.extend(item for item in hierarchy.get(query, ()) if item not in entries)
    if not entries:
        queries = (gene.lower(), family.lower())
        aliases = [
            alias
            for alias in hierarchy
            if len(alias) >= 4 and any(q.startswith(alias) for q in queries)
        ]
        if aliases:
            longest = max(map(len, aliases))
            for alias in aliases:
                if len(alias) == longest:
                    entries.extend(item for item in hierarchy[alias] if item not in entries)
    amr = [item for item in entries if item["element_type"] == "AMR"]
    if not amr:
        return None
    types = {item["element_type"] for item in entries}
    if types == {"AMR"}:
        return amr[0]
    terms = {value.lower() for value in (header_class, header_subclass) if value}
    for item in amr:
        if terms & {item["drug_class"], item["subclass"]}:
            return item
    return None


def parse_amrprot_hits(
    path: Path, orfs: Mapping[str, Orf], hierarchy: Mapping[str, Sequence[AMRFinderEntry]]
) -> list[ArgHit]:
    """Parse the explicitly supplemental AMRProt DIAMOND route, excluding non-AMR nodes."""
    hits: list[ArgHit] = []
    for query, subject, identity, coverage, evalue, bitscore, title in _diamond_rows(path):
        fields = title.split("|", 9)
        if len(fields) < 10:
            raise ValueError("AMRProt header is not the supported modern pipe-delimited format")
        gene = fields[3].strip() or fields[4].strip() or subject
        family = fields[4].strip()
        subclass = fields[7].strip()
        drug_class = fields[8].strip()
        name = fields[9].replace("_", " ").strip() or gene
        hierarchy_item = _amr_hierarchy_entry(hierarchy, gene, family, drug_class, subclass)
        if hierarchy_item is None:
            continue
        _, values = _base_hit(
            orfs,
            query,
            source="AMRPROT_DIAMOND",
            subject=subject,
            identity=identity,
            coverage=coverage,
            evalue=evalue,
            bitscore=bitscore,
        )
        values.update(
            gene_symbol=gene,
            gene_name=name,
            accession=fields[0].strip() or subject,
            amr_family=family or hierarchy_item["description"],
            drug_class=(drug_class or subclass or hierarchy_item["drug_class"]).lower(),
            resistance_mechanism="unknown",
            method="DIAMOND_BLASTP_SUPPLEMENTAL",
        )
        hits.append(ArgHit(**values))  # type: ignore[arg-type]
    return hits


def run_amrfinderplus(
    *, executable: Path, proteins: Path, database: Path | None, output: Path, threads: int
) -> None:
    validate_threads(threads)
    attempts = [threads]
    while attempts[-1] > 1:
        reduced = max(1, attempts[-1] // 2)
        if reduced == attempts[-1]:
            break
        attempts.append(reduced)

    resource_markers = (
        "error creating thread",
        "cannot create thread",
        "resource temporarily unavailable",
        "cannot allocate memory",
        "std::bad_alloc",
        "segmentation fault",
    )
    failures: list[tuple[int, str]] = []
    for attempted_threads in attempts:
        output.unlink(missing_ok=True)
        command = [
            str(executable),
            "--protein",
            str(proteins),
            "--plus",
            "--print_node",
            "--threads",
            str(attempted_threads),
            "--output",
            str(output),
        ]
        if database is not None:
            command.extend(["--database", str(database)])
        with external_tool_environment("amrfinder") as (environment, _):
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
        if completed.returncode == 0:
            if not output.is_file():
                raise RuntimeError("AMRFinderPlus completed without producing its report")
            return
        detail = (completed.stderr or completed.stdout).strip()
        failures.append((attempted_threads, detail))
        resource_failure = any(marker in detail.lower() for marker in resource_markers)
        if not resource_failure:
            break

    attempted = ", ".join(str(value) for value, _ in failures)
    last_detail = failures[-1][1] if failures else "no diagnostic output"
    if len(failures) > 1:
        raise RuntimeError(
            "AMRFinderPlus could not start its BLAST workers after automatic retries "
            f"with {attempted} thread(s). Check available memory and WSL limits, or set "
            "MOBIORIGIN_TMPDIR to a writable Linux-native directory. "
            f"Last AMRFinderPlus error: {last_detail}"
        )
    raise RuntimeError(f"AMRFinderPlus failed: {last_detail}")


def parse_amrfinderplus_hits(path: Path, orfs: Mapping[str, Orf]) -> list[ArgHit]:
    """Parse official AMRFinderPlus output and retain only Type=AMR rows."""
    hits: list[ArgHit] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Protein id", "Element symbol", "Element name", "Type", "Class", "Method"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("AMRFinderPlus output schema is unsupported")
        for row in reader:
            if row.get("Type", "").strip().upper() != "AMR":
                continue
            query = row["Protein id"].strip()
            if query not in orfs:
                raise ValueError(f"AMRFinderPlus returned an unknown protein identifier: {query}")
            orf = orfs[query]

            def number(name: str, current_row: Mapping[str, str] = row) -> float | None:
                value = current_row.get(name, "").strip()
                return None if not value or value.upper() == "NA" else float(value)

            drug_classes = tuple(
                dict.fromkeys(
                    value.lower()
                    for value in (
                        row.get("Class", "").strip(),
                        row.get("Subclass", "").strip(),
                    )
                    if value
                )
            )

            hits.append(
                ArgHit(
                    sequence_id=orf.sequence_id,
                    orf_id=query,
                    orf_start=orf.start,
                    orf_end=orf.end,
                    orf_strand=orf.strand,
                    source="AMRFINDERPLUS",
                    gene_symbol=row["Element symbol"].strip() or "unknown",
                    gene_name=row["Element name"].strip() or "unknown",
                    accession=row.get("Closest reference accession", "").strip() or "unknown",
                    amr_family=row.get("Hierarchy node", "").strip() or "unknown",
                    drug_class="; ".join(drug_classes) or "unknown",
                    resistance_mechanism="unknown",
                    method=row["Method"].strip() or "AMRFINDERPLUS",
                    identity=number("% Identity to reference"),
                    query_coverage=number("% Coverage of reference"),
                    evalue=None,
                    bitscore=None,
                )
            )
    return hits


def _hit_rank(hit: ArgHit) -> tuple[object, ...]:
    return (
        SOURCE_PRIORITY[hit.source],
        -(hit.bitscore if hit.bitscore is not None else 0.0),
        -(hit.identity if hit.identity is not None else 0.0),
        -(hit.query_coverage if hit.query_coverage is not None else 0.0),
        hit.accession,
        hit.gene_symbol,
    )


def consensus_hits(hits: Sequence[ArgHit]) -> list[ArgHit]:
    selected: dict[str, ArgHit] = {}
    for hit in hits:
        previous = selected.get(hit.orf_id)
        if previous is None or _hit_rank(hit) < _hit_rank(previous):
            selected[hit.orf_id] = hit
    return sorted(selected.values(), key=lambda hit: (hit.sequence_id, hit.orf_start, hit.orf_id))


def _format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.10g}"


def write_hits(path: Path, hits: Sequence[ArgHit]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HIT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for hit in hits:
            row = asdict(hit)
            for key in ("identity", "query_coverage", "evalue", "bitscore"):
                row[key] = _format_number(row[key])
            writer.writerow(row)


def write_summary(
    path: Path,
    records: Sequence[FastaRecord],
    orfs: Mapping[str, Orf],
    all_hits: Sequence[ArgHit],
    consensus: Sequence[ArgHit],
) -> None:
    orf_counts = Counter(item.sequence_id for item in orfs.values())
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for hit in all_hits:
        source_counts[hit.sequence_id][hit.source] += 1
    consensus_by_sequence: dict[str, list[ArgHit]] = defaultdict(list)
    for hit in consensus:
        consensus_by_sequence[hit.sequence_id].append(hit)
    fields = (
        "sequence_id",
        "length_bp",
        "predicted_orfs",
        "card_hits",
        "amrfinderplus_hits",
        "amrprot_diamond_hits",
        "sarg_hits",
        "consensus_arg_orfs",
        "consensus_genes",
        "consensus_drug_classes",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            selected = consensus_by_sequence[record.identifier]
            writer.writerow(
                {
                    "sequence_id": record.identifier,
                    "length_bp": len(record.sequence),
                    "predicted_orfs": orf_counts[record.identifier],
                    "card_hits": source_counts[record.identifier]["CARD"],
                    "amrfinderplus_hits": source_counts[record.identifier]["AMRFINDERPLUS"],
                    "amrprot_diamond_hits": source_counts[record.identifier]["AMRPROT_DIAMOND"],
                    "sarg_hits": source_counts[record.identifier]["SARG"],
                    "consensus_arg_orfs": len(selected),
                    "consensus_genes": ";".join(sorted({hit.gene_symbol for hit in selected})),
                    "consensus_drug_classes": ";".join(
                        sorted({hit.drug_class for hit in selected})
                    ),
                }
            )


def _database_identity(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _directory_identity(path: Path) -> dict[str, object]:
    files = sorted(item for item in path.rglob("*") if item.is_file() and not item.is_symlink())
    if not files:
        raise ValueError(f"Database directory has no regular files: {path}")
    members = [
        {
            "path": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in files
    ]
    canonical = json.dumps(members, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "path": str(path.resolve()),
        "files": len(members),
        "bytes": sum(item.stat().st_size for item in files),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "members": members,
    }


def annotate(
    *,
    input_fasta: Path,
    output_dir: Path,
    database_dir: Path,
    threads: int = 1,
    diamond: Path = Path("diamond"),
    amrfinder_mode: str = "official",
    amrfinder_bin: Path = Path("amrfinder"),
    amrfinder_database: Path | None = None,
    profile: str = "arg",
    predictions_tsv: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Run independent ARG or comprehensive annotation and publish atomically."""
    notify = progress or (lambda message: None)
    validate_threads(threads)
    if amrfinder_mode not in {"official", "amrprot"}:
        raise ValueError("AMRFinder mode must be 'official' or 'amrprot'")
    if profile not in {"arg", "comprehensive"}:
        raise ValueError("Annotation profile must be 'arg' or 'comprehensive'")
    if predictions_tsv is not None and profile != "comprehensive":
        raise ValueError("--predictions-tsv requires --profile comprehensive")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    notify("Reading the input FASTA and checking annotation resources")
    records = read_fasta(input_fasta)
    diamond_executable = _executable(diamond, "DIAMOND")
    card_db = _required_file(database_dir / "card" / "card.dmnd", "CARD database")
    card_index = _required_file(database_dir / "card" / "aro_index.tsv", "CARD metadata")
    sarg_db = _required_file(database_dir / "sarg" / "sarg.dmnd", "SARG database")
    official_executable: Path | None = None
    amrprot_db: Path | None = None
    fam_path: Path | None = None
    if amrfinder_mode == "official":
        official_executable = _executable(amrfinder_bin, "AMRFinderPlus")
        if amrfinder_database is None:
            raise ValueError("Official mode requires --amrfinder-database for frozen provenance")
        _required_file(amrfinder_database / "version.txt", "AMRFinderPlus database version")
    else:
        amrprot_db = _required_file(
            database_dir / "amrfinder" / "amrprot.dmnd", "AMRProt DIAMOND database"
        )
        fam_path = _required_file(database_dir / "amrfinder" / "fam.tsv", "AMRFinder hierarchy")

    comprehensive_databases: dict[str, Path] = {}
    if profile == "comprehensive":
        comprehensive_databases = {
            "vfdb_dmnd": _required_file(
                database_dir / "vfdb" / "vfdb_setA.dmnd", "VFDB core database"
            ),
            "vfdb_metadata": _required_file(
                database_dir / "vfdb" / "vfdb_indx.txt", "VFDB core metadata"
            ),
            "mobileog_dmnd": _required_file(
                database_dir / "mge" / "mobileog.dmnd", "mobileOG-db database"
            ),
            "bacmet_dmnd": _required_file(
                database_dir / "bacmet" / "bacmet.dmnd", "BacMet2 experimental database"
            ),
            "bacmet_metadata": _required_file(
                database_dir / "bacmet" / "Bacmet_list.tsv", "BacMet2 metadata"
            ),
            "mob_rep_dmnd": _required_file(
                database_dir / "mob_suite" / "rep_proteins.dmnd", "MOB-suite replication database"
            ),
            "mob_relaxase_dmnd": _required_file(
                database_dir / "mob_suite" / "mob_proteins.dmnd", "MOB-suite relaxase database"
            ),
            "mob_mpf_dmnd": _required_file(
                database_dir / "mob_suite" / "mpf_proteins.dmnd", "MOB-suite MPF database"
            ),
        }
        if predictions_tsv is not None:
            _required_file(predictions_tsv, "MobiOrigin predictions")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        raw = temporary / "raw_evidence"
        raw.mkdir()
        proteins = temporary / "predicted_proteins.faa"
        notify(f"Predicting protein-coding regions in {len(records):,} contigs")
        orfs = predict_annotation_orfs(records, proteins)
        card_raw = raw / "card_diamond.tsv"
        sarg_raw = raw / "sarg_diamond.tsv"
        if orfs:
            notify(f"Searching CARD across {len(orfs):,} predicted proteins")
            run_arg_diamond(
                diamond=diamond_executable,
                proteins=proteins,
                database=card_db,
                output=card_raw,
                threads=threads,
            )
            notify("Searching SARG for independent ARG evidence")
            run_arg_diamond(
                diamond=diamond_executable,
                proteins=proteins,
                database=sarg_db,
                output=sarg_raw,
                threads=threads,
            )
        else:
            card_raw.write_text("", encoding="utf-8")
            sarg_raw.write_text("", encoding="utf-8")
        card_hits = parse_card_hits(card_raw, orfs, load_card_metadata(card_index))
        sarg_hits = parse_sarg_hits(sarg_raw, orfs)
        amr_hits: list[ArgHit]
        if amrfinder_mode == "official":
            official_raw = raw / "amrfinderplus.tsv"
            if orfs:
                notify("Running official AMRFinderPlus annotation")
                assert official_executable is not None
                run_amrfinderplus(
                    executable=official_executable,
                    proteins=proteins,
                    database=amrfinder_database,
                    output=official_raw,
                    threads=threads,
                )
            else:
                official_raw.write_text(
                    "Protein id\tElement symbol\tElement name\tType\tClass\tMethod\n",
                    encoding="utf-8",
                )
            amr_hits = parse_amrfinderplus_hits(official_raw, orfs)
        else:
            amr_raw = raw / "amrprot_diamond.tsv"
            assert amrprot_db is not None and fam_path is not None
            if orfs:
                run_arg_diamond(
                    diamond=diamond_executable,
                    proteins=proteins,
                    database=amrprot_db,
                    output=amr_raw,
                    threads=threads,
                )
            else:
                amr_raw.write_text("", encoding="utf-8")
            amr_hits = parse_amrprot_hits(amr_raw, orfs, load_amrfinder_hierarchy(fam_path))

        all_hits = sorted(
            [*card_hits, *amr_hits, *sarg_hits],
            key=lambda hit: (
                hit.sequence_id,
                hit.orf_start,
                SOURCE_PRIORITY[hit.source],
                hit.accession,
            ),
        )
        consensus = consensus_hits(all_hits)
        hits_path = temporary / "arg_hits.tsv"
        consensus_path = temporary / "arg_consensus.tsv"
        summary_path = temporary / "annotation_summary.tsv"
        write_hits(hits_path, all_hits)
        write_hits(consensus_path, consensus)
        write_summary(summary_path, records, orfs, all_hits, consensus)

        comprehensive_outputs: list[Path] = []
        comprehensive_counts: dict[str, int] = {}
        mobileog_warning_count = 0
        if profile == "comprehensive":
            from mobiorigin.biological_evidence import (
                BACMET_MIN_IDENTITY,
                BACMET_MIN_QUERY_COVERAGE,
                MGE_MIN_IDENTITY,
                MGE_MIN_QUERY_COVERAGE,
                MOB_MIN_IDENTITY,
                MOB_MIN_QUERY_COVERAGE,
                VFDB_MIN_IDENTITY,
                VFDB_MIN_QUERY_COVERAGE,
                MobileOGParseWarning,
                arg_evidence,
                load_predictions,
                load_vfdb_metadata,
                parse_amrfinderplus_non_amr,
                parse_bacmet,
                parse_mge,
                parse_mob_marker,
                parse_mobileog,
                parse_vfdb,
                run_evidence_diamond,
                write_evidence,
                write_html_report,
                write_integrated_results,
                write_mobileog_warnings,
                write_publication_summary,
            )

            mobileog_warnings: list[MobileOGParseWarning] = []

            search_routes = {
                "vfdb": (
                    comprehensive_databases["vfdb_dmnd"],
                    VFDB_MIN_IDENTITY,
                    VFDB_MIN_QUERY_COVERAGE,
                ),
                "mobileog": (
                    comprehensive_databases["mobileog_dmnd"],
                    MGE_MIN_IDENTITY,
                    MGE_MIN_QUERY_COVERAGE,
                ),
                "bacmet": (
                    comprehensive_databases["bacmet_dmnd"],
                    BACMET_MIN_IDENTITY,
                    BACMET_MIN_QUERY_COVERAGE,
                ),
                "mob_rep": (
                    comprehensive_databases["mob_rep_dmnd"],
                    MOB_MIN_IDENTITY,
                    MOB_MIN_QUERY_COVERAGE,
                ),
                "mob_relaxase": (
                    comprehensive_databases["mob_relaxase_dmnd"],
                    MOB_MIN_IDENTITY,
                    MOB_MIN_QUERY_COVERAGE,
                ),
                "mob_mpf": (
                    comprehensive_databases["mob_mpf_dmnd"],
                    MOB_MIN_IDENTITY,
                    MOB_MIN_QUERY_COVERAGE,
                ),
            }
            legacy_isfinder_db = database_dir / "mge" / "legacy_isfinder.dmnd"
            legacy_isfinder_metadata = database_dir / "mge" / "legacy_isfinder_metadata.tsv"
            if legacy_isfinder_db.is_file() or legacy_isfinder_metadata.is_file():
                if not legacy_isfinder_db.is_file() or not legacy_isfinder_metadata.is_file():
                    raise FileNotFoundError(
                        "Legacy ISfinder installation is incomplete; both the database and "
                        "metadata files are required"
                    )
                search_routes["legacy_isfinder"] = (
                    legacy_isfinder_db,
                    MGE_MIN_IDENTITY,
                    MGE_MIN_QUERY_COVERAGE,
                )
            extended_raw: dict[str, Path] = {}
            for route, (database, identity, coverage) in search_routes.items():
                display_name = {
                    "vfdb": "VFDB virulence factors",
                    "mobileog": "mobileOG mobile-genetic-element proteins",
                    "bacmet": "BacMet biocide and metal resistance",
                    "mob_rep": "MOB-suite replicon markers",
                    "mob_relaxase": "MOB-suite relaxase markers",
                    "mob_mpf": "MOB-suite mating-pair-formation markers",
                    "legacy_isfinder": "authorized legacy ISfinder evidence",
                }[route]
                notify(f"Searching {display_name}")
                route_path = raw / f"{route}_diamond.tsv"
                if orfs:
                    run_evidence_diamond(
                        diamond=diamond_executable,
                        proteins=proteins,
                        database=database,
                        output=route_path,
                        threads=threads,
                        min_identity=identity,
                        min_query_coverage=coverage,
                    )
                else:
                    route_path.write_text("", encoding="utf-8")
                extended_raw[route] = route_path

            extended_hits = [*arg_evidence(all_hits)]
            if amrfinder_mode == "official":
                extended_hits.extend(parse_amrfinderplus_non_amr(official_raw, orfs))
            extended_hits.extend(
                parse_vfdb(
                    extended_raw["vfdb"],
                    orfs,
                    load_vfdb_metadata(comprehensive_databases["vfdb_metadata"]),
                )
            )
            extended_hits.extend(
                parse_mobileog(
                    extended_raw["mobileog"],
                    orfs,
                    warnings=mobileog_warnings,
                )
            )
            if "legacy_isfinder" in extended_raw:
                extended_hits.extend(
                    parse_mge(extended_raw["legacy_isfinder"], orfs, legacy_isfinder_metadata)
                )
            extended_hits.extend(
                parse_bacmet(
                    extended_raw["bacmet"],
                    orfs,
                    comprehensive_databases["bacmet_metadata"],
                )
            )
            for route, family in (
                ("mob_rep", "rep"),
                ("mob_relaxase", "mob"),
                ("mob_mpf", "mpf"),
            ):
                extended_hits.extend(parse_mob_marker(extended_raw[route], orfs, family))
            extended_hits.sort(
                key=lambda hit: (
                    hit.sequence_id,
                    hit.orf_start,
                    hit.evidence_group,
                    hit.source,
                    hit.accession,
                )
            )

            evidence_path = temporary / "biological_evidence.tsv"
            integrated_path = temporary / "mobiorigin_annotated_results.tsv"
            publication_path = temporary / "publication_summary.json"
            report_path = temporary / "mobiorigin_report.html"
            warnings_path = temporary / "annotation_warnings.tsv"
            write_mobileog_warnings(warnings_path, mobileog_warnings)
            mobileog_warning_count = len(mobileog_warnings)
            write_evidence(evidence_path, extended_hits)
            predictions = load_predictions(predictions_tsv, records) if predictions_tsv else {}
            notify("Building the integrated contig-level annotation table")
            integrated_rows = write_integrated_results(
                integrated_path,
                records,
                extended_hits,
                predictions,
                consensus_arg_hits=arg_evidence(consensus),
            )
            write_publication_summary(publication_path, integrated_rows, extended_hits)
            notify("Writing the biological-evidence HTML report")
            write_html_report(report_path, integrated_rows, publication_path)
            comprehensive_outputs = [
                evidence_path,
                integrated_path,
                publication_path,
                report_path,
                warnings_path,
            ]
            comprehensive_counts = Counter(hit.evidence_group for hit in extended_hits)

        database_identities = {
            "card_dmnd": _database_identity(card_db),
            "card_aro_index": _database_identity(card_index),
            "sarg_dmnd": _database_identity(sarg_db),
        }
        if amrfinder_mode == "official":
            assert official_executable is not None
            database_identities["amrfinderplus_executable"] = _database_identity(
                official_executable
            )
            assert amrfinder_database is not None
            database_identities["amrfinderplus_database"] = _directory_identity(amrfinder_database)
        else:
            assert amrprot_db is not None and fam_path is not None
            database_identities["amrprot_dmnd"] = _database_identity(amrprot_db)
            database_identities["amrfinder_fam"] = _database_identity(fam_path)
        if profile == "comprehensive":
            database_identities.update(
                {name: _database_identity(path) for name, path in comprehensive_databases.items()}
            )
            legacy_isfinder_db = database_dir / "mge" / "legacy_isfinder.dmnd"
            legacy_isfinder_metadata = database_dir / "mge" / "legacy_isfinder_metadata.tsv"
            if legacy_isfinder_db.is_file() and legacy_isfinder_metadata.is_file():
                database_identities["legacy_isfinder_dmnd"] = _database_identity(legacy_isfinder_db)
                database_identities["legacy_isfinder_metadata"] = _database_identity(
                    legacy_isfinder_metadata
                )

        evidence_counts = Counter(hit.source for hit in all_hits)
        provenance = {
            "schema_version": "mobiorigin-biological-annotation-v5",
            "mobiorigin_version": __version__,
            "input_fasta": _database_identity(input_fasta),
            "records": len(records),
            "bases": sum(len(record.sequence) for record in records),
            "predicted_orfs": len(orfs),
            "annotation_is_prediction_independent": True,
            "classification_labels_or_probabilities_accessed": False,
            "classification_labels_or_probabilities_changed": False,
            "amrfinder_route": amrfinder_mode,
            "annotation_profile": profile,
            "predictions_integrated": predictions_tsv is not None,
            "prediction_input_identity": (
                _database_identity(predictions_tsv) if predictions_tsv is not None else None
            ),
            "official_amrfinderplus_executed": amrfinder_mode == "official",
            "amrprot_diamond_is_supplemental_not_official_amrfinderplus": amrfinder_mode
            == "amrprot",
            "runtime": {
                "threads": threads,
                "diamond_executable": _database_identity(diamond_executable),
            },
            "thresholds": {
                "card_identity_percent": MIN_IDENTITY,
                "card_query_coverage_percent": MIN_QUERY_COVERAGE,
                "sarg_identity_percent": MIN_IDENTITY,
                "sarg_query_coverage_percent": MIN_QUERY_COVERAGE,
                "amrprot_identity_percent": MIN_IDENTITY if amrfinder_mode == "amrprot" else None,
                "amrprot_query_coverage_percent": (
                    MIN_QUERY_COVERAGE if amrfinder_mode == "amrprot" else None
                ),
                "amrfinderplus": (
                    "official curated database rules" if amrfinder_mode == "official" else None
                ),
            },
            "consensus_priority": list(AMR_SOURCES),
            "all_database_evidence_retained": True,
            "normalized_gene_vocabulary": {
                "schema_version": "mobiorigin-normalized-gene-v1",
                "source_specific_fields_retained": True,
                "changes_evidence_calls": False,
                "changes_origin_prediction": False,
            },
            "evidence_counts": {source: evidence_counts[source] for source in AMR_SOURCES},
            "consensus_arg_orfs": len(consensus),
            "comprehensive_evidence_counts": dict(sorted(comprehensive_counts.items())),
            "annotation_warnings": (
                {
                    "mobileog_rows_excluded": mobileog_warning_count,
                    "policy": (
                        "Unresolved mobileOG rows are excluded from evidence, retained in "
                        "annotation_warnings.tsv, and never interpreted by inference."
                    ),
                }
                if profile == "comprehensive"
                else None
            ),
            "evidence_priority_policy": (
                {
                    "type": "categorical_research_priority_not_clinical_risk",
                    "tiers": {
                        "A": "ARG plus relaxase and mating-pair-formation evidence",
                        "B": "ARG plus partial mobility, replication, or MGE evidence",
                        "C": "ARG without detected mobility context",
                        "D": "non-ARG biological evidence only",
                        "E": "no retained evidence",
                    },
                    "changes_origin_prediction": False,
                }
                if profile == "comprehensive"
                else None
            ),
            "third_party_database_payloads_bundled": False,
            "database_identities": database_identities,
            "output_identities": {
                path.name: _database_identity(path)
                for path in (
                    proteins,
                    hits_path,
                    consensus_path,
                    summary_path,
                    *comprehensive_outputs,
                )
            },
        }
        provenance_path = temporary / "annotation_provenance.json"
        atomic_json(provenance_path, provenance)
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        sums = "".join(
            f"{sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n" for path in files
        )
        atomic_text(temporary / "SHA256SUMS.txt", sums)
        notify("Publishing annotation outputs and checksum inventory")
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
