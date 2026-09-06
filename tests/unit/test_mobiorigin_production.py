"""Focused synthetic tests for the standalone MobiOrigin production package."""

from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mobiorigin import cli, model_setup, runtime
from mobiorigin.annotate import (
    ArgHit,
    Orf,
    _diamond_rows,
    annotate,
    consensus_hits,
    load_amrfinder_hierarchy,
    parse_amrfinderplus_hits,
    parse_amrprot_hits,
    parse_card_hits,
    parse_sarg_hits,
    predict_annotation_orfs,
    run_amrfinderplus,
)
from mobiorigin.annotation_database_retrieval import (
    download,
    prepare_official_annotation_sources,
)
from mobiorigin.annotation_database_setup import (
    ANNOTATION_MANIFEST_NAME,
    COMPREHENSIVE_DATABASE_FILES,
    MOBILEOG_COMPATIBILITY_FILE,
    check_annotation_databases,
    default_annotation_database_dir,
    setup_annotation_databases,
)
from mobiorigin.biological_evidence import (
    EvidenceHit,
    arg_evidence,
    load_predictions,
    parse_amrfinderplus_non_amr,
    parse_mobileog,
    write_integrated_results,
    write_publication_summary,
)
from mobiorigin.biological_evidence import (
    _rows as _evidence_rows,
)
from mobiorigin.database_setup import (
    DATABASE_FILENAMES,
    MANIFEST_NAME,
    check_databases,
    setup_databases,
)
from mobiorigin.fasta import FastaRecord, read_fasta, resolve_fasta_input
from mobiorigin.marker_database_builder import (
    DIAMOND_VERSION,
    _resolve_diamond,
    build_rep_proteins,
    translate,
)
from mobiorigin.marker_database_builder import (
    read_fasta as read_marker_fasta,
)
from mobiorigin.marker_features import (
    DATABASE_SHA256,
    OrfSummary,
    interval_union_length,
    load_database_manifest,
    marker_family_values,
    orf_values,
    parse_hits,
    predict_orfs,
    run_diamond,
)
from mobiorigin.mobileog import audit_mobileog_fasta, mobileog_fields
from mobiorigin.model import MobiOriginMLP, ModelLoadError, load_model
from mobiorigin.model_setup import (
    MODEL_ARCHIVE_ROOT,
    check_models,
    default_model_dir,
    resolve_model_dir,
    setup_models,
)
from mobiorigin.predict import (
    _write_predictions,
    configure_runtime,
    ensemble_probabilities,
    fuse_features,
    predict,
    selective_labels,
)
from mobiorigin.provenance import atomic_json, atomic_text, sha256_file
from mobiorigin.runtime import (
    MAX_THREADS,
    external_tool_environment,
    is_wsl_windows_mount,
    temporary_storage_report,
    validate_threads,
)
from mobiorigin.sequence_features import (
    FEATURE_DIM,
    extract_sequence_features,
    k7_canonical_vector,
    kmer_vector,
)
from mobiorigin.visualize import visualize
from mobiorigin.workflow import (
    default_database_dir,
    demo,
    doctor,
    resolve_database_dir,
    run_analysis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, value: str) -> Path:
    path.write_text(value, encoding="ascii")
    return path


@pytest.mark.parametrize("threads", [1, 8, 64, 128])
def test_thread_validation_accepts_supported_counts(threads: int) -> None:
    assert MAX_THREADS == 128
    assert validate_threads(threads) == threads


@pytest.mark.parametrize("threads", [0, 129])
def test_thread_validation_rejects_out_of_range_counts(threads: int) -> None:
    with pytest.raises(ValueError, match="Threads must be between 1 and 128"):
        validate_threads(threads)


def test_fasta_preserves_order_and_iupac(tmp_path: Path) -> None:
    path = write(tmp_path / "x.fasta", ">first description\nACGTRYSWKMBDHVN\n>second\nacgt\n")
    records = read_fasta(path)
    assert [record.identifier for record in records] == ["first", "second"]
    assert records[1].sequence == "ACGT"


def test_gzip_compressed_fasta_is_read_directly(tmp_path: Path) -> None:
    path = tmp_path / "assembly.fasta.gz"
    with gzip.open(path, "wt", encoding="ascii") as handle:
        handle.write(">compressed\n" + "ACGT" * 250 + "\n")
    records = read_fasta(path)
    assert records == [FastaRecord("compressed", "ACGT" * 250)]
    assert records[0].supported


def test_missing_fasta_reports_working_directory_and_nearby_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write(tmp_path / "final.contigs.fasta", ">seq\nACGT\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as captured:
        resolve_fasta_input(Path("final.contigs.fa"))
    message = str(captured.value)
    assert "final.contigs.fa" in message
    assert "final.contigs.fasta" in message
    assert f"Current directory: {tmp_path}" in message
    assert "absolute path" in message


def test_external_tool_environment_replaces_stale_temporary_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "native-temp"
    monkeypatch.setenv("MOBIORIGIN_TMPDIR", str(root))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "deleted"))
    with external_tool_environment("test") as (environment, temporary):
        assert temporary.parent == root
        assert temporary.is_dir()
        assert all(environment[name] == str(temporary) for name in ("TMPDIR", "TEMP", "TMP"))
    assert not temporary.exists()
    report = temporary_storage_report()
    assert report["status"] == "PASS"
    assert report["directory"] == str(root.resolve())


def test_wsl_windows_temporary_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "is_wsl", lambda: True)
    assert is_wsl_windows_mount(Path("/mnt/g/mobiorigin_tmp"))
    monkeypatch.setenv("MOBIORIGIN_TMPDIR", "/mnt/g/mobiorigin_tmp")
    report = temporary_storage_report()
    assert report["status"] == "FAIL"
    assert "Linux-native" in report["error"]


@pytest.mark.parametrize(
    "payload, message",
    [
        ("", "no records"),
        ("ACGT\n", "before its first header"),
        (">x\n", "empty"),
        (">x\nACGT-\n", "unsupported symbols"),
        (">x\nACGT\n>x\nACGT\n", "unique"),
    ],
)
def test_fasta_rejects_invalid_input(tmp_path: Path, payload: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        read_fasta(write(tmp_path / "bad.fasta", payload))


def test_supported_length_boundaries() -> None:
    assert FastaRecord("a", "A" * 1_000).supported
    assert FastaRecord("b", "A" * 500_000).supported
    assert not FastaRecord("c", "A" * 999).supported
    assert not FastaRecord("d", "A" * 500_001).supported


def test_sequence_features_are_deterministic_and_normalized() -> None:
    sequences = ["ACGT" * 260, "NRYKMSWBDHVACGT" * 70]
    first = extract_sequence_features(sequences)
    second = extract_sequence_features(sequences)
    assert first.shape == (2, FEATURE_DIM)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.isclose(np.linalg.norm(kmer_vector(sequences[0], 5)), 1.0)
    assert np.isclose(np.linalg.norm(k7_canonical_vector(sequences[0])), 1.0)


def test_short_and_invalid_kmer_behaviour() -> None:
    assert not kmer_vector("AC", 5).any()
    assert not k7_canonical_vector("ACGT").any()
    with pytest.raises(ValueError, match="Unsupported"):
        kmer_vector("ACGT", 6)


def test_marker_helper_semantics(tmp_path: Path) -> None:
    assert interval_union_length([]) == 0
    assert interval_union_length([(1, 10), (8, 20), (25, 30)]) == 26
    summary = OrfSummary(3, 270, (30, 50, 70), (1, 1, -1))
    assert np.allclose(orf_values(summary, 1_000), [0.27, 3, np.log1p(50), 0.5, 2 / 3])
    hits_path = write(
        tmp_path / "hits.tsv",
        "q\ts2\t70\t80\t1e-10\t90\t50\nq\ts1\t70\t80\t1e-10\t100\t50\n",
    )
    hit = parse_hits(hits_path)["q"]
    assert hit.subject_id == "s1"
    assert marker_family_values({"q": hit}, {"q": "x"}, "x", 2, 1.0) == [1, 0.5, 1, 2]


def test_malformed_marker_hit_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_hits(write(tmp_path / "hits.tsv", "too\tfew\n"))


def test_database_manifest_rejects_missing_or_changed_payload(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "mobiorigin-mob-suite-database-manifest-v1",
        "databases": {
            key: {"path": f"{key}.dmnd", "sha256": value} for key, value in DATABASE_SHA256.items()
        },
    }
    (tmp_path / "mobiorigin_mob_suite_database_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest identity"):
        load_database_manifest(tmp_path)


def test_database_setup_is_atomic_and_identity_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    hashes: dict[str, str] = {}
    for family, filename in DATABASE_FILENAMES.items():
        path = write(source / filename, f"{family}-database\n")
        hashes[family] = sha256_file(path)
    monkeypatch.setattr("mobiorigin.database_setup.DATABASE_SHA256", hashes)
    monkeypatch.setattr("mobiorigin.marker_features.DATABASE_SHA256", hashes)
    output = tmp_path / "installed"
    setup_databases(output, source_dir=source)
    assert load_database_manifest(output) == {
        family: output / filename for family, filename in DATABASE_FILENAMES.items()
    }
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["network_accessed"] is False
    assert (output / "THIRD_PARTY_DATABASE_NOTICE.txt").is_file()
    with pytest.raises(FileExistsError):
        setup_databases(output, source_dir=source)
    (source / DATABASE_FILENAMES["rep"]).write_text("changed\n", encoding="ascii")
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        setup_databases(failed, source_dir=source)
    assert not failed.exists()


def test_annotation_database_setup_is_atomic_and_identity_checked(tmp_path: Path) -> None:
    source = tmp_path / "authorized_source"
    for relative in COMPREHENSIVE_DATABASE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, f"database resource: {relative}\n")
    output = tmp_path / "annotation_databases"
    amrfinder = tmp_path / "amrfinder" / "2026-08-07.1"
    amrfinder.mkdir(parents=True)
    write(amrfinder / "version.txt", "2026-08-07.1\n")
    write(amrfinder / "AMR.LIB", "official AMRFinderPlus test database\n")
    with pytest.raises(PermissionError, match="accept-third-party-terms"):
        setup_annotation_databases(
            output,
            source_dir=source,
            amrfinder_database=amrfinder,
        )
    result = setup_annotation_databases(
        output,
        source_dir=source,
        amrfinder_database=amrfinder,
        accept_third_party_terms=True,
    )
    assert result["status"] == "PASS"
    assert result["profile"] == "comprehensive"
    assert result["resources_verified"] == len(COMPREHENSIVE_DATABASE_FILES) + 3
    assert result["mobileog_compatibility"]["status"] == "LEGACY_SOURCE_NOT_AUDITED"
    assert (output / MOBILEOG_COMPATIBILITY_FILE).is_file()
    assert (output / ANNOTATION_MANIFEST_NAME).is_file()
    assert (output / "THIRD_PARTY_ANNOTATION_DATABASE_NOTICE.txt").is_file()
    manifest = json.loads((output / ANNOTATION_MANIFEST_NAME).read_text())
    assert manifest["network_accessed"] is False
    assert manifest["third_party_terms_accepted"] is True
    with pytest.raises(FileExistsError):
        setup_annotation_databases(
            output,
            source_dir=source,
            amrfinder_database=amrfinder,
            accept_third_party_terms=True,
        )
    (output / COMPREHENSIVE_DATABASE_FILES[0]).write_text("changed\n", encoding="ascii")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        check_annotation_databases(output)


def test_model_setup_is_atomic_and_identity_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "seed.pt": b"frozen model bytes\n",
        "normalization.npy": b"frozen normalization bytes\n",
    }
    contracts = {
        name: {"bytes": len(payload), "sha256": __import__("hashlib").sha256(payload).hexdigest()}
        for name, payload in payloads.items()
    }
    monkeypatch.setattr(model_setup, "MODEL_ARTIFACTS", contracts)
    archive = tmp_path / "models.tar"
    with tarfile.open(archive, "w") as handle:
        for name, payload in payloads.items():
            information = tarfile.TarInfo(f"{MODEL_ARCHIVE_ROOT}/{name}")
            information.size = len(payload)
            handle.addfile(information, io.BytesIO(payload))
    monkeypatch.setattr(model_setup, "MODEL_ARCHIVE_SHA256", sha256_file(archive))

    output = tmp_path / "models"
    result = setup_models(output, archive=archive)
    assert result["status"] == "PASS"
    assert result["network_accessed"] is False
    assert result["artifacts_verified"] == 2
    assert check_models(output)["status"] == "PASS"
    assert (output / model_setup.MODEL_INSTALLATION_MANIFEST).is_file()
    with pytest.raises(FileExistsError):
        setup_models(output, archive=archive)
    (output / "seed.pt").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="identity changed"):
        check_models(output)


def test_model_resolution_supports_slim_and_legacy_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MOBIORIGIN_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    empty_package = tmp_path / "package_models"
    empty_package.mkdir()
    monkeypatch.setattr(model_setup, "packaged_model_dir", lambda: empty_package)
    assert default_model_dir() == tmp_path / "data/mobiorigin/models/dev1"
    assert resolve_model_dir() == default_model_dir()
    monkeypatch.setenv("MOBIORIGIN_MODEL_DIR", "~/frozen_models")
    assert resolve_model_dir() == Path("~/frozen_models").expanduser()


def test_annotation_database_setup_fails_without_complete_source(tmp_path: Path) -> None:
    source = tmp_path / "incomplete"
    source.mkdir()
    amrfinder = tmp_path / "amrfinder"
    amrfinder.mkdir()
    write(amrfinder / "version.txt", "test\n")
    output = tmp_path / "annotation_databases"
    with pytest.raises(FileNotFoundError, match="Missing annotation database file"):
        setup_annotation_databases(
            output,
            source_dir=source,
            amrfinder_database=amrfinder,
            accept_third_party_terms=True,
        )
    assert not output.exists()


def test_annotation_database_setup_adds_optional_legacy_isfinder(tmp_path: Path) -> None:
    source = tmp_path / "authorized_source"
    for relative in COMPREHENSIVE_DATABASE_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, f"database resource: {relative}\n")
    amrfinder = tmp_path / "amrfinder"
    amrfinder.mkdir()
    write(amrfinder / "version.txt", "test\n")
    legacy = tmp_path / "authorized_legacy_isfinder"
    legacy.mkdir()
    write(legacy / "isfinder.dmnd", "authorized legacy database\n")
    write(legacy / "mge_database.tsv", "ID\tClass\tSub_class\tgene_name\n")
    output = tmp_path / "annotation_databases"

    result = setup_annotation_databases(
        output,
        source_dir=source,
        amrfinder_database=amrfinder,
        legacy_isfinder_source_dir=legacy,
        accept_third_party_terms=True,
    )

    assert result["legacy_isfinder_installed"] is True
    assert (output / "mge/legacy_isfinder.dmnd").read_text() == ("authorized legacy database\n")
    manifest = json.loads((output / ANNOTATION_MANIFEST_NAME).read_text())
    assert manifest["mge_default_provider"] == "mobileOG-db"
    assert manifest["legacy_isfinder_installed"] is True


def test_legacy_isfinder_is_rejected_for_arg_only_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires the comprehensive profile"):
        setup_annotation_databases(
            tmp_path / "output",
            legacy_isfinder_source_dir=tmp_path / "legacy",
            profile="arg",
            accept_third_party_terms=True,
        )


def test_default_annotation_database_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MOBIORIGIN_ANNOTATION_DATABASE_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_annotation_database_dir() == tmp_path / "mobiorigin/annotation_databases"
    monkeypatch.setenv("MOBIORIGIN_ANNOTATION_DATABASE_DIR", "~/custom_annotation")
    assert default_annotation_database_dir() == Path("~/custom_annotation").expanduser()


def test_annotation_download_verifies_and_reuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"official database payload\n"

    class Response(io.BytesIO):
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    monkeypatch.setattr(
        "mobiorigin.annotation_database_retrieval.urllib.request.urlopen",
        lambda *args, **kwargs: Response(payload),
    )
    expected = __import__("hashlib").sha256(payload).hexdigest()
    destination = tmp_path / "database.bin"
    first = download("https://example.test/database", destination, expected_sha256=expected)
    second = download("https://example.test/database", destination, expected_sha256=expected)
    assert first["reused"] is False
    assert second["reused"] is True
    assert destination.read_bytes() == payload


def test_prepare_official_annotation_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()

    def make_tar(path: Path, members: dict[str, bytes], mode: str) -> None:
        with tarfile.open(path, mode) as archive:
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    card = fixtures / "card.tar.bz2"
    make_tar(
        card,
        {
            "protein_fasta_protein_homolog_model.fasta": b">card\nMAAA\n",
            "aro_index.tsv": b"ARO Accession\tARO Name\nARO:1\ttest\n",
        },
        "w:bz2",
    )
    sarg = fixtures / "sarg.tar.gz"
    make_tar(sarg, {"ARGs_OAP/DB/SARG.2.2.fasta": b">sarg\nMAAA\n"}, "w:gz")
    vfdb = fixtures / "vfdb.gz"
    with gzip.open(vfdb, "wb") as handle:
        handle.write(b">VFG000001(gb|WP_1) gene [Toxin] [Bacterium]\nMAAA\n")
    mobileog = write(
        fixtures / "mobileog.faa",
        ">mobileOG_000000007|xis|A0A653FUH0|IE|RRR|N/A|Multiple|Manual\nMAAA\n",
    )
    bacmet_fasta = write(fixtures / "bacmet.faa", ">BAC0001|abeM|tr|Q1\nMAAA\n")
    bacmet_metadata = write(
        fixtures / "bacmet.tsv", "Accession\tBacMet_ID\tGene_name\nQ1\tBAC0001\tabeM\n"
    )
    sources = {
        "CARD": card,
        "SARG": sarg,
        "VFDB": vfdb,
        "mobileOG-db": mobileog,
        "BacMet-fasta": bacmet_fasta,
        "BacMet-metadata": bacmet_metadata,
    }

    def fake_download(url: str, destination: Path, **kwargs: object) -> dict[str, object]:
        if "card" in url:
            source = sources["CARD"]
        elif "pythonhosted" in url:
            source = sources["SARG"]
        elif "VFDB" in url:
            source = sources["VFDB"]
        elif "zenodo" in url:
            source = sources["mobileOG-db"]
        elif url.endswith(".fasta"):
            source = sources["BacMet-fasta"]
        else:
            source = sources["BacMet-metadata"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return {"url": url, "path": str(destination), "bytes": destination.stat().st_size}

    def fake_build(diamond: Path, fasta: Path, destination: Path) -> None:
        assert fasta.is_file()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(f"built from {fasta.name}\n", encoding="ascii")

    monkeypatch.setattr("mobiorigin.annotation_database_retrieval.download", fake_download)
    monkeypatch.setattr(
        "mobiorigin.annotation_database_retrieval._diamond_executable", lambda value: value
    )
    monkeypatch.setattr("mobiorigin.annotation_database_retrieval._build_diamond", fake_build)
    marker = tmp_path / "marker"
    marker.mkdir()
    for filename in ("rep_proteins.dmnd", "mob_proteins.dmnd", "mpf_proteins.dmnd"):
        write(marker / filename, f"{filename}\n")
    amrfinder = tmp_path / "amrfinder"
    amrfinder.mkdir()
    write(amrfinder / "version.txt", "test\n")
    prepared = tmp_path / "prepared"
    resolved, details = prepare_official_annotation_sources(
        prepared,
        profile="comprehensive",
        cache_dir=tmp_path / "cache",
        diamond=Path("diamond"),
        marker_database_dir=marker,
        amrfinder_database=amrfinder,
        amrfinder_update=Path("amrfinder_update"),
    )
    assert resolved == amrfinder
    assert set(details["downloads"]) == set(sources)
    assert (prepared / "mge/mobileog.dmnd").is_file()
    compatibility = json.loads(
        (prepared / "mge/mobileog_compatibility.json").read_text(encoding="utf-8")
    )
    assert compatibility["status"] == "PASS"
    assert compatibility["headers_supported"] == 1
    assert not (prepared / "mge/legacy_isfinder.dmnd").exists()
    assert "Toxin" in (prepared / "vfdb/vfdb_indx.txt").read_text()


def test_installation_environments_keep_incompatible_stacks_separate() -> None:
    runtime = (PROJECT_ROOT / "environment.yml").read_text(encoding="utf-8")
    database = (PROJECT_ROOT / "environment.mob-database.yml").read_text(encoding="utf-8")
    marker_build = (PROJECT_ROOT / "environment.marker-build.yml").read_text(encoding="utf-8")
    assert "name: mobiorigin\n" in runtime
    assert "numpy=1.26.4" in runtime
    assert "pytorch=2.5.1=cpu*" in runtime
    assert "diamond>=2.1" in runtime
    assert "ncbi-amrfinderplus=4.2.7" in runtime
    assert "-e ." not in runtime
    assert "mob_suite" not in runtime
    assert "pandas" not in runtime
    assert "name: mobiorigin-db\n" in database
    assert "mob_suite=3.1.8" in database
    assert "numpy>=1.11.1,<1.23.5" in database
    assert "pandas>=0.22,<=1.5.3" in database
    assert "blast>=2.9,<2.16" in database
    assert "diamond=2.0.15" not in database
    assert "pytorch" not in database
    assert "name: mobiorigin-marker-build\n" in marker_build
    assert "diamond=2.0.15" in marker_build
    assert "mob_suite" not in marker_build


def test_guided_installer_is_non_errexit_and_runs_demo() -> None:
    installer = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")
    assert "set -e" not in installer
    assert "env create --file" in installer
    assert "mobiorigin doctor --software-only" in installer
    assert "scripts/setup_mobiorigin_databases.sh" in installer
    assert "mobiorigin demo" in installer
    assert "--comprehensive" in installer
    assert "--annotation-database-dir" in installer
    assert "--skip-annotation-databases" in installer
    assert "--software-only" in installer
    assert "${HOME}/mobiorigin_demo" in installer
    assert not any(line.lstrip().startswith("rm ") for line in installer.splitlines())


def test_pypi_workflow_is_oidc_only_and_fail_closed() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/publish-pypi.yml").read_text(encoding="utf-8")
    assert "id-token: write" in workflow
    assert "environment:\n      name: pypi" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "password:" not in workflow
    assert "PYPI_TOKEN" not in workflow
    assert "persist-credentials: false" in workflow
    assert "100_000_000" in workflow
    assert "Tag/package mismatch" in workflow


def test_default_database_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MOBIORIGIN_DATABASE_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    assert default_database_dir() == tmp_path / "data" / "mobiorigin" / "marker_databases"
    monkeypatch.setenv("MOBIORIGIN_DATABASE_DIR", str(tmp_path / "custom"))
    assert resolve_database_dir(None) == tmp_path / "custom"
    assert resolve_database_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_doctor_reports_software_and_database_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "mobiorigin.workflow._command_version",
        lambda command, arguments: {
            "status": "PASS",
            "executable": f"/bin/{command}",
            "version": "test",
        },
    )
    monkeypatch.setattr(
        "mobiorigin.workflow.check_databases", lambda path: {"status": "PASS", "path": str(path)}
    )
    monkeypatch.setattr(
        "mobiorigin.workflow.check_models", lambda path: {"status": "PASS", "path": str(path)}
    )
    monkeypatch.setattr(
        "mobiorigin.workflow.check_annotation_databases",
        lambda path, **kwargs: {"status": "PASS", "path": str(path), **kwargs},
    )
    result = doctor(
        database_dir=tmp_path / "db",
        annotation_database_dir=tmp_path / "annotation",
    )
    assert result["status"] == "PASS"
    assert result["database"]["status"] == "PASS"
    assert result["annotation_database"]["status"] == "PASS"
    assert result["temporary_storage"]["status"] == "PASS"
    assert doctor(software_only=True)["status"] == "PASS"


def test_compact_cli_output_omits_large_provenance_inventories() -> None:
    result = {
        "status": "PASS",
        "resources_verified": 174,
        "database_sha256": {"large/file": "digest"},
        "mobileog_compatibility": {
            "status": "PASS_WITH_EXCLUSIONS",
            "headers_excluded": 3,
            "unsupported_examples": ["long header"],
        },
    }
    compact = cli._compact_result(result)
    assert compact == {
        "status": "PASS",
        "resources_verified": 174,
        "mobileog_compatibility": {
            "status": "PASS_WITH_EXCLUSIONS",
            "headers_excluded": 3,
        },
    }


def test_atomic_run_and_demo_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_fasta = write(tmp_path / "input.fasta", ">demo\n" + "A" * 1200 + "\n")
    predictions_text = (
        "sequence_id\tlength_bp\tprediction\tp_chromosome\tp_plasmid\tp_phage\t"
        "plasmid_score\tabstention_reason\n"
        "demo\t1200\tplasmid\t0.1\t0.8\t0.1\t0.7\t\n"
    )

    def fake_predict(**kwargs: object) -> None:
        output = Path(str(kwargs["output_dir"]))
        output.mkdir()
        write(output / "predictions.tsv", predictions_text)
        write(output / "provenance.json", "{}\n")

    def fake_annotate(**kwargs: object) -> None:
        output = Path(str(kwargs["output_dir"]))
        output.mkdir()
        write(
            output / "mobiorigin_annotated_results.tsv",
            "sequence_id\tprediction\tconsensus_arg_orfs\tmge_hits\t"
            "mobility_marker_hits\tevidence_priority_tier\n"
            "demo\tplasmid\t1\t1\t1\tB\n",
        )
        write(output / "mobiorigin_report.html", "<html></html>\n")
        write(output / "biological_evidence.tsv", "sequence_id\n")

    monkeypatch.setattr("mobiorigin.workflow.predict", fake_predict)
    monkeypatch.setattr("mobiorigin.workflow.annotate", fake_annotate)
    output = tmp_path / "analysis"
    run_analysis(
        input_fasta=input_fasta,
        output_dir=output,
        database_dir=tmp_path / "db",
    )
    progress = capsys.readouterr().err
    assert "[MobiOrigin 1/5] Preflight" in progress
    assert "[MobiOrigin 2/5] Prediction" in progress
    assert "[MobiOrigin 3/5] Annotation" in progress
    assert "[MobiOrigin 4/5] Visualization" in progress
    assert "[MobiOrigin 5/5] Finalization" in progress
    assert (output / "README_RESULTS.txt").is_file()
    assert (output / "annotation" / "mobiorigin_report.html").is_file()
    assert (output / "visualization" / "mobiorigin_dashboard.html").is_file()
    assert (
        json.loads(
            (output / "visualization" / "visualization_summary.json").read_text(encoding="utf-8")
        )["annotated_results_sha256"]
        is not None
    )
    with pytest.raises(FileExistsError):
        run_analysis(
            input_fasta=input_fasta,
            output_dir=output,
            database_dir=tmp_path / "db",
        )
    demo_output = tmp_path / "demo"
    result = demo(output_dir=demo_output, database_dir=tmp_path / "db")
    assert result["status"] == "PASS"
    assert result["verification_profile"] == "basic"
    assert result["records"] == 1
    assert not (demo_output / "annotation").exists()
    comprehensive_demo = tmp_path / "comprehensive-demo"
    comprehensive_result = demo(
        output_dir=comprehensive_demo,
        database_dir=tmp_path / "db",
        annotation_database_dir=tmp_path / "annotation-db",
        comprehensive=True,
    )
    assert comprehensive_result["verification_profile"] == "comprehensive"
    assert comprehensive_result["annotation_report"].endswith("mobiorigin_report.html")
    assert (comprehensive_demo / "annotation" / "mobiorigin_report.html").is_file()

    def failing_annotation(**kwargs: object) -> None:
        raise ValueError("synthetic annotation failure")

    monkeypatch.setattr("mobiorigin.workflow.annotate", failing_annotation)
    failed_output = tmp_path / "failed-analysis"
    with pytest.raises(RuntimeError, match="incomplete state was retained"):
        run_analysis(
            input_fasta=input_fasta,
            output_dir=failed_output,
            database_dir=tmp_path / "db",
        )
    retained = tmp_path / "failed-analysis.failed"
    assert not failed_output.exists()
    assert (retained / "predictions" / "predictions.tsv").is_file()
    failure = json.loads((retained / "ANALYSIS_FAILED.json").read_text(encoding="utf-8"))
    assert failure["completed_stages"] == ["prediction"]
    assert failure["scientific_status"] == "incomplete_not_for_interpretation"


def test_cli_dispatches_run_doctor_and_demo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "run_analysis", lambda **kwargs: observed.update(kwargs))
    cli.main(
        [
            "run",
            "--input-fasta",
            str(tmp_path / "input.fasta"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert observed["database_dir"] is None
    assert observed["annotation_database_dir"] is None
    assert observed["annotation_profile"] == "comprehensive"
    assert observed["skip_annotation"] is False
    monkeypatch.setattr(cli, "doctor", lambda **kwargs: {"status": "PASS"})
    cli.main(["doctor", "--software-only"])
    assert '"status": "PASS"' in capsys.readouterr().out
    demo_arguments: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "demo",
        lambda **kwargs: demo_arguments.update(kwargs) or {"status": "PASS"},
    )
    cli.main(
        [
            "demo",
            "--output-dir",
            str(tmp_path / "demo"),
            "--annotation-database-dir",
            str(tmp_path / "annotation"),
            "--comprehensive",
        ]
    )
    assert '"status": "PASS"' in capsys.readouterr().out
    assert demo_arguments["annotation_database_dir"] == tmp_path / "annotation"
    assert demo_arguments["comprehensive"] is True


def test_database_helper_is_guided_non_destructive_and_non_errexit() -> None:
    helper = PROJECT_ROOT / "scripts/setup_mobiorigin_databases.sh"
    payload = helper.read_text(encoding="utf-8")
    assert payload.startswith("#!/usr/bin/env bash\n")
    assert "set -e" not in payload
    assert "environment.mob-database.yml" in payload
    assert "environment.marker-build.yml" in payload
    assert "run -n mobiorigin-db mob_init" in payload
    assert 'root / "databases"' in payload
    assert "marker_database_builder.py" in payload
    assert "run -n mobiorigin-marker-build python" in payload
    assert "PYTHONNOUSERSITE=1" in payload
    assert "unset PYTHONPATH" in payload
    assert '--diamond "$MARKER_DIAMOND"' in payload
    assert 'Path(sys.prefix) / "bin" / "diamond"' in payload
    assert "run -n mobiorigin mobiorigin setup-databases" in payload
    assert "--platform osx-64" in payload
    assert "Rosetta" in payload
    assert "Existing MobiOrigin models and marker databases are valid" in payload
    assert not any(line.lstrip().startswith("rm ") for line in payload.splitlines())


def test_bundled_four_class_example_and_runner_are_public_and_portable() -> None:
    example = PROJECT_ROOT / "src/mobiorigin/data/examples/annotated_assembly_example.fasta"
    records = read_fasta(example)
    assert [record.identifier for record in records] == [
        "assembly_example_chromosome_01",
        "assembly_example_chromosome_02",
        "assembly_example_plasmid_01",
        "assembly_example_plasmid_02",
        "assembly_example_phage_01",
        "assembly_example_phage_02",
        "assembly_example_unclassified_01",
        "assembly_example_unclassified_02",
    ]
    assert all(record.supported for record in records)
    assert sum(len(record.sequence) for record in records) == 160_054

    runner = PROJECT_ROOT / "scripts/run_mobiorigin_assembly_example.sh"
    payload = runner.read_text(encoding="utf-8")
    assert payload.startswith("#!/usr/bin/env bash\n")
    assert "set -e" not in payload
    assert "annotated_assembly_example.fasta" in payload
    assert "chromosome=2, plasmid=2, phage=2, unclassified=2" in payload
    assert "ANNOTATION_DATABASE" in payload
    assert "accuracy or prevalence" in payload


def test_frozen_marker_translation_is_deterministic(tmp_path: Path) -> None:
    source = write(tmp_path / "rep.dna.fas", ">record description\nATG" + "GCT" * 30 + "TAA")
    destination = tmp_path / "rep_proteins.faa"
    build_rep_proteins(source, destination)
    records = list(read_marker_fasta(destination))
    assert records[0] == ("record description_s1_f0_o0", "M" + "A" * 30)
    repeated = tmp_path / "repeated.faa"
    build_rep_proteins(source, repeated)
    assert repeated.read_bytes() == destination.read_bytes()
    assert translate("ATGGCTTAA") == "MA*"
    assert translate("GCN") == "A"
    assert DIAMOND_VERSION == "2.0.15"


def test_database_setup_missing_source_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="setup_mobiorigin_databases.sh"):
        setup_databases(tmp_path / "output", source_dir=tmp_path / "missing")


def test_database_check_missing_manifest_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="setup_mobiorigin_databases.sh"):
        check_databases(tmp_path / "missing")


def test_model_round_trip_and_rejections(tmp_path: Path) -> None:
    configure_runtime()
    model = MobiOriginMLP(input_dim=8)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    loaded = load_model(path, input_dim=8)
    assert tuple(loaded.state_dict()) == tuple(model.state_dict())
    wrong = tmp_path / "wrong.pt"
    torch.save(model.state_dict(), wrong)
    with pytest.raises(ModelLoadError, match="architecture"):
        load_model(wrong, input_dim=9)
    unsafe = tmp_path / "unsafe.pt"
    torch.save({"bad": "not a tensor"}, unsafe)
    with pytest.raises(ModelLoadError):
        load_model(unsafe, input_dim=8)
    other_architecture = tmp_path / "other_architecture.pt"
    torch.save(
        MobiOriginMLP(input_dim=8, hidden_dims=(512, 128, 32)).state_dict(),
        other_architecture,
    )
    with pytest.raises(ModelLoadError, match="architecture"):
        load_model(other_architecture, input_dim=8)


def test_fusion_and_selective_policy() -> None:
    sequence = np.zeros((2, 9_557), dtype=np.float32)
    marker = np.ones((2, 17), dtype=np.float32)
    normalization = np.vstack([np.zeros(17, dtype=np.float32), np.ones(17, dtype=np.float32)])
    fused = fuse_features(sequence, marker, normalization)
    assert fused.shape == (2, 9_574)
    probabilities = np.asarray(
        [[0.1, 0.54, 0.36], [0.1, 0.8, 0.1], [0.6, 0.3, 0.1]], dtype=np.float32
    )
    labels, scores = selective_labels(probabilities)
    assert labels == ["unclassified", "plasmid", "chromosome"]
    assert scores.shape == (3,)
    with pytest.raises(ValueError):
        fuse_features(sequence, np.ones((3, 17), dtype=np.float32), normalization)
    with pytest.raises(ValueError):
        selective_labels(np.zeros((2, 4), dtype=np.float32))


class ConstantModel(torch.nn.Module):
    def __init__(self, logits: tuple[float, float, float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(logits, dtype=torch.float32))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.values.repeat(len(values), 1)


def test_ensemble_is_equal_weight_and_normalized() -> None:
    values = np.zeros((2, 9_574), dtype=np.float32)
    models = [ConstantModel((1, 2, 3)), ConstantModel((3, 2, 1)), ConstantModel((2, 3, 1))]
    first = ensemble_probabilities(models, values)  # type: ignore[arg-type]
    second = ensemble_probabilities(models, values)  # type: ignore[arg-type]
    assert np.array_equal(first, second)
    assert np.allclose(first.sum(axis=1), 1.0)


def test_synthetic_orf_prediction(tmp_path: Path) -> None:
    proteins = tmp_path / "proteins.faa"
    summaries, query_map = predict_orfs([FastaRecord("x", "ATG" + "GCC" * 400)], proteins)
    assert summaries["x"].count >= 0
    assert all(identifier.startswith("x__orf_") for identifier in query_map)
    assert proteins.is_file()


def test_diamond_transport_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "hits.tsv"
    monkeypatch.setattr(
        "mobiorigin.marker_features.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""),
    )
    run_diamond(Path("diamond"), tmp_path / "proteins.faa", tmp_path / "rep.dmnd", output, 1)
    assert output.read_text() == ""
    monkeypatch.setattr(
        "mobiorigin.marker_features.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="failed"),
    )
    with pytest.raises(RuntimeError, match="failed"):
        run_diamond(
            Path("diamond"),
            tmp_path / "proteins.faa",
            tmp_path / "rep.dmnd",
            output,
            1,
        )


def test_prediction_table_schema_and_abstention(tmp_path: Path) -> None:
    path = tmp_path / "predictions.tsv"
    records = [FastaRecord("a", "A" * 1_000), FastaRecord("b", "A" * 999)]
    probabilities = np.asarray([[0.1, 0.8, 0.1], [1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
    _write_predictions(
        path,
        records,
        probabilities,
        ["plasmid", "unclassified"],
        np.asarray([0.7, 0], dtype=np.float32),
    )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["sequence_id"] for row in rows] == ["a", "b"]
    assert rows[1]["abstention_reason"] == "unsupported_length"


def test_atomic_helpers_and_hash(tmp_path: Path) -> None:
    text = tmp_path / "x.txt"
    atomic_text(text, "hello\n")
    assert len(sha256_file(text)) == 64
    payload = tmp_path / "x.json"
    atomic_json(payload, {"value": 1})
    assert json.loads(payload.read_text()) == {"value": 1}


def test_predict_synthetic_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fasta = write(
        tmp_path / "input.fasta",
        f">supported\n{'ACGT' * 250}\n>short\n{'ACGT' * 249}\n",
    )
    normalization = np.vstack([np.zeros(17, dtype=np.float32), np.ones(17, dtype=np.float32)])
    models = [ConstantModel((0.1, 3.0, 0.1))] * 3
    monkeypatch.setattr("mobiorigin.predict.configure_runtime", lambda: None)
    monkeypatch.setattr("mobiorigin.predict.load_artifacts", lambda _: (models, normalization))
    monkeypatch.setattr("mobiorigin.predict.load_database_manifest", lambda _: {})
    monkeypatch.setattr(
        "mobiorigin.predict.extract_marker_features",
        lambda records, **kwargs: np.zeros((len(records), 17), dtype=np.float32),
    )
    output = tmp_path / "output"
    predict(
        input_fasta=fasta,
        output_dir=output,
        database_dir=tmp_path,
        threads=1,
        model_dir=tmp_path,
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "SHA256SUMS.txt",
        "predictions.tsv",
        "provenance.json",
    ]
    with (output / "predictions.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["prediction"] == "plasmid"
    assert rows[1]["prediction"] == "unclassified"
    assert rows[1]["abstention_reason"] == "unsupported_length"
    assert json.loads((output / "provenance.json").read_text())["unsupported_length_records"] == 1
    with pytest.raises(FileExistsError):
        predict(
            input_fasta=fasta,
            output_dir=output,
            database_dir=tmp_path,
            threads=1,
            model_dir=tmp_path,
        )


def test_cli_dispatches_predict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "predict", lambda **kwargs: observed.update(kwargs))
    cli.main(
        [
            "predict",
            "--input-fasta",
            str(tmp_path / "input.fasta"),
            "--output-dir",
            str(tmp_path / "out"),
            "--database-dir",
            str(tmp_path / "db"),
            "--threads",
            "4",
        ]
    )
    assert observed["threads"] == 4


def test_cli_dispatches_database_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "setup_databases", lambda **kwargs: observed.update(kwargs))
    cli.main(
        [
            "setup-databases",
            "--source-dir",
            str(tmp_path / "official_source"),
            "--output-dir",
            str(tmp_path / "db"),
        ]
    )
    assert observed == {
        "output_dir": tmp_path / "db",
        "source_dir": tmp_path / "official_source",
    }


def test_cli_dispatches_annotation_database_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "setup_annotation_databases",
        lambda **kwargs: observed.update(kwargs) or {"status": "PASS"},
    )
    cli.main(
        [
            "setup-databases",
            "--component",
            "annotation",
            "--source-dir",
            str(tmp_path / "authorized_source"),
            "--output-dir",
            str(tmp_path / "annotation_databases"),
            "--profile",
            "arg",
            "--amrfinder-database",
            str(tmp_path / "amrfinder"),
            "--accept-third-party-terms",
        ]
    )
    assert observed == {
        "output_dir": tmp_path / "annotation_databases",
        "source_dir": tmp_path / "authorized_source",
        "amrfinder_database": tmp_path / "amrfinder",
        "marker_database_dir": Path.home() / ".local/share/mobiorigin/marker_databases",
        "legacy_isfinder_source_dir": None,
        "cache_dir": None,
        "diamond": Path("diamond"),
        "amrfinder_update": Path("amrfinder_update"),
        "profile": "arg",
        "accept_third_party_terms": True,
    }


def test_cli_dispatches_annotation_database_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "check_annotation_databases",
        lambda database_dir, **kwargs: observed.update({"database_dir": database_dir, **kwargs})
        or {"status": "PASS"},
    )
    cli.main(
        [
            "setup-databases",
            "--component",
            "annotation",
            "--check",
            "--output-dir",
            str(tmp_path / "annotation_databases"),
        ]
    )
    assert observed == {
        "database_dir": tmp_path / "annotation_databases",
        "profile": "comprehensive",
    }


def test_database_check_verifies_diamond_and_frozen_databases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    database_dir = tmp_path / "db"
    database_dir.mkdir()
    write(database_dir / MANIFEST_NAME, "{}\n")
    monkeypatch.setattr(
        "mobiorigin.database_setup.resolve_executable",
        lambda *args, **kwargs: Path("/bin/diamond"),
    )
    monkeypatch.setattr(
        "mobiorigin.database_setup.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="diamond version 2.1.9\n", stderr=""
        ),
    )
    monkeypatch.setattr(
        "mobiorigin.database_setup.load_database_manifest",
        lambda path: {family: path / filename for family, filename in DATABASE_FILENAMES.items()},
    )
    result = check_databases(database_dir)
    assert result["status"] == "PASS"
    assert result["databases_verified"] == 3
    assert result["diamond_version"] == "diamond version 2.1.9"


def test_resolve_executable_prefers_active_python_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = tmp_path / "environment"
    environment_bin = environment / "bin"
    host_bin = tmp_path / "host-bin"
    environment_bin.mkdir(parents=True)
    host_bin.mkdir()
    environment_diamond = environment_bin / "diamond"
    host_diamond = host_bin / "diamond"
    environment_diamond.write_text("#!/bin/sh\necho environment\n", encoding="utf-8")
    host_diamond.write_text("#!/bin/sh\necho host\n", encoding="utf-8")
    environment_diamond.chmod(0o755)
    host_diamond.chmod(0o755)
    monkeypatch.setattr(runtime.sys, "prefix", str(environment))
    monkeypatch.setenv("PATH", str(host_bin))

    assert runtime.resolve_executable("diamond") == environment_diamond.resolve()


def test_resolve_executable_honors_an_explicit_tool_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = tmp_path / "tools" / "diamond"
    explicit.parent.mkdir()
    explicit.write_text("#!/bin/sh\necho explicit\n", encoding="utf-8")
    explicit.chmod(0o755)
    monkeypatch.setattr(runtime.sys, "prefix", str(tmp_path / "environment"))

    assert runtime.resolve_executable(explicit) == explicit.resolve()


def test_marker_builder_prefers_its_isolated_environment_diamond(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = tmp_path / "marker-build"
    environment_diamond = environment / "bin" / "diamond"
    host_diamond = tmp_path / "host-bin" / "diamond"
    environment_diamond.parent.mkdir(parents=True)
    host_diamond.parent.mkdir(parents=True)
    environment_diamond.write_text("#!/bin/sh\necho 2.0.15\n", encoding="utf-8")
    host_diamond.write_text("#!/bin/sh\necho 2.1.11\n", encoding="utf-8")
    environment_diamond.chmod(0o755)
    host_diamond.chmod(0o755)
    monkeypatch.setattr("mobiorigin.marker_database_builder.sys.prefix", str(environment))
    monkeypatch.setenv("PATH", str(host_diamond.parent))

    assert _resolve_diamond(Path("diamond")) == environment_diamond.resolve()


def test_cli_dispatches_database_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "check_databases",
        lambda database_dir, **kwargs: observed.update({"database_dir": database_dir, **kwargs})
        or {"status": "PASS"},
    )
    cli.main(["setup-databases", "--check", "--output-dir", str(tmp_path / "db")])
    assert observed == {"database_dir": tmp_path / "db", "diamond": Path("diamond")}


def test_visualization_outputs_tables_svg_and_html(tmp_path: Path) -> None:
    predictions = write(
        tmp_path / "predictions.tsv",
        "sequence_id\tlength_bp\tprediction\tp_chromosome\tp_plasmid\tp_phage\t"
        "plasmid_score\tabstention_reason\n"
        "a\t1500\tplasmid\t0.1\t0.8\t0.1\t0.7\t\n"
        "b\t3000\tchromosome\t0.8\t0.1\t0.1\t-0.7\t\n"
        "c\t8000\tphage\t0.1\t0.1\t0.8\t-0.7\t\n"
        "d\t60000\tunclassified\t0.3\t0.4\t0.3\t0.1\tlow_plasmid_margin\n",
    )
    annotated = write(
        tmp_path / "annotated.tsv",
        "sequence_id\tprediction\tconsensus_arg_orfs\tmge_hits\tmobility_marker_hits\t"
        "evidence_priority_tier\tlength_bp\targ_genes\targ_gene_families\t"
        "arg_drug_classes\targ_mechanisms\tvirulence_classes\tmge_classes\t"
        "stress_classes\tbacmet_gene_families\tbacmet_classes\tmobility_class\t"
        "mobility_marker_types\t"
        "evidence_priority_label\trecommended_follow_up\n"
        "a\tplasmid\t1\t1\t2\tA\t1500\tblaX\tclass A beta-lactamase\t"
        "beta-lactam\tantibiotic inactivation\ttoxin\tintegration_excision\t\t"
        "efflux pump\tTriclosan\tconjugative\trelaxase;mating_pair_formation\t"
        "ARG-bearing conjugative "
        "candidate\tConfirm transferability\n"
        "b\tchromosome\t0\t0\t0\tE\t3000\t\t\t\t\t\t\t\t\t\t"
        "not_applicable\t\tNo retained annotation evidence\tReview absence\n"
        "c\tphage\t0\t0\t1\tD\t8000\t\t\t\t\ttoxin\t\t\t\t\t"
        "not_applicable\treplicon\tNon-ARG biological-evidence candidate\tReview hit\n"
        "d\tunclassified\t0\t0\t0\tE\t60000\t\t\t\t\t\t\t\t\t\t"
        "not_applicable\t\tNo retained annotation evidence\tReview absence\n",
    )
    output = tmp_path / "visualization"
    visualize(
        predictions_tsv=predictions,
        annotated_results_tsv=annotated,
        output_dir=output,
    )
    assert {
        "prediction_summary.tsv",
        "prediction_by_length_bin.tsv",
        "visualization_summary.json",
        "mobiorigin_summary.svg",
        "mobiorigin_annotation_summary.svg",
        "mobiorigin_priority_candidates.svg",
        "mobiorigin_arg_classes.svg",
        "mobiorigin_mge_classes.svg",
        "mobiorigin_virulence_classes.svg",
        "mobiorigin_bacmet_categories.svg",
        "annotation_class_summary.tsv",
        "evidence_tier_summary.tsv",
        "priority_candidates.tsv",
        "mobiorigin_dashboard.html",
        "SHA256SUMS.txt",
    } == {path.name for path in output.iterdir()}
    summary = json.loads((output / "visualization_summary.json").read_text())
    assert summary["records"] == 4
    assert summary["evidence_priority_tier_counts"]["A"] == 1
    assert summary["conjugative_candidates"] == 1
    assert summary["accuracy_metrics_calculated"] is False
    dashboard = (output / "mobiorigin_dashboard.html").read_text()
    assert "Interpretation boundary" in dashboard
    assert "Annotation class summary" in dashboard
    assert "Annotation results" in dashboard
    assert "BacMet resistance categories by predicted origin" in dashboard
    assert "Download class summary" not in dashboard
    assert "Recommended review" not in dashboard
    assert "beta-lactam" in dashboard


def test_cli_dispatches_visualize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "visualize", lambda **kwargs: observed.update(kwargs))
    cli.main(
        [
            "visualize",
            "--predictions-tsv",
            str(tmp_path / "predictions.tsv"),
            "--output-dir",
            str(tmp_path / "visualization"),
        ]
    )
    assert observed == {
        "predictions_tsv": tmp_path / "predictions.tsv",
        "output_dir": tmp_path / "visualization",
        "annotated_results_tsv": None,
    }


def test_arg_parsers_preserve_independent_evidence_and_filter_non_amr(tmp_path: Path) -> None:
    orfs = {"seq__orf_1": Orf("seq__orf_1", "seq", 1, 300, 1, 100)}
    card = write(
        tmp_path / "card.tsv",
        "seq__orf_1\tgb|ABC.1|ARO:3000001|blaX\t91\t95\t1e-20\t200\t"
        "gb|ABC.1|ARO:3000001|blaX description\n",
    )
    card_hits = parse_card_hits(
        card,
        orfs,
        {
            "ARO:3000001": {
                "gene": "blaX",
                "family": "class A beta-lactamase",
                "drug_class": "beta-lactam",
                "mechanism": "antibiotic inactivation",
            }
        },
    )
    sarg = write(
        tmp_path / "sarg.tsv",
        "seq__orf_1\tSARG|beta-lactam|bla*|WP_1.1\t90\t92\t1e-15\t180\t"
        "SARG|beta-lactam|bla*|WP_1.1 protein\n",
    )
    sarg_hits = parse_sarg_hits(sarg, orfs)
    assert card_hits[0].source == "CARD"
    assert card_hits[0].resistance_mechanism == "antibiotic inactivation"
    assert sarg_hits[0].source == "SARG"
    assert sarg_hits[0].drug_class == "beta-lactam"
    assert consensus_hits([sarg_hits[0], card_hits[0]]) == [card_hits[0]]

    official = write(
        tmp_path / "official.tsv",
        "Protein id\tElement symbol\tElement name\tType\tClass\tSubclass\tMethod\t"
        "% Coverage of reference\t% Identity to reference\tClosest reference accession\t"
        "Hierarchy node\n"
        "seq__orf_1\tblaX\tbeta-lactamase\tAMR\tBETA-LACTAM\tBETA-LACTAM\t"
        "BLASTP\t98\t99\tWP_1.1\tblaX_fam\n"
        "seq__orf_1\tstxA\tShiga toxin\tVIRULENCE\tSTX2\tstxA\tEXACTP\t100\t100\t"
        "WP_2.1\tstxA\n",
    )
    official_hits = parse_amrfinderplus_hits(official, orfs)
    assert len(official_hits) == 1
    assert official_hits[0].source == "AMRFINDERPLUS"
    assert official_hits[0].gene_symbol == "blaX"
    assert official_hits[0].drug_class == "beta-lactam"


def test_annotation_orf_coordinates_and_amrprot_hierarchy(tmp_path: Path) -> None:
    proteins = tmp_path / "proteins.faa"
    orfs = predict_annotation_orfs([FastaRecord("seq", "ATG" + "GCC" * 400 + "TAA")], proteins)
    assert orfs
    first = next(iter(orfs.values()))
    assert first.sequence_id == "seq"
    assert first.start >= 1
    assert first.end > first.start
    assert proteins.read_text(encoding="ascii").startswith(">seq__orf_")

    hierarchy_path = write(
        tmp_path / "fam.tsv",
        "#node_id\tparent_node_id\tgene_symbol\ttype\tclass\tsubclass\tfamily_name\n"
        "ALL\t\t-\t\t\t\t\n"
        "AMR\tALL\t-\tAMR\t\t\t\n"
        "BETA\tAMR\tblaX\t\tBETA-LACTAM\tCEPHALOSPORIN\tclass A family\n"
        "VIR\tALL\tstxA\tVIRULENCE\tSTX2\tstxA\tShiga toxin\n",
    )
    hierarchy = load_amrfinder_hierarchy(hierarchy_path)
    diamond = write(
        tmp_path / "amrprot.tsv",
        f"{first.identifier}\tABC.1\t99\t100\t1e-30\t300\t"
        "ABC.1|1|1|blaX|blaX_fam||1|CEPHALOSPORIN|BETA-LACTAM|class_A_beta_lactamase\n"
        f"{first.identifier}\tVIR.1\t100\t100\t1e-40\t400\t"
        "VIR.1|1|1|stxA|stxA||1|stxA|STX2|Shiga_toxin\n",
    )
    hits = parse_amrprot_hits(diamond, orfs, hierarchy)
    assert len(hits) == 1
    assert hits[0].source == "AMRPROT_DIAMOND"
    assert hits[0].gene_symbol == "blaX"


def test_arg_annotation_integration_is_atomic_and_prediction_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = write(tmp_path / "input.fasta", ">seq\nATGGCCGCCGCC\n")
    database = tmp_path / "databases"
    for directory in (database / "card", database / "sarg"):
        directory.mkdir(parents=True)
    write(database / "card" / "card.dmnd", "card\n")
    write(
        database / "card" / "aro_index.tsv",
        "ARO Accession\tARO Name\tAMR Gene Family\tDrug Class\tResistance Mechanism\n"
        "ARO:3000001\tblaX\tclass A\tbeta-lactam\tantibiotic inactivation\n",
    )
    write(database / "sarg" / "sarg.dmnd", "sarg\n")
    official_database = tmp_path / "official_amrfinder"
    official_database.mkdir()
    write(official_database / "version.txt", "test-version\n")

    def fake_orfs(records: list[FastaRecord], output: Path) -> dict[str, Orf]:
        output.write_text(">seq__orf_1\nMAAA\n", encoding="ascii")
        return {"seq__orf_1": Orf("seq__orf_1", records[0].identifier, 1, 12, 1, 4)}

    def fake_diamond(**kwargs: object) -> None:
        output = kwargs["output"]
        database_path = kwargs["database"]
        assert isinstance(output, Path) and isinstance(database_path, Path)
        if database_path.name == "card.dmnd":
            output.write_text(
                "seq__orf_1\tgb|ABC|ARO:3000001|blaX\t99\t100\t1e-30\t300\t"
                "gb|ABC|ARO:3000001|blaX\n",
                encoding="utf-8",
            )
        else:
            output.write_text("", encoding="utf-8")

    def fake_amrfinder(**kwargs: object) -> None:
        output = kwargs["output"]
        assert isinstance(output, Path)
        output.write_text(
            "Protein id\tElement symbol\tElement name\tType\tClass\tMethod\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("mobiorigin.annotate.predict_annotation_orfs", fake_orfs)
    monkeypatch.setattr("mobiorigin.annotate.run_arg_diamond", fake_diamond)
    monkeypatch.setattr("mobiorigin.annotate.run_amrfinderplus", fake_amrfinder)
    output = tmp_path / "annotations"
    annotate(
        input_fasta=fasta,
        output_dir=output,
        database_dir=database,
        diamond=Path("true"),
        amrfinder_bin=Path("true"),
        amrfinder_database=official_database,
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "SHA256SUMS.txt",
        "annotation_provenance.json",
        "annotation_summary.tsv",
        "arg_consensus.tsv",
        "arg_hits.tsv",
        "predicted_proteins.faa",
        "raw_evidence",
    ]
    provenance = json.loads((output / "annotation_provenance.json").read_text())
    assert provenance["annotation_is_prediction_independent"] is True
    assert provenance["official_amrfinderplus_executed"] is True
    with pytest.raises(FileExistsError):
        annotate(
            input_fasta=fasta,
            output_dir=output,
            database_dir=database,
            diamond=Path("true"),
            amrfinder_bin=Path("true"),
            amrfinder_database=official_database,
        )


def test_amrfinderplus_retries_resource_failures_with_fewer_threads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MOBIORIGIN_TMPDIR", str(tmp_path / "temporary"))
    proteins = write(tmp_path / "proteins.faa", ">protein\nMAAA\n")
    output = tmp_path / "amrfinder.tsv"
    attempted_threads: list[int] = []
    temporary_directories: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        attempted = int(command[command.index("--threads") + 1])
        attempted_threads.append(attempted)
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        temporary_directories.append(str(environment["TMPDIR"]))
        if attempted > 2:
            return SimpleNamespace(
                returncode=139,
                stderr="CThread::Run() -- error creating thread\nSegmentation fault",
                stdout="",
            )
        Path(command[command.index("--output") + 1]).write_text(
            "Protein id\tElement symbol\tElement name\tType\tClass\tMethod\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("mobiorigin.annotate.subprocess.run", fake_run)
    run_amrfinderplus(
        executable=Path("amrfinder"),
        proteins=proteins,
        database=tmp_path / "database",
        output=output,
        threads=8,
    )
    assert attempted_threads == [8, 4, 2]
    assert len(set(temporary_directories)) == 3
    assert output.is_file()


def test_amrfinderplus_does_not_retry_nonresource_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MOBIORIGIN_TMPDIR", str(tmp_path / "temporary"))
    proteins = write(tmp_path / "proteins.faa", ">protein\nMAAA\n")
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=1, stderr="database schema is unsupported", stdout="")

    monkeypatch.setattr("mobiorigin.annotate.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="database schema is unsupported"):
        run_amrfinderplus(
            executable=Path("amrfinder"),
            proteins=proteins,
            database=tmp_path / "database",
            output=tmp_path / "output.tsv",
            threads=16,
        )
    assert calls == 1


def test_cli_dispatches_annotate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(cli, "annotate", lambda **kwargs: observed.update(kwargs))
    cli.main(
        [
            "annotate",
            "--input-fasta",
            str(tmp_path / "input.fasta"),
            "--output-dir",
            str(tmp_path / "out"),
            "--database-dir",
            str(tmp_path / "db"),
            "--amrfinder-database",
            str(tmp_path / "amrfinder"),
            "--threads",
            "4",
        ]
    )
    assert observed["threads"] == 4
    assert observed["amrfinder_mode"] == "official"
    assert observed["profile"] == "arg"
    assert observed["predictions_tsv"] is None


def test_cli_reports_expected_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(**kwargs: object) -> None:
        raise FileNotFoundError("Input FASTA was not found: /missing/input.fasta")

    monkeypatch.setattr(cli, "predict", fail)
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "predict",
                "--input-fasta",
                "/missing/input.fasta",
                "--output-dir",
                "/missing/output",
            ]
        )
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert "STOP: Input FASTA was not found" in captured.err
    assert "Traceback" not in captured.err
    assert "MOBIORIGIN_DEBUG=1" in captured.err


def test_cli_annotate_uses_default_annotation_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setenv("MOBIORIGIN_ANNOTATION_DATABASE_DIR", str(tmp_path / "annotation_db"))
    monkeypatch.setattr(cli, "annotate", lambda **kwargs: observed.update(kwargs))
    cli.main(
        [
            "annotate",
            "--input-fasta",
            str(tmp_path / "input.fasta"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert observed["database_dir"] == tmp_path / "annotation_db"
    assert observed["amrfinder_database"] == tmp_path / "annotation_db/amrfinderplus"


def test_comprehensive_evidence_priority_is_transparent_and_prediction_preserving(
    tmp_path: Path,
) -> None:
    records = [FastaRecord("seq", "A" * 2000)]
    predictions_path = write(
        tmp_path / "predictions.tsv",
        "sequence_id\tlength_bp\tprediction\tp_chromosome\tp_plasmid\tp_phage\t"
        "plasmid_score\tabstention_reason\n"
        "seq\t2000\tplasmid\t0.05\t0.9\t0.05\t0.85\t\n",
    )
    predictions = load_predictions(predictions_path, records)
    arg = ArgHit(
        "seq",
        "seq__orf_1",
        1,
        300,
        1,
        "CARD",
        "blaX",
        "beta-lactamase",
        "ARO:1",
        "class A",
        "beta-lactam",
        "inactivation",
        "DIAMOND_BLASTP",
        99.0,
        100.0,
        1e-30,
        300.0,
    )
    evidence = [
        *arg_evidence([arg]),
        EvidenceHit(
            "seq",
            "seq__orf_2",
            400,
            700,
            1,
            "MOBILITY",
            "MOB_SUITE_RELAXASE",
            "relaxase",
            "MOBF",
            "MOBF",
            "mob",
            "relaxase",
            "DIAMOND_BLASTP",
            70.0,
            90.0,
            1e-20,
            200.0,
        ),
        EvidenceHit(
            "seq",
            "seq__orf_3",
            800,
            1100,
            1,
            "MOBILITY",
            "MOB_SUITE_MPF",
            "mating_pair_formation",
            "MPF_F",
            "MPF_F",
            "mpf",
            "MPF",
            "DIAMOND_BLASTP",
            65.0,
            85.0,
            1e-12,
            150.0,
        ),
    ]
    normalized_arg = evidence[0]
    assert normalized_arg.gene_symbol == "blaX"
    assert normalized_arg.gene_name == "beta-lactamase"
    assert normalized_arg.gene_family == "class A"
    assert normalized_arg.functional_class == "antimicrobial_resistance"
    assert normalized_arg.functional_subclass == "beta-lactam"
    assert normalized_arg.mechanism == "inactivation"
    integrated = tmp_path / "integrated.tsv"
    rows = write_integrated_results(integrated, records, evidence, predictions)
    assert rows[0]["prediction"] == "plasmid"
    assert rows[0]["p_plasmid"] == "0.9"
    assert rows[0]["evidence_priority_tier"] == "A"
    assert rows[0]["mobility_class"] == "conjugative"
    assert rows[0]["annotated_gene_symbols"] == "blaX"
    assert rows[0]["annotated_gene_families"] == "class A"
    assert rows[0]["annotated_functional_classes"] == "antimicrobial_resistance"
    assert rows[0]["annotated_functional_subclasses"] == "beta-lactam"
    assert rows[0]["annotated_mechanisms"] == "inactivation"
    assert rows[0]["annotation_sources"] == "CARD;MOB_SUITE_MPF;MOB_SUITE_RELAXASE"
    summary = tmp_path / "summary.json"
    write_publication_summary(summary, rows, evidence)
    payload = json.loads(summary.read_text())
    assert payload["interpretation"]["priority_is_clinical_risk_score"] is False
    assert payload["interpretation"]["annotation_changes_origin_prediction"] is False
    assert payload["annotation_vocabulary"]["schema_version"] == ("mobiorigin-normalized-gene-v1")
    assert payload["annotation_vocabulary"]["source_specific_fields_retained"] is True


def test_amrfinderplus_non_amr_evidence_remains_outside_arg_consensus(tmp_path: Path) -> None:
    orfs = {"seq__orf_1": Orf("seq__orf_1", "seq", 1, 300, 1, 100)}
    output = write(
        tmp_path / "amrfinder.tsv",
        "Protein id\tElement symbol\tElement name\tType\tSubtype\tClass\tMethod\t"
        "% Coverage of reference\t% Identity to reference\tClosest reference accession\n"
        "seq__orf_1\tstxA\tShiga toxin\tVIRULENCE\tVIRULENCE\tTOXIN\tEXACTP\t"
        "100\t100\tWP_1.1\n",
    )
    hits = parse_amrfinderplus_non_amr(output, orfs)
    assert len(hits) == 1
    assert hits[0].evidence_group == "VIRULENCE"
    assert hits[0].source == "AMRFINDERPLUS"
    assert hits[0].feature_name == "stxA"
    assert hits[0].gene_symbol == "stxA"
    assert hits[0].gene_name == "Shiga toxin"
    assert hits[0].gene_family == "unknown"
    assert hits[0].functional_class == "TOXIN"
    assert hits[0].functional_subclass == "virulence"


def test_mobileog_parser_recovers_titles_and_excludes_unresolved_rows(tmp_path: Path) -> None:
    assert mobileog_fields(
        "truncated",
        "description mobileOG_000000007|xis|A0A653FUH0|IE|RRR|N/A|Multiple|Manual rest",
    ) == (
        "mobileOG_000000007",
        "xis",
        "A0A653FUH0",
        "IE",
        "RRR",
        "N/A",
        "Multiple",
        "Manual",
    )
    orfs = {
        "seq__orf_1": Orf("seq__orf_1", "seq", 1, 300, 1, 100),
        "seq__orf_2": Orf("seq__orf_2", "seq", 400, 700, 1, 100),
    }
    results = write(
        tmp_path / "mobileog.tsv",
        "seq__orf_1\tshort-id\t80\t90\t1e-10\t150\t"
        "protein mobileOG_000000007|xis|A0A653FUH0|IE|RRR|N/A|Multiple|Manual\n"
        "seq__orf_2\tunresolved-id\t80\t90\t1e-10\t150\tunsupported title\n",
    )
    warnings = []
    hits = parse_mobileog(results, orfs, warnings=warnings)
    assert len(hits) == 1
    assert hits[0].accession == "mobileOG_000000007"
    assert hits[0].feature_type == "integration_excision"
    assert hits[0].gene_symbol == "xis"
    assert hits[0].gene_family == "xis"
    assert hits[0].functional_class == "integration_excision"
    assert hits[0].functional_subclass == "RRR"
    assert len(warnings) == 1
    assert warnings[0].query == "seq__orf_2"
    assert warnings[0].subject == "unresolved-id"
    assert warnings[0].reason == "unsupported_mobileog_header_excluded"


def test_diamond_parsers_tolerate_legacy_bytes_in_free_text_titles(tmp_path: Path) -> None:
    result = tmp_path / "diamond.tsv"
    result.write_bytes(
        b"seq__orf_1\tBAC0001\t91.5\t87.0\t1e-20\t205.0\t" b"legacy\xa0database description\n"
    )

    arg_rows = list(_diamond_rows(result))
    evidence_rows = _evidence_rows(result)

    for rows in (arg_rows, evidence_rows):
        assert len(rows) == 1
        assert rows[0][:6] == ("seq__orf_1", "BAC0001", 91.5, 87.0, 1e-20, 205.0)
        assert rows[0][6] == "legacy\ufffddatabase description"


def test_mobileog_source_audit_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    source = write(
        tmp_path / "mobileog.faa",
        ">mobileOG_1|xis|A0A|IE|RRR|N/A|Multiple|Manual\nMAAA\n" ">unsupported_header\nMBBB\n",
    )
    output = tmp_path / "compatibility.json"
    result = audit_mobileog_fasta(source, output)
    assert result["status"] == "PASS_WITH_EXCLUSIONS"
    assert result["headers_total"] == 2
    assert result["headers_supported"] == 1
    assert result["headers_excluded"] == 1
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_comprehensive_annotation_publishes_integrated_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = write(tmp_path / "input.fasta", ">seq\n" + "ATGGCC" * 400 + "\n")
    predictions = write(
        tmp_path / "predictions.tsv",
        "sequence_id\tlength_bp\tprediction\tp_chromosome\tp_plasmid\tp_phage\t"
        "plasmid_score\tabstention_reason\n"
        "seq\t2400\tplasmid\t0.05\t0.9\t0.05\t0.85\t\n",
    )
    database = tmp_path / "databases"
    required = {
        "card/card.dmnd": "card\n",
        "card/aro_index.tsv": (
            "ARO Accession\tARO Name\tAMR Gene Family\tDrug Class\tResistance Mechanism\n"
            "ARO:3000001\tblaX\tclass A\tbeta-lactam\tantibiotic inactivation\n"
        ),
        "sarg/sarg.dmnd": "sarg\n",
        "vfdb/vfdb_setA.dmnd": "vfdb\n",
        "vfdb/vfdb_indx.txt": "VFG000001(gb|WP_1.1)\tVFC0001\tToxin\n",
        "mge/mobileog.dmnd": "mge\n",
        "bacmet/bacmet.dmnd": "bacmet\n",
        "bacmet/Bacmet_list.tsv": (
            "BacMet_ID\tGene_name\tClass\tAccession\tOrganism\tLength\tLocation\tCompound\n"
            "BAC0001\tabeM\tBio\tQ1\tTest\t100\tChromosome\tTriclosan [class: phenol]\n"
        ),
        "mob_suite/rep_proteins.dmnd": "rep\n",
        "mob_suite/mob_proteins.dmnd": "mob\n",
        "mob_suite/mpf_proteins.dmnd": "mpf\n",
    }
    for relative, content in required.items():
        destination = database / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        write(destination, content)
    official_database = tmp_path / "official_amrfinder"
    official_database.mkdir()
    write(official_database / "version.txt", "test\n")

    def fake_orfs(records: list[FastaRecord], output: Path) -> dict[str, Orf]:
        output.write_text(
            ">seq__orf_1\nMAAA\n>seq__orf_2\nMBBB\n>seq__orf_3\nMCCC\n",
            encoding="ascii",
        )
        return {
            "seq__orf_1": Orf("seq__orf_1", records[0].identifier, 1, 300, 1, 100),
            "seq__orf_2": Orf("seq__orf_2", records[0].identifier, 400, 700, 1, 100),
            "seq__orf_3": Orf("seq__orf_3", records[0].identifier, 800, 1100, -1, 100),
        }

    def fake_arg_search(**kwargs: object) -> None:
        output = kwargs["output"]
        database_path = kwargs["database"]
        assert isinstance(output, Path) and isinstance(database_path, Path)
        if database_path.name == "card.dmnd":
            output.write_text(
                "seq__orf_1\tgb|ABC|ARO:3000001|blaX\t99\t100\t1e-30\t300\t"
                "gb|ABC|ARO:3000001|blaX\n",
                encoding="utf-8",
            )
        else:
            output.write_text("", encoding="utf-8")

    def fake_amrfinder(**kwargs: object) -> None:
        output = kwargs["output"]
        assert isinstance(output, Path)
        output.write_text(
            "Protein id\tElement symbol\tElement name\tType\tSubtype\tClass\tSubclass\t"
            "Method\t% Coverage of reference\t% Identity to reference\t"
            "Closest reference accession\tHierarchy node\n"
            "seq__orf_1\tblaX\tbeta-lactamase\tAMR\tAMR\tBETA-LACTAM\t\tEXACTP\t"
            "100\t100\tWP_ARG\tblaX\n"
            "seq__orf_2\tstxA\tShiga toxin\tVIRULENCE\tVIRULENCE\tTOXIN\t\tEXACTP\t"
            "100\t100\tWP_VF\tstxA\n"
            "seq__orf_3\tqacX\tbiocide resistance\tSTRESS\tBIOCIDE\tQAC\t\tBLASTP\t"
            "95\t90\tWP_QAC\tqacX\n",
            encoding="utf-8",
        )

    def fake_evidence_search(**kwargs: object) -> None:
        output = kwargs["output"]
        database_path = kwargs["database"]
        assert isinstance(output, Path) and isinstance(database_path, Path)
        rows = {
            "vfdb_setA.dmnd": (
                "seq__orf_2\tVFG000001(gb|WP_1.1)\t90\t100\t1e-20\t200\t"
                "VFG000001(gb|WP_1.1) stxA [Shiga toxin (VF1)] [Escherichia coli]\n"
            ),
            "mobileog.dmnd": (
                "seq__orf_3\tunresolved-mobileog-id"
                "\t80\t90\t1e-10\t150\tunsupported mobileOG title\n"
            ),
            "bacmet.dmnd": ("seq__orf_3\tBAC0001|abeM|tr|Q1\t90\t95\t1e-15\t180\tAbeM pump\n"),
            "rep_proteins.dmnd": (
                "seq__orf_1\tNC_1|IncP_s1_f0_o0\t70\t80\t1e-12\t170\tIncP replicon\n"
            ),
            "mob_proteins.dmnd": ("seq__orf_2\tMOBF_1\t70\t90\t1e-15\t190\tMOBF relaxase\n"),
            "mpf_proteins.dmnd": (
                "seq__orf_3\tMPF_F_1\t70\t90\t1e-15\t190\tMPF_F coupling protein\n"
            ),
        }
        output.write_text(rows[database_path.name], encoding="utf-8")

    monkeypatch.setattr("mobiorigin.annotate.predict_annotation_orfs", fake_orfs)
    monkeypatch.setattr("mobiorigin.annotate.run_arg_diamond", fake_arg_search)
    monkeypatch.setattr("mobiorigin.annotate.run_amrfinderplus", fake_amrfinder)
    monkeypatch.setattr("mobiorigin.biological_evidence.run_evidence_diamond", fake_evidence_search)
    output = tmp_path / "comprehensive"
    annotate(
        input_fasta=fasta,
        output_dir=output,
        database_dir=database,
        diamond=Path("true"),
        amrfinder_bin=Path("true"),
        amrfinder_database=official_database,
        profile="comprehensive",
        predictions_tsv=predictions,
    )
    for name in (
        "biological_evidence.tsv",
        "mobiorigin_annotated_results.tsv",
        "publication_summary.json",
        "mobiorigin_report.html",
        "annotation_warnings.tsv",
    ):
        assert (output / name).is_file()
    integrated = (output / "mobiorigin_annotated_results.tsv").read_text()
    assert "\tplasmid\t" in integrated
    with (output / "mobiorigin_annotated_results.tsv").open(encoding="utf-8", newline="") as handle:
        integrated_row = next(csv.DictReader(handle, delimiter="\t"))
    assert integrated_row["evidence_priority_tier"] == "A"
    assert integrated_row["evidence_priority_label"] == "ARG-bearing conjugative candidate"
    assert integrated_row["priority_rationale"] == (
        "ARG plus relaxase and mating-pair-formation evidence"
    )
    assert integrated_row["conjugative_candidate"] == "true"
    assert integrated_row["arg_gene_families"] == "class A"
    assert integrated_row["arg_drug_classes"] == "beta-lactam"
    assert integrated_row["arg_mechanisms"] == "antibiotic inactivation"
    assert integrated_row["virulence_classes"]
    assert integrated_row["mge_classes"] == ""
    assert integrated_row["bacmet_gene_families"] == "Bio"
    assert integrated_row["bacmet_classes"] == "Triclosan"
    assert "recommended_follow_up" not in integrated_row
    assert integrated_row["mobility_marker_types"] == ("mating_pair_formation;relaxase;replication")
    assert "annotated_gene_symbols" in integrated.splitlines()[0]
    assert "annotated_gene_families" in integrated.splitlines()[0]
    assert "annotated_functional_classes" in integrated.splitlines()[0]
    evidence_header = (output / "biological_evidence.tsv").read_text().splitlines()[0]
    assert "gene_symbol" in evidence_header
    assert "gene_name" in evidence_header
    assert "gene_family" in evidence_header
    assert "functional_class" in evidence_header
    assert "functional_subclass" in evidence_header
    assert "mechanism" in evidence_header
    report = (output / "mobiorigin_report.html").read_text()
    assert "not clinical risk scores" in report
    provenance = json.loads((output / "annotation_provenance.json").read_text())
    assert provenance["annotation_profile"] == "comprehensive"
    assert provenance["predictions_integrated"] is True
    assert provenance["classification_labels_or_probabilities_changed"] is False
    assert provenance["schema_version"] == "mobiorigin-biological-annotation-v5"
    assert provenance["normalized_gene_vocabulary"]["schema_version"] == (
        "mobiorigin-normalized-gene-v1"
    )
    assert provenance["annotation_warnings"]["mobileog_rows_excluded"] == 1
    warnings = (output / "annotation_warnings.tsv").read_text(encoding="utf-8")
    assert "unresolved-mobileog-id" in warnings
    assert "unsupported_mobileog_header_excluded" in warnings
