# Changelog

All notable MobiOrigin changes are documented here. The project uses semantic versioning for the standalone `mobiorigin` package interface.

## Unreleased

### Changed

- Made all DIAMOND annotation-result readers tolerate isolated legacy bytes in
  third-party free-text descriptions. Valid evidence rows are retained with
  only undecodable title characters replaced, preventing a completed
  multi-database annotation run from failing during result parsing.
- Removed an unused internal instruction file from the public repository.
- Generalized the local build-artifact exclusion and operational script root so
  public files no longer expose development-environment names or a private
  checkout path.
- Limited duplicate push-triggered CI runs to maintained branches. Pull-request
  checks against `main` remain unchanged.

## 0.1.6 — 2026-09-03

### Added

- Added an input-path preflight that reports the resolved missing path, current
  working directory, and nearby FASTA files instead of producing a traceback.
- Added direct reading of gzip-compressed `.fa.gz`, `.fasta.gz`, `.fna.gz`, and
  `.fas.gz` inputs, removing the manual decompression step.
- Added a writable temporary-storage check to `mobiorigin doctor`, including
  the selected location, available space, WSL status, and ignored unsafe
  environment settings.
- Added concise success summaries for prediction, annotation, visualization,
  and integrated runs.
- Added live, flushed progress messages for preflight, feature generation,
  marker searches, individual annotation resources, visualization, and atomic
  result publication.
- Added evidence-group-specific gene, family, class, subclass, and mechanism
  summaries to the integrated per-contig table.
- Added annotation-class and evidence-tier summary tables plus dedicated ARG,
  MGE, virulence-factor, and BacMet SVGs split by predicted origin. The clean
  HTML dashboard provides tier filtering and annotation search without download
  controls.

### Changed

- External tools now receive a private temporary directory chosen by
  MobiOrigin. On WSL, Linux-native storage is preferred automatically and stale
  `TMPDIR`, `TEMP`, or `TMP` values on `/mnt/*` are ignored.
- AMRFinderPlus now retries resource-related thread failures with progressively
  fewer workers, down to one, while preserving immediate failure for database,
  input, and other non-resource errors.
- Expected command-line failures now end with a short actionable message.
  Developer tracebacks remain available by setting `MOBIORIGIN_DEBUG=1`.
- A to E tiers now carry stable descriptive labels in reports while retaining
  the existing frozen tier rules.

### Scientific boundaries

- These are reporting, visualization, runtime, and diagnostic changes only. No
  model, feature, normalization, ensemble weight, threshold, annotation rule,
  database content, or prediction semantic changed.

## 0.1.5 — 2026-08-28

Installation usability and normalized annotation-reporting update for the
unchanged frozen `mobiorigin-dev1-mob-selective-v1` classifier.

### Added

- Added a source-preserving normalized gene vocabulary across ARG, virulence,
  MGE, stress, and plasmid-mobility evidence. Detailed evidence now includes
  canonical gene symbol, gene name, family, functional class, subclass, and
  mechanism fields.
- Added deterministic per-contig summaries of normalized genes, families,
  classes, subclasses, mechanisms, and contributing annotation sources to the
  integrated TSV and HTML report.

### Changed

- Put the complete guided Conda or Mamba installation before the managed PyPI
  route in the README.
- Made setup and doctor output concise by default while retaining the complete
  per-file inventory through `--verbose`.
- Moved the installer demonstration to `~/mobiorigin_demo` and ignored a
  repository-local demonstration directory.
- Extended `mobiorigin doctor` to verify the comprehensive annotation database.
- Made the versioned release asset the sole distributed source of the frozen
  checkpoint payload. New source checkouts no longer contain a second copy.
- Clarified that Apple Silicon prediction remains native arm64 while isolated
  MOB-suite database helpers run under Rosetta.
- Preferred DIAMOND, AMRFinderPlus, and database-update executables installed
  beside MobiOrigin's active Python over unrelated same-named tools earlier on
  the host `PATH`. Explicit executable paths remain authoritative.
- Passed the frozen marker builder the absolute DIAMOND 2.0.15 path from its
  isolated helper environment, preventing `~/bin/diamond` from shadowing it.

### Scientific boundaries

- No model, feature, normalization, ensemble weight, decision threshold,
  annotation threshold, database content, or prediction semantic changed.
- Gene normalization is additive. Source-specific names remain retained and no
  unsupported alias, family, class, mechanism, phenotype, or risk is inferred.

## 0.1.4 — 2026-08-27

Runtime compatibility and installation-verification update for the unchanged
frozen `mobiorigin-dev1-mob-selective-v1` classifier.

### Fixed

- Recovered supported mobileOG identifiers from either DIAMOND subject or title
  fields and prevented one unusual external header from aborting all annotation.
- Excluded unresolved mobileOG rows from biological evidence and recorded their
  exact subject and title in `annotation_warnings.tsv` instead of guessing.
- Retained completed prediction output in an explicitly failed workspace when a
  later integrated annotation stage stops.

### Added

- A streaming mobileOG source-header compatibility audit bound to new annotation
  database manifests.
- A comprehensive installation verification that exercises prediction,
  annotation, visualization, and all standard annotation database routes.
- Explicit `build` and `twine` development dependencies so the documented
  release validation is reproducible in a prepared development environment.

### Scientific boundaries

- No classifier model, feature, normalization, ensemble weight, threshold,
  label, or prediction semantic changed.
- An unresolved mobileOG row contributes no MGE evidence and cannot change an
  A to E evidence tier. The exclusion remains visible for audit.

## 0.1.3 — 2026-08-27

PyPI transport and supply-chain update for the unchanged frozen
`mobiorigin-dev1-mob-selective-v1` classifier.

### Added

- Atomic, resumable retrieval of the exact frozen dev1 model bundle from a
  versioned GitHub release asset.
- SHA-256 and byte-count verification of all three checkpoints, marker
  normalization, model manifest, and the complete transport archive.
- PyPI Trusted Publishing workflow with tag/version matching, separate build and
  publishing jobs, OIDC authentication, attestations, and a fail-closed 100 MB
  per-file gate.
- Offline model setup through `--model-archive` and configurable model storage
  through `MOBIORIGIN_MODEL_DIR`.

### Changed

- Python distributions no longer duplicate the 121 MB frozen model payload.
  Guided installation retrieves the exact bytes once and verifies them before
  making them available to prediction.
- The database helper now prepares and checks model artifacts before marker
  databases, keeping the standard installation route automatic.
- `mobiorigin doctor` verifies the resolved model directory as part of the full
  installation check.

### Scientific boundaries

- Model transport changed; checkpoint bytes, hashes, architecture, feature
  definitions, ensemble, normalization, threshold, and predictions did not.
- Prediction remains offline after installation. Missing or changed model bytes
  stop execution rather than triggering a fallback model or network request.

## 0.1.2 — 2026-08-26

Installation, database automation, and integrated analysis update for the
unchanged frozen `mobiorigin-dev1-mob-selective-v1` classifier.

### Added

- Guided Conda or Mamba installation with a post-installation doctor check and
  bundled deterministic demonstration.
- Automatic, resumable setup and cryptographic verification of the comprehensive
  annotation resources, with explicit acceptance of applicable third-party terms.
- mobileOG-db as the default MGE resource, while retaining legacy ISfinder as an
  optional user-supplied resource.
- A bundled eight-contig assembly example that demonstrates chromosome, plasmid,
  phage, and unclassified outputs without presenting an accuracy claim.
- Deterministic SVG, HTML, and tabular visualizations for prediction and annotation
  outputs.

### Changed

- `mobiorigin run` now performs prediction, comprehensive biological annotation,
  and integrated visualization by default in one atomic output directory. A
  `--skip-annotation` option preserves the lightweight prediction-only route.
- Expanded `--threads` from the original validated 1–8 range to 1–128 for
  DIAMOND, AMRFinderPlus, and other external searches. Deterministic neural-network
  inference remains single-threaded, and the model and scientific policy are unchanged.

### Fixed

- Separated the CPU-only MobiOrigin runtime from MOB-suite's incompatible legacy
  NumPy/pandas database-building stack.
- Added a non-overwriting database setup helper with Linux/WSL, Intel macOS, and
  Apple Silicon/Rosetta handling plus actionable failure messages.
- Prevented Conda from selecting CUDA by pinning the documented runtime to a
  cross-platform CPU PyTorch build.
- Added CI installation smoke tests, an isolated MOB-suite dependency solve, and
  installation-contract unit tests.

### Scientific boundaries

- The classifier checkpoints, feature definitions, ensemble, selective threshold,
  and frozen external-validation results are unchanged from version 0.1.1.
- Biological annotations remain independent supporting evidence. They do not
  override prediction labels or probabilities and are not clinical risk scores.
- The bundled assembly is a software demonstration. It is not an accuracy,
  prevalence, or biological-discovery dataset.

## 0.1.1 — 2026-08-23

Publication and biological-annotation update for the unchanged frozen
`mobiorigin-dev1-mob-selective-v1` classifier.

### Added

- Prediction-independent `mobiorigin annotate` workflow integrating CARD, SARG,
  official AMRFinderPlus, VFDB, MGE, BacMet2, and MOB-suite evidence without
  changing classifier labels or probabilities.
- Publication-quality annotation tables, provenance, checksums, and HTML reports,
  including transparent A–E biological evidence-priority tiers that are explicitly
  not clinical risk scores.
- Post-hoc exploratory external comparisons with PlasClass, PlasFlow v1, PLASMe,
  and Platon under a separately frozen statistical contract.
- Label-free operational evidence from two deterministic real-assembly subsets,
  covering all 12 dataset–tool runs and 10 pairwise operational comparisons.
- Validation tables, editable vector figures, methods, limitations, and updated
  repository documentation.

### Changed

- Package and citation metadata now identify the expanded publication bundle as
  version 0.1.1.
- Distribution metadata includes the annotation and operational-validation
  documentation.

### Scientific boundaries

- The frozen classifier, three model checkpoints, marker normalization, ensemble,
  and selective threshold are unchanged from version 0.1.0.
- Secondary comparator findings are exploratory and do not alter the preregistered
  MobiOrigin-versus-geNomad co-primary evidence.
- Real-assembly results support runtime, call-rate, coverage, agreement, and
  biological-evidence reporting only; they do not support ground-truth accuracy or
  superiority claims.

## 0.1.0 — 2026-08-21

Initial research release of the frozen `mobiorigin-dev1-mob-selective-v1` candidate.

### Added

- Standalone `mobiorigin predict` interface for chromosome, plasmid, phage, and explicit unclassified predictions.
- Deterministic 9,557-dimensional sequence-feature extraction and 17-dimensional MOB protein-marker extraction.
- Three frozen neural-network checkpoints combined by an equal-weight softmax mean.
- Frozen plasmid selective-abstention rule with threshold `0.19835489988327026`.
- Safe tensor-only checkpoint loading and exact model, normalization, and database identity verification.
- Atomic prediction outputs with provenance and SHA-256 checksums.
- `mobiorigin setup-databases` for atomic retrieval or offline installation of the exact marker databases.
- Prospective external validation against geNomad 1.12.0/database 1.9 using 3,000 source-disjoint records.
- Aggregate validation tables, vector figure, methods, and claim boundaries.

### Scientific boundaries

- The frozen external cohort is closed to retrospective tuning and record-level error mining.
- geNomad outputs are not model features or training targets.
- MobiOrigin does not use hard biological overrides or post-hoc probability transfer.
- Third-party biological database records are retrieved for local use and are not bundled in the Python distribution.
