"""Integration tests for the full pipeline.

Day 28 target: full end-to-end pass on Citrobacter test data.
These tests are skipped until the pipeline is wired up (Day 22+).
"""

import pytest

pytestmark = pytest.mark.skip(reason="Pipeline not yet wired (Day 22 target)")


def test_full_pipeline_citrobacter(tmp_path):
    """Run the full pipeline on the Citrobacter test FASTA and assert known outputs."""
    # TODO (Day 26): point at data/test/citrobacter.fasta
    # Expected: ~N plasmids, ~M chromosomes, ARG hits on known contigs
    pass


def test_cli_run_command(tmp_path):
    """Smoke-test the CLI `plasflow2 run` command end-to-end."""
    from click.testing import CliRunner
    from plasflow2.cli import main

    runner = CliRunner()
    # TODO (Day 26): use a real small test FASTA
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
