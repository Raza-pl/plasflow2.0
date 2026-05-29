"""Fragment large FASTA sequences into fixed-size chunks.

Simulates the contig-length distribution of a real WGS assembly so the
classifier sees the same kind of input it was trained on (~10 kb chunks).

Usage:
    python scripts/fragment_fasta.py \
        --input  data/test/kpneu_mgh78578.fasta \
        --output data/test/kpneu_fragmented.fasta \
        --chunk  15000 \
        --min    2000
"""

from __future__ import annotations

import argparse

from Bio import SeqIO


def fragment(input_fasta: str, output_fasta: str, chunk_size: int, min_length: int) -> None:
    records_out = []
    for rec in SeqIO.parse(input_fasta, "fasta"):
        seq = str(rec.seq)
        for i, start in enumerate(range(0, len(seq), chunk_size)):
            chunk = seq[start : start + chunk_size]
            if len(chunk) < min_length:
                continue
            from Bio.Seq import Seq
            from Bio.SeqRecord import SeqRecord

            records_out.append(
                SeqRecord(
                    Seq(chunk),
                    id=f"{rec.id}_chunk{i:04d}",
                    description=f"fragment {start}-{start+len(chunk)} of {rec.id}",
                )
            )
    SeqIO.write(records_out, output_fasta, "fasta")
    print(f"Written {len(records_out)} fragments to {output_fasta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk", type=int, default=15000, help="Chunk size in bp")
    parser.add_argument("--min", type=int, default=2000, help="Minimum fragment length")
    args = parser.parse_args()
    fragment(args.input, args.output, args.chunk, args.min)
