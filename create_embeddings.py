"""
Extract ProtT5 embeddings for all proteins across one or more FASTA files and
store them in a single HDF5 file keyed by protein_id.

HDF5 layout is flat: each protein_id is a top-level dataset of shape
(emb_dim,) float32. Root attrs record provenance (model, pooling strategy).

The script is resumable: on restart, proteins already present in the HDF5 are
skipped. Sequences are sorted by length (ascending) before batching to reduce
padding waste, and pooling uses the attention mask so padding tokens do not
pollute the mean.

Defaults to `Rostlab/prot_t5_xl_uniref50` to match prior pipeline code. For
faster runs on CPU/MPS, pass `--model Rostlab/prot_t5_xl_half_uniref50-enc`
(encoder-only, fp16, ~3 GB vs ~11 GB; pooled embeddings are nearly identical).

Usage:
    python -m src.extract_t5_embeddings \
        --fasta .../Train/train_sequences.fasta \
                .../Test/testsuperset.fasta \
        --output data/final_data/embeddings.h5 \
        --batch-size 4 \
        --device auto
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


DEFAULT_MODEL = 'Rostlab/prot_t5_xl_uniref50'
AMINO_ACID_REPLACE = re.compile(r'[UZOB]')


# ============================================================================
# FASTA loading
# ============================================================================

def parse_fasta(path):
    """Yield (protein_id, sequence) tuples from a FASTA file."""
    current_id = None
    parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id is not None:
                    yield current_id, ''.join(parts)
                current_id = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if current_id is not None:
        yield current_id, ''.join(parts)


def load_all_sequences(paths):
    """Merge sequences from multiple FASTAs. Duplicate IDs keep first occurrence."""
    seqs = {}
    for p in paths:
        before = len(seqs)
        for pid, seq in parse_fasta(p):
            if pid not in seqs:
                seqs[pid] = seq
        print(f"  {p}: +{len(seqs) - before:,} new sequences")
    return seqs


# ============================================================================
# Model
# ============================================================================

def resolve_device(arg):
    """auto -> cuda > mps > cpu; otherwise honor user."""
    if arg != 'auto':
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(model_name, device):
    print(f"Loading {model_name} on {device}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model


# ============================================================================
# Embedding extraction
# ============================================================================

def preprocess_sequence(seq):
    """ProtT5 quirks: non-standard residues -> X, spaces between residues."""
    seq = AMINO_ACID_REPLACE.sub('X', seq.upper())
    return ' '.join(list(seq))


@torch.no_grad()
def embed_batch(protein_ids, sequences, tokenizer, model, device):
    """Run one forward pass and mean-pool with the attention mask."""
    spaced = [preprocess_sequence(s) for s in sequences]
    enc = tokenizer(
        spaced, add_special_tokens=True, padding='longest', return_tensors='pt',
    )
    input_ids      = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden = out.last_hidden_state                         # (B, L, D)

    mask   = attention_mask.unsqueeze(-1).to(hidden.dtype) # (B, L, 1)
    summed = (hidden * mask).sum(dim=1)                    # (B, D)
    counts = mask.sum(dim=1).clamp(min=1)                  # (B, 1)
    pooled = (summed / counts).float().cpu().numpy()       # (B, D) fp32

    return dict(zip(protein_ids, pooled))


def batched(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ============================================================================
# HDF5 I/O
# ============================================================================

def existing_keys(h5_path):
    if not h5_path.exists():
        return set()
    with h5py.File(h5_path, 'r') as f:
        return set(f.keys())


def write_embeddings(h5_path, embeddings, model_name):
    with h5py.File(h5_path, 'a') as f:
        for pid, vec in embeddings.items():
            if pid in f:
                continue
            f.create_dataset(pid, data=vec, compression='gzip', compression_opts=4)
        f.attrs.setdefault('model', model_name)
        f.attrs.setdefault('pooling', 'mean_with_mask')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fasta', nargs='+', required=True,
                        help="One or more FASTA files (train + test).")
    parser.add_argument('--output', required=True,
                        help="HDF5 output file (created or appended).")
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f"HuggingFace model (default: {DEFAULT_MODEL}).")
    parser.add_argument('--device', default='auto',
                        help="cpu / cuda / mps / auto (default).")
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=2000,
                        help="Truncate sequences longer than this (residues).")
    parser.add_argument('--flush-every', type=int, default=50,
                        help="Flush buffered embeddings to disk every N batches.")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- load sequences ----
    print("Loading FASTA files...")
    sequences = load_all_sequences(args.fasta)
    print(f"Total unique proteins: {len(sequences):,}")

    # ---- resume: skip anything already in the h5 ----
    done = existing_keys(out_path)
    if done:
        print(f"Resuming: {len(done):,} already present in {out_path}")
    todo = {pid: s for pid, s in sequences.items() if pid not in done}
    print(f"To process: {len(todo):,}")
    if not todo:
        print("Nothing to do.")
        return

    # ---- truncate over-long sequences ----
    truncated = 0
    for pid, seq in todo.items():
        if len(seq) > args.max_length:
            todo[pid] = seq[:args.max_length]
            truncated += 1
    if truncated:
        print(f"Truncated {truncated:,} sequences to {args.max_length} residues.")

    # ---- sort by length ascending -> minimizes padding in each batch ----
    sorted_items = sorted(todo.items(), key=lambda kv: len(kv[1]))

    # ---- model ----
    device = resolve_device(args.device)
    tokenizer, model = load_model(args.model, device)

    # ---- process ----
    buffer = {}
    n_done = 0
    t_start = time.time()
    all_batches = list(batched(sorted_items, args.batch_size))

    for i, batch in enumerate(all_batches, start=1):
        ids  = [pid for pid, _ in batch]
        seqs = [seq for _, seq in batch]

        try:
            buffer.update(embed_batch(ids, seqs, tokenizer, model, device))
        except RuntimeError as e:
            # OOM fallback: retry this batch one sequence at a time
            msg = str(e).lower()
            if 'out of memory' in msg or 'alloc' in msg or 'mps' in msg:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                max_len = max(len(s) for s in seqs)
                print(f"  OOM on batch {i} (max len {max_len}); retrying singly")
                for pid, seq in zip(ids, seqs):
                    buffer.update(embed_batch([pid], [seq], tokenizer, model, device))
            else:
                raise

        n_done += len(batch)

        if i % args.flush_every == 0 or i == len(all_batches):
            write_embeddings(out_path, buffer, args.model)
            buffer.clear()
            elapsed   = time.time() - t_start
            rate      = n_done / elapsed if elapsed > 0 else 0.0
            remaining = (len(sorted_items) - n_done) / rate if rate > 0 else float('inf')
            print(f"  [{n_done:,}/{len(sorted_items):,}] "
                  f"{rate:.1f} prot/s  eta {remaining / 60:.1f} min")

    if buffer:
        write_embeddings(out_path, buffer, args.model)

    print(f"\nDone. Wrote {out_path}")


if __name__ == '__main__':
    main()