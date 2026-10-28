"""Rewrite a Qwen2 GGUF so the output layer only scores a token shortlist.

Kept tokens move to ids [0, n) (tokenizer metadata and embedding rows are
permuted together, so tokenization is unchanged in meaning) and a separate
`output.weight` holds only their rows. The patched llama.cpp loader accepts the
shorter output matrix and pads the remaining logits with zeros; the engine
reads `nes.output_vocab` and only scans the first n logits.

Usage: python3 prune_output_vocab.py IN.gguf OUT.gguf ascii
       python3 prune_output_vocab.py IN.gguf OUT.gguf counts.tsv [min_count]

`ascii` keeps every token whose bytes are all ASCII, which is exact for ASCII
source code. A corpus shortlist (counts.tsv: "<token id>\t<count>") is smaller
but hurts: the model copies identifiers from the window, and an identifier
whose token was never seen in the corpus must then be spelled differently.
"""

import sys

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter


def byte_decoder():
    """GPT-2 byte-level BPE: token characters back to bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {c: b for b, c in zip(bs, cs)}


def main():
    src, dst, counts_path = sys.argv[1:4]
    min_count = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    r = GGUFReader(src)
    f = r.fields
    tokens = [bytes(f["tokenizer.ggml.tokens"].parts[i]).decode("utf-8") for i in f["tokenizer.ggml.tokens"].data]
    types = [int(f["tokenizer.ggml.token_type"].parts[i][0]) for i in f["tokenizer.ggml.token_type"].data]
    n_vocab = len(tokens)
    keep = set()
    if counts_path == "ascii":
        byte_of = byte_decoder()
        for i, t in enumerate(tokens):
            if all(ord(ch) in byte_of and byte_of[ord(ch)] < 128 for ch in t):
                keep.add(i)
    else:
        for line in open(counts_path):
            tid, c = line.split()
            if int(c) >= min_count and int(tid) < n_vocab:
                keep.add(int(tid))
    for i, (t, ty) in enumerate(zip(tokens, types)):
        # Control/user-defined tokens and every single byte (so any text can be spelled).
        if ty != 1 or len(t) == 1:
            keep.add(i)
    order = sorted(keep) + [i for i in range(n_vocab) if i not in keep]
    new_id = np.empty(n_vocab, dtype=np.int64)
    new_id[np.array(order)] = np.arange(n_vocab)
    n_short = len(keep)
    n_short_padded = (n_short + 31) // 32 * 32  # keep row blocks aligned for repacking
    print(f"vocab {n_vocab} -> output rows {n_short_padded} ({n_short} kept)")

    arch = bytes(f["general.architecture"].parts[-1]).decode()
    w = GGUFWriter(dst, arch=arch, endianess=r.endianess)
    for name, field in f.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        vtype = field.types[0]
        if name == "tokenizer.ggml.tokens":
            w.add_key_value(name, [tokens[i] for i in order], vtype, sub_type=field.types[-1])
        elif name == "tokenizer.ggml.token_type":
            w.add_key_value(name, [types[i] for i in order], vtype, sub_type=field.types[-1])
        elif name == "tokenizer.ggml.scores":
            scores = [float(field.parts[i][0]) for i in field.data]
            w.add_key_value(name, [scores[i] for i in order], vtype, sub_type=field.types[-1])
        elif name.startswith("tokenizer.ggml.") and name.endswith("_token_id"):
            w.add_key_value(name, int(new_id[int(field.parts[-1][0])]), vtype)
        else:
            w.add_key_value(name, field.contents(), vtype, sub_type=field.types[-1] if vtype == GGUFValueType.ARRAY else None)
    w.add_key_value("nes.output_vocab", n_short_padded, GGUFValueType.UINT32)

    tensors = []
    for t in r.tensors:
        data = np.asarray(t.data)
        if t.name == "token_embd.weight":
            if data.shape[0] != n_vocab:
                raise SystemExit(f"unexpected token_embd shape {data.shape}")
            data = data[np.array(order)]
            tensors.append((t.name, data, t.tensor_type))
            tensors.append(("output.weight", np.ascontiguousarray(data[:n_short_padded]), t.tensor_type))
        elif t.name == "output.weight":
            raise SystemExit("model already has a separate output.weight; permute it too")
        else:
            tensors.append((t.name, data, t.tensor_type))
    for name, data, qtype in tensors:
        w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _, data, _ in tensors:
        w.write_tensor_data(data)
    w.close()


if __name__ == "__main__":
    main()
