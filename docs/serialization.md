# Serialization: compact event encoding for model prompts

Line-based tag format (`src/tinycomplete/protocol/serialize.py`).
Byte offsets into UTF-8. Escaping: content lines starting with `<` or `\`
gain one `\` prefix; the parser strips exactly one leading `\`.
`<P path offset>` is always the last line.

## Realistic session

Events: open `src/parser.py`, snapshot, move cursor to byte 57, insert
`item = `, diagnostic `E0308`, retrieved `src/lex.py` via tree-sitter,
then a predict request at byte 64.

```text
<F src/parser.py>
<S>
def parse(toks):
    out = []
    for t in toks:
        item = out.append(t)
    return out

</S>
<C 57>
<D 57 64 error>
E0308 expected `;`
</D>
</F>
<X tree-sitter src/lex.py>
def lex(s):
    return s.split()

</X>
<P src/parser.py 64>
```

Measured (2026-09-16, tokenizer `Qwen/Qwen3.5-0.8B-Base` from hub,
config/tokenizer only — no weights downloaded):

- serialized characters: 260
- Qwen tokens: 111 (~0.43 tokens/char on code)
- structural overhead vs raw source (~150 chars): ~110 chars, dominated by
  the retrieved-context block and tags.

Takeaway: full snapshots cost ~1x source size; per-keystroke deltas
(`<I>`/`<R>` style updates or re-serialization after replay) stay small.
`qwen_token_count()` falls back to character counts when the hub is
unreachable (it raises `RuntimeError`; `count_chars()` always works).
