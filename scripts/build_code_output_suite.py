"""Build the frozen causal code-output behavior suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tinycomplete.eval.code_output_benchmark import CodeLanguage, CodeOutputCase

CODE_PREFIXES: dict[CodeLanguage, list[tuple[str, str, str]]] = {
    "python": [
        (
            "function",
            "src/math_utils.py",
            "def clamp(value: float, low: float, high: float) -> float:\n",
        ),
        (
            "class",
            "src/cache.py",
            "class CacheEntry:\n    def __init__(self, key: str, value: bytes):\n",
        ),
        ("control_flow", "src/chunks.py", "def nonempty_chunks(items):\n    for item in items:\n"),
    ],
    "typescript": [
        (
            "function",
            "src/math.ts",
            "export function clamp(value: number, low: number, high: number): number {\n",
        ),
        (
            "class",
            "src/cache.ts",
            "export class CacheEntry {\n"
            "  constructor(readonly key: string, readonly value: Uint8Array) {\n",
        ),
        (
            "control_flow",
            "src/chunks.ts",
            "export function nonemptyChunks(items: string[]): string[] {\n"
            "  const out: string[] = [];\n"
            "  for (const item of items) {\n",
        ),
    ],
    "javascript": [
        ("function", "src/math.js", "export function clamp(value, low, high) {\n"),
        ("class", "src/cache.js", "export class CacheEntry {\n  constructor(key, value) {\n"),
        (
            "control_flow",
            "src/chunks.js",
            "export function nonemptyChunks(items) {\n"
            "  const out = [];\n"
            "  for (const item of items) {\n",
        ),
    ],
    "java": [
        (
            "function",
            "src/MathUtils.java",
            "final class MathUtils {\n  static int clamp(int value, int low, int high) {\n",
        ),
        (
            "class",
            "src/CacheEntry.java",
            "final class CacheEntry {\n  private final String key;\n  CacheEntry(String key) {\n",
        ),
        (
            "control_flow",
            "src/Chunks.java",
            "import java.util.*;\n"
            "final class Chunks {\n"
            "  static List<String> nonempty(List<String> items) {\n"
            "    var out = new ArrayList<String>();\n"
            "    for (var item : items) {\n",
        ),
    ],
    "cpp": [
        (
            "function",
            "src/math.cpp",
            "#include <algorithm>\nint clamp_value(int value, int low, int high) {\n",
        ),
        (
            "class",
            "src/cache.cpp",
            "#include <string>\n"
            "class CacheEntry {\n public:\n"
            "  explicit CacheEntry(std::string key) : key_(std::move(key)) {}\n"
            " private:\n",
        ),
        (
            "control_flow",
            "src/chunks.cpp",
            "#include <string>\n#include <vector>\n"
            "std::vector<std::string> nonempty(const std::vector<std::string>& items) {\n"
            "  std::vector<std::string> out;\n"
            "  for (const auto& item : items) {\n",
        ),
    ],
    "rust": [
        ("function", "src/math.rs", "pub fn clamp(value: i32, low: i32, high: i32) -> i32 {\n"),
        (
            "class",
            "src/cache.rs",
            "pub struct CacheEntry {\n    key: String,\n}\n\n"
            "impl CacheEntry {\n    pub fn new(key: String) -> Self {\n",
        ),
        (
            "control_flow",
            "src/chunks.rs",
            "pub fn nonempty(items: &[String]) -> Vec<String> {\n"
            "    let mut out = Vec::new();\n"
            "    for item in items {\n",
        ),
    ],
    "go": [
        ("function", "math.go", "package sample\n\nfunc clamp(value, low, high int) int {\n"),
        (
            "class",
            "cache.go",
            "package sample\n\n"
            "type CacheEntry struct {\n\tKey string\n}\n\n"
            "func NewCacheEntry(key string) CacheEntry {\n",
        ),
        (
            "control_flow",
            "chunks.go",
            "package sample\n\nfunc nonempty(items []string) []string {\n"
            "\tout := make([]string, 0, len(items))\n"
            "\tfor _, item := range items {\n",
        ),
    ],
    "c": [
        ("function", "src/math.c", "int clamp_value(int value, int low, int high) {\n"),
        (
            "class",
            "src/cache.c",
            "#include <stddef.h>\n"
            "typedef struct {\n    const char *key;\n    size_t size;\n} cache_entry;\n\n"
            "cache_entry make_entry(const char *key, size_t size) {\n",
        ),
        (
            "control_flow",
            "src/chunks.c",
            "#include <stddef.h>\n"
            "size_t count_nonempty(const char **items, size_t count) {\n"
            "    size_t total = 0;\n"
            "    for (size_t i = 0; i < count; i++) {\n",
        ),
    ],
    "csharp": [
        (
            "function",
            "src/MathUtils.cs",
            "static class MathUtils {\n"
            "    public static int Clamp(int value, int low, int high) {\n",
        ),
        (
            "class",
            "src/CacheEntry.cs",
            "sealed class CacheEntry {\n"
            "    public string Key { get; }\n"
            "    public CacheEntry(string key) {\n",
        ),
        (
            "control_flow",
            "src/Chunks.cs",
            "using System.Collections.Generic;\n"
            "static class Chunks {\n"
            "    public static List<string> Nonempty(IEnumerable<string> items) {\n"
            "        var output = new List<string>();\n"
            "        foreach (var item in items) {\n",
        ),
    ],
}

MARKDOWN_PREFIXES = [
    ("configuration", "README.md", "# Configuration\n\nThe `timeout_ms` option controls"),
    ("release_notes", "CHANGELOG.md", "## 0.4.0\n\nThis release fixes"),
    ("architecture", "docs/architecture.md", "## Cache ownership\n\nEach request owns"),
    (
        "troubleshooting",
        "docs/troubleshooting.md",
        "## The server exits at startup\n\nFirst, check",
    ),
]


def build_cases() -> list[CodeOutputCase]:
    cases = [
        CodeOutputCase(
            id=f"{language}/{index:02d}-{category}",
            language=language,
            path=path,
            prefix=prefix,
            target="code",
            category=category,
        )
        for language, entries in CODE_PREFIXES.items()
        for index, (category, path, prefix) in enumerate(entries, 1)
    ]
    cases.extend(
        CodeOutputCase(
            id=f"markdown/{index:02d}-{category}",
            language="markdown",
            path=path,
            prefix=prefix,
            target="markdown",
            category=category,
        )
        for index, (category, path, prefix) in enumerate(MARKDOWN_PREFIXES, 1)
    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/code_output_v1.jsonl"))
    args = parser.parse_args()
    cases = build_cases()
    payload = "".join(
        json.dumps(case.model_dump(mode="json"), sort_keys=True) + "\n" for case in cases
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "version": 1,
        "protocol": "causal-code-output-v1",
        "case_count": len(cases),
        "code_cases": sum(case.target == "code" for case in cases),
        "markdown_cases": sum(case.target == "markdown" for case in cases),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "languages": {
            language: sum(case.language == language for case in cases)
            for language in sorted({case.language for case in cases})
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
