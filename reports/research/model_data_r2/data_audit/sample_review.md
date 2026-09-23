# Deterministic sample review

The frozen manifests select ten files per language and arm by the registered
hash ordering. `reconciled-audit.json` verifies all 200 sample identities against
the retained source pool. A content preview of the first selected file in every
language/arm was inspected before either full pilot began. Smoke/restart checks
were already running; no selection or policy was changed after this review.

The sample includes declarations, function bodies, tests, comments, docstrings,
SQL parsing, numerical image processing, and shell build commands. Shared files
between arms are expected: this is a sampling-policy treatment, not disjoint
training sets. The review does not establish code quality.

Two limitations are visible in the actual source:

- The standard C sample from `shaojiankui/iOS10-Runtime-Headers` is a generated
  Objective-C header categorized upstream as C. Its C parser failure is not
  evidence that the file is invalid Objective-C. Parser filtering can remove
  language-label mismatches and unsupported extensions, not just broken code.
- Repository-level license metadata does not reliably identify each file's
  license. The filtered Ruby-GNOME2 C sample carries an LGPL notice although
  its repository metadata says Ruby; the standard ScriptDev2 C++ sample carries
  a GPL notice although its metadata lists other licenses. The campaign retains
  the existing authorized Stack-dedup source policy and original provenance;
  it does not claim a newly verified permissive-only corpus. Checkpoints and
  packed source remain private. A future release needs a separate file-level
  license review.

The filtered policy reduces repository concentration and removes declared
markers/parser failures. Those measurable changes are the treatment. The
experiment must decide whether they help; this audit does not call the result
“high-quality code.”
