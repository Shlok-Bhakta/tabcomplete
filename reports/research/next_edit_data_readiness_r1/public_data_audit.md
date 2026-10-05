# Public next-edit data readiness audit

## Scope and decision

This is a private research note for a future bounded Q25 next-edit study. The active R3 completion-scale campaign remains frozen. This audit inspected existing authorized artifacts and primary-source documentation. Hub metadata observations were refreshed on 2026-10-05 at 03:47 UTC. No new dataset shards, repository code, model weights, teacher outputs, or training examples were acquired. No GPU allocation occurred. Existing development, test, and private-answer reservations remain sealed.

There is no newly cleared large bank of human next-edit intent. Continue Instinct is the strongest already-cached candidate for observed human edits with context. Joseph Gentle's cached traces support exact sequential text reconstruction, with narrower task information. Neither currently supplies a verified broad next-edit training set. Large event counts and issue-solving trajectories need separate reconstruction and intent audits.

The user has requested finishing the currently active run, preparing a comprehensive handoff, and pausing. The trajectory bank described below is a future proposal. It was not implemented.

## Existing authorized evidence

| Existing artifact | Verified inventory | Readiness limits |
|---|---|---|
| Synthetic action pilot | 4,096 training, 1,024 development, 1,024 held-out states; insertion, replacement, deletion, no edit | Four action structures repeated with language/name/number variants. Existing family identifiers do not establish unseen semantic task families. Reserved splits remain reserved. |
| Cached Instinct TypeScript train | 4,371 official training rows; existing conversion retains 720 training and 210 development states | No no-edit states; full intermediate history and source-file licensing unverified; path split alone does not establish independent session/repository groups. |
| Personal feedback snapshot | 96 sessions, 292 proposals, eight sequence gaps, zero defensible preference pairs | Historical export observed 2026-10-04 14:31 UTC. These are not fresh live database totals. Synthetic editor tests are not human feedback. |
| CommitPack sibling provenance audit | Four objectively checked candidate edits, zero accepted training examples | Observable student intent and human chronology unverified. Private answers and reserved siblings were not opened. |
| Cached Rust/Svelte editing traces | 55,316 ordered transactions and 59,922 patches; both replay to their recorded final content | One author, two files; no explicit cursor, selection, acceptance, or task intent. |

The existing Instinct conversion retained 647 replacements, 53 deletions, and 20 insertions in training. Development has 182 replacements, 19 deletions, and nine insertions. Its filters removed 2,198 states outside the one-line cursor policy and 1,055 without uniquely attributable same-file history. Earlier task-readiness checks remained false. These numbers do not establish useful general next-edit accuracy.

The personal feedback artifact is `/mnt/ssd/tabcomplete-q25-completion-scale-r1/audit/current_feedback_evidence.json`, SHA-256 `f0f1a7e55b22bf0c6f7254acbd90024b8acd19ae22c3697b95249ed4216b8d36`. It exports no source content. Collection gaps invalidate preference derivation until a valid anchor restores reconstruction. Navigation, unseen cancellation, and later unrelated edits must stay separate from explicit rejection. Automatic personalization remains disabled.

`source_inventory.json` records paths, sizes, hashes, provenance, reservations, and individual limitations for these existing artifacts.

## Public source comparison

| Source | Observed task | License/access evidence | Appropriate future use |
|---|---|---|---|
| Continue Instinct | Chunked human TypeScript edits with five recent edits and code context; translated languages synthetic | Apache-2.0 dataset metadata; source-file rights still require audit | Re-audit existing official TypeScript training cache |
| Zed Zeta | Editable-region examples, events, outputs, rejection/assertion fields | Apache-2.0 metadata; 418 viewer training examples | Small format/control set, with example provenance checks |
| Josephg editing-traces | Ordered real text patches with exact initial/final states | Selected Rust/Svelte traces CC-BY-4.0 | Replay mechanics or a narrowly described sequential-patch task |
| crowd-code 0.1 | Raw IDE edits mixed with navigation and terminal activity | CC0 metadata; collector MIT does not clear every recorded source file | Reconstruction audit only; no training ingestion now |
| NVIDIA Open-SWE-Traces | Ordered agent tool trajectories conditioned on issues | CC-BY-4.0 dataset metadata plus repository rights | Separate agent-edit proxy after state and context checks |
| DECODE | Human changes to AI completions | Manual approval, research/privacy terms; no standard license exposed | Unavailable without approval and task-format audit |
| EditPackFT / Coeditor | Commit source pairs or constructed commit-edit order | Dataset/code metadata and per-source licenses differ | Instructional or commit-grouped proxy, clearly labeled |
| Educational IDE logs / SWE-chat | Unresolved schemas or gated agent traces | Access, rights, or reconstruction remain unknown | Metadata candidates only |

### Instinct and Zeta

Instinct's publisher describes actual team edits in a public TypeScript project. The data includes recent changes, nearby context, and a rewrite target. Its collection filters repetitive back-and-forth changes, so the particular rename/reversal behavior of interest may be underrepresented. Python, C, Rust, and Java rows are translated synthetic versions. Their original and translated examples must share a split group. Native official test data remains sealed. [Publisher description](https://blog.continue.dev/instinct/), [pinned dataset documentation](https://huggingface.co/datasets/continuedev/instinct-data/blob/e557e3ed1ea2c28b9c6b7cc46d670aa3cd451c29/README.md).

Zeta's published schema resembles the serving task, but examples need individual history and provenance checks. Constructed projects do not become independent real repositories through different names. Its train, evaluation, and DPO assignments must remain distinct. Zed documents opt-in data collection for eligible open-source projects; that policy alone does not prove the origin of every released row. [Dataset](https://huggingface.co/datasets/zed-industries/zeta), [collection policy](https://github.com/zed-industries/zed/blob/main/docs/src/ai/ai-improvement.md).

### Exact replay and damaged histories

The cached Josephg traces use Unicode codepoint offsets. Convert offsets to UTF-8 byte positions from the exact pre-state before building editor examples. Preserve transaction order and atomic multi-patch transactions. Exclude the initial large bootstrap insertion. A region chosen from the future patch location supplies an oracle location and must be identified as such; the traces do not establish a normal cursor policy or idle no-edit labels. Attribution is required. [Pinned trace format](https://github.com/josephg/editing-traces/blob/762fa6c51605c88a05ebe5c4b9d4540caca30b97/sequential_traces/README.md).

A fresh finding concerns crowd-code. Upstream's current documentation explicitly describes version 1.0 cache corruption from agent writes and external Git changes, missing actor attribution, and limited workspace coverage. The newer collector fixes do not repair the historical 0.1 dataset. That release cannot be treated as exactly reconstructable solely from its event count. Terminal content, machine paths, and unclear source rights also require filtering before any future ingestion. [Collector documentation](https://github.com/p-doom/crowd-code), [historical dataset](https://huggingface.co/datasets/p-doom/crowd-code-dataset-0.1).

### Agent and commit proxies

A prior fixed 100-row Open-SWE-Traces probe found 24 single-line replacement calls across 12 trajectories and zero reconstructed accepted labels. A resolved agent trajectory does not prove each intermediate change is inferable from current code and recent edits. Any future reconstruction must use a pinned public pre-state and explicit prior deltas without executing recorded commands. [Dataset](https://huggingface.co/datasets/nvidia/Open-SWE-Traces).

EditPackFT exposes before/after source, instructions, commit/repository identities, and row licenses. Its MIT dataset metadata coexists with source rows under other licenses, including AGPL. A future commit message is unavailable to a natural next-edit model unless explicitly supplied as an instruction. Coeditor's history construction likewise does not reveal a person's actual within-commit editing sequence. [EditPackFT](https://huggingface.co/datasets/nuprl/EditPackFT), [Coeditor](https://github.com/MrVPlusOne/Coeditor).

DECODE is promising observed post-completion editing data, but gated approval and privacy commitments apply. Its task also differs from unconstrained next-edit prediction. Educational IDE-log metadata does not yet establish exact source-state reconstruction. [DECODE](https://huggingface.co/datasets/jtliang/decode), [Aalto log metadata](https://research.aalto.fi/en/datasets/ide-action-log-dataset-from-a-cs1-mooc/).

## Future deterministic trajectory proposal

A small proposed bank would contain approximately 256 distinct states across 16 semantic task families and four languages. It would test latest-intent propagation, call-site changes, parameter/field consistency, deletions, imports, configuration literals, assertions, Unicode, whitespace, undo/reversal boundaries, and completed-task no edit. All variants of a semantic task family stay together before splitting. Source aliases, repositories, sessions, and commits add grouping constraints where available.

For `value0 → value1 → value2 → value3`, history should contain actual reconstructed synthetic declaration changes. The target updates remaining references to the currently declared `value3`. No target should predict `value4` merely from the numeric progression. Counterexamples include a direct rename to `value3`, a reversal to `value2`, unrelated numeric literals, unrelated editable regions, and obsolete history. Once the required references agree, a completed-task no-edit example is defensible. Arbitrary inactivity, navigation, cancellation, and transport failures are not no-edit ground truth.

Each state would preserve current file/filetype, byte cursor, a deterministic region chosen from current state, exact ordered history, pre-state hash, action, after-state hash, and context-policy version. Insertion uses a zero-width range; deletion uses an empty replacement. The canonical v2 actions and compact `N\n` or `R\n` plus exact replacement and EOS remain shared. No whitespace trimming, fence removal, or token-cutoff rescue is permitted.

Before any future training, verify full history replay, UTF-8 boundaries, target application, source rights, duplicate groups, and student-visible intent. Keep edit-required and no-edit denominators separate. Header compliance and a successful synthetic task do not establish human editing quality. This audit created zero examples and authorizes no new training.

## Audit artifacts

- `source_inventory.json` contains immutable dataset revisions and existing-artifact hashes.
- `hub_metadata_observations.json` contains fresh public metadata receipts only.
- No repository source, current campaign plan, collector database, or editor configuration changed.
