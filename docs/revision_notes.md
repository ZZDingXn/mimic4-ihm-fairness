# English documentation revision, 25 September 2026

The project homepage has been restructured for first-time readers. The revision implements the supplied README assessment while keeping experimental results and implementation unchanged.

## Changes

- Replaced the Chinese handover-style homepage with an English research overview, a complete twelve-comparison mitigation table and two embedded result figures.
- Moved detailed methods, figure interpretation, data management and reproduction into five focused English documents under `new/`.
- Added an implementation-level cleaning guide covering joins, labels, conversions, categorical fallback, exact-time GCS derivation, repeated observations, bin boundaries and train-only preprocessing.
- Added a complete raw-data construction command and an input/output description for every reproduction step. Historical checkpoint repair is separated from a fresh run.
- Preserved the previous README, distributed protocol and package manifest verbatim under `old/`. The new homepage is retained under `new/README.md`, with relative links adjusted for that location.

The root README and `new/README.md` contain the same current text; the root copy serves GitHub's landing page. Archived documents retain their original language and relative links as historical snapshots. The current English documents are the maintained reading path.

## Terminology ledger

| Term | Meaning and editorial choice |
|---|---|
| ICU stay | Unit of prediction and group counts |
| Patient-cluster bootstrap | Patient is the sampling unit; all associated stays are retained |
| Mortality risk | Model probability of the recorded mortality outcome |
| Alert probability | Probability that a randomized policy emits a high-risk decision |
| Equal opportunity | TPR parity; implementation key `opportunity` |
| Equalized odds | Joint TPR/FPR parity; implementation key `equalized_odds` |
| Reweighing | Group-by-outcome training weights; implementation key `reweight` |
| Percentage points | Unit for differences in rates, distinct from relative percentage change |
| Glasgow Coma Scale | Prose spelling; `Glascow` remains unchanged in historical code/feature names |

## Evidence and preservation

Numeric entries in the homepage table are formatted directly from the existing comparison summary and paired-difference exports. Cleaning statements were checked against `prepare.py`, `constants.py`, `features.py` and `train.py`. No new physiological exclusion rule or deduplication rule was introduced.

The documentation does not assert an established code license, a measured hardware budget or a public manuscript release. Source papers and dataset access links are attributed in the homepage and methods. No manuscript, model, patient-level data or experimental result was edited in this revision.

- `old/README_2026-09-24_zh.md`: SHA-256 `d1ff329f51d66f6f884713ca513e820bde58d6f121aee9b06c1458af13fb6ba5`.
- `old/protocol_2026-09-23.md`: SHA-256 `5d155948537203aa8032e66568de79ca8f3dc31345f9f273878fe9ae68b30f9d`.
- `old/public_package_manifest_2026-09-24.json`: SHA-256 `74e00fb38daee1ec17b969053a0e620d956309d9fb471ec5c60ee290f877c374`.

## Verification

Current English documents passed local link and Markdown-fence checks. All twelve homepage comparison rows were checked against the aggregate source tables. Archived originals were checked byte-for-byte. No training or statistical evaluation was rerun.
