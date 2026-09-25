# Data management and contributions

The public repository contains implementation code, synthetic tests, aggregate statistical results and figures. MIMIC access and use remain governed by the conditions attached to the [PhysioNet dataset](https://physionet.org/content/mimiciv/3.1/).

## Public and local material

| Material | Storage and release rule |
|---|---|
| Source code, methods, synthetic tests | Suitable for the repository after review |
| Group metrics, paired intervals, group-by-outcome weight tables | Publish the reviewed aggregate tables with their definitions |
| Overall calibration display | Publish bin summaries and fixed-grid LOWESS display values |
| Training curves | Publish epoch, loss and selected-epoch summaries |
| Raw MIMIC files, event databases, metadata, feature arrays, patient-level labels or predictions | Keep in controlled local storage |
| Sample weights, checkpoints, fitted policy objects and preprocessing artifacts | Keep local |
| Raw logs, raw-value audit strings, credentials, local configuration and intermediate files | Keep local; review any proposed summary separately |

The distributed calibration export contains overall curves only. Finer subgroup bins and individual prediction coordinates are excluded. Aggregate status alone does not establish that every possible export is appropriate for release; new groupings and small-cell summaries require a fresh disclosure review.

## Before proposing a change

Keep preparation inputs and derived patient data outside the repository. Local `fairness_study/experiments/` and `fairness_study/tem/` are ignored. CSV files are ignored by default, with exact-path exceptions for the reviewed aggregate tables.

Before committing, inspect `git status --short`, `git diff --cached --stat` and the actual staged contents. Check data granularity as well as filenames: overwriting an approved aggregate filename with patient-level data bypasses a filename-based allowlist. Avoid force-adding ignored research artifacts.

When a public result changes, document the cohort, configuration and statistical convention that produced it. Maintain the file inventory in `fairness_study/results/public_package_manifest.json`, and retain the prior release record when replacing a documented snapshot.

## Documentation versions and licensing

Current English documents are in `docs/new/`; `docs/old/` retains the previous README and the original distributed protocol verbatim. Historical copies are reference records, not the current reproduction instructions. Documentation revision notes are kept in [`docs/revision_notes.md`](../revision_notes.md).

No code license is currently declared in the repository. License selection requires the maintainer to establish rights and upstream obligations; this documentation does not assign one. Data-access conditions and code licensing are separate matters.
