# Reproduction bundle

Code and derived result tables for the manuscript:

*An overall-survival-derived gene-expression score reads proliferation and
prognosis, not acquired drug resistance: an eight-cohort audit in breast
cancer.*

Every input is a public accession (see the manuscript's Data Availability
Statement), so all tables and figures regenerate end to end once the raw
public data are placed in the expected locations.

## Contents

```
code_release/
├── README.md                  this file
├── requirements.txt           exact package versions (Python 3.14.7)
├── MANIFEST.txt               SHA256 + size for every file in this bundle
├── download_gse.py            fetch GEO series (soft-files / counts) for C1/C2/C5 and GEO validation cohorts
├── download_gpl.py            fetch GEO platform annotation files
├── download_fix.py            repair/re-download incomplete GEO fetches
├── score_cohort.py            apply the REP30 score to a prepared expression matrix
├── analysis/                  all audit and analysis scripts (56 files, see Pipeline order)
├── manuscript/
│   ├── build_manuscript_figures.py   generate Figures 1–4 and S1–S4 (PDF/EPS/PNG)
│   ├── export_panels.py              export individual figure panels
│   └── tables/                       DERIVED RESULT TABLES (Table 1–3, S1–S5, csv + xlsx)
└── results/manuscript_support/build_tables.py   generator for the submission tables
```

## Environment

- Python 3.14.7
- Install exact versions: `pip install -r requirements.txt`
- Key packages: numpy 2.5.3, pandas 2.3.3, scipy 1.18.1, scikit-learn 1.9.1,
  statsmodels 0.15.0, lifelines 0.30.3, scanpy 1.12.4, anndata 0.13.3.post0,
  harmonypy 0.2.0, matplotlib 3.11.2, openpyxl 3.1.5, python-docx 1.2.0

## Data (not included; all public)

Fetch the accessions below with `download_gse.py` / `download_gpl.py`
or the cBioPortal API, and arrange them as:

```
data/
├── 01_core/                 C1 GSE26459 (in vitro); C3 TCGA-BRCA; C4 METABRIC (cBioPortal brca_metabric)
├── 02_acquired_resistance/  C2 GSE45255 (treatment-dose series)
├── 03_single_cell/          C5 GSE264205 (scRNA-seq); GSE249361; GSE161529 (derived synthesis S1)
├── 04_api_reference/        cBioPortal API reference downloads (brca_aurora_2023, brca_mbcproject_2022)
├── 05_gpl/                  GEO platform annotation files
└── 06_metastatic/           C6 AURORA US (GSE209998); C7 MET500; C8 MBCproject/Count Me In
```

Cohort codes C1–C8 and the derived synthesis S1 follow the manuscript's
Table 1 (cohort inventory).

## Pipeline order (main analyses)

1. **Data acquisition** — `download_gse.py`, `download_gpl.py`,
   `download_fix.py`; cBioPortal queries (see `analysis/survey_cbioportal_cohorts.py`).
2. **Score construction** — `analysis/rep_build.py` (Cox-weighted REP30
   assembly; METABRIC = fitting cohort, TCGA-BRCA held out),
   `analysis/rep30_resistance_validation.py` and
   `analysis/rep30_resistance_within_er.py` (the 461-comparison audit).
3. **Statistical toolkit** — `analysis/stats_toolkit.py` is imported by all
   audit scripts (Firth penalised logistic regression, permutation tests,
   Cliff's delta, Hodges–Lehmann, Newcombe CI). Run nothing without it.
4. **Prognostic layer** — `analysis/prognostic_metrics_hrpos.py`,
   `analysis/rep52_percohort_performance.py`, `analysis/subtype_rep30.py`.
5. **Null benchmarks** — `analysis/audit_venet_null.py` …
   `audit_venet_null5.py` (random-gene-set nulls A–D),
   `analysis/audit_cindex_cost*.py`, `analysis/audit_transfer_stability.py`,
   `analysis/audit_prolif_control.py`, `analysis/method_sensitivity_demo.py`.
6. **Single-cell layer** — `analysis/sc_pipeline.py`,
   `analysis/sc_harmony_integrate.py`, `analysis/sc_gse264205_*.py`,
   `analysis/sc_rep_de*.py`.
7. **Metastatic / clinical layer** — `analysis/met500_*.py`,
   `analysis/aurora_*.py`, `analysis/mbcproject_*.py`,
   `analysis/fetch_metastatic_rna.py`, `analysis/cdk46_resistance_genomic.py`,
   `analysis/igr2015_resistance_landscape.py`, `analysis/ras_analysis.py`.
8. **Tables and figures** — `results/manuscript_support/build_tables.py`
   (submission tables), `manuscript/build_manuscript_figures.py` +
   `manuscript/export_panels.py` (figures), and the ready-made derived
   tables in `manuscript/tables/`.

## Important note on paths

The scripts are deposited exactly as used for the published results. Some
scripts contain machine-specific absolute paths in their input-configuration
section; to rerun on another machine, adjust those path constants to your
local layout (the relative-layout convention `data/`, `results/`,
`manuscript/tables/` is preserved in this bundle). `score_cohort.py` is the
generic entry point for applying the REP30 score to any prepared expression
matrix.
