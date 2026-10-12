# Submission-grade table package (README_tables.md)

> Directory: `manuscript/tables/`
> This package is an **aggregation and formatting** product: every number comes from CSV / XLSX files
> already present under `results/`, and **no analysis was re-run**.
> Build script: `results/manuscript_support/build_tables.py` (reads `results/` only, writes `manuscript/tables/` only).

---

## 1. File list and mapping to the manuscript

| File | Manuscript table | Sheets | Content |
|---|---|---|---|
| `Table1_cohort_inventory.xlsx` (+ `.csv`) | **Table 1** (main) | 1 | 8 independent data sources + 1 derived single-cell integration (C1-C8, S1) |
| `Table2_headline_evidence.xlsx` (+ `.csv`) | **Table 2** (main) | 1 | 23 non-redundant headline comparisons (columns prettified; `effect_str` split into `effect_metric`/`effect`) |
| `Table3_hrposher2neg_performance.xlsx` (+ `.csv`) | **Table 3** (main) | 1 (with `block` sections) | HR+/HER2- predictive performance: primary result / C-index comparison / multivariable / proliferation tertiles / external GEO |
| `TableS1_rep30_genes.xlsx` (+ `.csv`, `TableS1_metadata.csv`) | **Table S1** (supplementary) | 2 | REP30 30-gene panel (Cox coefficients / p values) + 52-gene panel membership; metadata in `TableS1_metadata.csv` |
| `TableS2_full_evidence.xlsx` (+ `.csv`) | **Table S2** (supplementary) | 1 | Full 472 rows (461 include=yes + 11 include=no, with `exclude_reason`) |
| `TableS3_prognostic_layer.xlsx` (+ `.csv`) | **Table S3** (supplementary) | 1 | Prognostic OS layer + single-cell variance decomposition (eta-squared: study 0.025 / cell type 0.036 / sample 0.103; 96,368 cells) |
| `TableS4_subtype_and_hrpos.xlsx` | **Table S4** (supplementary) | 8 | Subtype layer + all HR+/HER2- tests |
| `TableS5_genomic_and_metastatic.xlsx` | **Table S5** (supplementary) | 6 | Genomic and metastatic clinical layer |
| `csv/` (subdirectory) | - | - | Every sheet of every xlsx also saved as a same-named csv for easy review |

---

## 2. Per-table data-source mapping ("which number comes from which source file")

### Table 1 - Cohort inventory
- Source: `results/evidence_table/submission/Table_S1_cohort_inventory.csv` (columns `cohort_id, cohort, cohort_type, specimen, access_route, n_patients, n_samples, evidential_role, comparisons_in_supplement`)
- Mapping: `dataset<-cohort`, `type<-cohort_type`, `n_comparisons<-comparisons_in_supplement`; `n<-n_patients` (cell-type cohorts use the cell count from `n_samples`).
- C6 = GEO `GSE209998` + cBioPortal `brca_aurora_2023`, **the same 53 patients**, merged into one cohort.
- Sum of `n_comparisons` = 461, matching the `include=yes` row count in Table S2.
- The table note records the exclusion of `brca_mbcproject_wagle_2017` (99.1% patient overlap with C8; pseudo-replication).

### Table 2 - Headline evidence (23 rows)
- Source: `results/evidence_table/submission/Table_S2_headline_evidence.csv`
- Mapping: `effect_str` (e.g. `cliffs_delta=-0.375`) split into `effect_metric` (`cliffs_delta`/`spearman_r`) and `effect` (sign and precision kept); `p_str->p`; `n_label/control->n_label_control`.

### Table 3 - HR+/HER2- performance
| block | source file |
|---|---|
| Primary result | `results/subtype_rep30/subtype_cox.csv` (METABRIC `HRposHER2neg` row: n=1398, 812 ev, HR=1.9289, p=3.86e-62, C=0.6766) |
| C-index vs comparator predictors | `results/subtype_rep30/hrposher2neg_cindex.csv` (REP30 0.677 / Age 0.642 / NPI 0.617 / PROLIF 0.587 / MKI67 0.569 / PAM50 0.501) |
| Multivariable models | `results/subtype_rep30/hrposher2neg_multivariable.csv` (REP30+PROLIF / +MKI67 / +clinical) |
| HR by proliferation tertile | `results/subtype_rep30/hrposher2neg_prolif_tertiles.csv` |
| External GEO validation | `results/subtype_rep30/geo_hrpos_external.csv` (GSE9195 / GSE26971 / GSE17705 / GSE45255) |

### Table S2
- S2: `TableS2_full_evidence.xlsx` (461 include=yes + 11 include=no; contains all raw columns, including source_file)

### Table S3 - Prognostic + single-cell variance
- Source: `results/evidence_table/submission/Table_S4_prognostic_layer.csv` (= `.xlsx`)
- OS HR: `results/rep52_trim/backtest_results.csv`; variance decomposition: `results/sc/integrated/report/integrated_rep_tests.csv`

### Table S4 - Subtype & HR+/HER2- (8 sheets)
| sheet | source file |
|---|---|
| `subtype_cox` | `results/subtype_rep30/subtype_cox.csv` |
| `interaction_tests` | `results/subtype_rep30/interaction_tests.csv` |
| `score_correlations` | `results/subtype_rep30/score_correlations.csv` |
| `multivariable` | `results/subtype_rep30/hrposher2neg_multivariable.csv` |
| `prolif_tertiles` | `hrposher2neg_prolif_tertiles.csv` + `geo_hrpos_prolif_tertiles.csv` (merged with a `source` column) |
| `cindex` | `results/subtype_rep30/hrposher2neg_cindex.csv` |
| `geo_external` | `results/subtype_rep30/geo_hrpos_external.csv` |
| `resistance_labels` | `results/subtype_rep30/resistance_labels_hrposHER2neg.csv` |

### Table S5 - Genomic & metastatic (6 sheets)
| sheet | source file |
|---|---|
| `igr2015_landscape` | `results/brca_igr_2015/mutation_frequency.csv`, `cna_frequency.csv`, `cohort_comparison.csv`, `subtype_alteration.csv` (ESR1 mut 10.2%, HR+/HER2- 14.0%, RB1 loss 34.7%; TCGA primary ESR1 mut 0.8%) |
| `dfci_cdk46` | `results/metastatic/cdk46_resistance/dfci_sample_table.csv` (phenotype counts and duration medians), `dfci_gene_vs_phenotype.csv` (TP53 21/41 vs 0/18, Fisher p=6.6e-5, q=0.0074), `dfci_esr1_hotspot.csv`, `dfci_gene_vs_treatment_duration.csv` (CDKN2A delta=-0.80 p=0.022); method sensitivity (mid-p 4.0e-5, Firth OR=33.1 p=0.015) from `results/method_sensitivity/` |
| `met500_brca_labels` | `results/metastatic/MET500_clinical/resistance_label_tests.csv` (tumour content r=+0.371 p=1.5e-6; proliferation r=+0.099 p=0.21) |
| `mbcproject_labels` | `results/metastatic/MBCproject/rep30_tests.csv` (REP30~proliferation r=+0.410) |
| `aurora_labels` | `results/metastatic/AURORA_cBioPortal/rep30_tests.csv` |
| `exposure_axis_replication` | **Archived, not re-derived**: `_archive/2026-09-21_deadends/results/resistance_axis/` (see `results/manuscript_support/exposure_axis_source.txt`); MBCproject delta=+0.717 p=1.8e-13 -> AURORA delta=+0.086 p=0.414; cross-cohort delta-vector r=-0.021 |

---

## 3. Conventions and formatting rules
- **Effect-size convention**: Cliff's delta > 0 means the **more resistant/exposed group** has a higher REP30 (resistant-minus-control direction); Spearman r denotes a **dose axis** (treatment exposure / duration).
- **Negligible band**: `|effect| < 0.15` treated as negligible.
- **p values**: stored as **text** with the scientific notation preserved verbatim (e.g. `3.864586741297403e-62`) to avoid Excel displaying `6.6E-05` with loss of precision; numeric columns keep full precision, no rounding.
- **Unbounded ORs**: infinite ORs caused by zero cells are stored as the text `inf` (e.g. RB1 4/41 vs 0/18).
- **Formatting**: bold header + light grey fill, row height 15, frozen first row, auto column width (CJK characters counted as width 2), table notes below the table (italic, small font).
- **Figures**: this package contains no figures (figures are handled separately).

---

## 4. Known data gaps (recorded honestly)
1. **Table 3**: the multivariable `REP30+Age+NPI` requested by the task appears in the source file as `REP30+clinical` (partner=`AGE+NPI`); presented as-is from the source.
2. **Table S5 / `exposure_axis_replication`**: the source files live under `_archive/2026-09-21_deadends/` and were **not re-derived**; the sheet lists the summary and per-gene deltas, with the source path recorded in `results/manuscript_support/exposure_axis_source.txt`.
3. **Table S5 / `dfci_cdk46`**: the `mid-p 4.0e-5` and `Firth OR=33.1 p=0.015` values are not in the two DFCI result CSVs; they come from `results/method_sensitivity/` (`method_matrix_all.csv` / `interpretation.md`) and are annotated in the `TP53 method sensitivity` block.
4. **Table 1**: the `n` column counts cells for C5/S1 and samples for C1 (cell lines have no patients), as stated in the table note; this reflects differing count units across cohorts, not missing data.
5. All other supplementary tables have no gaps; the source files under `results/` are complete.
