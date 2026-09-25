# Ti-Alloy Surrogate Model Benchmark - Comparison Hub (Capstone)
Neutral dashboard linking the four repositories of the aerospace Ti-alloy
screening benchmark. Role assignment: one validated primary screener plus
two red-team OOD stress-test models.

## Repository map
- CrabNet-TiAlloy-Screening : primary screener; 8-test validation suite (A-H);
  dual-vintage hull gating; 39,480-candidate exhaustive screen; 244.33 samples/s E2E
- QwenRHEA-TiAlloy-Test     : red-team LLM (RHEA-trained); 4-bit NF4 on RTX 2000 Ada;
  1.711 samples/s @ batch 8; density/phase hallucinations
- TRIADS-TiAlloy-Test       : red-team ML (steel-trained, 5-fold ensemble);
  3,783.5 samples/s tensor-only vs 27.3 samples/s E2E (matminer tax)
- TiAlloy-Model-Comparison  : this repo - shared eval set, standardized summaries, audit scripts

## Contents
eval_set_20k.csv / controls.csv      - fixed inputs shared by ALL models (seed 42)
model_results_*.json                 - standardized per-model summaries (schema-aligned)
compare_all_models.py / _v2.py       - merge + full property audit scripts
grand_comparison_table(_v2).csv      - merged property tables (3 and 20 alloys)
slide_grand_comparison(_v2).png      - presentation figures

## Headline findings
| Metric                        | CrabNet          | Qwen-8B (LLM)      | TRIADS (steel ML)   |
|-------------------------------|------------------|--------------------|---------------------|
| Role                          | primary screener | red-team reviewer  | red-team reviewer   |
| E2E throughput (samples/s)    | 244.33           | 1.711              | 27.3 (steels) / 25.98 (Ti) |
| Density vs physics            | anchor (RoM-consistent) | mean +29.7%, max +60.6% | n/a |
| Yield strength behaviour      | n/a (elastic props)  | 1000-1200 MPa guesses | flat ~833 MPa steel-mean |
| Internal consistency          | deterministic    | Tabor self-inconsistency mean 43% | deterministic ensemble |
| OOD flags on Ti-rich inputs   | gates pass       | RHEA_EXTRAPOLATION / PHASE_HALLUCINATION | OOD_STEEL_BIAS (100%) |
| Verdict                       | PRODUCTION       | stress-test only   | stress-test only    |

## Reproduce
python -m venv venv ; activate ; pip install -r requirements_comparison.txt
python compare_all_models_v2.py

## Provenance
CrabNet weights: Zenodo 10.5281/zenodo.4633866 (vintages 2020-12-04 / 2021-02-23)
Qwen weights: HF tianchuang/Qwen-3-8B-RHEA-property-predictor (4-bit NF4 at load)
TRIADS: code github.com/Rtx09x/TRIADS ; weights HF Rtx09x/TRIADS weights/steels/weights.pt
