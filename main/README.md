# Main 24GB Ensemble Plan

This folder contains the main execution plan for the image emotion recognition competition.

The plan targets the highest final Macro-F1 under an initial 24GB GPU memory budget. It avoids oversized backbones as the default path and instead uses frozen feature extraction, lightweight heads, 5-fold OOF probability fusion, tail-class experts, safe pseudo-labeling, and a final submission gate.

## Core Strategy

- Use the Conda environment `dl` for Python commands.
- Do not full fine-tune large vision backbones by default.
- Cache frozen image features with `eval`, `torch.no_grad`, and fp16/bf16.
- Train lightweight heads on cached features.
- Use 5-fold OOF probabilities to learn fusion weights.
- Optimize Macro-F1, not Accuracy.
- Use tail-class experts only when OOF simulation proves a global Macro-F1 gain.
- Use pseudo-labels only as a final gated step with class quotas and low sample weight.

## Primary Model Pool

| Role | Model | Purpose |
|---|---|---|
| CLIP-like semantic mainstay | `facebook/PE-Core-L14-336` | Strong medium CLIP-like semantic feature source |
| Scene/global vision mainstay | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Scene, composition, and non-language visual representation |
| High-resolution detail source | `OpenGVLab/InternViT-300M-448px` | 24GB-friendly InternViT route for 448px details |
| EVA complement | `EVA02-CLIP-L-14-336` or available equivalent | EVA-family feature source without EVA-CLIP-18B cost |
| Local texture complement | DINOv3 ConvNeXt-L or ConvNeXtV2-L 384 | Color, texture, and local subject details |
| Existing anchors | Current Cambrian/SigLIP2/OpenCLIP outputs | Historical score anchors and fusion candidates |

Deferred by default:

- `EVA-CLIP-18B`: too large for the approved 24GB main route.
- `InternViT-6B`: replaced by `InternViT-300M-448px` unless larger hardware is approved.
- `DINOv3 ViT-H+`: optional probe only; does not block the main route.

## Files

- `schema.json`: JSON schema for each executable subtask.
- `index.json`: full project plan, model pool, phases, dependencies, and task registry.
- `tasks/*.json`: independently executable subtask plans.

Each task file includes:

- `goal`
- `inputs`
- `outputs`
- `execution_steps`
- `acceptance_criteria`
- `depends_on`
- `resource_profile`
- `implementation_notes`

## Execution Order

1. Foundation:
   - `001_project_foundation`
   - `002_data_split_and_audit`
   - `003_metric_submission_contract`

2. Feature cache framework:
   - `004_feature_cache_framework`

3. Parallel feature caching:
   - `005_cache_pe_core_features`
   - `006_cache_dinov3_vitl_features`
   - `007_cache_internvit300m_features`
   - `008_cache_eva02_clip_features`
   - `009_cache_convnext_features`
   - `010_import_existing_candidates`

4. Single-model screening:
   - `011_train_single_feature_heads`
   - `012_single_model_error_analysis`

5. Route B, 5-fold OOF fusion:
   - `013_build_fivefold_oof_features`
   - `014_train_oof_heads`
   - `015_oof_probability_fusion`

6. Route C, TTA and calibration:
   - `016_tta_probability_generation`
   - `017_macro_f1_calibration`

7. Route D, tail experts:
   - `018_tail_confusion_experts`
   - `019_expert_override_gate`

8. Safe pseudo-labeling and final package:
   - `020_safe_pseudolabel_selection`
   - `021_retrain_with_pseudolabels`
   - `022_final_model_selection`
   - `023_final_submission_package`

9. Optional upper-bound probe:
   - `024_optional_dinov3_vith_probe`

## Parallelization Notes

After `004_feature_cache_framework`, feature caching tasks can run independently as long as they do not write to the same model output directory.

After all selected feature caches complete, `011_train_single_feature_heads` can train heads per model independently.

After `013_build_fivefold_oof_features`, OOF head training can be parallelized by model and fold. The fusion, calibration, expert gate, pseudo-label, and final selection steps are sequential gates.

## Score Targets

- Current anchor: Cambrian fusion head around 73.
- Main 5-fold OOF fusion target: 76.5-78.5.
- With TTA, calibration, and tail experts: 78.5-80.
- With successful safe pseudo-labeling: 79-81 stretch target.

## Hard Constraints

- Use `conda run -n dl ...` for Python commands.
- All test predictions must follow numeric order from `1.jpg` to `1391.jpg`.
- Final `test.txt` must contain exactly 1391 lines, one legal label per line.
- All images must be loaded with RGB conversion.
- No final strategy is accepted unless it improves or preserves OOF Macro-F1 under the shared metric contract.
