# Result Provenance Table

| Paper location | Quantity | Value | Source file | Script / command | Status |
| --- | --- | ---: | --- | --- | --- |
| Abstract / Table 1 | SC-Select CSR-All | 0.5122 | `server_audits/server_004_self300_20260513/outputs/self300_candidate_scores_all_rewards.csv` | scripts/07_run_baselines.py plus local recomputation | Verified |
| Table 1 | Fixed Seed-0 CSR-All | 0.3819 | `outputs/selections/self300_sdxl_qwen_all_rewards_20260514/baseline_summary.csv` | scripts/07_run_baselines.py | Verified |
| Table 1 | CLIPScore-Select CSR-All | 0.3779 | `outputs/selections/self300_sdxl_qwen_all_rewards_20260514/baseline_summary.csv` | scripts/07_run_baselines.py | Verified |
| Table 1 | ImageReward-Select CSR-All | 0.3686 | `outputs/selections/self300_sdxl_qwen_all_rewards_20260514/baseline_summary.csv` | scripts/07_run_baselines.py | Verified |
| Table 1 | PickScore-Select CSR-All | 0.3695 | `outputs/selections/self300_sdxl_qwen_all_rewards_20260514/baseline_summary.csv` | scripts/07_run_baselines.py | Verified |
| Table 1 | HPSv2-Select CSR-All | 0.3808 | `outputs/selections/self300_sdxl_qwen_all_rewards_20260514/baseline_summary.csv` | scripts/07_run_baselines.py | Verified |
| Table 1 / Ablation | VLM-Direct-Select CSR-All | 0.3991 | `server_audits/server_004_vlm_direct_select_300_20260515/ablation_vqa_select.json` | server_004 scripts/27_vqa_select_baseline.py | Verified |
| GenEval table | GenEval official Fixed / CLIP / SC | 0.53885 / 0.57826 / 0.62232 | `paper/manuscript plus server audit GenEval outputs` | GenEval official detector | Verified from copied summaries |
| Cross-generator table | FLUX stratified-60 SC-Select | 0.6667 | `outputs/selections/flux_self300_stratified_60_all_rewards_20260514/baseline_summary.csv` | scripts/07_run_baselines.py | Verified |
| Cross-generator table | T2I-CompBench SDXL-60 SC-Select | 0.5625 | `server_audits/server_004_t2i_compbench_sdxl_60_20260514/t2i_compbench_sdxl_60_all_rewards_20260514/baseline_summary.csv` | scripts/07_run_baselines.py | Verified |
| Parser robustness | InternVL2.5-4B branch CSR-Select | 0.7031 | `server_audits/server_004_parser_robustness_internvl_flux60_20260514/parser_robustness_internvl_flux60_20260514/baseline_summary.csv` | InternVL parser branch | Verified |
| Human preference | Majority valid CSR win rate | 69.4% | `outputs/human_eval/pairwise_20260514/multi_annotator_3x_20260515/multi_annotator_report.md` | scripts/15_analyze_pairwise_human_eval.py | Verified |
| Human preference | Fleiss' kappa | 0.565 / 0.656 | `outputs/human_eval/pairwise_20260514/multi_annotator_3x_20260515/multi_annotator_report.md` | scripts/15_analyze_pairwise_human_eval.py | Verified |
| Ablation | Constraint-family leave-one-out table | See table | `paper/tables/constraint_family_ablation_self300_20260515.csv` | local family-ablation recomputation | Verified |
| Preference trade-off | Reward score table | See table | `paper/tables/quality_tradeoff_selected_rewards_20260515.csv` | local selected reward aggregation | Verified |
