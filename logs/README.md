# logs/

Training runs, evals and helper tools for the Microduck work on this fork
(branch `local/velstand-own-teachers`). Reorganized 2026-11-13; before that
everything below sat loose at the top of `logs/`.

| Folder | What's in it |
|---|---|
| `rsl_rl/<experiment>/<run>/` | Training runs: checkpoints, exported ONNX, params, videos. **Fixed path**: `train --agent.load-run` resumes from here. |
| `kick_mission/<stamp>/` | Ball-kick mission runs, headless and viewer. **Fixed path**: `scripts/kick_mission.py` writes here. |
| `goto/<stamp>/` | Go-to-point suite and viewer runs. **Fixed path**: `scripts/goto_point.py` writes here. |
| `tb_views/` | TensorBoard views: relative symlinks into `rsl_rl/`, which is why this folder stays at this depth. |
| `training/<task>/` | Console logs of the training chunks, smoke tests and memory probes |
| `evals/<task>/` | Evaluation output: loose result files plus dated eval folders, each with its own `SUMMARY.txt` where one exists |
| `tools/` | Helper scripts. Run them from the repo root, e.g. `bash logs/tools/export_ckpt.sh <run_dir> <it>`. |

Tasks: `velocity_walk` (our walk, "walkv2"), `standup` (spin-capped StandUp
teacher), `velstand` (walk + get-up, anchor remap, round 3), `ballkick` (both
feet, stance/aim calibrations), `precision_step` (v1 and v2).

The export scripts in `tools/` share the lock file `logs/.export.lock`.
`scripts/export.py` always writes `./output.onnx`, so exports must never run
in parallel.

## Old path → new path

Briefs, policy cards and commits from before 2026-11-13 cite the old paths.
Those paths are still valid at those commits.

| Old (`logs/…`) | New (`logs/…`) |
|---|---|
| `anchor_cmd_response_4750.txt` | `evals/velstand/anchor_cmd_response_4750.txt` |
| `anchor_remap_validation_4750.txt` | `evals/velstand/anchor_remap_validation_4750.txt` |
| `ballkick_eval_2026-11-09` | `evals/ballkick/ballkick_eval_2026-11-09` |
| `ballkick_left_chunk1.log` | `training/ballkick/ballkick_left_chunk1.log` |
| `ballkick_left_eval_2026-11-12` | `evals/ballkick/ballkick_left_eval_2026-11-12` |
| `ballkick_memprobe_4096.log` | `training/ballkick/ballkick_memprobe_4096.log` |
| `ballkick_smoke.log` | `training/ballkick/ballkick_smoke.log` |
| `ballkick_smoke_export.log` | `training/ballkick/ballkick_smoke_export.log` |
| `ballkick_ts0p25_chunk1.log` | `training/ballkick/ballkick_ts0p25_chunk1.log` |
| `ballkick_ts1p0_chunk1.log` | `training/ballkick/ballkick_ts1p0_chunk1.log` |
| `build_anchor_remap.py` | `tools/build_anchor_remap.py` |
| `cmd_calibrate.py` | `tools/cmd_calibrate.py` |
| `cmd_probe.py` | `tools/cmd_probe.py` |
| `cpu_rehearsal_remap_6000.txt` | `evals/velstand/cpu_rehearsal_remap_6000.txt` |
| `cpu_rehearsal_remap_6750.txt` | `evals/velstand/cpu_rehearsal_remap_6750.txt` |
| `cpu_rehearsal_round3_3999.txt` | `evals/velstand/cpu_rehearsal_round3_3999.txt` |
| `drive_remap.py` | `tools/drive_remap.py` |
| `eval_left_ckpt.sh` | `tools/eval_left_ckpt.sh` |
| `export_ckpt.sh` | `tools/export_ckpt.sh` |
| `export_pstep_ckpt.sh` | `tools/export_pstep_ckpt.sh` |
| `headless_rehearsal.py` | `tools/headless_rehearsal.py` |
| `obs_compare.py` | `tools/obs_compare.py` |
| `precision_step_chunk1.log` | `training/precision_step/precision_step_chunk1.log` |
| `precision_step_chunk2.log` | `training/precision_step/precision_step_chunk2.log` |
| `precision_step_memprobe2048.log` | `training/precision_step/precision_step_memprobe2048.log` |
| `precision_step_memprobe_gpu.txt` | `training/precision_step/precision_step_memprobe_gpu.txt` |
| `precision_step_smoke.log` | `training/precision_step/precision_step_smoke.log` |
| `precision_step_smoke_export.log` | `training/precision_step/precision_step_smoke_export.log` |
| `precision_step_smoke_pose.log` | `training/precision_step/precision_step_smoke_pose.log` |
| `precision_step_v2_chunk1.log` | `training/precision_step/precision_step_v2_chunk1.log` |
| `precision_step_v2_smoke.log` | `training/precision_step/precision_step_v2_smoke.log` |
| `pstep_eval_2026-11-11` | `evals/precision_step/pstep_eval_2026-11-11` |
| `pstep_v2_eval_2026-11-13` | `evals/precision_step/pstep_v2_eval_2026-11-13` |
| `remap4000_response.txt` | `evals/velstand/remap4000_response.txt` |
| `remap4500_response.txt` | `evals/velstand/remap4500_response.txt` |
| `remap5500_response.txt` | `evals/velstand/remap5500_response.txt` |
| `remap6000_response.txt` | `evals/velstand/remap6000_response.txt` |
| `remap6500_response.txt` | `evals/velstand/remap6500_response.txt` |
| `remap6750_response.txt` | `evals/velstand/remap6750_response.txt` |
| `remap6999_response.txt` | `evals/velstand/remap6999_response.txt` |
| `remap_getup_eval.txt` | `evals/velstand/remap_getup_eval.txt` |
| `remap_verify.py` | `tools/remap_verify.py` |
| `round3_2000_response.txt` | `evals/velstand/round3_2000_response.txt` |
| `round3_3000_response.txt` | `evals/velstand/round3_3000_response.txt` |
| `round3_3500_response.txt` | `evals/velstand/round3_3500_response.txt` |
| `round3_3999_response.txt` | `evals/velstand/round3_3999_response.txt` |
| `round3_getup_eval.txt` | `evals/velstand/round3_getup_eval.txt` |
| `spincap_chunk1.log` | `training/standup/spincap_chunk1.log` |
| `spincap_chunk2.log` | `training/standup/spincap_chunk2.log` |
| `spincap_chunk2_eval.txt` | `evals/standup/spincap_chunk2_eval.txt` |
| `spincap_eval.txt` | `evals/standup/spincap_eval.txt` |
| `stance_aim_2026-11-11` | `evals/ballkick/stance_aim_2026-11-11` |
| `velstand3000_response.txt` | `evals/velstand/velstand3000_response.txt` |
| `velstand_chunk1.log` | `training/velstand/velstand_chunk1.log` |
| `velstand_remap_chunk1.log` | `training/velstand/velstand_remap_chunk1.log` |
| `velstand_round3_chunk1.log` | `training/velstand/velstand_round3_chunk1.log` |
| `verify_onnx.py` | `tools/verify_onnx.py` |
| `walkv2_3000_response.txt` | `evals/velocity_walk/walkv2_3000_response.txt` |
| `walkv2_3500_response.txt` | `evals/velocity_walk/walkv2_3500_response.txt` |
| `walkv2_3999_response.txt` | `evals/velocity_walk/walkv2_3999_response.txt` |
| `walkv2_6000_response.txt` | `evals/velocity_walk/walkv2_6000_response.txt` |
| `walkv2_7000_response.txt` | `evals/velocity_walk/walkv2_7000_response.txt` |
| `walkv2_7998_remap_table.txt` | `evals/velocity_walk/walkv2_7998_remap_table.txt` |
| `walkv2_7998_response.txt` | `evals/velocity_walk/walkv2_7998_response.txt` |
| `walkv2_chunk1.log` | `training/velocity_walk/walkv2_chunk1.log` |
| `walkv2_chunk2.log` | `training/velocity_walk/walkv2_chunk2.log` |
| `warp_cmd_response.py` | `tools/warp_cmd_response.py` |
| `warp_standup_eval.py` | `tools/warp_standup_eval.py` |
| `warp_standup_render.py` | `tools/warp_standup_render.py` |
| `warp_velstand_eval.py` | `tools/warp_velstand_eval.py` |
