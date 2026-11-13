#!/bin/bash
# usage: logs/eval_left_ckpt.sh <iteration>: export + battery (50) + prestand battery (50) + missions 2x30 (precision, uncalibrated left)
set -e; cd ~/microduck_rl
it=$1; D=logs/rsl_rl/ball_kick_left/2026-11-12_18-13-14_ts0p25_left_chunk1; E=logs/ballkick_left_eval_2026-11-12; mkdir -p $E
V=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
P=logs/rsl_rl/precision_step/2026-11-11_16-15-45_chunk2_pose/2026-11-11_16-15-45_chunk2_pose_model_3999.onnx
K=$D/$(basename $D)_model_$it.onnx
[ -f $K ] || MICRODUCK_KICK_FOOT=left MICRODUCK_BALL_TARGET_SPEED=0.25 bash logs/export_ckpt.sh $D $it >/dev/null
export CUDA_VISIBLE_DEVICES=""
for s in 0 1; do env -u PYTHONPATH uv run python scripts/kick_mission.py $V $K --foot left --precision $P --runs 30 --seed $s > $E/m${it}_mission_s$s.txt 2>&1 & done
env -u PYTHONPATH uv run python scripts/kick_battery.py $K --foot left --target 0.25 --episodes 50 --out $E/m${it}_battery.csv 2>&1 | tail -10 > $E/m${it}_battery.txt
env -u PYTHONPATH uv run python scripts/kick_battery.py $K --foot left --target 0.25 --episodes 50 --prestand $V --handoff $V --out $E/m${it}_prestand_handoff.csv 2>&1 | tail -10 > $E/m${it}_prestand_handoff.txt
wait
echo "=== model_$it"; grep -E "peak ball|signed|GOOD" $E/m${it}_battery.txt; echo "-- prestand+handoff"; grep -E "peak ball|signed|GOOD" $E/m${it}_prestand_handoff.txt
echo "-- missions 2x30"; env -u PYTHONPATH uv run python scripts/mission_stats.py $E/m${it}_mission_s0.txt $E/m${it}_mission_s1.txt | head -7
