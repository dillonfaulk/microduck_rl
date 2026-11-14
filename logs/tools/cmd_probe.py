"""Probe achieved velocity for a list of commands on the CPU BAM harness (projected gravity,
delay 1, training solver, all-collisions scene). Run: python logs/tools/cmd_probe.py <onnx>"""
import sys
sys.argv = [sys.argv[0], sys.argv[1], "--suite", "walk", "--delay", "1", "--train-solver",
            "--scene", "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"]
src = open("logs/tools/headless_rehearsal.py").read().split('print(f"onnx={args.onnx}')[0]
exec(compile(src, "headless_rehearsal.py", "exec"))
CMDS = [(-0.1, 0, 0), (-0.2, 0, 0), (-0.3, 0, 0), (-0.4, 0, 0), (0.1, 0, 0), (0.2, 0, 0),
        (0, -0.1, 0), (0, -0.2, 0), (0, 0.2, 0), (-0.3, -0.2, 0)]
for c in CMDS:
    run_walk(f"cmd {c}", c, 8.0)
