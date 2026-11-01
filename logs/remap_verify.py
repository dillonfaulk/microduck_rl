import sys
sys.path.insert(0, "logs"); import drive_remap  # noqa: F401  (patches PolicyInference)
sys.argv = [sys.argv[0], sys.argv[1], "--suite", "walk", "--delay", "1", "--train-solver",
            "--scene", "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"]
src = open("logs/headless_rehearsal.py").read().split('print(f"onnx={args.onnx}')[0]
exec(compile(src, "headless_rehearsal.py", "exec"))
for c in [(0.3, 0, 0), (0.15, 0, 0), (-0.3, 0, 0), (-0.15, 0, 0), (0, 0.2, 0), (0, -0.2, 0),
          (0, 0, 1.0), (0, 0, -1.0), (0, 0, 0.3), (0.3, 0, 0.5), (0, 0, 0)]:
    run_walk(f"target {c}", c, 10.0)
