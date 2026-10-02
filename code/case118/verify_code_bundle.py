from pathlib import Path
import compileall

root = Path(__file__).resolve().parent
required = [
    root / "stage3b_controller_evaluation/src/case118_stage3b.py",
    root / "stage4_pic_certificate/src/case118_stage4.py",
    root / "stage4c_activity_audit/scripts/final_h3_activity_refinement.py",
    root / "stage4d_curvature_prototype/src/cellwise_stage4d.py",
    root / "stage5a_control_metric/src/case118_stage5a3.py",
    root / "stage5b1_exact_pic_frontier/scripts/run_stage5b1.py",
    root / "stage5b2_sparse_reinforcement/src/case118_stage5b2.py",
]
missing = [str(p.relative_to(root)) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing required code files: " + ", ".join(missing))
if not compileall.compile_dir(str(root), quiet=1):
    raise SystemExit("Python compilation check failed")
print("Code bundle OK; required stage files exist and all Python sources compile.")
