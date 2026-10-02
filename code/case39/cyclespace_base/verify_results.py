"""Compare a completed run against the values printed in the manuscript."""
from pathlib import Path
import json
import numpy as np
from paper_results import STATIC, TRANSIENT_MW


def main(results_dir="results"):
    d=Path(results_dir)
    s=json.loads((d/"static_results.json").read_text())
    t=json.loads((d/"transient_results.json").read_text())
    print("STATIC")
    for case in ("case118","case39"):
        print(case)
        for q in ("dcb","cohesive","continued"):
            got=float(s[case][q]); exp=float(STATIC[case][q])
            print(f"  {q:10s}: got {got:.6f}, paper {exp:.6f}, diff {got-exp:+.3e}")
    got=np.asarray(t["rows_MW"])
    exp=np.column_stack([TRANSIENT_MW[k] for k in ["certified","simulated","static_exact","static_dcb"]])
    print("\nTRANSIENT [MW] -- computed minus paper")
    print(np.round(got-exp,3))
    print("\nAt the 1 MW bisection resolution, differences of order 1 MW are expected.")

if __name__=="__main__":
    main()
