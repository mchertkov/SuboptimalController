"""Run the complete numerical suite and regenerate the paper figures."""
from exp1_theory import main as static_main
from exp2_transient import main as transient_main
from figures import make_figures

if __name__ == "__main__":
    static_main("results")
    transient_main("results")
    make_figures("figures", "results/static_results.json", "results/transient_results.json")
    print("Done. Results are in results/ and figures/.")
