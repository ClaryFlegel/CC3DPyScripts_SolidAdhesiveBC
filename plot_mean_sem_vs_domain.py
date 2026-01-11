from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

# -----------------------------
# Configuration
# -----------------------------
AVERAGES_DIR = Path("SolidRuns/Averages")
OUTPUT_FIG = AVERAGES_DIR / "mean_closure_vs_domain_by_R.png"

# -----------------------------
# Helper functions
# -----------------------------
def parse_domain_size(name):
    """
    Extract L from strings like:
    Lx306_Ly306  -> 306
    """
    match = re.search(r"Lx(\d+)_Ly(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse domain size from {name}")
    return int(match.group(1))


def read_average_file(avg_file):
    """
    Reads mean and SEM from an average file.
    """
    mean = None
    sem = None

    with open(avg_file, "r") as f:
        for line in f:
            if line.startswith("mean_closure_mcs"):
                mean = float(line.strip().split(",")[1])
            elif line.startswith("sem_closure_mcs"):
                sem = float(line.strip().split(",")[1])

    if mean is None or sem is None:
        raise ValueError(f"Missing mean or SEM in {avg_file}")

    return mean, sem


# -----------------------------
# Main data collection
# -----------------------------
data_by_R = {}   # R -> list of (L, mean, sem)

for domain_dir in sorted(AVERAGES_DIR.glob("Lx*_Ly*")):
    if not domain_dir.is_dir():
        continue

    L = parse_domain_size(domain_dir.name)
    # find wound-size subfolders (Rxx)
    wound_dirs = [
        f for f in domain_dir.iterdir()
        if f.is_dir() and re.match(r"R\d+", f.name)
    ]

    if not wound_dirs:
        print("  No wound size folders found, skipping domain")
        continue
    
    for wound_dir in sorted(wound_dirs):


        for avg_file in sorted(wound_dir.glob("*_R*_avg.txt")):

            match = re.search(r"R(\d+)", avg_file.name)
            if not match:
                continue

            R = int(match.group(1))

            mean, sem = read_average_file(avg_file)

            data_by_R.setdefault(R, []).append((L, mean, sem))


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6, 4))

for R, values in sorted(data_by_R.items()):

    # sort by domain size L
    values = sorted(values, key=lambda x: x[0])
    Ls, means, sems = map(np.array, zip(*values))

    plt.errorbar(
        Ls,
        means,
        yerr=sems,
        fmt='o-',
        capsize=5,
        label=f"R = {R}"
    )

plt.xlabel("Domain size L")
plt.ylabel("Mean wound closure time (MCS)")
plt.title("Wound closure time vs domain size\n(one curve per R)")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_FIG, dpi=300)
#plt.show()

print(f"Saved figure to {OUTPUT_FIG}")