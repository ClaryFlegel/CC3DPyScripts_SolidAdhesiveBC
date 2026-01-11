from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import re

# ============================================================
# User-defined parameters
# ============================================================

V_t = 100 #target volume
woundMakerTime = 10 #check parameters file
BIN_WIDTH = int(np.sqrt(V_t))  # = 10

RUNS_ROOT = Path("SolidRuns")
AVG_ROOT = RUNS_ROOT / "Averages"

# ============================================================
# Helper functions
# ============================================================

def parse_domain_and_radius(path):
    """
    Extract Lx, Ly and R from folder names:
    Runs/Lx*_Ly*/R*/
    """
    lxly_match = re.search(r"Lx(\d+)_Ly(\d+)", path.as_posix())
    r_match = re.search(r"/R(\d+)", path.as_posix())

    if lxly_match is None or r_match is None:
        return None

    Lx = int(lxly_match.group(1))
    Ly = int(lxly_match.group(2))
    R = int(r_match.group(1))

    return Lx, Ly, R


# ============================================================
# Main processing loop
# ============================================================

for lxly_dir in RUNS_ROOT.glob("Lx*_Ly*"):

    print(f"\nProcessing domain {lxly_dir.name}")

    for r_dir in lxly_dir.glob("R*"):

        print(f"  Processing {r_dir.name}")

        parsed = parse_domain_and_radius(r_dir)
        if parsed is None:
            continue

        Lx, Ly, R = parsed
        domain_name = lxly_dir.name
        wound_name = r_dir.name

        for data_file in r_dir.glob("cell_field_data_*.txt"):

            # ------------------------------------------------------------
            # Read file and validate rows
            # ------------------------------------------------------------

            rows = []
            invalid_file = False

            with data_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split(",")

                    # We require at least columns 0, 4, 5
                    if len(parts) <= 5:
                        invalid_file = True
                        break

                    try:
                        mcs = int(parts[0])
                        volume = float(parts[4])
                        radial_distance = float(parts[5])
                    except ValueError:
                        invalid_file = True
                        break

                    rows.append((mcs, volume, radial_distance))

            if invalid_file:
                print(
                    f"File {domain_name}_{wound_name} skipped because invalid values in columns."
                )
                continue

            if len(rows) == 0:
                print(
                    f"File {domain_name}_{wound_name} skipped because no valid data rows."
                )
                continue

            # ------------------------------------------------------------
            # Convert to arrays
            # ------------------------------------------------------------

            rows = np.array(rows) 
            mcs_values = rows[:, 0].astype(int)
            volumes = rows[:, 1]
            radial_distances = rows[:, 2]

            relative_strain = (volumes - V_t) / V_t

            unique_mcs = np.unique(mcs_values)
            max_radius = radial_distances.max()

            n_bins = int(np.floor(max_radius / BIN_WIDTH)) + 1
            bin_edges = np.arange(0, (n_bins + 1) * BIN_WIDTH, BIN_WIDTH)
            bin_centers = bin_edges[:-1] + BIN_WIDTH / 2

            # ------------------------------------------------------------
            # Build 2D matrix: rows = bins, cols = mcs
            # ------------------------------------------------------------

            strain_matrix = np.full((n_bins, len(unique_mcs)), np.nan)

            for j, mcs in enumerate(unique_mcs):
                mask_mcs = mcs_values == mcs

                r_mcs = radial_distances[mask_mcs]
                strain_mcs = relative_strain[mask_mcs]

                bin_indices = np.floor(r_mcs / BIN_WIDTH).astype(int)

                for b in range(n_bins):
                    mask_bin = bin_indices == b
                    if np.any(mask_bin):
                        strain_matrix[b, j] = np.mean(strain_mcs[mask_bin])

            # ------------------------------------------------------------
            # Create output directories
            # ------------------------------------------------------------

            out_dir = AVG_ROOT / domain_name / wound_name / "Bins"
            out_dir.mkdir(parents=True, exist_ok=True)

            # ------------------------------------------------------------
            # Save averages file (2D matrix)
            # ------------------------------------------------------------

            out_file = out_dir / data_file.name.replace(".txt", "_bin_average.txt")

            with out_file.open("w") as f:
                f.write("# Bin-averaged relative strain\n")
                f.write("# Rows: radial distance bins (increasing radius)\n")
                f.write("# Columns: mcs values (in increasing order)\n")
                f.write(f"# Bin width = {BIN_WIDTH} pixels\n")
                f.write(f"# Domain Size: Lx={Lx}, Ly={Ly}\n")
                f.write(f"# Wound Radius: R={R}\n")
                f.write(f"# woundMakerTime = {woundMakerTime}\n")
                f.write(f"# V_t = {V_t}\n")
                f.write("#\n")
                f.write("# mcs values:\n")
                f.write("# " + " ".join(map(str, unique_mcs)) + "\n")

                np.savetxt(f, strain_matrix, fmt="%.6e")
            
            #print(f"    Saved → {out_file.name}")

            # ------------------------------------------------------------
            # Plotting
            # ------------------------------------------------------------

            cmap = plt.cm.seismic.copy()
            cmap.set_bad(color="black")

            vmax = np.nanmax(np.abs(strain_matrix))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

            fig, ax = plt.subplots(figsize=(8, 6))

            im = ax.imshow(
                strain_matrix,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
                extent=[
                    unique_mcs.min(),
                    unique_mcs.max(),
                    bin_edges[0],
                    bin_edges[-1],
                ],
            )

            ax.axvline(
                woundMakerTime,
                color="red",
                linestyle="-",
                linewidth=1.5,
                label="woundMakerTime",
            )

            ax.set_xlabel("MCS")
            ax.set_ylabel("Radial distance (pixels)")
            ax.set_title(
                f"Relative strain\n{domain_name}, {wound_name}, {data_file.name}"
            )

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Mean relative strain")

            ax.legend(loc="upper right")

            plot_file = out_dir / data_file.name.replace(".txt", "_bin_average.png")

            fig.tight_layout()
            fig.savefig(plot_file, dpi=300)
            #print(f"    Saved plot → {plot_file.name}")
            plt.close(fig)

        print(f"    Saved all files and plots in → {domain_name}_{wound_name}_Bins")