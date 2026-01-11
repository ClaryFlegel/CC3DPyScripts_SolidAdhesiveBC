import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import csv
import os 
import re 

MCS_MIN = 20   # ignore early transient

# ----------------------
# Define models
# ----------------------
def exp_decay(t, A0, tau, Ainf):
    return Ainf + (A0 - Ainf) * np.exp(-t / tau)

def logistic(t, A0, t0, k):
    return A0 / (1 + np.exp((t - t0) / k))

def power_law(t, A0, alpha):
    return A0 * t**(-alpha)

# --- NEW: stretched exponential ---
def stretched_exp(t, A0, tau, beta):
    return A0 * np.exp(-(t / tau)**beta)

# ----------------------
# Characteristic times
# ----------------------
def closing_time_exp(A0, tau, Ainf, threshold=0.05):
    return -tau * np.log(threshold * A0 / (A0 - Ainf))

def closing_time_logistic(A0, t0, k, threshold=0.05):
    return t0 + k * np.log((1/threshold - 1))

def closing_time_power(A0, alpha, Ainf, threshold=0.05):
    return (A0 / (threshold * A0 - Ainf))**(1/alpha)

# --- NEW: stretched exponential closing time ---
def closing_time_stretched(A0, tau, beta, threshold=0.05):
    return tau * (-np.log(threshold))**(1/beta)

# ----------------------
# Parent folder containing domain folders
# ----------------------
RUNS_ROOT = Path("SolidRuns")
AVG_ROOT = RUNS_ROOT / "Averages"

# ----------------------
# Create output folder
# ----------------------
output_folder = RUNS_ROOT / "Curve Fit"
output_folder.mkdir(exist_ok=True)

# ----------------------
# Store results
# ----------------------
results = []

for domain_dir in sorted(AVG_ROOT.glob("Lx*_Ly*")):

    match = re.search(r"Lx(\d+)_Ly(\d+)", domain_dir.name)
    if not match:
        continue

    Lx, Ly = map(int, match.groups())

    for r_dir in sorted(domain_dir.glob("R*")):

        avg_file = r_dir / "simulation_results_averages.txt"
        if not avg_file.exists():
            continue

        match_r = re.search(r"R(\d+)", r_dir.name)
        if not match_r:
            continue

        R = int(match_r.group(1))

        data = np.loadtxt(avg_file, delimiter=",", comments="#") 
        mcs = data[:, 0]
        wound_area = data[:, 1]
    
        #mcs_nonzero = mcs[wound_area > 0]
        #wound_nonzero = wound_area[wound_area > 0]

        fit_mask = (wound_area > 0) & (mcs >= MCS_MIN)

        mcs_fit = mcs[fit_mask]
        wound_fit = wound_area[fit_mask]

        # ----------------------
        # Fit exponential
        # ----------------------
        try:
            popt_exp, _ = curve_fit(
                exp_decay, mcs_fit, wound_fit,
                p0=[wound_area[0], 100, 0]
            )
            tau_exp = popt_exp[1]
            t_exp = closing_time_exp(*popt_exp)
        except:
            popt_exp = [np.nan]*3
            tau_exp = np.nan
            t_exp = np.nan

        # ----------------------
        # Fit logistic
        # ----------------------
        try:
            popt_log, _ = curve_fit(
                logistic, mcs_fit, wound_fit,
                p0=[wound_area[0], np.median(mcs), 10]
            )
            t0_log = popt_log[1]
            t_log = closing_time_logistic(*popt_log)
        except:
            popt_log = [np.nan]*3
            t0_log = np.nan
            t_log = np.nan

        # ----------------------
        # Fit power law
        # ----------------------
        try:
            popt_pow, _ = curve_fit(
                power_law, mcs_fit, wound_fit,
                p0=[wound_fit[0], 0.5]
            )
            A0_pow, alpha_pow = popt_pow
            t_pow = closing_time_power(A0_pow, alpha_pow, 0)
        except:
            popt_pow = [np.nan]*2
            alpha_pow = np.nan
            t_pow = np.nan

        # ----------------------
        # NEW: Fit stretched exponential
        # ----------------------
        try:
            popt_str, _ = curve_fit(
                stretched_exp, mcs_fit, wound_fit,
                p0=[wound_fit[0], 100, 0.7],
                bounds=([0, 0, 0], [np.inf, np.inf, 2])
            )
            A0_str, tau_str, beta_str = popt_str
            t_str = closing_time_stretched(A0_str, tau_str, beta_str)
        except:
            popt_str = [np.nan]*3
            tau_str = np.nan
            beta_str = np.nan
            t_str = np.nan

        results.append({
            "domain": f"{domain_dir.name}_{r_dir.name}",
            "tau_exp": tau_exp,
            "t_exp": t_exp,
            "t0_log": t0_log,
            "t_log": t_log,
            "alpha_pow": alpha_pow,
            "t_pow": t_pow,
            "tau_stretch": tau_str,
            "beta_stretch": beta_str,
            "t_stretch": t_str
        })

        # ----------------------
        # Plot fits
        # ----------------------
        plt.figure(figsize=(6,4))
        plt.scatter(mcs_fit, wound_fit, label="Data", color="black", s=1)
        #plt.plot(mcs_fit, exp_decay(mcs_fit, *popt_exp), label=f"Exp (τ={tau_exp:.1f})")
        #plt.plot(mcs_fit, logistic(mcs_fit, *popt_log), label=f"Logistic (t0={t0_log:.1f})")
        #plt.plot(mcs_fit, power_law(mcs_fit, *popt_pow), label=f"Power (α={alpha_pow:.2f})")
        plt.plot(mcs_fit, stretched_exp(mcs_fit, *popt_str),
                 label=f"Stretch (β={beta_str:.2f})")
        plt.xlabel("MCS")
        plt.ylabel("Wound Area")
        plt.title(f"Fits for {domain_dir.name}_{r_dir.name}")
        plt.legend()

        domain_out = output_folder / domain_dir.name
        domain_out.mkdir(exist_ok=True)

        fig_path = domain_out / f"{r_dir.name}_curve_fit.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()

# ----------------------
# Save results CSV
# ----------------------
csv_path = output_folder / "wound_closing_times.csv"

with open(csv_path, "w", newline="") as csvfile:
    fieldnames = [
        "domain",
        "tau_exp","t_exp",
        "t0_log","t_log",
        "alpha_pow","t_pow",
        "tau_stretch","beta_stretch","t_stretch"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Done! Figures and CSV saved in {output_folder}")