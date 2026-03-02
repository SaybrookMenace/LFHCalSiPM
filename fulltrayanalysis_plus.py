#!/usr/bin/env python3
"""
Massive SiPM IV analysis script for tray 250821-1301.

What it does:
- Reads IV_250821-1301_COL_ROW.txt from test_data/
- For each device:
    * parses IV curve
    * computes breakdown voltage using 5 methods
    * creates a 6-panel figure (IV + 5 panels) and stores in a multi-page PDF
- Builds a 2D data structure iv_matrix[row][col]
- Generates a summary PDF with:
    * 5 breakdown heatmaps (one per method)
    * 5 difference-from-mean heatmaps (+ outlier count > ±50 mV)
    * 1 histogram: within / outside ±50 mV of mean Vbd for each method
    * 1 sensitivity bar chart
    * 1 timing bar chart
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI / no figure windows
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter

# ------------------------- Config -------------------------

DATA_DIR = "test_data"
TRAY_NUMBER = "250821-1301"   # use ONLY this tray
NUM_COLS = 23                 # columns: 0..22
NUM_ROWS = 20                 # rows: 0..19

METHOD_LABELS = [
    "Tangent",
    "Relative Derivative",
    "Inverse Relative Derivative",
    "Second Derivative",
    "Parabolic Fit",
]

DEVICE_PDF_NAME = f"IV_breakdown_plots_{TRAY_NUMBER}.pdf"
SUMMARY_PDF_NAME = f"IV_summary_{TRAY_NUMBER}.pdf"

DELTA_VBD_CUT = 0.05   # 50 mV

# -------------------- File reading helpers --------------------


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def getIVLists(filepath):
    """
    Read a text IV file and return [V_list, I_list].
    Expects each data line like: V ... I  (current in 3rd column).
    """
    raw_list = []
    with open(filepath) as f:
        lines = [line.strip() for line in f]
        for line in lines:
            raw_list.append(line.split())

    x_list = []
    y_list = []
    for line in raw_list:
        if is_number(line[0]):
            x_list.append(float(line[0]))   # voltage
            y_list.append(float(line[2]))   # current (3rd column)
    return [x_list, y_list]


# ----------------------- Core analysis -----------------------


def compute_vbds(
    V,
    I,
    smooth=True,
    N_factor=1 / 6,
    M_factor=1 / 3,
    win=10,
):
    """
    Compute breakdown voltages by 5 methods on one IV curve.

    Returns:
        vbds: dict of method -> Vbd
        extras: dict with arrays and fits for plotting
        timings: dict of method -> time (seconds), plus 'Preprocessing'
    """
    timings = {}

    t0 = time.perf_counter()
    V = np.array(V, dtype=float)
    I = np.array(I, dtype=float)

    # avoid log(0)
    lnI = np.log(I + 1e-30)

    if smooth and len(lnI) >= 11:
        lnI_smooth = savgol_filter(lnI, 11, 3)
    else:
        lnI_smooth = lnI.copy()

    dlnI_dV = np.gradient(lnI_smooth, V)
    d2lnI_dV2 = np.gradient(dlnI_dV, V)
    dI_dV = np.gradient(I, V)

    t1 = time.perf_counter()
    timings["Preprocessing"] = t1 - t0

    vbds = {}

    # 1) Tangent Method
    t_start = time.perf_counter()
    N = max(10, int(len(V) * N_factor))
    baseline = np.poly1d(np.polyfit(V[:N], lnI_smooth[:N], 1))

    idx_max_slope = int(np.argmax(dlnI_dV))
    slope = dlnI_dV[idx_max_slope]
    tangent = np.poly1d([slope, lnI_smooth[idx_max_slope] - slope * V[idx_max_slope]])

    roots_tan = np.roots(tangent - baseline)
    VT_tangent = np.real(roots_tan[0]) if roots_tan.size > 0 else np.nan
    vbds["Tangent"] = float(VT_tangent)
    timings["Tangent"] = time.perf_counter() - t_start

    # 2) Relative derivative max
    t_start = time.perf_counter()
    VT_rel = float(V[int(np.argmax(dlnI_dV))])
    vbds["Relative Derivative"] = VT_rel
    timings["Relative Derivative"] = time.perf_counter() - t_start

    # 3) Inverse relative derivative
    t_start = time.perf_counter()
    inv_rel = I / (dI_dV + 1e-30)
    M = max(5, int(len(V) * M_factor))
    inv_fit = np.poly1d(np.polyfit(V[-M:], inv_rel[-M:], 1))
    roots_inv = np.roots(inv_fit)
    VT_inv = np.real(roots_inv[0]) if roots_inv.size > 0 else np.nan
    vbds["Inverse Relative Derivative"] = float(VT_inv)
    timings["Inverse Relative Derivative"] = time.perf_counter() - t_start

    # 4) Second derivative max
    t_start = time.perf_counter()
    VT_second = float(V[int(np.argmax(d2lnI_dV2))])
    vbds["Second Derivative"] = VT_second
    timings["Second Derivative"] = time.perf_counter() - t_start

    # 5) Parabolic fit near max slope
    t_start = time.perf_counter()
    win = max(5, win)
    start = max(0, idx_max_slope - win)
    end = min(len(V), idx_max_slope + win)
    if end - start >= 3:
        parabola = np.poly1d(np.polyfit(V[start:end], lnI_smooth[start:end], 2))
        roots_para = np.roots(parabola - baseline)
        VT_para = np.real(roots_para[0]) if roots_para.size > 0 else np.nan
    else:
        parabola = np.poly1d([0, 0, 0])
        VT_para = np.nan
    vbds["Parabolic Fit"] = float(VT_para)
    timings["Parabolic Fit"] = time.perf_counter() - t_start

    extras = {
        "V": V,
        "I": I,
        "lnI": lnI_smooth,
        "dlnI_dV": dlnI_dV,
        "d2lnI_dV2": d2lnI_dV2,
        "dI_dV": dI_dV,
        "inv_rel": inv_rel,
        "baseline": baseline,
        "tangent": tangent,
        "parabola": parabola,
        "M": M,
        "idx_max_slope": idx_max_slope,
    }

    return vbds, extras, timings


def analyze_breakdown(V, I, smooth=True):
    """
    Thin wrapper returning only the breakdown voltages.
    (Kept for API compatibility.)
    """
    vbds, _, _ = compute_vbds(V, I, smooth=smooth)
    return vbds


# --------------------- Plotting helpers ---------------------


def plot_six_panel(extras, vbds, fig_title, pdf_obj):
    """
    Create a 2x3 panel figure:
    0) IV curve (semi-log)
    1) Tangent method
    2) Relative derivative
    3) Inverse relative derivative
    4) Second derivative
    5) Parabolic fit

    Save one page to the given PdfPages object.
    """
    V = extras["V"]
    I = extras["I"]
    lnI = extras["lnI"]
    dlnI_dV = extras["dlnI_dV"]
    d2lnI_dV2 = extras["d2lnI_dV2"]
    inv_rel = extras["inv_rel"]
    baseline = extras["baseline"]
    tangent = extras["tangent"]
    parabola = extras["parabola"]
    M = extras["M"]

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs = axs.ravel()

    # Panel 0: Base IV curve
    axs[0].semilogy(V, I, "k", label="I(V)")
    axs[0].set_title("IV Curve")
    axs[0].set_xlabel("Bias (V)")
    axs[0].set_ylabel("I (A)")
    axs[0].grid(True)
    axs[0].legend()

    # Panel 1: Tangent method
    axs[1].plot(V, lnI, "k", label="ln(I)")
    axs[1].plot(V, baseline(V), "r--", label="Baseline")
    axs[1].plot(V, tangent(V), "g--", label="Tangent")
    vbd = vbds["Tangent"]
    axs[1].axvline(vbd, linestyle=":", color="blue", label=f"Vbd = {vbd:.3f} V")
    axs[1].set_title("Tangent Method")
    axs[1].set_xlabel("Bias (V)")
    axs[1].set_ylabel("ln(I)")
    axs[1].grid(True)
    axs[1].legend()

    # Panel 2: Relative derivative
    vbd = vbds["Relative Derivative"]
    axs[2].plot(V, dlnI_dV, "k", label="d(ln I)/dV")
    axs[2].axvline(vbd, linestyle=":", color="blue", label=f"Vbd = {vbd:.3f} V")
    axs[2].set_title("Relative Derivative")
    axs[2].set_xlabel("Bias (V)")
    axs[2].set_ylabel("d(ln I)/dV")
    axs[2].grid(True)
    axs[2].legend()

    # Panel 3: Inverse relative derivative
    vbd = vbds["Inverse Relative Derivative"]
    axs[3].plot(V, inv_rel, "k", label="I / I'")
    inv_fit = np.poly1d(np.polyfit(V[-M:], inv_rel[-M:], 1))
    axs[3].plot(V[-M:], inv_fit(V[-M:]), "r--", label="Linear Fit")
    axs[3].axvline(vbd, linestyle=":", color="blue", label=f"Vbd = {vbd:.3f} V")
    axs[3].set_title("Inverse Relative Derivative")
    axs[3].set_xlabel("Bias (V)")
    axs[3].set_ylabel("I / I'")
    axs[3].grid(True)
    axs[3].legend()

    # Panel 4: Second derivative
    vbd = vbds["Second Derivative"]
    axs[4].plot(V, d2lnI_dV2, "k", label="d²(ln I)/dV²")
    axs[4].axvline(vbd, linestyle=":", color="blue", label=f"Vbd = {vbd:.3f} V")
    axs[4].set_title("Second Derivative")
    axs[4].set_xlabel("Bias (V)")
    axs[4].set_ylabel("d²(ln I)/dV²")
    axs[4].grid(True)
    axs[4].legend()

    # Panel 5: Parabolic fit
    vbd = vbds["Parabolic Fit"]
    axs[5].plot(V, lnI, "k", label="ln(I)")
    axs[5].plot(V, baseline(V), "r--", label="Baseline")
    axs[5].plot(V, parabola(V), "b--", label="Parabolic Fit")
    axs[5].axvline(vbd, linestyle=":", color="blue", label=f"Vbd = {vbd:.3f} V")
    axs[5].set_title("Parabolic Fit")
    axs[5].set_xlabel("Bias (V)")
    axs[5].set_ylabel("ln(I)")
    axs[5].grid(True)
    axs[5].legend()

    fig.suptitle(fig_title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf_obj.savefig(fig)
    plt.close(fig)


# -------------- Sensitivity study for one curve --------------


def sensitivity_study_single_curve(V, I, smooth=True):
    """
    Simple sensitivity measure: run compute_vbds with three different
    fit-range parameter sets and compute the std dev of Vbd for each method.
    """
    configs = [
        {"N_factor": 1 / 8, "M_factor": 0.25, "win": 8},
        {"N_factor": 1 / 6, "M_factor": 1 / 3, "win": 10},
        {"N_factor": 1 / 5, "M_factor": 0.5, "win": 12},
    ]

    method_to_values = {m: [] for m in METHOD_LABELS}

    for cfg in configs:
        vbds, _, _ = compute_vbds(
            V,
            I,
            smooth=smooth,
            N_factor=cfg["N_factor"],
            M_factor=cfg["M_factor"],
            win=cfg["win"],
        )
        for m in METHOD_LABELS:
            method_to_values[m].append(vbds[m])

    # std dev across parameter sets
    sensitivities = {}
    for m in METHOD_LABELS:
        vals = np.array(method_to_values[m], dtype=float)
        sensitivities[m] = float(np.nanstd(vals))

    return sensitivities


# -------------- Build full matrix + per-device PDFs --------------


def build_iv_matrix_and_device_pdf():
    """
    Loop over all COL,ROW combinations, analyze curves, store results
    into iv_matrix[row][col], and generate per-device 6-panel plots
    into a multi-page PDF.
    """
    iv_matrix = [[None for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]

    print(f"Creating device PDF: {DEVICE_PDF_NAME}")
    with PdfPages(DEVICE_PDF_NAME) as device_pdf:
        for col in range(NUM_COLS):
            for row in range(NUM_ROWS):
                filename = f"IV_{TRAY_NUMBER}_{col}_{row}.txt"
                filepath = os.path.join(DATA_DIR, filename)

                if not os.path.exists(filepath):
                    # Missing devices are just left as None
                    continue

                try:
                    V, I = getIVLists(filepath)
                    vbds, extras, timings = compute_vbds(V, I, smooth=True)

                    iv_matrix[row][col] = {
                        "row": row,
                        "col": col,
                        "file": filename,
                        "V": V,
                        "I": I,
                        "results": vbds,
                        "timings": timings,
                    }

                    fig_title = f"{filename} (col={col}, row={row})"
                    plot_six_panel(extras, vbds, fig_title, device_pdf)
                    print(f"Processed {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    return iv_matrix


# ----------------- Summary: heatmaps & metrics -----------------


def make_breakdown_heatmaps(iv_matrix, tray_number, pdf_obj):
    """
    Create 5 breakdown heatmaps (one per method) of Vbd across tray.
    iv_matrix[row][col], rows=0..19, cols=0..22
    """
    n_rows = len(iv_matrix)
    n_cols = len(iv_matrix[0])

    for method in METHOD_LABELS:
        data = np.full((n_rows, n_cols), np.nan)
        for row in range(n_rows):
            for col in range(n_cols):
                cell = iv_matrix[row][col]
                if cell is not None:
                    data[row, col] = cell["results"][method]

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(
            data,
            origin="upper",   # row 0 at top
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(f"{method} Breakdown Heatmap")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row (0 at top)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Vbd (V)")
        fig.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)


def make_diff_heatmaps_and_counts(iv_matrix, tray_number, pdf_obj):
    """
    For each method:
      - compute global mean Vbd for that method
      - plot heatmap of Vbd - mean(Vbd)
      - tally # of devices with |ΔVbd| > DELTA_VBD_CUT
    """
    n_rows = len(iv_matrix)
    n_cols = len(iv_matrix[0])

    for method in METHOD_LABELS:
        data = np.full((n_rows, n_cols), np.nan)
        for row in range(n_rows):
            for col in range(n_cols):
                cell = iv_matrix[row][col]
                if cell is not None:
                    data[row, col] = cell["results"][method]

        # Flatten, ignoring NaNs, to get mean per method
        flat_vals = data[~np.isnan(data)]
        if flat_vals.size == 0:
            continue
        mean_vbd = np.mean(flat_vals)
        diff = data - mean_vbd

        # outliers beyond ±50 mV
        outlier_mask = np.abs(diff) > DELTA_VBD_CUT
        outlier_count = np.sum(outlier_mask & ~np.isnan(diff))

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(
            diff,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
        )
        ax.set_title(
            f"{method}: Difference-from-Mean Heatmap\n"
            f"Outliers (>±{int(DELTA_VBD_CUT*1000)} mV): {outlier_count}"
        )
        ax.set_xlabel("Column")
        ax.set_ylabel("Row (0 at top)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("ΔVbd (V)")
        fig.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)


def make_range_histogram(iv_matrix, pdf_obj):
    """
    For each method, count how many SiPMs are within ±50 mV of that
    method's *global mean* Vbd, and how many are outside.
    Plot a grouped bar chart: Within vs Outside for each method.
    """
    within_counts = []
    outside_counts = []

    for method in METHOD_LABELS:
        vals = []
        for row in range(len(iv_matrix)):
            for col in range(len(iv_matrix[0])):
                cell = iv_matrix[row][col]
                if cell is not None:
                    vals.append(cell["results"][method])
        vals = np.array(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            within_counts.append(0)
            outside_counts.append(0)
            continue

        mean_vbd = np.mean(vals)
        diff = vals - mean_vbd
        within = np.sum(np.abs(diff) <= DELTA_VBD_CUT)
        outside = np.sum(np.abs(diff) > DELTA_VBD_CUT)

        within_counts.append(int(within))
        outside_counts.append(int(outside))

    x = np.arange(len(METHOD_LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, within_counts, width, label=f"|ΔVbd| ≤ {int(DELTA_VBD_CUT*1000)} mV")
    ax.bar(x + width / 2, outside_counts, width, label=f"|ΔVbd| > {int(DELTA_VBD_CUT*1000)} mV")

    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, rotation=20)
    ax.set_ylabel("Number of SiPMs")
    ax.set_title("SiPM Counts Within / Outside ±50 mV of Method-Specific Mean Vbd")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    pdf_obj.savefig(fig)
    plt.close(fig)


def make_sensitivity_plot(iv_matrix, pdf_obj):
    """
    For each device, run a small sensitivity study and aggregate
    average std dev in Vbd vs fit-range parameters for each method.
    """
    sensitivities_per_method = {m: [] for m in METHOD_LABELS}

    for row in range(len(iv_matrix)):
        for col in range(len(iv_matrix[0])):
            cell = iv_matrix[row][col]
            if cell is None:
                continue
            V = cell["V"]
            I = cell["I"]
            sens = sensitivity_study_single_curve(V, I, smooth=True)
            for m in METHOD_LABELS:
                sensitivities_per_method[m].append(sens[m])

    avg_sensitivity = []
    for m in METHOD_LABELS:
        vals = np.array(sensitivities_per_method[m], dtype=float)
        avg_sensitivity.append(np.nanmean(vals))

    x = np.arange(len(METHOD_LABELS))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, avg_sensitivity)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, rotation=20)
    ax.set_ylabel("Std Dev of Vbd (V)")
    ax.set_title("Average Sensitivity of Vbd to Fit-Range Parameters")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    pdf_obj.savefig(fig)
    plt.close(fig)


def make_timing_plot(iv_matrix, pdf_obj):
    """
    Aggregate average compute time per method across all devices.
    """
    times_per_method = {m: [] for m in METHOD_LABELS}

    for row in range(len(iv_matrix)):
        for col in range(len(iv_matrix[0])):
            cell = iv_matrix[row][col]
            if cell is None:
                continue
            timings = cell["timings"]
            for m in METHOD_LABELS:
                if m in timings:
                    times_per_method[m].append(timings[m])

    avg_times = []
    for m in METHOD_LABELS:
        vals = np.array(times_per_method[m], dtype=float)
        avg_times.append(np.nanmean(vals))

    x = np.arange(len(METHOD_LABELS))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, avg_times)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, rotation=20)
    ax.set_ylabel("Time per device (s)")
    ax.set_title("Average Compute Time per Method")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    pdf_obj.savefig(fig)
    plt.close(fig)


def make_summary_pdf(iv_matrix):
    """
    Build summary PDF with:
    - 5 breakdown heatmaps
    - 5 difference-from-mean heatmaps (+ outlier counts)
    - histogram of within/outside ±50 mV
    - sensitivity plot
    - timing plot
    """
    print(f"Creating summary PDF: {SUMMARY_PDF_NAME}")
    with PdfPages(SUMMARY_PDF_NAME) as summary_pdf:
        make_breakdown_heatmaps(iv_matrix, TRAY_NUMBER, summary_pdf)
        make_diff_heatmaps_and_counts(iv_matrix, TRAY_NUMBER, summary_pdf)
        make_range_histogram(iv_matrix, summary_pdf)
        make_sensitivity_plot(iv_matrix, summary_pdf)
        make_timing_plot(iv_matrix, summary_pdf)


# ------------------------------ main ------------------------------


def main():
    iv_matrix = build_iv_matrix_and_device_pdf()
    make_summary_pdf(iv_matrix)
    print("Done. Device PDF and summary PDF written.")


if __name__ == "__main__":
    main()
