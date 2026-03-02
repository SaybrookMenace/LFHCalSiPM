#!/usr/bin/env python3
"""
Full-tray SiPM IV analysis for tray 250821-1301 (NO PARABOLIC FIT).

What this script does:

1. Reads IV text files named:
       test_data/IV_250821-1301_COL_ROW.txt
   with COL = 0..22 and ROW = 0..19.

2. For each device:
   - Parses IV curve (V, I)
   - Computes breakdown voltage using 4 methods:
       * Tangent
       * Relative Derivative
       * Inverse Relative Derivative
       * Second Derivative
   - Stores results + timing in iv_matrix[row][col]
   - Writes a 5-panel figure (IV + 4 methods, last panel blank)
     into a multi-page PDF:
       IV_breakdown_plots_250821-1301_noParabolic.pdf

3. Reads IV_result.txt in the SAME directory as this script
   and extracts RAW_VBD for SIPMIDs of the form:
       250821-1301_COL_ROW
   Adds this RAW_VBD into iv_matrix[row][col]["raw_vbd"].

4. Builds a summary PDF:
       IV_summary_250821-1301_noParabolic.pdf
   containing:
       - 4 breakdown heatmaps (one per method)
       - 1 RAW_VBD heatmap
       - 4 heatmaps of (method - RAW_VBD)
         with outlier counts (> ±50 mV) in titles
       - 1 bar chart: outlier counts per method (> ±50 mV vs RAW_VBD)
       - 1 bar chart: sensitivity to fit-range parameters
       - 1 bar chart: timing per method
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter

# ------------------------- Config -------------------------

DATA_DIR = "test_data"
TRAY_NUMBER = "250821-1301"
NUM_COLS = 23  # columns 0..22
NUM_ROWS = 20  # rows 0..19

METHOD_LABELS = [
    "Tangent",
    "Relative Derivative",
    "Inverse Relative Derivative",
    "Second Derivative",
]

DEVICE_PDF_NAME = f"IV_breakdown_plots_{TRAY_NUMBER}_noParabolic.pdf"
SUMMARY_PDF_NAME = f"IV_summary_{TRAY_NUMBER}_noParabolic.pdf"

OUTLIER_THRESHOLD = 0.05  # 50 mV


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
    Assumes each data line is like:
        V  ...  I
    with the current in the 3rd column.
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
):
    """
    Compute breakdown voltages by 4 methods on one IV curve.

    Methods:
        - Tangent
        - Relative Derivative
        - Inverse Relative Derivative
        - Second Derivative

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

    # smooth ln(I) for derivatives
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
        "M": M,
    }

    return vbds, extras, timings


def analyze_breakdown(V, I, smooth=True):
    """Thin wrapper returning only the breakdown voltages."""
    vbds, _, _ = compute_vbds(V, I, smooth=smooth)
    return vbds


# --------------------- Plotting helpers ---------------------


def plot_device_panels(extras, vbds, fig_title, pdf_obj):
    """
    Create a 3x2 panel figure (5 panels used, last one blank):

        0) IV curve (semi-log)
        1) Tangent method
        2) Relative derivative
        3) Inverse relative derivative
        4) Second derivative
        5) (unused)

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
    vbd = vbds["Tangent"]
    axs[1].plot(V, lnI, "k", label="ln(I)")
    axs[1].plot(V, baseline(V), "r--", label="Baseline")
    axs[1].plot(V, tangent(V), "g--", label="Tangent")
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

    # Panel 5: unused
    axs[5].axis("off")

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
        {"N_factor": 1 / 8, "M_factor": 0.25},
        {"N_factor": 1 / 6, "M_factor": 1 / 3},
        {"N_factor": 1 / 5, "M_factor": 0.5},
    ]

    method_to_values = {m: [] for m in METHOD_LABELS}

    for cfg in configs:
        vbds, _, _ = compute_vbds(
            V,
            I,
            smooth=smooth,
            N_factor=cfg["N_factor"],
            M_factor=cfg["M_factor"],
        )
        for m in METHOD_LABELS:
            method_to_values[m].append(vbds[m])

    sensitivities = {}
    for m in METHOD_LABELS:
        vals = np.array(method_to_values[m], dtype=float)
        sensitivities[m] = float(np.nanstd(vals))

    return sensitivities


# ----------- Read RAW_VBD from IV_result.txt -----------------


def parse_raw_vbd_from_result(tray_number, filename="IV_result.txt"):
    """
    Parse RAW_VBD values from IV_result.txt.

    Each data line looks like:
        TRAYNOTE SIPMID AVG_TEMP TEMP_DEV RAW_VBD VBD25 ...

    where SIPMID is like: 250821-1301_COL_ROW

    Returns:
        raw_map[(row, col)] = RAW_VBD
    """
    raw_map = {}

    if not os.path.exists(filename):
        print(f"WARNING: {filename} not found; RAW_VBD will be missing.")
        return raw_map

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # skip lines that don't look like data lines
            if not parts[0].startswith(tray_number):
                continue
            if len(parts) < 6:
                continue

            tray_note = parts[0]
            sipmid = parts[1]

            if tray_number not in sipmid:
                continue

            # sipmid format: "250821-1301_COL_ROW"
            try:
                _, col_str, row_str = sipmid.split("_")
                col = int(col_str)
                row = int(row_str)
            except Exception:
                continue

            try:
                raw_vbd = float(parts[4])  # RAW_VBD column
            except ValueError:
                continue

            raw_map[(row, col)] = raw_vbd

    return raw_map


def add_raw_vbd_to_iv_matrix(iv_matrix, tray_number):
    """
    Read RAW_VBD from IV_result.txt and attach to iv_matrix[row][col]["raw_vbd"].
    """
    raw_map = parse_raw_vbd_from_result(tray_number, filename="IV_result.txt")
    if not raw_map:
        print("No RAW_VBD entries found.")
        return

    for row in range(len(iv_matrix)):
        for col in range(len(iv_matrix[0])):
            cell = iv_matrix[row][col]
            if cell is None:
                continue
            key = (row, col)
            if key in raw_map:
                cell["raw_vbd"] = raw_map[key]
            else:
                cell["raw_vbd"] = np.nan


# -------------- Build full matrix + per-device PDFs --------------


def build_iv_matrix_and_device_pdf():
    """
    Loop over all COL,ROW combinations, analyze curves, store results
    into iv_matrix[row][col], and generate per-device 5-panel plots
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
                        "raw_vbd": np.nan,  # filled later
                    }

                    fig_title = f"{filename} (col={col}, row={row})"
                    plot_device_panels(extras, vbds, fig_title, device_pdf)
                    print(f"Processed {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    return iv_matrix


# ----------------- Utility: gather arrays -----------------


def gather_method_and_raw_arrays(iv_matrix):
    """
    Collect Vbd arrays for methods and RAW_VBD.

    Returns:
        method_vals: array shape (N_dev, N_methods)
        raw_vals:    array shape (N_dev,)
    Only devices with non-NaN raw_vbd are included.
    """
    rows = len(iv_matrix)
    cols = len(iv_matrix[0])

    method_list = []
    raw_list = []

    for r in range(rows):
        for c in range(cols):
            cell = iv_matrix[r][c]
            if cell is None:
                continue
            raw_vbd = cell.get("raw_vbd", np.nan)
            if np.isnan(raw_vbd):
                continue
            vals = [cell["results"][m] for m in METHOD_LABELS]
            if any(np.isnan(vals)):
                continue
            method_list.append(vals)
            raw_list.append(raw_vbd)

    if not method_list:
        return None, None

    method_vals = np.array(method_list, dtype=float)
    raw_vals = np.array(raw_list, dtype=float)
    return method_vals, raw_vals


# ----------------- Summary: heatmaps & metrics -----------------


def make_breakdown_heatmaps(iv_matrix, tray_number, pdf_obj):
    """
    Create 4 heatmaps (one per method) of Vbd across tray, plus
    one heatmap of RAW_VBD.
    """
    n_rows = len(iv_matrix)
    n_cols = len(iv_matrix[0])

    # Heatmaps for the four methods
    for method in METHOD_LABELS:
        data = np.full((n_rows, n_cols), np.nan)
        for row in range(n_rows):
            for col in range(n_cols):
                cell = iv_matrix[row][col]
                if cell is not None:
                    val = cell["results"][method]
                    data[row, col] = val

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(
            data,
            origin="upper",        # row 0 at top
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(f"{method} Breakdown Heatmap (Tray {tray_number})")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row (0 at top)")
        ax.set_xticks(range(0, n_cols, max(1, n_cols // 5)))
        ax.set_yticks(range(0, n_rows, max(1, n_rows // 5)))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Vbd (V)")
        fig.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)

    # RAW_VBD heatmap
    data = np.full((n_rows, n_cols), np.nan)
    for row in range(n_rows):
        for col in range(n_cols):
            cell = iv_matrix[row][col]
            if cell is not None:
                data[row, col] = cell.get("raw_vbd", np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        data,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title(f"RAW_VBD Heatmap (Tray {tray_number})")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row (0 at top)")
    ax.set_xticks(range(0, n_cols, max(1, n_cols // 5)))
    ax.set_yticks(range(0, n_rows, max(1, n_rows // 5)))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RAW_VBD (V)")
    fig.tight_layout()
    pdf_obj.savefig(fig)
    plt.close(fig)


def make_diff_heatmaps_vs_raw(iv_matrix, tray_number, pdf_obj):
    """
    For each method, create a heatmap of (method - RAW_VBD).
    Title also shows outlier count (> ±50 mV).
    """
    n_rows = len(iv_matrix)
    n_cols = len(iv_matrix[0])

    for method in METHOD_LABELS:
        data = np.full((n_rows, n_cols), np.nan)
        for row in range(n_rows):
            for col in range(n_cols):
                cell = iv_matrix[row][col]
                if cell is None:
                    continue
                raw_vbd = cell.get("raw_vbd", np.nan)
                if np.isnan(raw_vbd):
                    continue
                val = cell["results"][method] - raw_vbd
                data[row, col] = val

        # Outlier count
        valid = ~np.isnan(data)
        outliers = np.sum(np.abs(data[valid]) > OUTLIER_THRESHOLD)

        # symmetric color scale around 0
        vmax = np.nanmax(np.abs(data))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = OUTLIER_THRESHOLD
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(
            data,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(
            f"{method}: (Method - RAW_VBD) Heatmap\n"
            f"Outliers (>±{OUTLIER_THRESHOLD:.2f} V): {outliers}"
        )
        ax.set_xlabel("Column")
        ax.set_ylabel("Row (0 at top)")
        ax.set_xticks(range(0, n_cols, max(1, n_cols // 5)))
        ax.set_yticks(range(0, n_rows, max(1, n_rows // 5)))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Vbd(method) - RAW_VBD (V)")
        fig.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)


def make_outlier_hist_vs_raw(iv_matrix, pdf_obj):
    """
    For each method, count how many devices have |method - RAW_VBD| > 50 mV
    and make a bar chart.
    """
    method_vals, raw_vals = gather_method_and_raw_arrays(iv_matrix)
    if method_vals is None:
        return

    diffs = method_vals - raw_vals[:, None]  # shape (N_dev, N_methods)
    outlier_counts = np.sum(np.abs(diffs) > OUTLIER_THRESHOLD, axis=0)

    x = np.arange(len(METHOD_LABELS))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, outlier_counts)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, rotation=20)
    ax.set_ylabel("Number of devices")
    ax.set_title(
        f"Outliers per Method (|Vbd_method - RAW_VBD| > {OUTLIER_THRESHOLD:.2f} V)"
    )
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
    ax.set_ylabel("Std dev of Vbd (V)")
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
        - 4 breakdown heatmaps
        - RAW_VBD heatmap
        - 4 (method - RAW_VBD) heatmaps with outlier counts
        - Outlier bar chart (> ±50 mV vs RAW_VBD)
        - Sensitivity bar chart
        - Timing bar chart
    """
    print(f"Creating summary PDF: {SUMMARY_PDF_NAME}")
    with PdfPages(SUMMARY_PDF_NAME) as summary_pdf:
        make_breakdown_heatmaps(iv_matrix, TRAY_NUMBER, summary_pdf)
        make_diff_heatmaps_vs_raw(iv_matrix, TRAY_NUMBER, summary_pdf)
        make_outlier_hist_vs_raw(iv_matrix, summary_pdf)
        make_sensitivity_plot(iv_matrix, summary_pdf)
        make_timing_plot(iv_matrix, summary_pdf)


# ------------------------------ main ------------------------------


def main():
    # 1) Build iv_matrix and per-device plots
    iv_matrix = build_iv_matrix_and_device_pdf()

    # 2) Attach RAW_VBD from IV_result.txt
    add_raw_vbd_to_iv_matrix(iv_matrix, TRAY_NUMBER)

    # 3) Build summary PDF
    make_summary_pdf(iv_matrix)

    print("Done. Device and summary PDFs written (no parabolic fit).")


if __name__ == "__main__":
    main()
