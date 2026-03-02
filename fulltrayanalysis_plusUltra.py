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
    iv_matrix[row][col] = {
        "row": row,
        "col": col,
        "file": filename,
        "V": V,
        "I": I,
        "results": {method: Vbd},
        "timings": {method: time_s, "Preprocessing": time_s},
        "RAW_VBD": raw_vbd_or_nan
    }
- Generates a summary PDF with:
    * 5 heatmaps of Vbd (one per method)
    * 5 heatmaps of ΔVbd (mV) vs method mean, + outlier counts (|Δ| > 50 mV)
    * Bar chart of outlier counts per method
    * Sensitivity bar chart (std dev of Vbd vs fit-range parameters)
    * Timing bar chart (average compute time per method)
    * Our Vbd vs RAW_VBD scatter plots for each method
    * Histograms of Our Vbd - RAW_VBD (mV) for each method
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI windows
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter

# ------------------------- Config -------------------------

DATA_DIR = "test_data"
TRAY_NUMBER = "250821-1301"  # use ONLY this tray
NUM_COLS = 23  # columns: 0..22
NUM_ROWS = 20  # rows: 0..19

RESULTS_FILE = "IV_result.txt"   # in same folder as this script

METHOD_LABELS = [
    "Tangent",
    "Relative Derivative",
    "Inverse Relative Derivative",
    "Second Derivative",
    "Parabolic Fit",
]

DEVICE_PDF_NAME = f"IV_breakdown_plots_{TRAY_NUMBER}.pdf"
SUMMARY_PDF_NAME = f"IV_summary_{TRAY_NUMBER}.pdf"

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


def parse_iv_results(results_file):
    """
    Parse IV_result.txt and return a dict:
        key: (col, row)
        val: RAW_VBD (float)

    File format (from IV_HEADERS.txt):
    TRAYID+NOTE, SIPMID, AVERAGE_TEMPERATURE, TEMPERATURE_DEVIATION,
    RAW_VBD, VBD@25C, ...

    Example line:
    250821-1301-set11-full 250821-1301_14_16 23.246 0.014 38.2861 ...
                            ^^^^^^^^^^^^^^^^ SIPMID
    """
    mapping = {}

    if not os.path.exists(results_file):
        print(f"WARNING: {results_file} not found, RAW_VBD will be NaN.")
        return mapping

    with open(results_file) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Some files may have header / count lines; skip those that don't look right
    for ln in lines:
        parts = ln.split()
        if len(parts) < 6:
            continue
        # Heuristic: SIPMID has form 250821-1301_col_row
        sipmid = None
        raw_vbd = None
        for p in parts:
            if f"{TRAY_NUMBER}_" in p:
                sipmid = p
                break
        if sipmid is None:
            continue

        # RAW_VBD is 2 floats after SIPMID: (avgT, dT, RAW_VBD, ...)
        idx = parts.index(sipmid)
        if idx + 3 >= len(parts):
            continue
        try:
            raw_vbd = float(parts[idx + 3 - 1])  # avgT, dT, RAW_VBD: RAW is idx+3-1
            # Wait: from header: TRAYID SIPMID AVERAGE TEMPERATURE_DEVIATION RAW_VBD ...
            # So RAW_VBD is at idx+3
            raw_vbd = float(parts[idx + 3])
        except Exception:
            continue

        # Parse col,row from SIPMID suffix
        try:
            suffix = sipmid.split("_")[-2:]  # ['col','row']
            col = int(suffix[0])
            row = int(suffix[1])
        except Exception:
            continue

        mapping[(col, row)] = raw_vbd

    print(f"Parsed {len(mapping)} RAW_VBD entries from {results_file}")
    return mapping


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


# --------------------- Plotting helpers ---------------------


def plot_six_panel(extras, vbds, fig_title, pdf_obj):
    """
    Create a 2x3 panel figure for one device and save it into device PDF.
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

    sensitivities = {}
    for m in METHOD_LABELS:
        vals = np.array(method_to_values[m], dtype=float)
        sensitivities[m] = float(np.nanstd(vals))

    return sensitivities


# -------------- Build full matrix + per-device PDFs --------------


def build_iv_matrix_and_device_pdf(raw_vbd_map):
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
                    # print(f"Missing file, skipping: {filename}")
                    continue

                try:
                    V, I = getIVLists(filepath)
                    vbds, extras, timings = compute_vbds(V, I, smooth=True)
                    raw_vbd = raw_vbd_map.get((col, row), np.nan)

                    iv_matrix[row][col] = {
                        "row": row,
                        "col": col,
                        "file": filename,
                        "V": V,
                        "I": I,
                        "results": vbds,
                        "timings": timings,
                        "RAW_VBD": raw_vbd,
                    }

                    fig_title = f"{filename} (col={col}, row={row})"
                    plot_six_panel(extras, vbds, fig_title, device_pdf)
                    print(f"Processed {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    return iv_matrix


# ----------------- Summary: heatmaps & metrics -----------------


def make_vbd_heatmaps(iv_matrix, tray_number, pdf_obj):
    """
    Create 5 heatmaps (one per method) of Vbd across tray.
    iv_matrix[row][col], rows=0..19 (top), cols=0..22
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
            origin="upper",  # row 0 at top
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(f"{method} Breakdown Heatmap (Tray {tray_number})")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row (0 at top)")
        ax.set_xticks(range(0, n_cols, max(1, n_cols // 12)))
        ax.set_yticks(range(0, n_rows, max(1, n_rows // 10)))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Vbd (V)")
        fig.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)


def make_diff_from_method_mean_heatmaps(iv_matrix, pdf_obj, threshold_mV=50.0):
    """
    For each method, compute:
      - method-wide mean Vbd across all devices
      - ΔVbd (mV) per device = (Vbd - method_mean)*1e3
      - heatmap of ΔVbd(mV)
      - count of outliers |ΔVbd| > threshold_mV

    Also returns a dict method -> outlier_count.
    """
    n_rows = len(iv_matrix)
    n_cols = len(iv_matrix[0])

    outlier_counts = {}

    for method in METHOD_LABELS:
        data = np.full((n_rows, n_cols), np.nan)

        # collect all Vbd values for this method
        all_vals = []
        for row in range(n_rows):
            for col in range(n_cols):
                cell = iv_matrix[row][col]
                if cell is not None:
                    v = cell["results"][method]
                    data[row, col] = v
                    all_vals.append(v)

        all_vals = np.array(all_vals, dtype=float)
        if all_vals.size == 0:
            outlier_counts[method] = 0
            continue

        method_mean = np.nanmean(all_vals)
        diff_mV = (data - method_mean) * 1e3  # in mV

        # mask for outliers
        mask_out = np.abs(diff_mV) > threshold_mV
        outlier_counts[method] = int(np.nansum(mask_out))

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(
            diff_mV,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(
            f"{method}: ΔVbd from Method Mean (mV)\n"
            f"Outliers (|Δ| > {threshold_mV:.0f} mV): {outlier_counts[method]}"
        )
        ax.set_xlabel("Column")
        ax.set_ylabel("Row (0 at top)")
        ax.set_xticks(range(0, n_cols, max(1, n_cols // 12)))
        ax.set_yticks(range(0, n_rows, max(1, n_rows // 10)))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("ΔVbd (mV)")
        fig.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)

    # Bar chart of outlier counts per method
    methods = METHOD_LABELS
    counts = [outlier_counts[m] for m in methods]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylabel("Number of devices")
    ax.set_title(f"Count of Devices with |ΔVbd| > {threshold_mV:.0f} mV (vs method mean)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    pdf_obj.savefig(fig)
    plt.close(fig)

    return outlier_counts


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

    fig, ax = plt.subplots(figsize=(6, 4))
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

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, avg_times)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, rotation=20)
    ax.set_ylabel("Time per device (s)")
    ax.set_title("Average Compute Time per Method")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    pdf_obj.savefig(fig)
    plt.close(fig)


def make_raw_vbd_comparison_plots(iv_matrix, pdf_obj):
    """
    Make scatter plots and histograms comparing each method's Vbd
    to RAW_VBD from IV_result.txt.
    """
    # collect per-method arrays
    for method in METHOD_LABELS:
        our_vals = []
        raw_vals = []
        for row in range(len(iv_matrix)):
            for col in range(len(iv_matrix[0])):
                cell = iv_matrix[row][col]
                if cell is None:
                    continue
                raw = cell["RAW_VBD"]
                if np.isnan(raw):
                    continue
                vbd = cell["results"][method]
                our_vals.append(vbd)
                raw_vals.append(raw)

        if not our_vals:
            continue

        our_vals = np.array(our_vals, dtype=float)
        raw_vals = np.array(raw_vals, dtype=float)

        diff_mV = (our_vals - raw_vals) * 1e3
        mean_diff = float(np.nanmean(diff_mV))
        rms_diff = float(np.sqrt(np.nanmean(diff_mV**2)))

        # Scatter plot
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(raw_vals, our_vals, s=8, alpha=0.6)
        lims = [
            min(raw_vals.min(), our_vals.min()) - 0.01,
            max(raw_vals.max(), our_vals.max()) + 0.01,
        ]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("RAW_VBD (V)")
        ax.set_ylabel("Our Vbd (V)")
        ax.set_title(
            f"{method}: Our Vbd vs RAW_VBD\n"
            f"Mean Δ = {mean_diff:.1f} mV, RMS Δ = {rms_diff:.1f} mV"
        )
        fig.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)

        # Histogram of differences
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.hist(diff_mV, bins=40, alpha=0.7, edgecolor="black")
        ax.axvline(mean_diff, color="r", linestyle="--", label="Mean Δ")
        ax.set_xlabel("ΔVbd = Our Vbd - RAW_VBD (mV)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{method}: Our Vbd - RAW_VBD\n"
            f"Mean Δ = {mean_diff:.1f} mV, RMS Δ = {rms_diff:.1f} mV"
        )
        ax.legend()
        fig.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)


def make_summary_pdf(iv_matrix):
    """
    Build summary PDF with:
    - 5 breakdown heatmaps
    - 5 ΔVbd-from-method-mean heatmaps + outlier counts
    - Bar chart of outlier counts
    - Sensitivity plot
    - Timing plot
    - RAW_VBD comparison plots
    """
    print(f"Creating summary PDF: {SUMMARY_PDF_NAME}")
    with PdfPages(SUMMARY_PDF_NAME) as summary_pdf:
        make_vbd_heatmaps(iv_matrix, TRAY_NUMBER, summary_pdf)
        make_diff_from_method_mean_heatmaps(iv_matrix, summary_pdf, threshold_mV=50.0)
        make_sensitivity_plot(iv_matrix, summary_pdf)
        make_timing_plot(iv_matrix, summary_pdf)
        make_raw_vbd_comparison_plots(iv_matrix, summary_pdf)


# ------------------------------ main ------------------------------


def main():
    raw_vbd_map = parse_iv_results(RESULTS_FILE)
    iv_matrix = build_iv_matrix_and_device_pdf(raw_vbd_map)
    make_summary_pdf(iv_matrix)
    print("Done. Device PDF and summary PDF written.")


if __name__ == "__main__":
    main()
