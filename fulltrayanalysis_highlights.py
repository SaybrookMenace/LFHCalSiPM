#!/usr/bin/env python3
"""
Full SiPM IV analysis script for tray 250821-1301.

What it does
------------
- Reads IV_250821-1301_COL_ROW.txt from test_data/ (COL=0..22, ROW=0..19)
- Parses IV_result.txt (in same folder) to get RAW_VBD for each SIPMID
- For each device:
    * loads IV data
    * computes breakdown voltages using 5 methods:
        1. Tangent
        2. Relative Derivative
        3. Inverse Relative Derivative
        4. Second Derivative
        5. Parabolic Fit (positive-curvature, fitted to I(V) on the
           *right* side of the knee; baseline also in I(V))
    * stores all results in iv_matrix[row][col]
    * appends a 6-panel figure into a multi-page PDF

- Builds summary PDF with:
    * 5 method heatmaps + RAW_VBD heatmap
    * Difference-from-mean heatmaps (per method, ±50 mV band highlighted)
    * Per-method bar chart of outlier counts (> ±50 mV)
    * For RAW_VBD vs each method:
        - Scatter plot (RAW vs method, y=x line)
        - Histogram of (method - RAW) residuals
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
TRAY_NUMBER = "250821-1301"
IV_RESULT_FILE = "IV_result.txt"

NUM_COLS = 23  # 0..22
NUM_ROWS = 20  # 0..19

METHOD_LABELS = [
    "Tangent",
    "Relative Derivative",
    "Inverse Relative Derivative",
    "Second Derivative",
    "Parabolic Fit",
]

DEVICE_PDF_NAME = f"IV_breakdown_plots_{TRAY_NUMBER}_parab_fixed.pdf"
SUMMARY_PDF_NAME = f"IV_summary_{TRAY_NUMBER}_parab_fixed.pdf"

OUTLIER_THRESHOLD = 0.05  # 50 mV

# -------------------- Basic helpers --------------------


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def getIVLists(filepath):
    """
    Read a text IV file and return [V_list, I_list].
    Expects lines like: V ... I  (I in 3rd column).
    """
    raw_list = []
    with open(filepath) as f:
        lines = [line.strip() for line in f]
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            raw_list.append(parts)

    x_list = []
    y_list = []
    for line in raw_list:
        if is_number(line[0]):
            x_list.append(float(line[0]))   # voltage
            y_list.append(float(line[2]))   # current (3rd column)
    return [x_list, y_list]


# ------------------ RAW_VBD parsing ------------------


def parse_raw_vbd_matrix():
    """
    Parse IV_result.txt and build a NUM_ROWS x NUM_COLS matrix
    of RAW_VBD values. Indexing uses SIPMID suffix: TRAY_COL_ROW.
    """
    raw_vbd = np.full((NUM_ROWS, NUM_COLS), np.nan)

    if not os.path.exists(IV_RESULT_FILE):
        print(f"WARNING: {IV_RESULT_FILE} not found; RAW_VBD will be NaN.")
        return raw_vbd

    with open(IV_RESULT_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            sipm_id = parts[1]
            # Expect something like 250821-1301_14_16
            if not sipm_id.startswith(TRAY_NUMBER + "_"):
                continue

            try:
                _, col_str, row_str = sipm_id.split("_")
                col = int(col_str)
                row = int(row_str)
            except Exception:
                continue

            if not (0 <= col < NUM_COLS and 0 <= row < NUM_ROWS):
                continue

            try:
                raw_val = float(parts[4])  # RAW_VBD column
            except ValueError:
                continue

            raw_vbd[row, col] = raw_val

    return raw_vbd


# ----------------------- Core analysis -----------------------


def compute_vbds(
    V,
    I,
    smooth=True,
    N_factor=1 / 6,
    M_factor=1 / 3,
    tangent_win=10,
    parab_win_max=20,
):
    """
    Compute breakdown voltages by 5 methods on one IV curve.
    Parabolic Fit:
        - Baseline: linear fit to I(V) in low-current region
        - Parabola: fit to I(V) on the *right* side of the knee,
          restricted to points with positive curvature (d²I/dV² > 0)
        - Intersection solved in I(V), then reported as Vbd

    Returns:
        vbds   : dict method -> Vbd
        extras : dict with arrays and fits for plotting
        timings: dict method -> time in seconds (plus 'Preprocessing')
    """
    timings = {}

    t0 = time.perf_counter()
    V = np.array(V, dtype=float)
    I = np.array(I, dtype=float)

    # Avoid log(0)
    lnI = np.log(I + 1e-30)

    if smooth and len(lnI) >= 11:
        lnI_smooth = savgol_filter(lnI, 11, 3)
    else:
        lnI_smooth = lnI.copy()

    dlnI_dV = np.gradient(lnI_smooth, V)
    d2lnI_dV2 = np.gradient(dlnI_dV, V)
    dI_dV = np.gradient(I, V)
    d2I_dV2 = np.gradient(dI_dV, V)

    t1 = time.perf_counter()
    timings["Preprocessing"] = t1 - t0

    vbds = {}

    # --- Common baseline in ln(I) for tangent-related methods ---
    N = max(10, int(len(V) * N_factor))
    baseline_ln = np.poly1d(np.polyfit(V[:N], lnI_smooth[:N], 1))

    # --- 1) Tangent method ---
    t_start = time.perf_counter()
    idx_max_slope = int(np.argmax(dlnI_dV))
    slope = dlnI_dV[idx_max_slope]
    tangent_ln = np.poly1d(
        [slope, lnI_smooth[idx_max_slope] - slope * V[idx_max_slope]]
    )
    roots_tan = np.roots(tangent_ln - baseline_ln)
    VT_tangent = np.real(roots_tan[0]) if roots_tan.size > 0 else np.nan
    vbds["Tangent"] = float(VT_tangent)
    timings["Tangent"] = time.perf_counter() - t_start

    # --- 2) Relative derivative max ---
    t_start = time.perf_counter()
    VT_rel = float(V[int(np.argmax(dlnI_dV))])
    vbds["Relative Derivative"] = VT_rel
    timings["Relative Derivative"] = time.perf_counter() - t_start

    # --- 3) Inverse relative derivative ---
    t_start = time.perf_counter()
    inv_rel = I / (dI_dV + 1e-30)
    M = max(5, int(len(V) * M_factor))
    inv_fit = np.poly1d(np.polyfit(V[-M:], inv_rel[-M:], 1))
    roots_inv = np.roots(inv_fit)
    VT_inv = np.real(roots_inv[0]) if roots_inv.size > 0 else np.nan
    vbds["Inverse Relative Derivative"] = float(VT_inv)
    timings["Inverse Relative Derivative"] = time.perf_counter() - t_start

    # --- 4) Second derivative max ---
    t_start = time.perf_counter()
    VT_second = float(V[int(np.argmax(d2lnI_dV2))])
    vbds["Second Derivative"] = VT_second
    timings["Second Derivative"] = time.perf_counter() - t_start

    # --- 5) Parabolic Fit with positive curvature on I(V) ---
    t_start = time.perf_counter()

    # Baseline in I(V)
    baseline_I = np.poly1d(np.polyfit(V[:N], I[:N], 1))

    # Knee index from ln(I) curvature
    idx_knee = int(np.argmax(d2lnI_dV2))

    # Candidate region: right side of knee, up to parab_win_max points
    start = idx_knee
    end = min(len(V), idx_knee + parab_win_max)
    cand_idx = np.arange(start, end)

    # Restrict to positive curvature in I(V)
    cand_idx = cand_idx[d2I_dV2[cand_idx] > 0]

    # If too few points, fallback to all positive-curvature points
    if len(cand_idx) < 5:
        cand_idx = np.where(d2I_dV2 > 0)[0]
        if len(cand_idx) >= 5:
            # take last section (high-voltage side)
            cand_idx = cand_idx[-min(len(cand_idx), parab_win_max):]

    if len(cand_idx) >= 5:
        parabola_I = np.poly1d(np.polyfit(V[cand_idx], I[cand_idx], 2))
        # Intersection: parabola_I(V) = baseline_I(V)
        diff_poly = parabola_I - baseline_I
        roots_para = diff_poly.r
        real_roots = roots_para[np.isreal(roots_para)].real
        # Choose root within V-range and nearest to knee
        mask_range = (real_roots >= V[0]) & (real_roots <= V[-1])
        candidates = real_roots[mask_range]
        if candidates.size > 0:
            VT_para = candidates[np.argmin(np.abs(candidates - V[idx_knee]))]
        else:
            VT_para = np.nan
    else:
        parabola_I = None
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
        "baseline_ln": baseline_ln,
        "tangent_ln": tangent_ln,
        "inv_fit": inv_fit,
        "idx_max_slope": idx_max_slope,
        "baseline_I": baseline_I,
        "parabola_I": parabola_I,
        "parab_idx": cand_idx,
    }

    return vbds, extras, timings


def analyze_breakdown(V, I, smooth=True):
    vbds, _, _ = compute_vbds(V, I, smooth=smooth)
    return vbds


# --------------------- Plotting helpers ---------------------


def plot_six_panel(extras, vbds, raw_vbd, fig_title, pdf_obj):
    """
    2x3 panel figure:
    0) IV curve (semi-log)
    1) Tangent method
    2) Relative derivative
    3) Inverse relative derivative
    4) Second derivative
    5) Parabolic fit (positive-curvature, in I(V) but shown as ln(I))
    """
    V = extras["V"]
    I = extras["I"]
    lnI = extras["lnI"]
    dlnI_dV = extras["dlnI_dV"]
    d2lnI_dV2 = extras["d2lnI_dV2"]
    inv_rel = extras["inv_rel"]
    baseline_ln = extras["baseline_ln"]
    tangent_ln = extras["tangent_ln"]
    inv_fit = extras["inv_fit"]
    baseline_I = extras["baseline_I"]
    parabola_I = extras["parabola_I"]

    fig, axs = plt.subplots(3, 2, figsize=(13, 10))
    axs = axs.ravel()

    # Panel 0: IV curve
    axs[0].semilogy(V, I, "k", label="I(V)")
    if not np.isnan(raw_vbd):
        axs[0].axvline(raw_vbd, color="magenta", linestyle="--",
                       label=f"RAW Vbd = {raw_vbd:.3f} V")
    axs[0].set_title("IV Curve")
    axs[0].set_xlabel("Bias (V)")
    axs[0].set_ylabel("I (A)")
    axs[0].grid(True)
    axs[0].legend(fontsize=8)

    # Panel 1: Tangent
    vbd = vbds["Tangent"]
    axs[1].plot(V, lnI, "k", label="ln(I)")
    axs[1].plot(V, baseline_ln(V), "r--", label="Baseline (lnI)")
    axs[1].plot(V, tangent_ln(V), "g--", label="Tangent (lnI)")
    axs[1].axvline(vbd, linestyle=":", color="blue", label=f"Vbd = {vbd:.3f} V")
    axs[1].set_title("Tangent Method")
    axs[1].set_xlabel("Bias (V)")
    axs[1].set_ylabel("ln(I)")
    axs[1].grid(True)
    axs[1].legend(fontsize=8)

    # Panel 2: Relative derivative
    vbd = vbds["Relative Derivative"]
    axs[2].plot(V, dlnI_dV, "k", label="d(ln I)/dV")
    axs[2].axvline(vbd, linestyle=":", color="blue", label=f"Vbd = {vbd:.3f} V")
    axs[2].set_title("Relative Derivative")
    axs[2].set_xlabel("Bias (V)")
    axs[2].set_ylabel("d(ln I)/dV")
    axs[2].grid(True)
    axs[2].legend(fontsize=8)

    # Panel 3: Inverse relative derivative
    vbd = vbds["Inverse Relative Derivative"]
    M = extras["inv_rel"].shape[0]
    last_M = max(5, M // 3)
    axs[3].plot(V, inv_rel, "k", label="I / I'")
    axs[3].plot(V[-last_M:], inv_fit(V[-last_M:]), "r--", label="Linear Fit")
    axs[3].axvline(vbd, linestyle=":", color="blue", label=f"Vbd = {vbd:.3f} V")
    axs[3].set_title("Inverse Relative Derivative")
    axs[3].set_xlabel("Bias (V)")
    axs[3].set_ylabel("I / I'")
    axs[3].grid(True)
    axs[3].legend(fontsize=8)

    # Panel 4: Second derivative
    vbd = vbds["Second Derivative"]
    axs[4].plot(V, d2lnI_dV2, "k", label="d²(ln I)/dV²")
    axs[4].axvline(vbd, linestyle=":", color="blue", label=f"Vbd = {vbd:.3f} V")
    axs[4].set_title("Second Derivative")
    axs[4].set_xlabel("Bias (V)")
    axs[4].set_ylabel("d²(ln I)/dV²")
    axs[4].grid(True)
    axs[4].legend(fontsize=8)

    # Panel 5: Parabolic fit (curvature-based, fitted to I(V))
    vbd = vbds["Parabolic Fit"]
    axs[5].plot(V, lnI, "k", label="ln(I)")
    # Baseline and parabola converted from I(V) to ln(I) for plotting
    if baseline_I is not None:
        I_base = baseline_I(V)
        lnI_base = np.log(np.clip(I_base, 1e-30, None))
        axs[5].plot(V, lnI_base, "r--", label="Baseline (I)")
    if parabola_I is not None:
        I_par = parabola_I(V)
        lnI_par = np.log(np.clip(I_par, 1e-30, None))
        axs[5].plot(V, lnI_par, "b--", label="Parabolic Fit (I)")
    if not np.isnan(vbd):
        axs[5].axvline(vbd, linestyle=":", color="blue",
                       label=f"Vbd = {vbd:.3f} V")
    axs[5].set_title("Parabolic Fit (positive curvature)")
    axs[5].set_xlabel("Bias (V)")
    axs[5].set_ylabel("ln(I)")
    axs[5].grid(True)
    axs[5].legend(fontsize=8)

    fig.suptitle(fig_title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf_obj.savefig(fig)
    plt.close(fig)


# -------------- Build matrix & per-device PDF --------------


def build_iv_matrix_and_device_pdf(raw_vbd_matrix):
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
                    # silent skip if that position doesn't exist
                    continue

                try:
                    V, I = getIVLists(filepath)
                    vbds, extras, timings = compute_vbds(V, I, smooth=True)
                    raw_vbd = raw_vbd_matrix[row, col]

                    iv_matrix[row][col] = {
                        "row": row,
                        "col": col,
                        "file": filename,
                        "V": V,
                        "I": I,
                        "results": vbds,
                        "timings": timings,
                        "raw_vbd": raw_vbd,
                    }

                    fig_title = (f"{filename}  (col={col}, row={row})\n"
                                 f"RAW Vbd = {raw_vbd:.3f} V"
                                 if not np.isnan(raw_vbd)
                                 else f"{filename}  (col={col}, row={row})")
                    plot_six_panel(extras, vbds, raw_vbd, fig_title, device_pdf)
                    print(f"Processed {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    return iv_matrix


# ----------------- Summary helpers -----------------


def make_method_heatmaps(iv_matrix, tray_number, pdf_obj):
    """
    Create heatmaps of Vbd across tray for each method.
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
        ax.set_title(f"{method} Heatmap (Tray {tray_number})")
        ax.set_xlabel("Column (0–22)")
        ax.set_ylabel("Row (0–19, top to bottom)")
        ax.set_xticks(range(0, n_cols, max(1, n_cols // 12)))
        ax.set_yticks(range(0, n_rows, max(1, n_rows // 10)))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Vbd (V)")
        fig.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)


def make_raw_heatmap(iv_matrix, tray_number, pdf_obj):
    """
    Heatmap of RAW_VBD values across tray.
    """
    n_rows = len(iv_matrix)
    n_cols = len(iv_matrix[0])
    data = np.full((n_rows, n_cols), np.nan)

    for row in range(n_rows):
        for col in range(n_cols):
            cell = iv_matrix[row][col]
            if cell is not None:
                data[row, col] = cell["raw_vbd"]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        data,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title(f"RAW_VBD Heatmap (Tray {tray_number})")
    ax.set_xlabel("Column (0–22)")
    ax.set_ylabel("Row (0–19, top to bottom)")
    ax.set_xticks(range(0, n_cols, max(1, n_cols // 12)))
    ax.set_yticks(range(0, n_rows, max(1, n_rows // 10)))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RAW Vbd (V)")
    fig.tight_layout()
    pdf_obj.savefig(fig)
    plt.close(fig)


def gather_method_matrix(iv_matrix, method):
    """
    Return 2D array of Vbd for a given method.
    """
    n_rows = len(iv_matrix)
    n_cols = len(iv_matrix[0])
    data = np.full((n_rows, n_cols), np.nan)
    for row in range(n_rows):
        for col in range(n_cols):
            cell = iv_matrix[row][col]
            if cell is not None:
                data[row, col] = cell["results"][method]
    return data


def make_diff_from_mean_heatmaps(iv_matrix, pdf_obj):
    """
    For each method, compute global mean Vbd, build heatmap of
    (Vbd - mean), highlight outliers (>±50 mV), and count them.
    """
    n_rows = len(iv_matrix)
    n_cols = len(iv_matrix[0])

    for method in METHOD_LABELS:
        data = gather_method_matrix(iv_matrix, method)
        mean_vbd = np.nanmean(data)
        diff = data - mean_vbd

        # Outliers: abs(diff) > threshold
        outlier_mask = np.abs(diff) > OUTLIER_THRESHOLD
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
            f"Mean Vbd = {mean_vbd:.3f} V, "
            f"Outliers (>±{OUTLIER_THRESHOLD*1000:.0f} mV): {int(outlier_count)}"
        )
        ax.set_xlabel("Column (0–22)")
        ax.set_ylabel("Row (0–19, top to bottom)")
        ax.set_xticks(range(0, n_cols, max(1, n_cols // 12)))
        ax.set_yticks(range(0, n_rows, max(1, n_rows // 10)))
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Vbd - mean (V)")

        # Overlay red dots on outlier cells
        ys, xs = np.where(outlier_mask)
        ax.scatter(xs, ys, facecolors="none", edgecolors="black", linewidths=0.8)

        fig.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)


def make_outlier_count_bar(iv_matrix, pdf_obj):
    """
    Bar chart of number of outliers (>±50 mV) per method.
    """
    counts = []
    for method in METHOD_LABELS:
        data = gather_method_matrix(iv_matrix, method)
        mean_vbd = np.nanmean(data)
        diff = data - mean_vbd
        outlier_mask = np.abs(diff) > OUTLIER_THRESHOLD
        outlier_count = np.sum(outlier_mask & ~np.isnan(diff))
        counts.append(outlier_count)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(METHOD_LABELS))
    ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, rotation=20)
    ax.set_ylabel("Number of outliers (>±50 mV)")
    ax.set_title("Outlier Count per Method (relative to per-method mean)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    pdf_obj.savefig(fig)
    plt.close(fig)


def gather_flat_raw_and_methods(iv_matrix):
    """
    Flatten iv_matrix to 1D arrays:
        raw_vals, method_vals[method]
    Only entries where RAW_VBD and method Vbd are both finite are kept.
    """
    raw_vals = []
    method_vals = {m: [] for m in METHOD_LABELS}

    for row in range(len(iv_matrix)):
        for col in range(len(iv_matrix[0])):
            cell = iv_matrix[row][col]
            if cell is None:
                continue
            raw = cell["raw_vbd"]
            if not np.isfinite(raw):
                continue
            for m in METHOD_LABELS:
                vbd = cell["results"][m]
                if np.isfinite(vbd):
                    raw_vals.append(raw)
                    method_vals[m].append(vbd)
    raw_vals = np.array(raw_vals)
    for m in METHOD_LABELS:
        method_vals[m] = np.array(method_vals[m])
    return raw_vals, method_vals


def make_raw_comparison_plots(iv_matrix, pdf_obj):
    """
    For each method, make a page with:
        - Scatter (RAW vs method, y=x line)
        - Histogram of residuals (method - RAW)
    """
    raw_vals, method_vals = gather_flat_raw_and_methods(iv_matrix)
    if raw_vals.size == 0:
        return

    for method in METHOD_LABELS:
        vals = method_vals[method]
        if vals.size == 0:
            continue
        # Align lengths (they should already match, but just in case)
        n = min(raw_vals.size, vals.size)
        r = raw_vals[:n]
        v = vals[:n]
        residuals = v - r

        fig, axes = plt.subplots(2, 1, figsize=(7, 8))

        # Scatter
        ax = axes[0]
        ax.scatter(r, v, s=10, alpha=0.7)
        mn = min(r.min(), v.min())
        mx = max(r.max(), v.max())
        ax.plot([mn, mx], [mn, mx], "r--", label="y = x")
        ax.set_xlabel("RAW Vbd (V)")
        ax.set_ylabel(f"{method} Vbd (V)")
        ax.set_title(f"{method} vs RAW Vbd")
        ax.grid(True)
        ax.legend()

        # Histogram of residuals
        ax = axes[1]
        ax.hist(residuals, bins=30, alpha=0.8)
        mu = np.mean(residuals)
        sigma = np.std(residuals)
        ax.axvline(0.0, color="k", linestyle="--", label="0")
        ax.axvline(mu, color="r", linestyle="--", label=f"mean = {mu:.3f} V")
        ax.set_xlabel(f"{method} - RAW (V)")
        ax.set_ylabel("Count")
        ax.set_title(f"Residuals ({method} - RAW): mean={mu:.3f} V, σ={sigma:.3f} V")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        pdf_obj.savefig(fig)
        plt.close(fig)


def make_summary_pdf(iv_matrix):
    """
    Build summary PDF with:
      - 5 method heatmaps
      - RAW_VBD heatmap
      - Difference-from-mean heatmaps + outlier highlighting
      - Outlier count bar chart
      - RAW vs method scatter + residual histograms
    """
    print(f"Creating summary PDF: {SUMMARY_PDF_NAME}")
    with PdfPages(SUMMARY_PDF_NAME) as summary_pdf:
        make_method_heatmaps(iv_matrix, TRAY_NUMBER, summary_pdf)
        make_raw_heatmap(iv_matrix, TRAY_NUMBER, summary_pdf)
        make_diff_from_mean_heatmaps(iv_matrix, summary_pdf)
        make_outlier_count_bar(iv_matrix, summary_pdf)
        make_raw_comparison_plots(iv_matrix, summary_pdf)


# ------------------------------ main ------------------------------


def main():
    raw_vbd_matrix = parse_raw_vbd_matrix()
    iv_matrix = build_iv_matrix_and_device_pdf(raw_vbd_matrix)
    make_summary_pdf(iv_matrix)
    print("Done. Device PDF and summary PDF written.")


if __name__ == "__main__":
    main()
