#!/usr/bin/env python3
"""
Build a matrix of adjusted Vbd values (VBD@25C) from IV_results.txt.
Make heatmaps and compute outlier statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =====================
# CONFIG
# =====================

RESULT_FILE = "250717-1303-results/IV_result.txt"      # Already in the same folder
TRAY_NUMBER = "250821-1301"
NUM_COLS = 23   # 0–22
NUM_ROWS = 20   # 0–19
OUTLIER_THRESHOLD = 0.050   # 50 mV

HEATMAP_PDF = f"AdjustedVBD_Heatmaps_{TRAY_NUMBER}.pdf"

# =====================
# PARSE RESULTS
# =====================

def parse_results(filename):
    """
    Read IV_results.txt and extract:
    SIPMID → (col,row)
    adjusted_Vbd (VBD@25C)
    """
    data = {}

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("TRAYID"):
                continue

            parts = line.split()
            # Format from your file:
            # TRAY NOTE | SIPMID | AvgTemp | TempDev | RAW_VBD | VBD@25C | ...

            tray_note = parts[0]
            sipmid = parts[1]
            raw_vbd = float(parts[4])
            adj_vbd = float(parts[5])      # <<< ADJUSTED VBD@25C (your request)

            # Extract col,row from SIPMID like: 250821-1301_0_2 → col=0,row=2
            try:
                suffix = sipmid.split("_")
                col = int(suffix[-2])
                row = int(suffix[-1])
            except:
                print(f"Warning: cannot parse SIPMID {sipmid}")
                continue

            data[(row, col)] = {
                "sipmid": sipmid,
                "raw_vbd": raw_vbd,
                "adj_vbd": adj_vbd,
            }

    return data


# =====================
# BUILD MATRIX
# =====================

def build_matrix(parsed):
    """
    Build (row,col) matrix of adjusted Vbd values.
    """
    mat = np.full((NUM_ROWS, NUM_COLS), np.nan)

    for (row, col), info in parsed.items():
        if 0 <= row < NUM_ROWS and 0 <= col < NUM_COLS:
            mat[row, col] = info["adj_vbd"]

    return mat


# =====================
# HEATMAP PLOTTER
# =====================

def plot_heatmap(data, title, pdf, cmap="viridis", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(data, origin="lower", aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, label="Vbd (V)")

    pdf.savefig(fig)
    plt.close(fig)


# =====================
# OUTLIER ANALYSIS
# =====================

def compute_outlier_stats(mat):
    """
    Compute mean Vbd, differences, and count outliers.
    """
    mean_vbd = np.nanmean(mat)
    diff = mat - mean_vbd
    outlier_mask = np.abs(diff) > OUTLIER_THRESHOLD
    num_outliers = np.nansum(outlier_mask)

    return mean_vbd, diff, outlier_mask, num_outliers


# =====================
# MAIN
# =====================

def main():
    parsed = parse_results(RESULT_FILE)
    mat = build_matrix(parsed)

    mean_vbd, diff, outlier_mask, num_outliers = compute_outlier_stats(mat)

    print("====================================")
    print(f"Tray {TRAY_NUMBER} Adjusted Vbd Summary")
    print("====================================")
    print(f"Mean Adjusted Vbd: {mean_vbd:.4f} V")
    print(f"Outlier threshold: ±{OUTLIER_THRESHOLD:.3f} V")
    print(f"Number of outliers: {num_outliers}")
    print(f"Total SiPMs found: {np.sum(~np.isnan(mat))}")
    print("====================================\n")

    # Produce PDF of heatmaps
    with PdfPages(HEATMAP_PDF) as pdf:

        # Heatmap of adjusted Vbd
        plot_heatmap(
            mat,
            title=f"Adjusted Vbd (VBD@25C) – Tray {TRAY_NUMBER}",
            pdf=pdf
        )

        # Heatmap of difference from mean
        plot_heatmap(
            diff,
            title=f"Difference from Mean Adjusted Vbd (Mean={mean_vbd:.4f}V)",
            pdf=pdf,
            cmap="coolwarm"
        )

        # Heatmap of outlier mask
        plot_heatmap(
            outlier_mask.astype(float),
            title=f"Outlier Locations (> ±50mV from mean)",
            pdf=pdf,
            cmap="gray"
        )

    print(f"PDF written: {HEATMAP_PDF}")


if __name__ == "__main__":
    main()
