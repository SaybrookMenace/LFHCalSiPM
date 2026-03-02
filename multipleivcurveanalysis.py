'''
All Import statements go here
'''
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter

'''
ALL FUNCTIONS GO HERE
'''
def getIVLists(path):
    raw_list = []
    with open(path) as f:
        l = [line.strip() for line in f]
        for line in l:
            raw_list.append(line.split())
    # by this point, we have the file completely read. let's return the IV arrays
    x_list = []
    y_list = []
    for line in raw_list:
        if is_number(line[0]):
            x_list.append(float(line[0]))   # Voltage
            y_list.append(float(line[2]))   # Current (per your format)
    return [x_list, y_list]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def _safe_savgol(y, window=11, poly=3):
    n = len(y)
    if n < (poly + 2):
        return np.array(y, dtype=float)
    # window must be odd and <= n
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < poly + 2:
        # fallback: no smoothing if too few points
        return np.array(y, dtype=float)
    return savgol_filter(y, w, poly)

def analyze_breakdown(V, I, smooth=True):
    # Convert inputs
    V = np.array(V, dtype=float)
    I = np.array(I, dtype=float)

    # avoid log(0)
    lnI = np.log(I + 1e-30)

    # smoothing for cleaner derivatives
    if smooth:
        lnI = _safe_savgol(lnI, 11, 3)

    # Derivatives
    dlnI_dV = np.gradient(lnI, V)
    d2lnI_dV2 = np.gradient(dlnI_dV, V)
    dI_dV = np.gradient(I, V)

    # Baseline = linear region at low current
    N = max(10, len(V)//6) if len(V) >= 12 else max(3, len(V)//3)
    baseline = np.poly1d(np.polyfit(V[:N], lnI[:N], 1))

    # Tangent line at max slope
    idx_max_slope = int(np.argmax(dlnI_dV))
    slope = dlnI_dV[idx_max_slope]
    tangent = np.poly1d([slope, lnI[idx_max_slope] - slope*V[idx_max_slope]])
    VT_tangent = float(np.roots(tangent - baseline)[0])

    # Relative derivative max
    VT_rel = float(V[np.argmax(dlnI_dV)])

    # Inverse relative derivative
    inv_rel = I / (dI_dV + 1e-30)
    M = max(5, len(V)//3)
    M = min(M, len(V))  # bounds
    inv_fit = np.poly1d(np.polyfit(V[-M:], inv_rel[-M:], 1))
    VT_inv = float(np.roots(inv_fit)[0])

    # Second derivative max
    VT_second = float(V[np.argmax(d2lnI_dV2)])

    # Parabolic fit near max slope
    win = min(10, len(V)//5)
    start = max(0, idx_max_slope - win)
    end = min(len(V), idx_max_slope + win)
    if end - start < 3:  # ensure enough points
        start = max(0, idx_max_slope - 2)
        end = min(len(V), idx_max_slope + 2)
    parabola = np.poly1d(np.polyfit(V[start:end], lnI[start:end], 2))
    VT_para = float(np.roots(parabola - baseline)[0])

    results = {
        "VT_tangent": VT_tangent,
        "VT_rel": VT_rel,
        "VT_inv": VT_inv,
        "VT_second": VT_second,
        "VT_para": VT_para,
        # keep intermediates needed for plotting
        "_baseline": baseline,
        "_tangent": tangent,
        "_parabola": parabola,
        "_dlnI_dV": dlnI_dV,
        "_d2lnI_dV2": d2lnI_dV2,
        "_inv_rel": inv_rel,
        "_inv_fit": inv_fit,
        "_M": M,
        "_lnI": lnI
    }
    return results

def plot_six_panel(V, I, res, fig_title=None):
    V = np.array(V, dtype=float)
    I = np.array(I, dtype=float)
    lnI = res["_lnI"]
    baseline = res["_baseline"]
    tangent = res["_tangent"]
    parabola = res["_parabola"]
    dlnI_dV = res["_dlnI_dV"]
    d2lnI_dV2 = res["_d2lnI_dV2"]
    inv_rel = res["_inv_rel"]
    inv_fit = res["_inv_fit"]
    M = res["_M"]

    VT_tangent = res["VT_tangent"]
    VT_rel     = res["VT_rel"]
    VT_inv     = res["VT_inv"]
    VT_second  = res["VT_second"]
    VT_para    = res["VT_para"]

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs = axs.ravel()
    if fig_title:
        fig.suptitle(fig_title, fontsize=14, y=0.98)

    # 0) Base IV Curve (Semi-log)
    axs[0].semilogy(V, I, label="I(V)")
    axs[0].set_title("IV Curve")
    axs[0].set_xlabel("Bias (V)")
    axs[0].set_ylabel("I (A)")
    axs[0].grid(True, which='both')
    axs[0].legend(loc='best')

    # 1) Tangent Method
    axs[1].plot(V, lnI, label="ln(I)")
    axs[1].plot(V, baseline(V), linestyle='--', label="Baseline")
    axs[1].plot(V, tangent(V), linestyle='--', label="Tangent")
    axs[1].axvline(VT_tangent, linestyle=':', label=f"Vbd={VT_tangent:.2f}V")
    axs[1].set_title("Tangent Method")
    axs[1].set_xlabel("Bias (V)")
    axs[1].set_ylabel("ln(I)")
    axs[1].grid(True)
    axs[1].legend(loc='best')

    # 2) Relative Derivative
    axs[2].plot(V, dlnI_dV, label="d(lnI)/dV")
    axs[2].axvline(VT_rel, linestyle=':', label=f"Vbd={VT_rel:.2f}V")
    axs[2].set_title("Relative Derivative")
    axs[2].set_xlabel("Bias (V)")
    axs[2].set_ylabel("d(lnI)/dV")
    axs[2].grid(True)
    axs[2].legend(loc='best')

    # 3) Inverse Relative Derivative
    axs[3].plot(V, inv_rel, label="I/I'")
    axs[3].plot(V[-M:], inv_fit(V[-M:]), linestyle='--', label="Linear Fit")
    axs[3].axvline(VT_inv, linestyle=':', label=f"Vbd={VT_inv:.2f}V")
    axs[3].set_title("Inverse Relative Derivative")
    axs[3].set_xlabel("Bias (V)")
    axs[3].set_ylabel("I / I'")
    axs[3].grid(True)
    axs[3].legend(loc='best')

    # 4) Second Derivative
    axs[4].plot(V, d2lnI_dV2, label="d²(lnI)/dV²")
    axs[4].axvline(VT_second, linestyle=':', label=f"Vbd={VT_second:.2f}V")
    axs[4].set_title("Second Derivative")
    axs[4].set_xlabel("Bias (V)")
    axs[4].set_ylabel("d²(lnI)/dV²")
    axs[4].grid(True)
    axs[4].legend(loc='best')

    # 5) Parabolic Fit
    axs[5].plot(V, lnI, label="ln(I)")
    axs[5].plot(V, baseline(V), linestyle='--', label="Baseline")
    axs[5].plot(V, parabola(V), linestyle='--', label="Parabolic Fit")
    axs[5].axvline(VT_para, linestyle=':', label=f"Vbd={VT_para:.2f}V")
    axs[5].set_title("Parabolic Fit")
    axs[5].set_xlabel("Bias (V)")
    axs[5].set_ylabel("ln(I)")
    axs[5].grid(True)
    axs[5].legend(loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

'''
GREAT! AT THIS POINT, FUNCTIONS PERTAINING TO THE READING IN OF FILES IS COMPLETE

ALL FUNCTIONS PERTAINING TO IV ANALYSIS GO HERE
'''

if __name__ == "__main__":
    # Point this glob to all the .txt files you want to analyze.
    # Example: files from test_reader2.cpp runs:
    #   test_data/IV_250821-1301_0_0.txt ... IV_250821-1301_22_19.txt
    TXT_GLOB = "test_data/IV_*.txt"

    txt_files = sorted(glob.glob(TXT_GLOB))
    if not txt_files:
        raise SystemExit(f"No text files matched glob: {TXT_GLOB}")

    out_pdf = "IV_breakdown_report.pdf"
    with PdfPages(out_pdf) as pdf:
        for path in txt_files:
            try:
                V, I = getIVLists(path)
                if len(V) < 5:
                    print(f"Skipping (too few points): {path}")
                    continue

                res = analyze_breakdown(V, I, smooth=True)
                title = f"{os.path.basename(path)}"
                fig = plot_six_panel(V, I, res, fig_title=title)
                pdf.savefig(fig)
                plt.close(fig)

                # Optional progress
                vt_summary = (res["VT_tangent"], res["VT_rel"], res["VT_inv"], res["VT_second"], res["VT_para"])
                print(f"Saved page for {path}; Vbd(s): {tuple(round(v,2) for v in vt_summary)}")

            except Exception as e:
                print(f"Error on {path}: {e}")

    print(f"Report written to: {out_pdf}")
