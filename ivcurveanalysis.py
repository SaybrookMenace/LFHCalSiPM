'''
All Import statements go here
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

'''
ALL FUNCTIONS GO HERE
'''
def getIVLists(directory):
    raw_list = []
    with open(directory) as f:
        l = [line.strip() for line in f]
        for line in l:
            raw_list.append(line.split())
    #by this point, we have the file completely read. let's return the IV arrays
    x_list = []
    y_list = []
    for line in raw_list:
        if(is_number(line[0])):
            x_list.append(float(line[0]))
            y_list.append(float(line[2]))
    # x_list is voltage, y_list is current
    return([x_list, y_list])

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
'''
GREAT! AT THIS POINT, FUNCTIONS PERTAINING TO THE READING IN OF FILES IS COMPLETE

ALL FUNCTIONS PERTAINING TO IV ANALYSIS GO HERE
'''
def analyze_breakdown(V, I, smooth=True):
    # Convert inputs
    V = np.array(V, dtype=float)
    I = np.array(I, dtype=float)

    # avoid log(0)
    lnI = np.log(I + 1e-30)

    # smoothing for cleaner derivatives
    if smooth:
        lnI = savgol_filter(lnI, 11, 3)

    # Derivatives
    dlnI_dV = np.gradient(lnI, V)
    d2lnI_dV2 = np.gradient(dlnI_dV, V)
    dI_dV = np.gradient(I, V)

    # Baseline = linear region at low current
    N = max(10, len(V)//6)
    baseline = np.poly1d(np.polyfit(V[:N], lnI[:N], 1))

    # Tangent line at max slope
    idx_max_slope = np.argmax(dlnI_dV)
    slope = dlnI_dV[idx_max_slope]
    tangent = np.poly1d([slope, lnI[idx_max_slope] - slope*V[idx_max_slope]])
    VT_tangent = np.roots(tangent - baseline)[0]

    # Relative derivative max
    VT_rel = V[np.argmax(dlnI_dV)]

    # Inverse relative derivative
    inv_rel = I / (dI_dV + 1e-30)
    M = len(V)//3
    inv_fit = np.poly1d(np.polyfit(V[-M:], inv_rel[-M:], 1))
    VT_inv = np.roots(inv_fit)[0]

    # Second derivative max
    VT_second = V[np.argmax(d2lnI_dV2)]

    # Parabolic fit near max slope
    win = 10
    start = max(0, idx_max_slope - win)
    end = min(len(V), idx_max_slope + win)
    parabola = np.poly1d(np.polyfit(V[start:end], lnI[start:end], 2))
    VT_para = np.roots(parabola - baseline)[0]

    # ----------- ALL SIX PLOTS (2 columns × 3 rows) -----------

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs = axs.ravel()

    # 0) Base IV Curve (Semi-log)
    axs[0].semilogy(V, I, 'k', label="I(V)")
    axs[0].set_title("IV Curve")
    axs[0].set_xlabel("Bias (V)")
    axs[0].set_ylabel("I (A)")
    axs[0].grid(True)
    axs[0].legend()

    # 1) Tangent Method
    axs[1].plot(V, lnI, 'k', label="ln(I)")
    axs[1].plot(V, baseline(V), 'r--', label="Baseline")
    axs[1].plot(V, tangent(V), 'g--', label="Tangent")
    axs[1].axvline(VT_tangent, linestyle=':', color='blue',
                   label=f"Vbd = {VT_tangent:.2f} V")
    axs[1].set_title("Tangent Method")
    axs[1].set_xlabel("Bias (V)")
    axs[1].set_ylabel("ln(I)")
    axs[1].grid(True)
    axs[1].legend()

    # 2) Relative Derivative
    axs[2].plot(V, dlnI_dV, 'k', label="d(lnI)/dV")
    axs[2].axvline(VT_rel, linestyle=':', color='blue',
                   label=f"Vbd = {VT_rel:.2f} V")
    axs[2].set_title("Relative Derivative")
    axs[2].set_xlabel("Bias (V)")
    axs[2].set_ylabel("d(lnI)/dV")
    axs[2].grid(True)
    axs[2].legend()

    # 3) Inverse Relative Derivative
    axs[3].plot(V, inv_rel, 'k', label="I / I'")
    axs[3].plot(V[-M:], inv_fit(V[-M:]), 'r--', label="Linear Fit")
    axs[3].axvline(VT_inv, linestyle=':', color='blue',
                   label=f"Vbd = {VT_inv:.2f} V")
    axs[3].set_title("Inverse Relative Derivative")
    axs[3].set_xlabel("Bias (V)")
    axs[3].set_ylabel("I / I'")
    axs[3].grid(True)
    axs[3].legend()

    # 4) Second Derivative
    axs[4].plot(V, d2lnI_dV2, 'k', label="d²(lnI)/dV²")
    axs[4].axvline(VT_second, linestyle=':', color='blue',
                   label=f"Vbd = {VT_second:.2f} V")
    axs[4].set_title("Second Derivative")
    axs[4].set_xlabel("Bias (V)")
    axs[4].set_ylabel("d²(lnI)/dV²")
    axs[4].grid(True)
    axs[4].legend()

    # 5) Parabolic Fit
    axs[5].plot(V, lnI, 'k', label="ln(I)")
    axs[5].plot(V, baseline(V), 'r--', label="Baseline")
    axs[5].plot(V, parabola(V), 'b--', label="Parabolic Fit")
    axs[5].axvline(VT_para, linestyle=':', color='blue',
                   label=f"Vbd = {VT_para:.2f} V")
    axs[5].set_title("Parabolic Fit")
    axs[5].set_xlabel("Bias (V)")
    axs[5].set_ylabel("ln(I)")
    axs[5].grid(True)
    axs[5].legend()

    plt.tight_layout()
    plt.show()

    return {
        "Base IV Curve": None,
        "Tangent": float(VT_tangent),
        "Relative Derivative": float(VT_rel),
        "Inverse Relative Derivative": float(VT_inv),
        "Second Derivative": float(VT_second),
        "Parabolic Fit": float(VT_para)
    }

'''
Here, we run the actual python script! Nothing above runs unless something exists
underneath this comment.
'''
xylist = getIVLists("test_data/IV_250717_0_0.txt")
print(analyze_breakdown(xylist[0], xylist[1]))
