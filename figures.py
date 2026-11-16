"""Matplotlib figures shared by the Gradio app (app.py) and the in-browser page (web.py)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import interpolation as ip

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})


def time_label(minute):
    return f"{(570 + int(minute)) // 60}:{(570 + int(minute)) % 60:02d}"


def fig_prices(data):
    days = sorted(data["date"].unique())
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for i, (_, day) in enumerate(data.groupby("date")):
        ax.plot(i * 390 + day["minute"], day["Close"], lw=0.7, color="tab:blue")
    ax.set_xticks([i * 390 + 195 for i in range(len(days))])
    ax.set_xticklabels([d.strftime("%b %d") for d in days], rotation=60, fontsize=7)
    ax.set_ylabel("MNQ price (index points)")
    ax.set_title("MNQ 1-minute closing prices, regular session (days side by side)")
    fig.tight_layout()
    return fig


def fig_error_vs_spacing(t1):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ip.METHOD_ORDER:
        r = t1.xs(name, level="method")["RMSE"]
        slope = np.polyfit(np.log(ip.SPACINGS), np.log(r), 1)[0]
        ax.loglog(ip.SPACINGS, r, "o-", label=f"{name} (slope {slope:.2f})")
    ax.set_xticks(ip.SPACINGS); ax.set_xticklabels(ip.SPACINGS)
    ax.set_xlabel("spacing between kept prices h (minutes)"); ax.set_ylabel("RMSE (index points)")
    ax.set_title("Error grows with spacing (slope 0.5 = random walk, 2 = smooth linear theory)", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout()
    return fig


def fig_daily_boxplots(daily):
    fig, axes = plt.subplots(1, len(ip.SPACINGS), figsize=(12, 3.8))
    for ax, h in zip(axes, ip.SPACINGS):
        d = daily.xs(h, level="h")
        ax.boxplot([d[m] for m in ip.METHOD_ORDER])
        ax.set_xticks(range(1, len(ip.METHOD_ORDER) + 1))
        ax.set_xticklabels([m.replace("Newton deg ", "N") for m in ip.METHOD_ORDER], rotation=45, fontsize=8)
        ax.set_title(f"h = {h} min"); ax.set_ylabel("daily RMSE (points)")
    fig.suptitle("Daily RMSE across days (N2, N3, N5 = Newton degree)"); fig.tight_layout()
    return fig


def fig_convergence():
    hs = 2 * np.pi / np.array([8, 16, 32, 64, 128])
    fine = np.linspace(0, 2 * np.pi, 5001)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, f in ip.METHODS.items():
        errs = []
        for h in hs:
            xn = np.arange(0, 2 * np.pi + h / 2, h)
            errs.append(np.max(np.abs(f(xn, np.sin(xn), fine) - np.sin(fine))))
        slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        ax.loglog(hs, errs, "o-", label=f"{name} (slope {slope:.1f})")
    ax.set_xlabel("node spacing h"); ax.set_ylabel("max error")
    ax.set_title("Convergence on sin x: slope = order of accuracy")
    ax.legend(fontsize=8); fig.tight_layout()
    return fig


def fig_runge():
    runge = lambda x: 1 / (1 + 25 * x ** 2)
    xn = np.linspace(-1, 1, 15); fine = np.linspace(-1, 1, 801)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(fine, runge(fine), "k", lw=2, label="true function")
    ax.plot(fine, ip.interp_newton(xn, runge(xn), fine, len(xn) - 1), "r--", label="one global polynomial (degree 14)")
    ax.plot(fine, ip.interp_newton(xn, runge(xn), fine, 3), "b-", label="local Newton, degree 3")
    ax.plot(xn, runge(xn), "ko", ms=4)
    ax.set_ylim(-1, 2); ax.legend(fontsize=8); ax.set_title("Runge phenomenon")
    fig.tight_layout()
    return fig


def fig_explore(data, date_str, h, methods, start):
    """One-hour close-up of a day's reconstruction, plus whole-day errors for each chosen method."""
    day = data[data["date"].astype(str) == date_str]
    xn, yn, xt, yt = ip.split_nodes(day, h)
    a, b = start, min(start + 60, 389)
    w = day[(day["minute"] >= a) & (day["minute"] <= b)]
    fine = np.linspace(a, b, 600)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(w["minute"], w["Close"], "k.-", lw=0.8, ms=4, label="true 1-minute prices")
    rows = []
    for name in methods:
        f = ip.METHODS[name]
        ax.plot(fine, f(xn, yn, fine), lw=1.3, label=name)
        err = f(xn, yn, xt) - yt
        rows.append([name, ip.rmse(err), float(np.abs(err).mean()), float(np.abs(err).max())])
    sel = (xn >= a) & (xn <= b)
    ax.plot(xn[sel], yn[sel], "s", color="red", ms=8, label=f"kept prices (every {h} min)", zorder=4)
    ticks = np.arange(a - a % 15, b + 1, 15)
    ax.set_xticks(ticks); ax.set_xticklabels([time_label(t) for t in ticks])
    ax.set_xlim(a - 1, b + 1); ax.set_ylabel("MNQ price"); ax.legend(fontsize=8, ncol=2)
    ax.set_title(f"{date_str}, h = {h} min, {time_label(a)} to {time_label(b)}")
    fig.tight_layout()
    table = pd.DataFrame(rows, columns=["method", "RMSE (whole day)", "MAE (whole day)", "Max (whole day)"]).round(3)
    return fig, table
