"""Interpolation methods and the hide-and-reconstruct experiment.

The same logic as interpolation_project.ipynb, packaged so the Hugging Face app
(app.py) can import it.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

TICK = 0.25                    # MNQ minimum price movement, in index points
SPACINGS = [5, 15, 30]         # minutes between kept prices (nodes)
NEWTON_DEGREES = [2, 3, 5]
OPEN_WINDOW = (0, 30)          # 9:30-10:00, as minutes after the open
MIDDAY_WINDOW = (120, 240)     # 11:30-13:30
WINDOW_LABELS = {"Open 9:30-10:00": OPEN_WINDOW, "Midday 11:30-13:30": MIDDAY_WINDOW}
MIN_BARS_PER_DAY = 370         # a full session has 390; shorter days are excluded
DATA_FILE = Path(__file__).resolve().parent / "data" / "mnq_1m_raw.csv"


# ---------------- Methods ----------------

def interp_linear(xn, yn, xt):
    return np.interp(xt, xn, yn)


def interp_spline(xn, yn, xt):
    return CubicSpline(xn, yn, bc_type="natural")(xt)


def interp_pchip(xn, yn, xt):
    return PchipInterpolator(xn, yn)(xt)


def divided_differences(x, y):
    """Newton coefficients f[x0], f[x0,x1], ..., f[x0,...,xn]."""
    coef = np.array(y, dtype=float)
    for j in range(1, len(x)):
        coef[j:] = (coef[j:] - coef[j - 1:-1]) / (x[j:] - x[:-j])
    return coef


def newton_eval(coef, x, t):
    """Evaluate the Newton form in nested (Horner-like) order."""
    result = coef[-1]
    for k in range(len(coef) - 2, -1, -1):
        result = result * (t - x[k]) + coef[k]
    return result


def newton_window(xn, i, size, t):
    """Indices lo..hi of `size` nodes: the bracketing pair xn[i], xn[i+1] plus the nearest others."""
    lo, hi = i, i + 1
    while hi - lo + 1 < size:
        can_left, can_right = lo > 0, hi < len(xn) - 1
        if can_left and (not can_right or t - xn[lo - 1] <= xn[hi + 1] - t):
            lo -= 1   # ties go to the earlier node
        else:
            hi += 1
    return lo, hi


def interp_newton(xn, yn, xt, degree):
    size = min(degree + 1, len(xn))
    brackets = np.clip(np.searchsorted(xn, xt, side="right") - 1, 0, len(xn) - 2)
    out = np.empty(len(xt))
    for k, (t, i) in enumerate(zip(xt, brackets)):
        lo, hi = newton_window(xn, i, size, t)
        xw, yw = xn[lo:hi + 1], yn[lo:hi + 1]
        center, scale = (xw[0] + xw[-1]) / 2, (xw[-1] - xw[0]) / 2
        u = (xw - center) / scale
        out[k] = newton_eval(divided_differences(u, yw), u, (t - center) / scale)
    return out


METHODS = {
    "Linear": interp_linear,
    "Natural spline": interp_spline,
    "PCHIP": interp_pchip,
    **{f"Newton deg {d}": (lambda xn, yn, xt, d=d: interp_newton(xn, yn, xt, d)) for d in NEWTON_DEGREES},
}
METHOD_ORDER = list(METHODS)


# ---------------- Data ----------------

def load_session_data(path=DATA_FILE):
    """Regular-session closing prices on full days, with a `minute` column (0 = 9:30)."""
    raw = pd.read_csv(path, index_col="Datetime")
    raw.index = pd.to_datetime(raw.index, utc=True).tz_convert("America/New_York")
    prices = raw[~raw.index.duplicated(keep="first")].sort_index()[["Close"]].dropna()
    session = prices.between_time("09:30", "15:59")
    session = session[session.index.dayofweek < 5].copy()
    session["minute"] = session.index.hour * 60 + session.index.minute - (9 * 60 + 30)
    session["date"] = session.index.date
    bars_per_day = session.groupby("date").size()
    kept = bars_per_day[bars_per_day >= MIN_BARS_PER_DAY].index
    info = {
        "raw_bars": len(raw),
        "first_bar": str(raw.index.min()),
        "last_bar": str(raw.index.max()),
        "days_kept": len(kept),
        "days_excluded": {str(d): int(n) for d, n in bars_per_day[bars_per_day < MIN_BARS_PER_DAY].items()},
    }
    data = session[session["date"].isin(kept)]
    info["bars_used"] = len(data)
    info["missing_minutes"] = 390 * len(kept) - len(data)
    return data, info


def split_nodes(day, h):
    """Node and target arrays for one day: nodes every h minutes plus the first and last price."""
    m = day["minute"].to_numpy(dtype=float)
    y = day["Close"].to_numpy()
    is_node = day["minute"].to_numpy() % h == 0
    is_node[0] = is_node[-1] = True
    return m[is_node], y[is_node], m[~is_node], y[~is_node]


# ---------------- Experiment ----------------

def run_day(d, day, timing_repeats=3):
    """Reconstruct one day's hidden prices with every method and spacing."""
    rows, timings = [], []
    for h in SPACINGS:
        xn, yn, xt, yt = split_nodes(day, h)
        gap = np.searchsorted(xn, xt) - 1
        for name, f in METHODS.items():
            runs = []
            for _ in range(timing_repeats):
                start = time.perf_counter()
                est = f(xn, yn, xt)
                runs.append(time.perf_counter() - start)
            timings.append({"date": d, "h": h, "method": name, "seconds": np.median(runs)})
            rows.append(pd.DataFrame({
                "date": d, "h": h, "method": name, "minute": xt, "true": yt, "est": est,
                "gap": gap, "left_node": yn[gap], "right_node": yn[gap + 1],
            }))
    return rows, timings


def run_experiment(data, timing_repeats=3):
    rows, timings = [], []
    for d, day in data.groupby("date"):
        day_rows, day_timings = run_day(d, day, timing_repeats)
        rows += day_rows
        timings += day_timings
    return finish_results(rows, timings)


def finish_results(rows, timings):
    """Combine per-day output and add the error, overshoot, and session-window columns."""
    res = pd.concat(rows, ignore_index=True)
    res["error"] = res["est"] - res["true"]
    lo = np.minimum(res["left_node"], res["right_node"]) - 1e-9
    hi = np.maximum(res["left_node"], res["right_node"]) + 1e-9
    res["outside"] = (res["est"] < lo) | (res["est"] > hi)
    res["window"] = "other"
    for label, (a, b) in WINDOW_LABELS.items():
        res.loc[(res["minute"] >= a) & (res["minute"] < b), "window"] = label
    return res, pd.DataFrame(timings)


def rmse(e):
    return float(np.sqrt((e ** 2).mean()))


def summary_tables(res, timing):
    g = res.groupby(["h", "method"])["error"]
    t1 = pd.DataFrame({
        "MAE": g.apply(lambda e: e.abs().mean()),
        "RMSE": g.apply(rmse),
        "Max": g.apply(lambda e: e.abs().max()),
    })
    t1["MAE ticks"] = t1["MAE"] / TICK
    t1["RMSE ticks"] = t1["RMSE"] / TICK
    t1["Overshoot %"] = res.groupby(["h", "method", "date", "gap"])["outside"].any().groupby(["h", "method"]).mean() * 100
    t1["ms/day"] = timing.groupby(["h", "method"])["seconds"].median() * 1000
    t1 = t1.reindex(pd.MultiIndex.from_product([SPACINGS, METHOD_ORDER], names=["h", "method"]))

    daily = res.groupby(["h", "method", "date"])["error"].apply(rmse).unstack("method")[METHOD_ORDER]
    beats = daily.lt(daily["Linear"], axis=0).groupby("h").mean() * 100
    by_h = daily.groupby("h")
    stats = {"median RMSE": by_h.median(), "IQR": by_h.quantile(0.75) - by_h.quantile(0.25), "% days beating linear": beats}
    t2 = pd.concat(stats, names=["statistic", "h"]).swaplevel()
    t2 = t2.reindex(pd.MultiIndex.from_product([SPACINGS, list(stats)], names=["h", "statistic"]))

    t3 = (res[res["window"] != "other"].groupby(["h", "window", "method"])["error"]
          .apply(rmse).unstack("method")[METHOD_ORDER])
    return t1, t2, t3, daily, beats


def window_volatility(data):
    vol = {}
    for label, (a, b) in WINDOW_LABELS.items():
        in_win = data[(data["minute"] >= a) & (data["minute"] < b)]
        vol[label] = float(in_win.groupby("date")["Close"].diff().std())
    return vol


def summarize(data, info, res, timing):
    """Everything the app and the browser page display, from finished experiment results."""
    t1, t2, t3, daily, beats = summary_tables(res, timing)
    vol = window_volatility(data)
    return {"data": data, "info": info, "res": res, "t1": t1, "t2": t2, "t3": t3, "daily": daily,
            "vol": vol, "findings": findings(t1, t3, beats, vol)}


def describe_data(info):
    excluded = ", ".join(f"{d} ({n} bars)" for d, n in info["days_excluded"].items()) or "none"
    return (f"Data snapshot: {info['raw_bars']:,} one-minute bars from {info['first_bar']} to {info['last_bar']}. "
            f"Used: {info['days_kept']} full trading days, {info['bars_used']:,} regular-session prices "
            f"({info['missing_minutes']} missing minutes). Excluded days: {excluded}.")


def findings(t1, t3, beats, vol):
    lines = []
    rm = t1["RMSE"].unstack("method")
    grows = {m: bool(np.all(np.diff(rm[m].to_numpy()) > 0)) for m in METHOD_ORDER}
    lines.append(f"H1 (error grows with spacing): holds for {sum(grows.values())} of {len(grows)} methods"
                 + ("" if all(grows.values()) else f"; exceptions: {[m for m, ok in grows.items() if not ok]}") + ".")
    for h in SPACINGS:
        best, lin = rm.loc[h].idxmin(), rm.loc[h, "Linear"]
        if best == "Linear":
            lines.append(f"   h = {h:>2} min: linear has the lowest RMSE ({lin:.3f} pts = {lin / TICK:.1f} ticks); no method beat the baseline.")
        else:
            diff = 100 * (rm.loc[h, best] - lin) / lin
            lines.append(f"   h = {h:>2} min: lowest RMSE is {best} ({rm.loc[h, best]:.3f} pts = {rm.loc[h, best] / TICK:.1f} ticks), "
                         f"{diff:+.1f}% versus linear ({lin:.3f} pts).")
    for m in ["Natural spline", "PCHIP"]:
        shares = ", ".join(f"h={h}: {beats.loc[h, m]:.0f}%" for h in SPACINGS)
        lines.append(f"H2 (no consistent gain over linear): {m} beat linear on {shares} of days.")
    ov = t1["Overshoot %"].unstack("method")
    lines.append(f"H3 (linear and PCHIP never overshoot): max overshoot rate linear = {ov['Linear'].max():.2f}%, "
                 f"PCHIP = {ov['PCHIP'].max():.2f}% (spline up to {ov['Natural spline'].max():.1f}%). "
                 + ("Check passed." if ov[["Linear", "PCHIP"]].to_numpy().max() == 0 else "CHECK FAILED: investigate the code."))
    newton = [f"Newton deg {d}" for d in NEWTON_DEGREES]
    for h in SPACINGS:
        r = [rm.loc[h, n] for n in newton]
        trend = "rises" if np.all(np.diff(r) > 0) else "does not rise steadily"
        lines.append(f"H4 (higher Newton degree, more oscillation): at h = {h:>2}, RMSE for degrees {NEWTON_DEGREES} = "
                     f"{', '.join(f'{v:.3f}' for v in r)} ({trend}).")
    o, mdy = vol["Open 9:30-10:00"], vol["Midday 11:30-13:30"]
    lines.append(f"Sessions: 1-minute volatility was {o:.3f} pts at the open versus {mdy:.3f} at midday "
                 f"({o / mdy:.1f}x). Linear RMSE at h=15: open {t3.loc[(15, 'Open 9:30-10:00'), 'Linear']:.3f}, "
                 f"midday {t3.loc[(15, 'Midday 11:30-13:30'), 'Linear']:.3f}.")
    return "\n".join(lines)
