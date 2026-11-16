"""Hugging Face Space: interpolation methods on intraday MNQ futures prices."""
import matplotlib

matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import download_mnq
import interpolation as ip

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
STATE = {}


def compute():
    """Load the data snapshot and run the full experiment (about a minute)."""
    data, info = ip.load_session_data()
    res, timing = ip.run_experiment(data)
    t1, t2, t3, daily, beats = ip.summary_tables(res, timing)
    vol = ip.window_volatility(data)
    STATE.update(data=data, info=info, res=res, t1=t1, t2=t2, t3=t3, daily=daily,
                 findings=ip.findings(t1, t3, beats, vol), vol=vol)


def show(df):
    return df.round(3).reset_index()


def time_label(minute):
    return f"{(570 + int(minute)) // 60}:{(570 + int(minute)) % 60:02d}"


# ---------------- Figures ----------------

def fig_prices():
    data = STATE["data"]
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


def fig_error_vs_spacing():
    t1 = STATE["t1"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ip.METHOD_ORDER:
        r = t1.xs(name, level="method")["RMSE"]
        slope = np.polyfit(np.log(ip.SPACINGS), np.log(r), 1)[0]
        ax.loglog(ip.SPACINGS, r, "o-", label=f"{name} (slope {slope:.2f})")
    ax.set_xticks(ip.SPACINGS); ax.set_xticklabels(ip.SPACINGS)
    ax.set_xlabel("spacing between kept prices h (minutes)"); ax.set_ylabel("RMSE (index points)")
    ax.set_title("Error grows with spacing (slope 0.5 = random walk, 2 = smooth linear theory)")
    ax.legend(fontsize=8); fig.tight_layout()
    return fig


def fig_daily_boxplots():
    daily = STATE["daily"]
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


# ---------------- Explore a day ----------------

def explore(date_str, h, methods, start):
    data = STATE["data"]
    day = data[data["date"].astype(str) == date_str]
    h, start = int(h), int(start)
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

    table =pd.DataFrame(rows, columns=["method", "RMSE (whole day)", "MAE (whole day)", "Max (whole day)"]).round(3)
    return fig, table


# ---------------- Data refresh ----------------

def refresh_data():
    try:
        download_mnq.main()
    except SystemExit:
        return ("Yahoo did not return any data. This often happens on cloud servers such as Hugging Face; "
                "the app is still using the saved snapshot. To update it, run download_mnq.py on your own "
                "computer and push the new CSV to GitHub.", *outputs_for_results())
    except Exception as error:
        return (f"Download failed ({type(error).__name__}: {error}). The saved snapshot is still in use.",
                *outputs_for_results())
    compute()
    return (f"Data refreshed: {STATE['info']['days_kept']} days, last bar {STATE['info']['last_bar']}. "
            "Note: on Hugging Face this update lasts only until the Space restarts.", *outputs_for_results())


def data_summary():
    i = STATE["info"]
    excluded = ", ".join(f"{d} ({n} bars)" for d, n in i["days_excluded"].items()) or "none"
    return (f"**Data snapshot:** {i['raw_bars']:,} one-minute bars from {i['first_bar']} to {i['last_bar']}  \n"
            f"**Used:** {i['days_kept']} full trading days, {i['bars_used']:,} regular-session prices "
            f"({i['missing_minutes']} missing minutes)  \n**Excluded days:** {excluded}")


def outputs_for_results():
    return (data_summary(), STATE["findings"], show(STATE["t1"]), show(STATE["t2"]), show(STATE["t3"]),
            fig_error_vs_spacing(), fig_daily_boxplots(), fig_prices())


# ---------------- Layout ----------------

compute()
dates = [str(d) for d in sorted(STATE["data"]["date"].unique())]

with gr.Blocks(title="Interpolation Methods on MNQ Futures") as demo:
    gr.Markdown("# Comparing Interpolation Methods for Reconstructing Intraday MNQ Futures Prices\n"
                "Lameck Nyakweba · Mathematics 5670-08, Fall 2026 · Austin Peay State University\n\n"
                "Prices are kept every *h* minutes and the prices in between are hidden; each method rebuilds "
                "them from the kept prices only, and the estimates are compared with the true prices.")
    summary_md = gr.Markdown(data_summary())

    with gr.Tab("Results"):
        findings_box = gr.Textbox(STATE["findings"], label="Findings (generated from the data)", lines=12)
        gr.Markdown("### Table 1. Accuracy (index points), overshoot, and speed")
        t1_df = gr.Dataframe(show(STATE["t1"]))
        err_plot = gr.Plot(fig_error_vs_spacing())
        gr.Markdown("### Table 2. Day-to-day consistency")
        t2_df = gr.Dataframe(show(STATE["t2"]))
        box_plot = gr.Plot(fig_daily_boxplots())
        gr.Markdown("### Table 3. RMSE at the open versus midday\n"
                    + ", ".join(f"{k}: volatility {v:.2f} pts/min" for k, v in STATE["vol"].items()))
        t3_df = gr.Dataframe(show(STATE["t3"]))
        price_plot = gr.Plot(fig_prices())

    with gr.Tab("Explore a day"):
        with gr.Row():
            date_in = gr.Dropdown(dates, value=dates[-1], label="Trading day")
            h_in = gr.Radio([str(h) for h in ip.SPACINGS], value="15", label="Spacing h (minutes)")
            start_in = gr.Slider(0, 329, value=0, step=1, label="Window start (minutes after 9:30)")
        methods_in = gr.CheckboxGroup(ip.METHOD_ORDER, value=["Linear", "Natural spline", "PCHIP", "Newton deg 5"], label="Methods")
        explore_plot = gr.Plot()
        explore_table = gr.Dataframe()
        inputs = [date_in, h_in, methods_in, start_in]
        for control in inputs:
            control.change(explore, inputs, [explore_plot, explore_table])
        demo.load(explore, inputs, [explore_plot, explore_table])

    with gr.Tab("Code checks"):
        gr.Markdown("Before trusting the MNQ results, the code is checked on functions with known answers. "
                    "On sin x, linear error should shrink with slope 2 and the natural spline with slope 4. "
                    "On the Runge function, one high-degree polynomial oscillates while local Newton does not.")
        gr.Plot(fig_convergence())
        gr.Plot(fig_runge())

    with gr.Tab("Data"):
        gr.Markdown("The app runs on the data snapshot saved in the GitHub repository (`data/mnq_1m_raw.csv`). "
                    "You can try fetching newer bars from Yahoo Finance, but Yahoo often blocks cloud servers.")
        refresh_btn = gr.Button("Try to fetch the latest data from Yahoo")
        refresh_msg = gr.Markdown()
        refresh_btn.click(refresh_data, None,
                          [refresh_msg, summary_md, findings_box, t1_df, t2_df, t3_df, err_plot, box_plot, price_plot])

if __name__ == "__main__":
    demo.launch()
