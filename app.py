"""Gradio app: interpolation methods on intraday MNQ futures prices. Run locally with `python app.py`.

The free Hugging Face Space uses index.html instead, which runs the same code in the browser.
"""
import matplotlib

matplotlib.use("Agg")
import gradio as gr

import figures as fg
import interpolation as ip

try:
    import download_mnq
except ImportError:
    download_mnq = None

STATE = {}


def compute():
    """Load the data snapshot and run the full experiment."""
    data, info = ip.load_session_data()
    res, timing = ip.run_experiment(data)
    STATE.update(ip.summarize(data, info, res, timing))


def show(df):
    return df.round(3).reset_index()


def explore(date_str, h, methods, start):
    return fg.fig_explore(STATE["data"], date_str, int(h), methods, int(start))


def results_outputs():
    return (ip.describe_data(STATE["info"]), STATE["findings"], show(STATE["t1"]), show(STATE["t2"]),
            show(STATE["t3"]), fg.fig_error_vs_spacing(STATE["t1"]), fg.fig_daily_boxplots(STATE["daily"]),
            fg.fig_prices(STATE["data"]))


def refresh_data():
    try:
        download_mnq.main()
    except SystemExit:
        return ("Yahoo did not return any data. This often happens on cloud servers; the app is still using "
                "the saved snapshot.", *results_outputs())
    except Exception as error:
        return (f"Download failed ({type(error).__name__}: {error}). The saved snapshot is still in use.",
                *results_outputs())
    compute()
    return f"Data refreshed: {STATE['info']['days_kept']} days, last bar {STATE['info']['last_bar']}.", *results_outputs()


compute()
dates = [str(d) for d in sorted(STATE["data"]["date"].unique())]

with gr.Blocks(title="Interpolation Methods on MNQ Futures") as demo:
    gr.Markdown("# Comparing Interpolation Methods for Reconstructing Intraday MNQ Futures Prices\n"
                "Lameck Nyakweba · Mathematics 5670-08, Fall 2026 · Austin Peay State University\n\n"
                "Prices are kept every *h* minutes and the prices in between are hidden; each method rebuilds "
                "them from the kept prices only, and the estimates are compared with the true prices.")
    summary_md = gr.Markdown(ip.describe_data(STATE["info"]))

    with gr.Tab("Results"):
        findings_box = gr.Textbox(STATE["findings"], label="Findings (generated from the data)", lines=12)
        gr.Markdown("### Table 1. Accuracy (index points), overshoot, and speed")
        t1_df = gr.Dataframe(show(STATE["t1"]))
        err_plot = gr.Plot(fg.fig_error_vs_spacing(STATE["t1"]))
        gr.Markdown("### Table 2. Day-to-day consistency")
        t2_df = gr.Dataframe(show(STATE["t2"]))
        box_plot = gr.Plot(fg.fig_daily_boxplots(STATE["daily"]))
        gr.Markdown("### Table 3. RMSE at the open versus midday\n"
                    + ", ".join(f"{k}: volatility {v:.2f} pts/min" for k, v in STATE["vol"].items()))
        t3_df = gr.Dataframe(show(STATE["t3"]))
        price_plot = gr.Plot(fg.fig_prices(STATE["data"]))

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
        gr.Plot(fg.fig_convergence())
        gr.Plot(fg.fig_runge())

    if download_mnq is not None:
        with gr.Tab("Data"):
            gr.Markdown("The app runs on the saved data snapshot (`data/mnq_1m_raw.csv`). "
                        "You can try fetching newer bars from Yahoo Finance, but Yahoo often blocks cloud servers.")
            refresh_btn = gr.Button("Try to fetch the latest data from Yahoo")
            refresh_msg = gr.Markdown()
            refresh_btn.click(refresh_data, None,
                              [refresh_msg, summary_md, findings_box, t1_df, t2_df, t3_df, err_plot, box_plot, price_plot])

if __name__ == "__main__":
    demo.launch()
