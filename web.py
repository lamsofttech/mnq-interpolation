"""Bridge between index.html and the project code; index.html runs it in the browser with Pyodide.

The experiment runs one day per call to step(), so the page can show progress instead of freezing.
"""
import base64
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figures as fg
import interpolation as ip

STATE = {}


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def table_html(df, index=True):
    return df.round(3).to_html(index=index, border=0, classes="tbl")


def start():
    data, info = ip.load_session_data()
    STATE.update(data=data, info=info, days=list(data.groupby("date")), rows=[], timings=[])
    return len(STATE["days"])


def step(i):
    d, day = STATE["days"][i]
    rows, timings = ip.run_day(d, day, timing_repeats=1)
    STATE["rows"] += rows
    STATE["timings"] += timings
    return str(d)


def finish():
    res, timing = ip.finish_results(STATE.pop("rows"), STATE.pop("timings"))
    STATE.update(ip.summarize(STATE["data"], STATE["info"], res, timing))
    s = STATE
    return json.dumps({
        "summary": ip.describe_data(s["info"]),
        "findings": s["findings"],
        "volatility": ", ".join(f"{k}: {v:.2f} points per minute" for k, v in s["vol"].items()),
        "t1": table_html(s["t1"]), "t2": table_html(s["t2"]), "t3": table_html(s["t3"]),
        "fig_error": png(fg.fig_error_vs_spacing(s["t1"])),
        "fig_box": png(fg.fig_daily_boxplots(s["daily"])),
        "fig_prices": png(fg.fig_prices(s["data"])),
        "fig_conv": png(fg.fig_convergence()),
        "fig_runge": png(fg.fig_runge()),
        "dates": [str(d) for d in sorted(s["data"]["date"].unique())],
        "methods": ip.METHOD_ORDER,
        "spacings": ip.SPACINGS,
    })


def explore(date_str, h, methods_json, start_minute):
    fig, table = fg.fig_explore(STATE["data"], date_str, int(h), json.loads(methods_json), int(start_minute))
    return json.dumps({"fig": png(fig), "table": table_html(table, index=False)})
