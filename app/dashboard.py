"""GlucoTwin clinician dashboard.

    streamlit run app/dashboard.py

Historical replay: a day and time slider set "now"; every value shown is computed from data up to
that moment only (tests/test_replay.py). The synthetic track always carries a banner.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from glucotwin import twin as tw  # noqa: E402
from glucotwin.config import load_config  # noqa: E402
from glucotwin.features import prepare_participant  # noqa: E402
from glucotwin.models import MODEL_NAMES  # noqa: E402
from glucotwin.pipeline import load_bundle, load_trackdata, synthetic_trackdata  # noqa: E402
from glucotwin.replay import precompute, state_at  # noqa: E402
from glucotwin.schema import provenance  # noqa: E402
from glucotwin.whatif import scenario  # noqa: E402

CFG = load_config(ROOT / "config.yaml")
THR = CFG["event"]["threshold_mgdl"]
DISCLAIMER = "Research prototype. Not a medical device. Not for diagnosis, treatment, dosing or emergencies."
FORECAST_NOTE = "Forecast from a simple personal meal-response model, not a physiological simulation of insulin, digestion or medication."

st.set_page_config(page_title="GlucoTwin", layout="wide")


def available_tracks() -> list[str]:
    tracks = []
    if (ROOT / "models" / "synthetic" / "bundle.json").exists():
        tracks.append("synthetic")
    if (ROOT / "models" / "evidence" / "bundle.json").exists() and (
        ROOT / "data" / "processed" / "evidence" / "meta.json"
    ).exists():
        tracks.append("evidence")
    return tracks


@st.cache_resource(show_spinner="Loading data and model ...")
def load_track(track: str):
    if track == "evidence":
        td = load_trackdata(ROOT / CFG["paths"]["processed"] / "evidence")
    else:
        folder = ROOT / CFG["paths"]["processed"] / "synthetic"
        td = load_trackdata(folder) if (folder / "meta.json").exists() else synthetic_trackdata(CFG)
    bundle, fitted = load_bundle(ROOT / CFG["paths"]["models"] / track)
    return td, bundle, fitted


@st.cache_resource(show_spinner="Preparing the virtual patient ...")
def virtual_patient(track: str, pid: str):
    td, bundle, fitted = load_track(track)
    p = prepare_participant(td, pid, CFG)
    if p is None:
        return None
    stored = bundle.get("daily_params", {}).get(pid)
    return precompute(p, bundle, fitted, CFG, None if stored is None else np.asarray(stored))


def minutes_at(pre, day: int, clock: int) -> float:
    start = pre.pdata.t0.normalize() + pd.Timedelta(days=day, minutes=clock)
    return float((start - pre.pdata.t0) / pd.Timedelta(minutes=1))


def badge(kind: str) -> str:
    colours = {
        "observed": "green",
        "derived": "blue",
        "imputed": "orange",
        "synthetic": "violet",
        "simulated": "violet",
        "estimated": "orange",
    }
    return f":{colours.get(kind, 'gray')}-badge[{kind}]"


tracks = available_tracks()
st.title("GlucoTwin")
st.caption(DISCLAIMER)
if not tracks:
    st.error("No trained bundle found. Run `python scripts/run_pipeline.py --demo` first.")
    st.stop()

with st.sidebar:
    labels = {"synthetic": "Synthetic demo", "evidence": "Evidence (CGMacros, local)"}
    track = st.radio("Track", tracks, format_func=labels.get)
    td, bundle, fitted = load_track(track)
    prof = td.profiles
    pids = [p for p in prof.index if bool(prof.loc[p].get("included", True))]
    pid = st.selectbox(
        "Participant",
        pids,
        format_func=lambda x: f"{x} · {prof.loc[x, 'group']} · {int(prof.loc[x, 'age'])} {prof.loc[x, 'sex']}",
    )
    pre = virtual_patient(track, pid)
    if pre is None:
        st.error("Not enough readings for this participant.")
        st.stop()
    n_days = int(np.ceil((pre.pdata.reading_t[-1] + pre.pdata.t0.hour * 60 + pre.pdata.t0.minute) / 1440))
    day = st.slider("Replay day", 1, max(n_days, 1), min(2, max(n_days, 1))) - 1
    clock = st.slider("Replay time", 0, 1435, 13 * 60, step=5, format="%d min")
    st.write(f"Replay time: **{pre.pdata.t0.normalize() + pd.Timedelta(days=day, minutes=clock):%a %d %b, %H:%M}**")
    model = bundle["selected_model"]
    st.markdown(
        f"**Model:** {model} {MODEL_NAMES[model]}  \n"
        f"**Alert threshold:** {bundle['thresholds'][model]:.3f}  \n"
        f"**Twin extensions:** {', '.join(bundle['spec']) or 'none'}  \n"
        f"**Bundle created:** {bundle['created'][:10]}"
    )

if track == "synthetic":
    st.warning(
        "SYNTHETIC DEMO: every value is simulated (Synthea EHRs + simulated wearables). Not evidence of accuracy."
    )

now = minutes_at(pre, day, clock)
state = state_at(pre, now, bundle, CFG)
tab_panel, tab_patient, tab_why, tab_whatif, tab_quality = st.tabs(
    ["Panel", "Patient", "Why", "What-if", "Data quality"]
)

with tab_panel:
    st.subheader("Participants ranked by risk at the replay time")
    panel_ids = list(dict.fromkeys([pid] + bundle.get("participants", [])[: CFG["dashboard"]["panel_size"]]))
    rows = []
    for other in panel_ids:
        opre = virtual_patient(track, other)
        if opre is None:
            continue
        s = state_at(opre, minutes_at(opre, day, clock), bundle, CFG, with_reasons=False)
        today = opre.pdata.day_starts[day] if day < len(opre.pdata.day_starts) else np.inf
        rows.append(
            {
                "Participant": other,
                "Group": prof.loc[other, "group"],
                "Glucose now (mg/dL)": None if np.isnan(s["g_now"]) else round(s["g_now"]),
                "Risk": None if np.isnan(s["risk"]) else round(s["risk"], 3),
                "Status": s["status"],
                "Alerts today": int((s["alerts_so_far"] >= today).sum()),
                "Readings in last 30 min": f"{s['cov30']:.0%}",
            }
        )
    table = pd.DataFrame(rows).sort_values("Risk", ascending=False, na_position="last")
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(
        "Risk is the model's score for a sustained excursion above 180 mg/dL starting within 2 hours. Alerts follow a 90-minute cooldown and a budget of 3 per patient-day."
    )

with tab_patient:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Glucose now", "–" if np.isnan(state["g_now"]) else f"{state['g_now']:.0f} mg/dL")
    c2.metric("Risk (2 h)", "–" if np.isnan(state["risk"]) else f"{state['risk']:.2f}")
    c3.metric("Status", state["status"])
    c4.metric("Readings, last 30 min", f"{state['cov30']:.0%}")
    if state["alert_now"]:
        st.error("Alert: model risk of a sustained excursion above 180 mg/dL within 2 hours is above the threshold.")

    left, right = st.columns([2, 1])
    with left:
        fig = go.Figure()
        trace = state["trace"]
        trace = trace[trace["t"] >= now - 12 * 60]
        x = lambda t: pre.pdata.t0 + pd.to_timedelta(np.asarray(t), unit="m")  # noqa: E731
        obs = trace[trace["src"] == "reading"]
        imp = trace[trace["src"] == "imputed"]
        fig.add_trace(
            go.Scatter(
                x=x(obs["t"]), y=obs["glucose"], mode="lines+markers", name="Sensor readings", marker={"size": 4}
            )
        )
        if len(imp):
            fig.add_trace(
                go.Scatter(
                    x=x(imp["t"]),
                    y=imp["glucose"],
                    mode="markers",
                    name="Imputed (short gap)",
                    marker={"symbol": "circle-open", "size": 7},
                )
            )
        if state["forecast"] is not None:
            hx = x(now + state["horizons"])
            fig.add_trace(go.Scatter(x=hx, y=state["band_hi"], line={"width": 0}, showlegend=False, hoverinfo="skip"))
            fig.add_trace(
                go.Scatter(
                    x=hx,
                    y=state["band_lo"],
                    line={"width": 0},
                    fill="tonexty",
                    name="80% band",
                    hoverinfo="skip",
                    fillcolor="rgba(99,110,250,0.15)",
                )
            )
            fig.add_trace(go.Scatter(x=hx, y=state["forecast"], line={"dash": "dash"}, name="Twin forecast"))
        meals = state["meals"]
        meals = meals[meals["t"] >= now - 12 * 60]
        if len(meals):
            hover = [f"{r.get('dish', r['meal_type'])}: {r['carbs_eaten']:.0f} g carbs" for _, r in meals.iterrows()]
            fig.add_trace(
                go.Scatter(
                    x=x(meals["t"]),
                    y=[60] * len(meals),
                    mode="markers",
                    marker={"symbol": "triangle-up", "size": 12},
                    name="Meals",
                    hovertext=hover,
                )
            )
        for a in state["alerts_so_far"][state["alerts_so_far"] >= now - 12 * 60]:
            fig.add_vline(x=x([a])[0], line={"color": "red", "width": 1, "dash": "dot"})
        fig.add_hline(y=THR, line={"color": "gray", "dash": "dot"}, annotation_text="180 mg/dL")
        fig.add_vline(x=x([now])[0], line={"color": "black", "width": 1})
        fig.update_layout(
            height=430, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title="mg/dL", legend={"orientation": "h"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(FORECAST_NOTE + " Red dotted lines mark alerts.")
    with right:
        st.markdown("**Profile**")
        profile = prof.loc[pid]
        fields = [
            ("age", "Age", "{:.0f}"),
            ("sex", "Sex", "{}"),
            ("bmi", "BMI", "{:.1f}"),
            ("hba1c_pct", "HbA1c (%)", "{:.1f}"),
            ("fasting_glucose", "Fasting glucose", "{:.0f}"),
            ("fasting_insulin", "Fasting insulin", "{:.1f}"),
            ("homa_ir", "HOMA-IR", "{:.2f}"),
            ("group", "Group", "{}"),
        ]
        for key, label, fmt in fields:
            value = profile.get(key, np.nan)
            if isinstance(value, float) and np.isnan(value):
                continue
            kind = profile.get(f"prov_{key}", provenance(track, key))
            kind = "simulated" if kind == "simulated" else provenance(track, key)
            st.markdown(f"{label}: **{fmt.format(value)}** {badge(kind)}")
        if track == "synthetic" and isinstance(profile.get("conditions"), str):
            st.markdown(f"Conditions (Synthea FHIR): {profile['conditions'].replace(';', ', ')} {badge('synthetic')}")
        st.markdown("**Personal twin today**")
        for k, v in state["params"].items():
            st.markdown(f"{k.replace('_', ' ')}: **{v:.2f}**")
        st.caption("Fitted on this person's data before today's midnight, starting from a lab-informed prior.")
    st.markdown("**Alert history**")
    if len(state["alerts_so_far"]):
        st.dataframe(pd.DataFrame({"Alert time": x(state["alerts_so_far"])}).iloc[::-1], hide_index=True)
    else:
        st.write("No alerts yet.")
    st.markdown("**Confirmed excursions so far** (above 180 mg/dL for at least 15 minutes)")
    ev = state["events"]
    if len(ev):
        st.dataframe(
            pd.DataFrame({"Onset": x(ev["onset"]), "Confirmed": x(ev["confirm"]), "Last high reading": x(ev["last"])}),
            hide_index=True,
        )
    else:
        st.write("None so far.")

with tab_why:
    st.subheader("Why the model rates the current risk this way")
    if state["reasons"]:
        for r in state["reasons"]:
            icon = "▲" if r["effect"] == "raises risk" else "▼"
            st.markdown(f"{icon} **{r['family'].title()}** ({r['effect']}): {r['text']}")
        st.caption(
            "Factors the model associated with higher or lower risk, from TreeSHAP contributions. Associations, not causes."
        )
    else:
        st.info(f"No prediction at this moment (status: {state['status']}).")

with tab_whatif:
    st.subheader("What-if: compare a scenario with the baseline")
    if state["forecast"] is None or np.isnan(state["g_now"]):
        st.info("No current reading to start from.")
    else:
        spec = tw.TwinSpec(tuple(bundle["spec"]))
        mode = st.radio(
            "Scenario",
            ["next", "last"],
            format_func={"next": "Add a meal now", "last": "Change the last meal's carbohydrate"}.get,
            horizontal=True,
        )
        carbs = st.slider("Carbohydrate (g)", 0, 150, 60, step=5)
        walk = 0
        if spec.has("E2"):
            walk = st.slider("Hypothetical walk after the meal (minutes)", 0, 45, 0, step=5)
        else:
            st.caption("The walk control appears only when the activity extension (E2) was selected for this twin.")
        day_idx = int(np.searchsorted(pre.pdata.day_starts, now, side="right") - 1)
        res = scenario(
            pre.pdata,
            spec,
            pre.params[day_idx],
            bundle["tau_r"],
            bundle["band"],
            now,
            state["g_now"],
            mode,
            float(carbs),
            float(walk),
            CFG,
        )
        hx = pre.pdata.t0 + pd.to_timedelta(now + res["horizons"], unit="m")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=hx, y=res["baseline"] + res["band_hi"], line={"width": 0}, showlegend=False, hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(
                x=hx,
                y=res["baseline"] + res["band_lo"],
                line={"width": 0},
                fill="tonexty",
                name="80% band (baseline)",
                fillcolor="rgba(99,110,250,0.12)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(go.Scatter(x=hx, y=res["baseline"], name="Baseline"))
        fig.add_trace(go.Scatter(x=hx, y=res["scenario"], name="Scenario", line={"dash": "dash"}))
        fig.add_hline(y=THR, line={"color": "gray", "dash": "dot"})
        fig.update_layout(
            height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title="mg/dL", legend={"orientation": "h"}
        )
        st.plotly_chart(fig, use_container_width=True)
        a, b = st.columns(2)
        a.metric(
            "Baseline peak",
            f"{res['baseline_summary']['peak']:.0f} mg/dL",
            f"{res['baseline_summary']['minutes_above']:.0f} min above 180",
            delta_color="off",
        )
        b.metric(
            "Scenario peak",
            f"{res['scenario_summary']['peak']:.0f} mg/dL",
            f"{res['scenario_summary']['minutes_above']:.0f} min above 180",
            delta_color="off",
        )
        st.markdown("**Changed inputs:** " + "; ".join(res["changed"]))
        for w in res["warnings"]:
            st.warning(w)
        st.info(res["disclaimer"])

with tab_quality:
    st.subheader("Data quality up to the replay time")
    p = pre.pdata
    rt = p.reading_t[p.reading_t <= now]
    tl = state["trace"]
    gaps = np.diff(rt) if len(rt) > 1 else np.array([])
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Coverage over wear period", f"{prof.loc[pid].get('primary_coverage', np.nan):.0%}")
    last24 = ((rt > now - 1440).sum()) / (1440 / p.interval)
    q2.metric("Coverage, last 24 h", f"{min(last24, 1):.0%}")
    q3.metric("Imputed share", f"{(tl['src'] == 'imputed').mean():.1%}" if len(tl) else "–")
    q4.metric("Gaps over 20 min", int((gaps > CFG["sensors"]["max_imputation_gap_min"]).sum()))
    cens = p.reading_c[p.reading_t <= now]
    st.write(
        f"Measured sampling interval: **{np.median(gaps):.0f} min** · censored readings: **{int((cens == -1).sum())} at <= 40, "
        f"{int((cens == 1).sum())} at >= 400**"
        if len(gaps)
        else "Not enough readings yet."
    )
    st.markdown("**Provenance legend**")
    st.markdown(
        f"{badge('observed')} measured or logged in the source data · {badge('derived')} calculated from observed values · "
        f"{badge('imputed')} filled in for a missing expected reading, never used for labels · {badge('synthetic')} generated by Synthea or our simulator"
    )
    st.markdown("**Validation checks for this track**")
    st.dataframe(pd.DataFrame([c.__dict__ for c in td.checks]), hide_index=True, use_container_width=True)
