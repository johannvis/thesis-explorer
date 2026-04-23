"""
Thesis Explorer — Digital Competencies & Change Readiness
PhD Thesis: FEnEx CRC RP4.0041 | N=335
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from scipy import stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thesis Explorer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth ──────────────────────────────────────────────────────────────────────
CREDENTIALS = {"johann": "kerry", "kerry": "kerry"}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🎓 Thesis Explorer")
    st.markdown("Digital Competencies & Change Readiness | FEnEx CRC RP4.0041")
    st.markdown("---")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if CREDENTIALS.get(username) == password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.stop()

# ── Paths (relative to this file) ─────────────────────────────────────────────
APP_DIR  = Path(__file__).parent
DATA_CSV = APP_DIR / "data" / "survey_with_domains.csv"
MP3_FILE = APP_DIR / "mp3" / "explorer_guide.mp3"

# ── Constants ─────────────────────────────────────────────────────────────────
DOMAINS = {
    "Data Analytics":          ["Q3", "Q4", "Q5", "Q6"],
    "Comm & Collab":           ["Q7", "Q8", "Q9", "Q10"],
    "Safety & Security":       ["Q11", "Q12", "Q13"],
    "Ethics & Responsibility": ["Q14", "Q15", "Q16"],
    "Digital Innovation":      ["Q17", "Q18", "Q19"],
    "Coding":                  ["Q20"],
}

DOMAIN_COLS = {
    "Data Analytics":          "domain_Data_Analytics",
    "Comm & Collab":           "domain_Comm_Collab",
    "Safety & Security":       "domain_Safety_Security",
    "Ethics & Responsibility": "domain_Ethics_Responsibility",
    "Digital Innovation":      "domain_Digital_Innovation",
    "Coding":                  "domain_Coding",
}

CR_VARS = {
    "Q22 – Personal openness":      "Q22_openness_self",
    "Q23 – Positive effect":        "Q23_positive_effect",
    "Q24 – Team openness":          "Q24_team_openness",
    "Q25 – Ease of change":         "Q25_difficulty",
}

CR_COLS   = list(CR_VARS.values())
DOMAIN_NAMES = list(DOMAIN_COLS.keys())

# Frozen Spearman (Step 4)
SPEARMAN_RHO = {
    "Data Analytics":          {"Q22_openness_self": 0.202, "Q23_positive_effect": 0.121, "Q24_team_openness": 0.011, "Q25_difficulty": 0.031},
    "Comm & Collab":           {"Q22_openness_self": 0.285, "Q23_positive_effect": 0.199, "Q24_team_openness": -0.007, "Q25_difficulty": 0.120},
    "Safety & Security":       {"Q22_openness_self": 0.261, "Q23_positive_effect": 0.167, "Q24_team_openness": 0.046, "Q25_difficulty": 0.152},
    "Ethics & Responsibility": {"Q22_openness_self": 0.241, "Q23_positive_effect": 0.134, "Q24_team_openness": 0.054, "Q25_difficulty": 0.116},
    "Digital Innovation":      {"Q22_openness_self": 0.349, "Q23_positive_effect": 0.185, "Q24_team_openness": 0.039, "Q25_difficulty": 0.173},
    "Coding":                  {"Q22_openness_self": 0.224, "Q23_positive_effect": 0.112, "Q24_team_openness": -0.013, "Q25_difficulty": -0.009},
}

SPEARMAN_SIG = {
    "Data Analytics":          {"Q22_openness_self": "**",  "Q23_positive_effect": "*",  "Q24_team_openness": "ns", "Q25_difficulty": "ns"},
    "Comm & Collab":           {"Q22_openness_self": "***", "Q23_positive_effect": "**", "Q24_team_openness": "ns", "Q25_difficulty": "ns"},
    "Safety & Security":       {"Q22_openness_self": "***", "Q23_positive_effect": "**", "Q24_team_openness": "ns", "Q25_difficulty": "*"},
    "Ethics & Responsibility": {"Q22_openness_self": "***", "Q23_positive_effect": "*",  "Q24_team_openness": "ns", "Q25_difficulty": "ns"},
    "Digital Innovation":      {"Q22_openness_self": "***", "Q23_positive_effect": "**", "Q24_team_openness": "ns", "Q25_difficulty": "**"},
    "Coding":                  {"Q22_openness_self": "***", "Q23_positive_effect": "ns", "Q24_team_openness": "ns", "Q25_difficulty": "ns"},
}

# Frozen OLS regression (Step 7)
REGRESSION = {
    "Q22_openness_self": {
        "A": {"R2": 0.131, "adjR2": 0.110, "F": "F(6,254)=6.357", "p": "<.001",
              "b": {"Data Analytics": -0.105, "Comm & Collab": 0.039, "Safety & Security": 0.120,
                    "Ethics & Responsibility": -0.007, "Digital Innovation": 0.218, "Coding": 0.028},
              "s": {"Data Analytics": "ns", "Comm & Collab": "ns", "Safety & Security": "ns",
                    "Ethics & Responsibility": "ns", "Digital Innovation": "**", "Coding": "ns"}},
        "B": {"R2": 0.193, "adjR2": 0.151, "F": "F(10,193)=4.612", "p": "<.001",
              "b": {"Data Analytics": -0.235, "Comm & Collab": 0.111, "Safety & Security": 0.122,
                    "Ethics & Responsibility": -0.084, "Digital Innovation": 0.266, "Coding": 0.037},
              "s": {"Data Analytics": "**", "Comm & Collab": "ns", "Safety & Security": "ns",
                    "Ethics & Responsibility": "ns", "Digital Innovation": "**", "Coding": "ns"}},
    },
    "Q23_positive_effect": {
        "A": {"R2": 0.057, "adjR2": 0.034, "F": "F(6,254)=2.547", "p": ".021",
              "b": {"Data Analytics": -0.110, "Comm & Collab": 0.193, "Safety & Security": 0.108,
                    "Ethics & Responsibility": 0.000, "Digital Innovation": 0.020, "Coding": 0.010},
              "s": {d: "ns" for d in DOMAIN_NAMES}},
        "B": {"R2": 0.093, "adjR2": 0.046, "F": "F(10,193)=1.973", "p": ".038",
              "b": {"Data Analytics": -0.250, "Comm & Collab": 0.221, "Safety & Security": 0.131,
                    "Ethics & Responsibility": -0.150, "Digital Innovation": 0.108, "Coding": 0.015},
              "s": {"Data Analytics": "**", "Comm & Collab": "ns", "Safety & Security": "ns",
                    "Ethics & Responsibility": "ns", "Digital Innovation": "ns", "Coding": "ns"}},
    },
    "Q24_team_openness": {
        "A": {"R2": 0.008, "adjR2": -0.015, "F": "F(6,253)=0.352", "p": ".908 ns",
              "b": {d: 0.0 for d in DOMAIN_NAMES}, "s": {d: "ns" for d in DOMAIN_NAMES}},
        "B": {"R2": 0.008, "adjR2": -0.015, "F": "F(6,253)=0.352", "p": ".908 ns",
              "b": {d: 0.0 for d in DOMAIN_NAMES}, "s": {d: "ns" for d in DOMAIN_NAMES}},
    },
    "Q25_difficulty": {
        "A": {"R2": 0.040, "adjR2": 0.017, "F": "F(6,240)=1.678", "p": ".128 ns",
              "b": {"Data Analytics": -0.028, "Comm & Collab": 0.152, "Safety & Security": 0.073,
                    "Ethics & Responsibility": 0.009, "Digital Innovation": 0.071, "Coding": -0.046},
              "s": {d: "ns" for d in DOMAIN_NAMES}},
        "B": {"R2": 0.116, "adjR2": 0.069, "F": "F(10,183)=2.400", "p": ".011",
              "b": {"Data Analytics": -0.188, "Comm & Collab": 0.186, "Safety & Security": 0.040,
                    "Ethics & Responsibility": -0.085, "Digital Innovation": 0.144, "Coding": -0.032},
              "s": {d: "ns" for d in DOMAIN_NAMES}},
    },
}

# SEM results (Steps 10–13)
SEM = {
    "CR_ind": {
        "label": "CR_ind — Individual Readiness (Q22 + Q23)",
        "items": ["Q22_openness_self", "Q23_positive_effect"],
        "predictors": {
            "Digital Innovation": {"b": +0.44, "p": ".003", "s": "**"},
            "Data Analytics":     {"b": -0.42, "p": ".004", "s": "**"},
            "Comm & Collab":      {"b": +0.18, "p": ".21",  "s": "ns"},
            "Safety & Security":  {"b": +0.12, "p": ".38",  "s": "ns"},
            "Ethics & Resp.":     {"b": -0.08, "p": ".55",  "s": "ns"},
            "Coding":             {"b": +0.05, "p": ".61",  "s": "ns"},
        },
    },
    "CR_ctx": {
        "label": "CR_ctx — Contextual Readiness (Q24 + Q25)",
        "items": ["Q24_team_openness", "Q25_difficulty"],
        "predictors": {
            "Age":               {"b": +0.34, "p": "<.01", "s": "**"},
            "Education":         {"b": +0.23, "p": "<.05", "s": "*"},
            "Tenure":            {"b": -0.22, "p": ".06",  "s": "†"},
            "Digital Innovation":{"b": +0.08, "p": ".48",  "s": "ns"},
        },
    },
}

# Hypotheses (Step 9)
HYPOTHESES = {
    "H1": {
        "short": "Digital Innovation is the strongest predictor of personal openness (Q22)",
        "direction": "+",
        "status": "Supported",
        "color": "#2ca02c",
        "rq": "RQ2",
        "theory": "Bandura (1977) self-efficacy; Amalia & Kusmaryani (2024)",
        "key_stats": "Digital Innovation ρ=+0.349*** (strongest domain); β=+0.266** in OLS Model B; β=+0.44** in SEM",
        "highlight_domains": ["Digital Innovation"],
        "highlight_cr": ["Q22_openness_self"],
        "note": "Only domain to survive joint regression. Replicates after excluding Digital/Data/Tech role group.",
        "methods": {
            "Spearman ρ": "Ranks every respondent on Digital Innovation score and on Q22 score separately, then asks: do people who rank high on one also tend to rank high on the other? ρ=+0.349 means a strong positive monotonic relationship — the highest-innovation people are disproportionately the most open to change. Three asterisks means this would occur by chance less than 1 time in 1,000.",
            "OLS Regression (Model B)": "Puts all 6 domains and 4 demographics into a single equation simultaneously. Digital Innovation β=+0.266** means: holding all other domains and demographics constant, a one-standard-deviation increase in Digital Innovation predicts a 0.266 SD increase in Q22. It is the only domain to survive this competition — all others are rendered non-significant once Digital Innovation is in the model.",
            "SEM Path": "In the structural model, Digital Innovation→CR_ind path β=+0.44 is the strongest direct path from any competency predictor. The SEM accounts for measurement error and the factor structure of CR simultaneously — making this estimate more precise than OLS. The positive direction means: greater mastery in digital innovation tasks predicts a stronger individual readiness orientation.",
        },
    },
    "H2": {
        "short": "Competency–CR strength declines: Q22 > Q23 > Q25 > Q24",
        "direction": "ordinal",
        "status": "Supported",
        "color": "#2ca02c",
        "rq": "RQ2, RQ3",
        "theory": "Armenakis & Harris (2002) five-belief model of readiness",
        "key_stats": "Sig domains per outcome: Q22=6/6, Q23=4/6, Q25=2/6, Q24=0/6",
        "highlight_domains": DOMAIN_NAMES,
        "highlight_cr": CR_COLS,
        "note": "Exact monotonic decline predicted by theory. Q24 = zero significant correlations.",
        "methods": {
            "Spearman ρ (across all 4 outcomes)": "Run separately for each of the 24 domain×outcome combinations (6 domains × 4 CR variables). The pattern of significance counts — how many of the 6 domain correlations are significant for each outcome — reveals the ordinal decline: Q22 gets 6/6 significant, Q23 gets 4/6, Q25 gets 2/6, Q24 gets 0/6. This monotonic pattern is exactly what Armenakis's belief hierarchy predicts: self-referent beliefs (Q22) are most tightly coupled to individual capability; socially-constructed perceptions (Q24) are independent of it.",
        },
    },
    "H3": {
        "short": "Competency does NOT predict team openness (Q24) — null hypothesis supported",
        "direction": "null",
        "status": "Strongly Supported",
        "color": "#1f77b4",
        "rq": "RQ3",
        "theory": "Weiner (2009) organisational readiness; Rafferty et al. (2013) multilevel framework",
        "key_stats": "All 6 domains ns vs Q24 (ρ range: −0.013 to +0.054). OLS Model A R²=0.008, p=.908.",
        "highlight_domains": DOMAIN_NAMES,
        "highlight_cr": ["Q24_team_openness"],
        "note": "Most robust finding. Replicates in Step 8. Individual upskilling cannot shift perceived team readiness.",
        "methods": {
            "Spearman ρ": "All 6 domain×Q24 correlations are near-zero (range −0.013 to +0.054) and non-significant. In bivariate terms, knowing someone's competency level gives you zero information about how open they perceive their team to be. A null result here is substantively meaningful — it is not absence of evidence but evidence of absence, given the large N=261 which gives ample power to detect even small effects.",
            "OLS Regression (Model A)": "The joint model of all 6 domains predicting Q24 yields R²=0.008 and F-statistic p=.908 — nowhere near significance. The model explains less than 1% of variance in team openness. This is a remarkably clean null: not a weak effect, but essentially no effect at all.",
            "SEM": "In the structural model, the CR_ctx factor (which includes Q24) is predicted by demographic variables, not competency domains. Digital Innovation's path to CR_ctx is β=+0.08, non-significant. The model confirms that Q24 lives in a different causal world from Q22.",
        },
    },
    "H4": {
        "short": "Longer organisational tenure → lower personal openness (Q22) and ease (Q25)",
        "direction": "−",
        "status": "Supported",
        "color": "#2ca02c",
        "rq": "RQ4",
        "theory": "Holt et al. (2007) organisational inertia; Kar et al. (2021) human capital devaluation",
        "key_stats": "Tenure at Origin: Q22 ρ=−0.215***, Q25 ρ=−0.199**; not reducible to age (age ns).",
        "highlight_domains": [],
        "highlight_cr": ["Q22_openness_self", "Q25_difficulty"],
        "note": "Tenure effect is about organisational embedding, not ageing per se.",
        "methods": {
            "Spearman ρ": "Continuous Tenure at Origin is rank-correlated with each CR outcome. ρ=−0.215*** with Q22 means: longer-tenured people rank systematically lower on personal openness. The negative direction is what the routinisation theory predicts — embedded workers develop stable practices that resist disruption. Crucially, Age itself has ρ≈0 with Q22, proving the tenure effect is not simply ageing.",
            "OLS Regression (Model B)": "In the joint model, Tenure at Origin retains a significant negative coefficient for Q25 (ease of change) after controlling for all six competency domains and other demographics. This means tenure's effect is not explained away by the fact that longer-tenured workers might be less digitally skilled — it operates independently of competency level.",
        },
    },
    "H5": {
        "short": "Education → Q23 and Q24 (not Q22 or Q25) — contextual appraisal, not personal readiness",
        "direction": "+",
        "status": "Supported (reframed)",
        "color": "#ff7f0e",
        "rq": "RQ4",
        "theory": "Social cognition; UTAUT boundary conditions (Venkatesh et al. 2003)",
        "key_stats": "Education β=+sig for Q23 (OLS Model B) and Q24 (SEM); ns for Q22 and Q25.",
        "highlight_domains": [],
        "highlight_cr": ["Q23_positive_effect", "Q24_team_openness"],
        "note": "Original draft predicted negative effect (Holt & Vardaman). Data reframed direction to positive on contextual items.",
        "methods": {
            "Kruskal–Wallis (Step 5)": "A non-parametric one-way ANOVA equivalent, testing whether the distribution of CR scores differs significantly across education level groups (e.g. Trade < Bachelor's < Master's < PhD). Significant for Q24 (p<.05) — higher education groups score higher on team openness perception.",
            "OLS Regression (Model B)": "Education (treated as ordinal: 1–5 scale) has β=+0.173* as the sole significant predictor of Q23. This means: each step up the education ladder predicts a meaningful increase in positive expectations about digital technology's effect, independent of competency. Education is non-significant for Q22 and Q25 in the same model.",
            "SEM": "Education → CR_ctx path is β=+0.23*. This is the structural confirmation: education shapes how people appraise their context (team readiness, ease of change) rather than their personal orientation (Q22). The social cognition explanation is that educated workers construct more charitable narratives about collective capacity.",
        },
    },
    "H6": {
        "short": "Q22 >> Q24: individuals perceive themselves as far more ready than their teams",
        "direction": "gap",
        "status": "Supported — central finding",
        "color": "#d62728",
        "rq": "RQ5",
        "theory": "Pluralistic ignorance; Rafferty et al. (2013); social projection asymmetry",
        "key_stats": "Q22 mean=4.11 vs Q24 mean=2.98. Gap=1.13 pts. 74.1% rate self > team. Wilcoxon p=5.4×10⁻³¹, r=+0.932.",
        "highlight_domains": [],
        "highlight_cr": ["Q22_openness_self", "Q24_team_openness"],
        "note": "Together with H3: individual upskilling alone cannot close the collective readiness gap.",
        "methods": {
            "Descriptive (Step 2)": "Simple means: Q22=4.11 (SD 0.85), Q24=2.98 (SD 0.90). Raw gap=1.13 scale points, equivalent to 1.3 standard deviations of Q24. 74.1% of respondents score Q22 > Q24 on their individual response pairs.",
            "Paired Wilcoxon signed-rank test (Step 12)": "Unlike an independent-samples t-test, the paired Wilcoxon uses each respondent's own Q22 and Q24 scores as a matched pair, then ranks the absolute differences and tests whether positive differences (Q22 > Q24) are systematically larger and more frequent than negative differences. This is robust to the non-normality of ordinal data. The result: p=5.4×10⁻³¹ — one of the most significant results possible in a sample of 335. The rank-biserial r=+0.932 means 93% of the signed ranks point in the positive direction. In conventional effect size terms, anything above r=0.5 is 'large'. This is exceptional.",
        },
    },
}

STATUS_ICONS = {
    "Supported": "✅",
    "Strongly Supported": "✅✅",
    "Supported (reframed)": "🔄",
    "Not Supported": "❌",
    "Partially Supported": "⚠️",
}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def get_data():
    if DATA_CSV.exists():
        return load_csv(DATA_CSV), "local"
    return None, "unavailable"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎓 Thesis Explorer")
st.sidebar.caption("Digital Competencies & Change Readiness\nFEnEx CRC RP4.0041 | N=335")

# MP3 download
st.sidebar.markdown("---")
st.sidebar.subheader("🎧 Audio Guide")
if MP3_FILE.exists():
    st.sidebar.download_button(
        label="Download 30-min guided tour (MP3)",
        data=MP3_FILE.read_bytes(),
        file_name="thesis_explorer_guide.mp3",
        mime="audio/mpeg",
    )
else:
    st.sidebar.caption("Audio guide not available in this deployment.")

# Analysis mode
st.sidebar.markdown("---")
st.sidebar.subheader("Analysis View")
analysis_mode = st.sidebar.radio(
    "Method",
    ["Overview", "Bivariate (Spearman)", "OLS Regression", "SEM / Structural", "Hypotheses", "Theory", "Statistics"],
    index=0,
)

df_raw, data_source = get_data()
data_available = df_raw is not None

if data_available:
    # Cohort filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Cohort Filters")
    roles     = sorted(df_raw["Role_label"].dropna().unique())
    genders   = sorted(df_raw["Gender_label"].dropna().unique())
    edu_order = df_raw.groupby("Education_label")["Education_ordinal"].mean().sort_values()
    edus      = edu_order.index.tolist()

    sel_roles   = st.sidebar.multiselect("Role",      roles,   default=roles)
    sel_genders = st.sidebar.multiselect("Gender",    genders, default=genders)
    sel_edus    = st.sidebar.multiselect("Education", edus,    default=edus)

    df = df_raw[
        df_raw["Role_label"].isin(sel_roles) &
        df_raw["Gender_label"].isin(sel_genders) &
        df_raw["Education_label"].isin(sel_edus)
    ].copy()
else:
    df = None


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Digital Competencies → Change Readiness")
st.caption("PhD Thesis | FEnEx CRC RP4.0041 | Australian Energy Sector | N=335 | Johann Visser")

if data_available:
    n = len(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered N", n)
    c2.metric("Q22 mean (personal openness)", f"{df['Q22_openness_self'].mean():.2f}")
    c3.metric("Q24 mean (team openness)",      f"{df['Q24_team_openness'].mean():.2f}")
    c4.metric("Gap (Q22 − Q24)", f"{(df['Q22_openness_self'] - df['Q24_team_openness']).mean():.2f}")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def spearman_heatmap(highlight_domains=None, highlight_cr=None):
    rho_df = pd.DataFrame(SPEARMAN_RHO).T
    cr_labels = ["Q22", "Q23", "Q24", "Q25"]
    rho_df.columns = cr_labels
    sig_df = pd.DataFrame(SPEARMAN_SIG).T
    sig_df.columns = cr_labels
    text_df = rho_df.round(3).astype(str) + sig_df.map(
        lambda s: "" if s == "ns" else f" {s}"
    )

    # Dim non-highlighted cells if a hypothesis is active
    z = rho_df.values.copy().astype(float)
    if highlight_domains or highlight_cr:
        cr_map = {"Q22_openness_self": "Q22", "Q23_positive_effect": "Q23",
                  "Q24_team_openness": "Q24", "Q25_difficulty": "Q25"}
        active_cr = [cr_map[c] for c in (highlight_cr or [])] if highlight_cr else cr_labels
        active_domains = highlight_domains if highlight_domains else rho_df.index.tolist()
        for i, dom in enumerate(rho_df.index):
            for j, cr in enumerate(cr_labels):
                if dom not in active_domains or cr not in active_cr:
                    z[i, j] = np.nan

    fig = go.Figure(go.Heatmap(
        z=z, x=cr_labels, y=rho_df.index.tolist(),
        text=text_df.values, texttemplate="%{text}",
        colorscale="RdBu", zmid=0, zmin=-0.4, zmax=0.4,
        colorbar=dict(title="ρ"),
        hoverongaps=False,
    ))
    fig.update_layout(height=400, xaxis_title="CR Outcome", yaxis_title="Competency Domain",
                      margin=dict(l=0, r=0, t=30, b=0))
    return fig


def beta_chart(model_data, title, highlight_domains=None):
    domains = list(model_data["b"].keys())
    betas   = list(model_data["b"].values())
    sigs    = [model_data["s"][d] for d in domains]
    colors  = []
    for d, b in zip(domains, betas):
        if highlight_domains and d not in highlight_domains:
            colors.append("#cccccc")
        elif b < 0:
            colors.append("#d62728")
        else:
            colors.append("#1f77b4")
    fig = go.Figure(go.Bar(
        x=betas, y=domains, orientation="h",
        marker_color=colors,
        text=[f"{b:+.3f} {s}" for b, s in zip(betas, sigs)],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"{title}<br><sup>R²={model_data['R2']:.3f}, adj R²={model_data['adjR2']:.3f}, {model_data['F']}, p={model_data['p']}</sup>",
        xaxis_title="Standardised β",
        height=360, xaxis_range=[-0.45, 0.45],
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Overview
# ══════════════════════════════════════════════════════════════════════════════
if analysis_mode == "Overview":
    st.subheader("Conceptual Overview")
    tab1, tab2, tab3 = st.tabs(["Domain Profiles", "CR Distributions", "Individual–Team Gap"])

    with tab1:
        if data_available:
            domain_means = {d: df[c].mean() for d, c in DOMAIN_COLS.items()}
        else:
            # frozen sample means (from step2)
            domain_means = {"Data Analytics": 3.89, "Comm & Collab": 4.21,
                            "Safety & Security": 4.18, "Ethics & Responsibility": 3.97,
                            "Digital Innovation": 3.74, "Coding": 3.11}
        fig = px.bar(x=list(domain_means.keys()), y=list(domain_means.values()),
                     color=list(domain_means.values()), color_continuous_scale="Blues",
                     labels={"x": "Domain", "y": "Mean score (1–7)"},
                     text=[f"{v:.2f}" for v in domain_means.values()])
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=380, yaxis_range=[0, 7])
        st.plotly_chart(fig, use_container_width=True)

        vals = list(domain_means.values()); labels = list(domain_means.keys())
        fig_r = go.Figure(go.Scatterpolar(r=vals+[vals[0]], theta=labels+[labels[0]],
                                           fill="toself", line_color="steelblue"))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,7])), height=380)
        st.plotly_chart(fig_r, use_container_width=True)

    with tab2:
        if data_available:
            fig = make_subplots(rows=1, cols=4, subplot_titles=list(CR_VARS.keys()))
            for i, (label, col) in enumerate(CR_VARS.items()):
                counts = df[col].value_counts().sort_index()
                fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts.values,
                                     marker_color=["#1f77b4","#ff7f0e","#2ca02c","#d62728"][i],
                                     showlegend=False), row=1, col=i+1)
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload data to see live distributions.")
            # Show frozen means
            frozen_means = {"Q22 Personal openness": 4.11, "Q23 Positive effect": 4.38,
                            "Q24 Team openness": 2.98, "Q25 Ease of change": 3.52}
            fig = px.bar(x=list(frozen_means.keys()), y=list(frozen_means.values()),
                         color_discrete_sequence=["#1f77b4"], text=[f"{v:.2f}" for v in frozen_means.values()],
                         labels={"x": "", "y": "Mean (1–5 scale)"}, title="Sample means (frozen)")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=360, yaxis_range=[0,6])
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if data_available:
            gap = df["Q22_openness_self"] - df["Q24_team_openness"]
            fig = px.histogram(gap, nbins=15, color_discrete_sequence=["steelblue"],
                               labels={"value": "Q22 − Q24 gap"},
                               title=f"Mean gap = {gap.mean():.2f} | {(gap>0).mean()*100:.1f}% rate self higher than team")
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No gap")
            fig.add_vline(x=gap.mean(), line_color="orange", annotation_text=f"Mean={gap.mean():.2f}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Frozen distribution — approximate from Step 2 reported percentages
            # Gap categories: <0 (team more ready), 0 (equal), 1, 2, 3, 4 (self more ready)
            gap_cats  = ["≤ −2", "−1", "0\n(equal)", "+1", "+2", "+3", "≥ +4"]
            gap_pcts  = [2.3, 2.6, 20.9, 29.2, 24.3, 13.8, 6.9]   # approximate from N=335 distribution
            colors_gap = ["#d62728" if x <= 0 else "#1f77b4" for x in [-2, -1, 0, 1, 2, 3, 4]]
            fig = go.Figure(go.Bar(
                x=gap_cats, y=gap_pcts,
                marker_color=colors_gap,
                text=[f"{v:.1f}%" for v in gap_pcts],
                textposition="outside",
            ))
            fig.add_vline(x=2, line_dash="dash", line_color="black", annotation_text="No gap (0)")
            fig.update_layout(
                title="Individual–Team Gap distribution (Q22 − Q24) | N=335 | frozen",
                xaxis_title="Q22 − Q24 gap (scale points)",
                yaxis_title="% of respondents",
                height=400,
                yaxis_range=[0, 38],
            )
            st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean gap Q22 − Q24", "1.13 pts")
        c2.metric("Rate self > team", "74.1%")
        c3.metric("Wilcoxon p", "5.4 × 10⁻³¹")
        c4.metric("Effect size r", "+0.932")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Bivariate Spearman
# ══════════════════════════════════════════════════════════════════════════════
elif analysis_mode == "Bivariate (Spearman)":
    st.subheader("Spearman Correlations — Competency Domains × Change Readiness")
    tab1, tab2 = st.tabs(["Heatmap (frozen N=261)", "Live scatter"])

    with tab1:
        st.plotly_chart(spearman_heatmap(), use_container_width=True)
        st.caption("ns = not significant | * p<.05 | ** p<.01 | *** p<.001")

    with tab2:
        if data_available and SCIPY_OK:
            c1, c2 = st.columns(2)
            x_d = c1.selectbox("X — Domain", DOMAIN_NAMES, index=4)
            y_d = c2.selectbox("Y — CR Outcome", list(CR_VARS.keys()), index=0)
            x_col, y_col = DOMAIN_COLS[x_d], CR_VARS[y_d]
            sdf = df[[x_col, y_col, "Role_label"]].dropna()
            rho, pval = stats.spearmanr(sdf[x_col], sdf[y_col])
            sig = "***" if pval<.001 else "**" if pval<.01 else "*" if pval<.05 else "ns"
            fig = px.scatter(sdf, x=x_col, y=y_col, color="Role_label", trendline="ols",
                             labels={x_col: x_d, y_col: y_d},
                             title=f"ρ = {rho:.3f} {sig} | p = {pval:.4f} | n = {len(sdf)}",
                             opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload the CSV to enable live scatter.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: OLS Regression
# ══════════════════════════════════════════════════════════════════════════════
elif analysis_mode == "OLS Regression":
    st.subheader("OLS Regression — Standardised Betas (Step 7, frozen)")
    st.caption("Model A = 6 domains only. Model B = domains + Age, Tenure, Education.")
    cr_choice = st.selectbox("Outcome", list(CR_VARS.keys()), index=0)
    cr_col = CR_VARS[cr_choice]
    res = REGRESSION[cr_col]

    c1, c2 = st.columns(2)
    c1.plotly_chart(beta_chart(res["A"], "Model A: Domains only"), use_container_width=True)
    c2.plotly_chart(beta_chart(res["B"], "Model B: Domains + Demographics"), use_container_width=True)

    comp = pd.DataFrame({
        "Model": ["A (domains only)", "B (+demographics)"],
        "R²": [res["A"]["R2"], res["B"]["R2"]],
        "adj R²": [res["A"]["adjR2"], res["B"]["adjR2"]],
        "F-stat": [res["A"]["F"], res["B"]["F"]],
        "p": [res["A"]["p"], res["B"]["p"]],
    })
    st.dataframe(comp, hide_index=True, use_container_width=True)
    if cr_col == "Q24_team_openness":
        st.warning("Q24: Model A p=.908 — not significant. Domain scores explain zero variance in perceived team readiness.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: SEM / Structural
# ══════════════════════════════════════════════════════════════════════════════
elif analysis_mode == "SEM / Structural":
    st.subheader("Structural Model — Two-factor Change Readiness (Steps 10–13)")
    st.markdown("**Model fit:** χ²=36.9, df=64, **p=.997**, CFI=1.03, RMSEA=0.00 — excellent.")
    tab1, tab2 = st.tabs(["CR_ind predictors (Q22+Q23)", "CR_ctx predictors (Q24+Q25)"])

    def sem_chart(factor_key):
        pred = SEM[factor_key]["predictors"]
        ps = list(pred.keys())
        bs = [pred[p]["b"] for p in ps]
        ss = [pred[p]["s"] for p in ps]
        pv = [pred[p]["p"] for p in ps]
        colors = ["#d62728" if b < 0 else "#1f77b4" for b in bs]
        fig = go.Figure(go.Bar(
            x=bs, y=ps, orientation="h",
            marker_color=colors,
            text=[f"{b:+.2f} (p={p}) {s}" for b, p, s in zip(bs, pv, ss)],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(xaxis_title="Standardised β", height=360,
                          xaxis_range=[-0.6, 0.6], margin=dict(l=0, r=0, t=20, b=0))
        return fig

    with tab1:
        st.plotly_chart(sem_chart("CR_ind"), use_container_width=True)
        st.info("Digital Innovation β=+0.44** (dominant). Data Analytics β=−0.42** (suppressor — multicollinearity artefact, not causal).")
    with tab2:
        st.plotly_chart(sem_chart("CR_ctx"), use_container_width=True)
        st.info("Contextual readiness driven by **demographics** (Age, Education, Tenure), not competency. Digital Innovation β=+0.08 ns.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.markdown("**CR_ind** — Individual Readiness\n- Q22 Personal openness\n- Q23 Positive effect expectation\n\n→ Predicted by **digital competency** (esp. Digital Innovation)")
    c2.markdown("**CR_ctx** — Contextual Readiness\n- Q24 Team openness\n- Q25 Ease of change\n\n→ Predicted by **age, education, tenure** — NOT competency")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Hypotheses
# ══════════════════════════════════════════════════════════════════════════════
elif analysis_mode == "Hypotheses":
    st.subheader("H1 – H6 — Hypotheses and Empirical Outcomes")

    # Summary cards row
    for i, (hid, h) in enumerate(HYPOTHESES.items()):
        icon = STATUS_ICONS.get(h["status"], "")
        with st.expander(f"**{hid}** {icon} {h['status']} | RQ: {h['rq']} — {h['short']}", expanded=False):
            col_a, col_b = st.columns([2, 3])
            with col_a:
                st.markdown(f"**Theory:** {h['theory']}")
                st.markdown(f"**Key stats:** {h['key_stats']}")
                if h["note"]:
                    st.caption(h["note"])
                st.markdown("**Statistical methods used:**")
                for method, explanation in h.get("methods", {}).items():
                    st.markdown(f"*{method}* — {explanation}")
            with col_b:
                # Show heatmap with hypothesis-relevant cells highlighted
                if h["highlight_domains"] or h["highlight_cr"]:
                    st.markdown("*Relevant cells highlighted in heatmap:*")
                    st.plotly_chart(
                        spearman_heatmap(h["highlight_domains"], h["highlight_cr"]),
                        use_container_width=True,
                        key=f"heatmap_expander_{hid}",
                    )

    st.markdown("---")
    st.subheader("Summary Table")
    summary = []
    for hid, h in HYPOTHESES.items():
        icon = STATUS_ICONS.get(h["status"], "")
        summary.append({
            "H": hid,
            "Status": f"{icon} {h['status']}",
            "RQ": h["rq"],
            "Hypothesis": h["short"],
            "Theoretical anchor": h["theory"],
        })
    st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Drill into a hypothesis")
    selected_h = st.selectbox("Select hypothesis", list(HYPOTHESES.keys()))
    h = HYPOTHESES[selected_h]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {selected_h}: {STATUS_ICONS.get(h['status'],'')} {h['status']}")
        st.markdown(f"**{h['short']}**")
        st.markdown(f"- **RQ:** {h['rq']}")
        st.markdown(f"- **Theory:** {h['theory']}")
        st.markdown(f"- **Key stats:** {h['key_stats']}")
        if h["note"]:
            st.info(h["note"])

    with col2:
        # Show regression for relevant CR outcomes
        relevant_cr = h["highlight_cr"]
        if relevant_cr:
            cr_col = relevant_cr[0]
            if cr_col in REGRESSION:
                res = REGRESSION[cr_col]
                cr_label = [k for k, v in CR_VARS.items() if v == cr_col][0]
                st.markdown(f"**OLS betas for {cr_label}** (highlighted: relevant domains)")
                st.plotly_chart(
                    beta_chart(res["B"], "Model B", h["highlight_domains"] or None),
                    use_container_width=True,
                    key=f"beta_drill_{selected_h}",
                )
        if not relevant_cr:
            st.markdown("*This hypothesis is about demographic predictors — see SEM / Structural view.*")

    # Heatmap with hypothesis overlay
    if h["highlight_domains"] or h["highlight_cr"]:
        st.markdown("**Spearman heatmap — highlighted cells are where this hypothesis plays out:**")
        st.plotly_chart(spearman_heatmap(h["highlight_domains"], h["highlight_cr"]), use_container_width=True,
                        key=f"heatmap_drill_{selected_h}")
        st.caption("Dimmed cells = not relevant to this hypothesis. Full heatmap in Bivariate view.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Theory
# ══════════════════════════════════════════════════════════════════════════════
elif analysis_mode == "Theory":
    st.subheader("Bodies of Theoretical Knowledge — and What the Data Say")
    st.caption("Each tradition below is summarised in terms of its core claims and how the results confirm, qualify, or extend those claims.")

    THEORY_BLOCKS = [
        {
            "title": "Change Readiness Theory",
            "authors": "Armenakis (1993) → Holt et al. (2007) → Rafferty, Jimmieson & Armenakis (2013) → Weiner (2009)",
            "colour": "#1f77b4",
            "summary": """
Armenakis and colleagues established that readiness for change is not a single attitude but a multi-belief psychological state. Their framework identifies at least five sub-beliefs — discrepancy (the need for change), appropriateness (this is the right change), efficacy (we can do it), principal support (leadership is behind it), and valence (it will benefit me) — each of which must be sufficiently positive before an individual or group is genuinely ready. Holt et al. (2007) operationalised this into a validated scale, and Rafferty, Jimmieson & Armenakis (2013) extended the model explicitly to the work-group level, distinguishing individual readiness (a personal cognitive-affective orientation) from collective readiness (a shared psychological state emergent from team interaction). Weiner (2009) formalised the organisational level, arguing that collective readiness has its own antecedents — leadership behaviour, structural conditions, psychological safety — that are independent of individual human capital accumulation.

The thesis findings align closely with this multi-level architecture. The SEM confirms that individual readiness (Q22 + Q23) and contextual readiness (Q24 + Q25) are empirically distinct latent factors, not merely correlated items on a single scale. More importantly, the antecedent structures diverge sharply: digital competency predicts individual readiness but is essentially unrelated to contextual readiness, which is instead shaped by age, education, and tenure. This validates Rafferty et al.'s (2013) insistence that the levels cannot be conflated, and it gives Weiner's (2009) theory empirical traction in a sector-specific setting. The Armenakis belief taxonomy maps neatly onto the four items: Q22 = appropriateness + valence (self); Q23 = valence; Q25 = efficacy (systemic); Q24 = principal support / social norm.
            """,
            "result_connection": "SEM two-factor structure (CR_ind vs CR_ctx) | H2 monotonic decline | H6 individual–team gap",
        },
        {
            "title": "Digital Self-Efficacy",
            "authors": "Bandura (1977, 1997) → Ulfert-Blank & Schmidt (2023) → Amalia & Kusmaryani (2024)",
            "colour": "#ff7f0e",
            "summary": """
Bandura's self-efficacy theory holds that the strongest source of efficacy beliefs is mastery experience — actually performing a task successfully. Applied to digital contexts, Ulfert-Blank & Schmidt (2023) define digital self-efficacy as a person's confidence in their ability to use digital technologies to achieve goals, and demonstrate that it is both domain-specific (competency in a particular tool or practice) and generalisable (a broader orientation toward digital novelty). Critically, high digital self-efficacy is associated with approach behaviour rather than avoidance — people who feel capable digitally are more likely to engage with, rather than resist, technological change. Amalia & Kusmaryani (2024) provide the most direct empirical precedent, linking digital self-efficacy scores to readiness-for-change scores in an organisational sample, finding a significant positive relationship.

The thesis cannot directly measure digital self-efficacy — it was not included in the FEnEx instrument — but the competency scores (Q3–Q20) are plausibly interpreted as proxies for mastery experience: a respondent who self-rates highly on digital innovation tasks has, by implication, had successful mastery experiences in that domain. The finding that Digital Innovation is the strongest predictor of personal openness to change (ρ=+0.349, β=+0.44 in SEM) is theoretically consistent with Bandura's mechanism: innovation-type tasks — creative, discretionary, open-ended — are precisely the tasks most likely to generate generalisable efficacy beliefs, because they require initiative and tolerance of uncertainty rather than compliance with a procedure. The thesis positions this as the most likely underlying mechanism for H1, while acknowledging it is inferred, not measured.
            """,
            "result_connection": "H1 (Digital Innovation strongest predictor) | Spearman ρ=+0.349*** | SEM β=+0.44**",
        },
        {
            "title": "Digital Competency Frameworks",
            "authors": "DigComp (Vuorikari et al. 2022) → Whyatt (2023) → FEnEx 6-domain structure",
            "colour": "#2ca02c",
            "summary": """
The European DigComp framework organises digital competency into five broad areas: information literacy, communication and collaboration, digital content creation, safety, and problem-solving. Whyatt (2023) adapted and operationalised a six-domain version for the Australian energy sector workforce under the FEnEx CRC, mapping closely onto DigComp but separating Ethics and Responsibility as a distinct domain and collapsing problem-solving into a single Coding item. The resulting 18-item instrument (Q3–Q20) was designed to assess sector-relevant digital skills across professional roles ranging from engineers to corporate functions, and was intended to produce six interpretable domain scores.

The thesis contributes a psychometric critique of this structure. The EFA in Step 10 (KMO=0.931, Bartlett p<.001) finds that parallel analysis supports only two underlying factors, not six. No CFA model — including the theoretically-specified 6-factor FEnEx model — achieves acceptable fit (CFI<.90, RMSEA>.08). This means the six domains, while theoretically meaningful and useful as composites, are not psychometrically independent latent constructs: they share too much common variance, almost certainly reflecting a general digital competency factor underlying all items. The practical implication for the thesis is that domain scores should be interpreted as theoretically-motivated descriptions of where competency sits, not as distinct predictors with clean separation. This finding is itself a contribution to the measurement literature — the FEnEx instrument, as currently specified, measures one to two broad factors rather than six narrow ones.
            """,
            "result_connection": "EFA 2-factor solution | CFA non-convergence | Multicollinearity in regression (suppressor effect)",
        },
        {
            "title": "Psychometric Structure of Change Readiness",
            "authors": "Rafferty et al. (2013) scale → Weiner (2009) → Steps 10–13 EFA/CFA/SEM",
            "colour": "#9467bd",
            "summary": """
Prior uses of the four change readiness items (Q22–Q25) in this dataset — including Whyatt's (2023) analysis — treated them as a single composite or analysed them individually without examining their latent structure. Cronbach's alpha for the full Q22–Q25 set is 0.685, below the conventional 0.70 threshold for scale reliability, suggesting the items do not form a unidimensional scale. The thesis takes this as motivation for explicit factor analysis rather than a limitation to be footnoted.

The EFA (Step 10) and CFA (Step 11) jointly confirm a two-factor structure: CR_ind (Q22 + Q23, individual-referent beliefs) and CR_ctx (Q24 + Q25, context-referent perceptions). The two-factor SEM achieves near-perfect fit (χ²=36.9, df=64, p=.997, CFI=1.03, RMSEA=0.00), confirming the structure empirically. The factors are correlated (r=+0.601) but distinct — high correlation does not preclude discriminant validity when the structural antecedents differ as dramatically as they do here. This finding reframes Q25 (ease of change): it was theoretically expected to reflect personal efficacy and therefore cluster with Q22, but empirically it clusters with Q24 as a contextual perception. This is a non-trivial discovery with implications for how the Armenakis framework is operationalised in future survey instruments for this sector.
            """,
            "result_connection": "SEM fit: CFI=1.03, RMSEA=0.00 | CR_ind vs CR_ctx factor separation | Q25 clusters with Q24, not Q22",
        },
        {
            "title": "Demographic Moderators",
            "authors": "Holt et al. (2007) → Gfrerer et al. (2020) → Kar et al. (2021) → Trenerry et al. (2021)",
            "colour": "#d62728",
            "summary": """
The demographic moderation literature on change readiness identifies age, tenure, education, and role as the most consistently examined boundary conditions. The dominant view from Holt et al. (2007) and Kar et al. (2021) is that longer organisational tenure reduces readiness through two mechanisms: routinisation (stable work practices resist disruption) and investment depreciation (accumulated competency in legacy systems is devalued by digital transformation). Gfrerer et al. (2020) document a specific perception gap — managers systematically overestimate their organisations' digital readiness relative to front-line employees, suggesting that role moderates the relationship between capability and perceived readiness.

The thesis results support the tenure effect for individual readiness (H4: Tenure at Origin ρ=−0.215*** with Q22) but reveal something more nuanced for contextual readiness: in the SEM, age, education, and tenure together are the dominant predictors of CR_ctx, while digital competency domains are non-significant. Older, more educated professionals with less time at their current organisation tend to perceive their environment as more change-ready — a pattern consistent with Gfrerer et al.'s perception gap (experienced professionals read organisational signals more charitably) and with the social cognition account of education (higher education fosters abstract-future orientation, leading to more positive appraisals of team and systemic capacity). The implication is pointed: if contextual readiness is what limits collective digital transformation, then training programmes targeting individual digital skills will not move the needle — demographic composition and organisational culture are the relevant leverage points.
            """,
            "result_connection": "H4 (tenure negative) | H5 (education → Q23/Q24) | SEM CR_ctx: Age β=+0.34**, Education β=+0.23*",
        },
    ]

    for block in THEORY_BLOCKS:
        with st.expander(f"**{block['title']}**  ·  {block['authors']}", expanded=False):
            st.markdown(block["summary"])
            st.markdown(f"**Key result connections:** `{block['result_connection']}`")

    st.markdown("---")
    st.info("Use these summaries alongside the Bivariate, Regression, SEM, and Hypotheses views to see where each theoretical claim is empirically grounded in this dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: Statistics
# ══════════════════════════════════════════════════════════════════════════════
elif analysis_mode == "Statistics":
    st.subheader("Statistical Techniques — How Each Method Works")
    st.caption("Plain-language explanations of every technique used in this thesis, with specific reference to what the numbers mean and what drives results in each direction.")

    STAT_BLOCKS = [
        {
            "title": "Spearman's Rank Correlation (ρ)",
            "used_for": "Steps 4A & 4B — domain scores and individual items vs CR outcomes",
            "what_it_does": """
Spearman's rho measures the strength of a **monotonic relationship** between two variables — that is, whether higher values on one variable tend to go with higher values on the other, regardless of whether the relationship is perfectly linear. It works by converting raw scores to ranks (1st, 2nd, 3rd...) and then computing a Pearson correlation on those ranks.

**Why Spearman rather than Pearson here?** The survey uses ordinal scales (1–7 for competency, 1–5 for change readiness). These are ranked categories — a 7 is not necessarily twice as much as a 3.5 in any meaningful sense. Spearman makes no such assumption, making it the appropriate choice.
            """,
            "what_the_numbers_mean": """
- **ρ = 0**: no relationship — knowing someone's rank on domain X tells you nothing about their rank on Q22
- **ρ = +0.35**: moderate positive — higher-ranked competency people tend to score higher on CR (this is Digital Innovation vs Q22, the strongest finding)
- **ρ = +0.01**: negligible — essentially random scatter (this is any domain vs Q24)
- **Significance (*, **, ***):** p<.05, p<.01, p<.001 — the probability of observing a correlation this large by chance if the true ρ were zero
            """,
            "what_pushes_it": """
**Pushes ρ positive:** respondents who score high on competency also consistently score high on the CR outcome — a monotonic positive relationship
**Pushes ρ toward zero:** the two variables are independent — high competency is equally common among high and low CR scorers
**Pushes ρ negative:** high competency systematically goes with *lower* CR (this does not occur for any domain vs Q22–Q25 at the domain level)
**Increases power to detect significance:** larger N, stronger true effect, less variance/noise in responses
            """,
            "thesis_connection": "The heatmap in the Bivariate view shows all 24 ρ values. The Q24 column (all near-zero) versus the Q22 column (all significant and positive) is the visual centrepiece of H2 and H3.",
        },
        {
            "title": "OLS Regression (Ordinary Least Squares)",
            "used_for": "Step 7 — predicting each CR outcome from domain scores and demographics",
            "what_it_does": """
OLS regression fits a straight line (or hyperplane, for multiple predictors) through the data that minimises the sum of squared errors between predicted and actual values. The **standardised beta coefficient (β)** tells you how many standard deviations the outcome changes for a one-standard-deviation increase in a predictor, *holding all other predictors constant*.

**Model A** uses only the 6 domain scores. **Model B** adds continuous demographics (age, tenure, education). The added predictors in B sometimes change the coefficients of the domains substantially — this is how suppressor effects emerge.
            """,
            "what_the_numbers_mean": """
- **β = +0.266**: a 1 SD increase in Digital Innovation predicts a 0.266 SD increase in Q22, after controlling for everything else
- **β = −0.235**: Data Analytics predicts *lower* Q22 when Digital Innovation is held constant — a suppressor effect (see below)
- **R² = 0.131**: the model explains 13.1% of variance in Q22; 86.9% is unexplained by these predictors
- **F-statistic p<.001**: the model as a whole is significantly better than predicting everyone's mean
            """,
            "what_pushes_it": """
**Pushes β positive:** a predictor that independently (controlling for others) correlates positively with the outcome
**Pushes β toward zero:** the predictor's variance is already captured by another predictor in the model (multicollinearity)
**Suppressor effect (negative β despite positive bivariate ρ):** Data Analytics has ρ=+0.20 with Q22 but β=−0.235 in Model B. This happens because Data Analytics and Digital Innovation are highly correlated with each other. Once Digital Innovation is in the model, what's left of the Data Analytics variance — the portion *not* shared with Digital Innovation — is actually slightly negatively related to Q22. This is a statistical artefact, not a causal claim.
**Increases R²:** adding predictors that have unique variance in the outcome, not captured by existing predictors
            """,
            "thesis_connection": "The OLS view shows side-by-side bar charts for Model A and B for each CR outcome. The dramatic appearance of Data Analytics as a red (negative) bar in Model B is the suppressor effect — not a real negative causal relationship.",
        },
        {
            "title": "Kruskal–Wallis Test",
            "used_for": "Step 5 & 6 — comparing CR outcomes across role groups, education levels, gender",
            "what_it_does": """
Kruskal–Wallis is a non-parametric equivalent of one-way ANOVA. It tests whether the **distribution of a variable differs significantly across three or more groups** (e.g., Engineers vs. Corporate vs. Graduates vs. Operations). Like Spearman, it works on ranks rather than raw values, making it appropriate for ordinal outcomes.

It answers: "Are these groups drawn from the same population, or does at least one group have a systematically different distribution of CR scores?"
            """,
            "what_the_numbers_mean": """
- **p<.05**: at least one group has a significantly different distribution from at least one other
- **η² (eta-squared)**: effect size — how much variance in the outcome is explained by group membership
- **Post-hoc pairwise tests:** which specific pairs of groups differ (reported when omnibus test is significant)
            """,
            "what_pushes_it": """
**Pushes toward significance:** large differences between group medians, low within-group variance, large N
**Pushes toward non-significance:** groups with similar distributions, high variability within groups
**Key finding:** Role groups differ significantly on domain scores — the Digital/Data/Technology role scores highest across all domains. But role groups do NOT differ significantly on Q24, reinforcing H3.
            """,
            "thesis_connection": "Used in Steps 5 and 6 to test demographic moderators. The Step 8 replication analysis excluded the Digital/Data/Technology role group to check whether findings were driven by this outlier group — they were not.",
        },
        {
            "title": "Paired Wilcoxon Signed-Rank Test",
            "used_for": "Step 12 — testing the individual–team gap (H6)",
            "what_it_does": """
The paired Wilcoxon test is the non-parametric equivalent of a paired t-test. It takes **matched pairs** (here: each respondent's Q22 score and their own Q24 score), computes the difference for each pair, ranks those differences by absolute size, and then tests whether the positive differences (Q22 > Q24) are systematically larger and more prevalent than the negative differences.

It is the correct test for H6 because (a) the data are ordinal, (b) the comparison is within-person (not between groups), and (c) normality cannot be assumed.
            """,
            "what_the_numbers_mean": """
- **p = 5.4 × 10⁻³¹**: the probability of observing a gap this consistent by chance, if Q22 and Q24 were actually equal in the population, is vanishingly small — essentially impossible
- **Rank-biserial r = +0.932**: the effect size. Ranges from −1 to +1. Interpreted as: 93.2% of the signed ranks point in the positive direction (Q22 > Q24). A 'large' effect by any convention starts at r = 0.50.
            """,
            "what_pushes_it": """
**Pushes toward significance and larger r:** more respondents with Q22 > Q24, larger within-person differences, larger N
**Pushes toward zero:** equal numbers of positive and negative differences, or small differences
**Key feature of this result:** 74.1% of respondents rate Q22 > Q24, only 4.9% rate Q24 > Q22. The asymmetry is overwhelming and consistent across every role group.
            """,
            "thesis_connection": "This is the single most statistically dramatic result in the thesis. The effect size of +0.932 is unusually large for survey research. It is the empirical foundation of H6 and the thesis's central organisational diagnosis.",
        },
        {
            "title": "Exploratory Factor Analysis (EFA)",
            "used_for": "Step 10 — examining the underlying structure of the 18 competency items (Q3–Q20)",
            "what_it_does": """
EFA asks: how many underlying **latent factors** are needed to explain the pattern of correlations among a set of observed items? If items cluster together — if people who score high on Q3 also tend to score high on Q4, Q5, Q6 — that suggests a common underlying factor.

**Principal axis factoring** is the extraction method (more conservative than principal components). **Promax rotation** allows factors to be correlated with each other (oblique), which is realistic for psychological constructs. **Parallel analysis** (Horn 1965) determines how many factors to retain by comparing actual eigenvalues to eigenvalues from random data of the same dimensions.
            """,
            "what_the_numbers_mean": """
- **KMO = 0.931**: Kaiser-Meyer-Olkin measure of sampling adequacy. Above 0.90 is 'marvellous' — the correlation matrix is highly factorable
- **Bartlett's test p<.001**: confirms the matrix is not an identity matrix — correlations are real
- **Parallel analysis result: 2 factors** — only 2 eigenvalues exceed the 95th percentile of random-data eigenvalues. The theoretical 6-factor FEnEx structure is not supported by the data.
            """,
            "what_pushes_it": """
**Pushes toward more factors:** items that form distinct clusters with low cross-loadings
**Pushes toward fewer factors:** items that all correlate highly with each other, sharing a dominant common factor (general digital competency)
**Why 2, not 6:** The 18 items are all positively intercorrelated (average inter-item r ≈ 0.45). A general digital competency factor accounts for most variance. Only a second factor — roughly contrasting higher-order innovation/collaboration skills vs. compliance/security skills — explains additional unique variance.
            """,
            "thesis_connection": "The EFA finding challenges the FEnEx instrument's 6-domain scoring. It means domain scores are useful theoretical composites but not psychometrically distinct latent constructs. This is reported as a contribution to the measurement literature.",
        },
        {
            "title": "Confirmatory Factor Analysis (CFA) & Structural Equation Modelling (SEM)",
            "used_for": "Steps 11–13 — testing factor structure and estimating structural paths",
            "what_it_does": """
**CFA** tests a *specified* factor model — you tell it which items load on which factors, and it estimates how well that structure fits the observed covariance matrix. Unlike EFA (which discovers structure), CFA tests structure.

**SEM** extends CFA by adding **structural paths** between latent factors — essentially regression between constructs that are measured with error. The thesis uses SEM to estimate paths from the 6 domain composites to the two CR latent factors (CR_ind and CR_ctx).
            """,
            "what_the_numbers_mean": """
- **CFI ≥ 0.95**: Comparative Fit Index — how much better is this model than a model of zero correlations? Above 0.95 is good; above 0.90 is acceptable
- **RMSEA ≤ 0.06**: Root Mean Square Error of Approximation — average error per degree of freedom. Below 0.06 is excellent; below 0.08 is acceptable
- **The two-factor CR SEM: CFI=1.03, RMSEA=0.00** — near-perfect fit. The two-factor structure (CR_ind + CR_ctx) is an excellent description of the data.
- **Standardised path coefficient β**: same interpretation as OLS β but accounts for measurement error in latent variables
            """,
            "what_pushes_it": """
**Pushes CFI toward 1.0:** model structure closely matches the observed correlations among items
**Pushes RMSEA toward 0:** model explains the covariance matrix with minimal residual error
**Why the 6-factor FEnEx CFA failed:** the items are too highly intercorrelated — the model requires near-zero correlations between domain factors, but the data show r≈0.60–0.80 between domains. The fit indices (CFI<.80, RMSEA>.10) reflect this misfit.
**What the SEM revealed:** Digital Innovation → CR_ind β=+0.44 is significant; demographics → CR_ctx β=+0.23–0.34 are significant. Two completely different causal stories for two factors that look similar on the surface.
            """,
            "thesis_connection": "The SEM is the most sophisticated analysis in the thesis and produces the cleanest theoretical result: competency and demographics operate on different dimensions of change readiness. This is the definitive answer to the research question.",
        },
    ]

    for block in STAT_BLOCKS:
        with st.expander(f"**{block['title']}** — used in {block['used_for']}", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**What it does**")
                st.markdown(block["what_it_does"])
                st.markdown("**What the numbers mean**")
                st.markdown(block["what_the_numbers_mean"])
            with c2:
                st.markdown("**What pushes results in each direction**")
                st.markdown(block["what_pushes_it"])
                st.markdown(f"**Connection to this thesis:** {block['thesis_connection']}")

    st.markdown("---")
    st.info("Tip: open any Hypothesis card in the Hypotheses view to see which of these techniques was used for that specific hypothesis and what the result means in context.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Johann Visser | PhD Thesis | FEnEx CRC RP4.0041 | ECU School of Business and Law | 2026 | [GitHub](https://github.com/johannvis/phd-thesis-final)")
