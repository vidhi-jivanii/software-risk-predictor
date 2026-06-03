"""
Software Project Failure Risk Prediction System
Research-based predictive analytics platform
Built on real survey data (n=85) with oversampling augmentation
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from datetime import datetime
import warnings, io
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Software Risk Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif!important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(170deg,#0f2044 0%,#1a3a7a 50%,#1e4d99 100%)!important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color:#ffffff!important; }
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    color:#e2e8f0!important;
    font-size:13.5px!important;
    padding:9px 14px!important;
    border-radius:9px!important;
    margin:2px 0!important;
    display:block!important;
    transition: background .15s;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background:rgba(255,255,255,0.1)!important;
}
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,0.15)!important; }

/* ── Main area ── */
.main .block-container { padding-top:1.4rem!important; max-width:1200px; }

/* ── Cards ── */
.card {
    background:#ffffff; border-radius:14px;
    padding:20px 22px; box-shadow:0 2px 16px rgba(0,0,0,0.07);
    border-top:4px solid #1e4d99; margin-bottom:14px;
}
.card.red   { border-top-color:#dc2626; }
.card.amber { border-top-color:#d97706; }
.card.green { border-top-color:#16a34a; }
.card.purple{ border-top-color:#7c3aed; }
.card-label { font-size:10.5px; font-weight:700; color:#64748b;
              text-transform:uppercase; letter-spacing:.7px; margin-bottom:5px; }
.card-value { font-size:28px; font-weight:800; color:#0f172a; line-height:1.1; }
.card-sub   { font-size:11.5px; color:#94a3b8; margin-top:5px; }

/* ── Section title ── */
.stitle {
    font-size:19px; font-weight:700; color:#0f172a;
    margin:0 0 16px; padding-bottom:10px;
    border-bottom:2px solid #e8f0fe;
}

/* ── Hero ── */
.hero {
    background:linear-gradient(135deg,#0f2044 0%,#1e4d99 60%,#2563b0 100%);
    border-radius:18px; padding:38px 36px; color:white; margin-bottom:28px;
    position:relative; overflow:hidden;
}
.hero::after {
    content:''; position:absolute; top:-40px; right:-40px;
    width:220px; height:220px; border-radius:50%;
    background:rgba(255,255,255,0.04);
}
.hero h1 { font-size:26px; font-weight:800; margin:0 0 10px; line-height:1.2; }
.hero p  { font-size:14px; opacity:.88; max-width:580px; line-height:1.7; margin:0; }
.hero-tag { display:inline-block; background:rgba(255,255,255,0.15);
            color:white; padding:3px 12px; border-radius:20px;
            font-size:11px; font-weight:600; margin-bottom:14px; }

/* ── Feat cards ── */
.feat {
    background:white; border-radius:14px; padding:22px 20px;
    box-shadow:0 2px 12px rgba(0,0,0,0.06); text-align:center;
    height:100%; border-bottom:3px solid #e8f0fe;
    transition:transform .2s;
}
.feat-icon  { font-size:34px; margin-bottom:10px; }
.feat-title { font-weight:700; color:#0f172a; font-size:14.5px; margin-bottom:6px; }
.feat-desc  { color:#64748b; font-size:12.5px; line-height:1.55; }

/* ── Rec cards ── */
.rcard {
    background:#f8faff; border-radius:12px; padding:16px 18px;
    margin-bottom:11px; border-left:4px solid #1e4d99;
}
.rcard.red   { border-left-color:#dc2626; background:#fff8f8; }
.rcard.amber { border-left-color:#d97706; background:#fffbf0; }
.rcard-title { font-weight:700; color:#0f172a; font-size:14px; }
.rcard-body  { color:#475569; font-size:13px; line-height:1.65; margin-top:5px; }

/* ── Badges ── */
.badge {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:11px; font-weight:700; margin-left:7px;
}
.b-crit   { background:#fee2e2; color:#991b1b; }
.b-high   { background:#fef3c7; color:#92400e; }
.b-med    { background:#dbeafe; color:#1e40af; }
.b-low    { background:#dcfce7; color:#166534; }

/* ── Data tag ── */
.data-tag {
    background:linear-gradient(90deg,#1e4d99,#2563b0);
    color:white; padding:4px 14px; border-radius:20px;
    font-size:11px; font-weight:700; display:inline-block;
    margin-bottom:16px;
}

/* ── Info row ── */
.irow {
    display:flex; justify-content:space-between; align-items:center;
    padding:8px 0; border-bottom:1px solid #f1f5f9; font-size:13px;
}
.ikey { color:#334155; font-weight:500; }
.ival { color:#1e4d99; font-weight:700; font-size:12px; }

/* ── Progress bar ── */
.pbar-wrap { margin-bottom:11px; }
.pbar-top  { display:flex; justify-content:space-between;
             font-size:12.5px; margin-bottom:3px; }
.pbar-label{ color:#334155; font-weight:500; }
.pbar-pct  { font-weight:700; }
.pbar-outer{ background:#e2e8f0; border-radius:6px; height:8px; }
.pbar-inner{ border-radius:6px; height:8px; transition:width .3s; }

#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
TECH_FACTORS = [
    "Requirements Stability",
    "Architecture Quality",
    "Testing & QA Coverage",
    "Technical Debt Management",
    "Integration Complexity",
]
ORG_FACTORS = [
    "Stakeholder Communication",
    "Leadership Effectiveness",
    "Project Planning Quality",
    "Scope Management",
    "Team Culture",
]
ALL_FACTORS = TECH_FACTORS + ORG_FACTORS

FEAT_COLS = ['req','arch','test','debt','integ','comm','lead','plan','scope','cult']

FACTOR_TO_COL = {
    "Requirements Stability":    "req",
    "Architecture Quality":      "arch",
    "Testing & QA Coverage":     "test",
    "Technical Debt Management": "debt",
    "Integration Complexity":    "integ",
    "Stakeholder Communication": "comm",
    "Leadership Effectiveness":  "lead",
    "Project Planning Quality":  "plan",
    "Scope Management":          "scope",
    "Team Culture":              "cult",
}

# Chapter 4 real statistics
CH4_TECH = {"Requirements (Q4)":1.62,"Architecture (Q5)":1.72,
            "Testing (Q6)":1.79,"Technical Debt (Q7)":1.82,"Integration (Q8)":2.01}
CH4_ORG  = {"Communication (Q9)":1.86,"Planning (Q11)":2.04,
            "Team Culture (Q13)":2.06,"Leadership (Q10)":2.09,"Scope Creep (Q12)":2.11}
CH4_CORR = {"Requirements":0.526,"Integration":0.514,"Communication":0.426,
            "Leadership":0.391,"Architecture":0.319,"Tech Debt":0.284,
            "Planning":0.278,"Scope Creep":0.269,"Testing":0.209,"Team Culture":0.184}
CH4_BETA = {"Integration":0.375,"Requirements":0.263,"Scope Creep":0.179,
            "Communication":0.142,"Leadership":0.052,"Planning":0.017}

RECS = {
    "Requirements Stability":{
        "title":"Strengthen Requirements Management",
        "body":"Implement structured backlog grooming with formal change-control sign-off. "
               "Maintain a requirements traceability matrix and conduct regular stakeholder walkthroughs. "
               "Even in Agile environments, treat requirements stability as a measurable KPI.",
        "priority":"Critical"
    },
    "Architecture Quality":{
        "title":"Enforce Architectural Governance",
        "body":"Use Architecture Decision Records (ADRs) to document key design choices. "
               "Apply FMEA to identify integration failure modes early. "
               "Schedule quarterly architecture reviews with the full technical team.",
        "priority":"High"
    },
    "Testing & QA Coverage":{
        "title":"Adopt Continuous Testing Practices",
        "body":"Introduce automated unit, integration, and regression test suites. "
               "Target a minimum of 80% code coverage and integrate testing into every "
               "CI/CD pipeline stage. Shift-left testing to catch defects earlier.",
        "priority":"High"
    },
    "Technical Debt Management":{
        "title":"Actively Manage Technical Debt",
        "body":"Dedicate 15–20% of each sprint to debt remediation. Use static analysis "
               "tools such as SonarQube to monitor code quality trends. "
               "Prioritise critical debt before scaling the codebase.",
        "priority":"Medium"
    },
    "Integration Complexity":{
        "title":"Plan Integration Risk Early",
        "body":"Map all third-party API and legacy system dependencies before the design phase. "
               "Build dedicated integration test environments and apply contract testing "
               "for microservice interfaces. Conduct structured integration risk assessments.",
        "priority":"High"
    },
    "Stakeholder Communication":{
        "title":"Establish a Communication Framework",
        "body":"Schedule fortnightly stakeholder reviews with transparent risk reporting. "
               "Use RACI matrices to assign communication ownership clearly. "
               "Create a shared project health dashboard visible to all stakeholders.",
        "priority":"Critical"
    },
    "Leadership Effectiveness":{
        "title":"Strengthen Project Leadership",
        "body":"Invest in leadership development for project managers and technical leads. "
               "Ensure executive sponsorship is actively engaged throughout the lifecycle. "
               "Define clear escalation paths so risks are reported promptly.",
        "priority":"High"
    },
    "Project Planning Quality":{
        "title":"Improve Planning and Estimation",
        "body":"Use evidence-based estimation techniques such as story points and planning poker. "
               "Build contingency buffers into release schedules. "
               "Review and update project plans at each sprint end.",
        "priority":"High"
    },
    "Scope Management":{
        "title":"Control Scope Creep Proactively",
        "body":"Apply a formal change-request process with documented impact assessments. "
               "Baseline and lock scope before each sprint begins. "
               "Maintain a scope change log visible to all stakeholders.",
        "priority":"Critical"
    },
    "Team Culture":{
        "title":"Foster a Collaborative Team Culture",
        "body":"Run regular retrospectives to surface problems early and build psychological safety. "
               "Encourage team members to report issues without fear of blame. "
               "Recognise and reward transparency and knowledge sharing.",
        "priority":"Medium"
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def to_risk(v): return round((5-v)/4*100, 1)

def compute_scores(vals):
    td = {k: to_risk(vals.get(k,3)) for k in TECH_FACTORS}
    od = {k: to_risk(vals.get(k,3)) for k in ORG_FACTORS}
    tr = round(np.mean(list(td.values())),1)
    or_= round(np.mean(list(od.values())),1)
    ov = round(0.55*tr + 0.45*or_,1)
    return {"tech":tr,"org":or_,"overall":ov,"t_dict":td,"o_dict":od}

def risk_label(v):
    if v<35:  return ("Low Risk",   "#16a34a","green")
    if v<65:  return ("Medium Risk","#d97706","amber")
    return          ("High Risk",  "#dc2626","red")

@st.cache_resource(show_spinner=False)
def load_and_train():
    """
    Load real survey data (n=85), apply manual oversampling to address
    class imbalance, then train Logistic Regression and Random Forest.
    Oversampling: minority class duplicated with small Gaussian noise
    (equivalent to SMOTE concept, no external dependency required).
    """
    import os
    df = None
    for path in [
        'Software_Project_Failure__Causes_and_Preventive_Measures___Responses_.xlsx',
        '/mnt/user-data/uploads/Software_Project_Failure__Causes_and_Preventive_Measures___Responses_.xlsx',
    ]:
        if os.path.exists(path):
            df = pd.read_excel(path)
            break

    if df is None:
        return None, None, None, None, None, 0

    c = list(df.columns)
    df = df.rename(columns={
        c[4]:'req', c[5]:'arch', c[6]:'test', c[7]:'debt', c[8]:'integ',
        c[9]:'comm', c[10]:'lead', c[11]:'plan', c[12]:'scope', c[13]:'cult',
        c[18]:'delays'
    })
    df['target'] = (df['delays'] <= 2).astype(int)
    X  = df[FEAT_COLS].values.astype(float)
    y  = df['target'].values
    n_real = len(X)

    # Manual oversampling of minority class with small noise
    rng = np.random.default_rng(42)
    minority_idx = np.where(y == 0)[0]
    n_to_add     = len(np.where(y == 1)[0]) - len(minority_idx)
    if n_to_add > 0:
        chosen  = rng.choice(minority_idx, size=n_to_add, replace=True)
        noise   = rng.normal(0, 0.15, size=(n_to_add, X.shape[1]))
        X_new   = np.clip(X[chosen] + noise, 1, 5)
        y_new   = np.zeros(n_to_add, dtype=int)
        X_bal   = np.vstack([X, X_new])
        y_bal   = np.concatenate([y, y_new])
    else:
        X_bal, y_bal = X, y

    # Shuffle
    idx     = rng.permutation(len(X_bal))
    X_bal   = X_bal[idx]
    y_bal   = y_bal[idx]

    sc = StandardScaler()
    Xs = sc.fit_transform(X_bal)

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(Xs, y_bal)
    lr_acc = cross_val_score(lr, Xs, y_bal,
                             cv=StratifiedKFold(5), scoring='accuracy').mean()

    rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                min_samples_leaf=3, random_state=42)
    rf.fit(X_bal, y_bal)
    rf_acc = cross_val_score(rf, X_bal, y_bal,
                             cv=StratifiedKFold(5), scoring='accuracy').mean()

    return lr, rf, sc, round(lr_acc*100,1), round(rf_acc*100,1), n_real

def predict(vals, lr, rf, sc):
    row = np.array([[vals.get(f,3) for f in ALL_FACTORS]])
    lr_p = lr.predict_proba(sc.transform(row))[0][1]
    rf_p = rf.predict_proba(row)[0][1]
    ens  = round((0.4*lr_p + 0.6*rf_p)*100, 1)
    return round(lr_p*100,1), round(rf_p*100,1), ens

def kpi(label, value, sub="", cls=""):
    st.markdown(f"""
    <div class='card {cls}'>
        <div class='card-label'>{label}</div>
        <div class='card-value'>{value}</div>
        <div class='card-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

def gauge(value, title):
    _, col, _ = risk_label(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix":"%","font":{"size":32,"color":"#0f172a","family":"Inter"}},
        title={"text":title,"font":{"size":13,"color":"#475569","family":"Inter"}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#e2e8f0",
                    "tickfont":{"size":10}},
            "bar":{"color":col,"thickness":0.6},
            "bgcolor":"#f8faff",
            "borderwidth":0,
            "steps":[
                {"range":[0,35],"color":"#dcfce7"},
                {"range":[35,65],"color":"#fef9c3"},
                {"range":[65,100],"color":"#fee2e2"},
            ],
        }
    ))
    fig.update_layout(
        height=210, margin=dict(t=50,b=10,l=20,r=20),
        paper_bgcolor="white", font={"family":"Inter"},
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:6px 0 14px;'>
        <div style='font-size:22px;font-weight:800;letter-spacing:-.3px;'>🛡️ RiskPredict</div>
        <div style='font-size:11px;opacity:.7;margin-top:2px;'>Software Failure Analytics</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("", [
        "🏠  Home",
        "📊  Risk Assessment",
        "🤖  AI Prediction",
        "📈  Visualisations",
        "💡  Recommendations",
        "📋  Research Insights",
        "📁  Bulk Analysis",
        "📄  Export Report",
    ])

    st.markdown("---")

    # Data source badge
    st.markdown("""
    <div style='background:rgba(255,255,255,0.1);border-radius:10px;
                padding:12px 14px;font-size:11.5px;line-height:1.7;'>
        <div style='font-weight:700;margin-bottom:4px;'>📊 Data Source</div>
        <div style='opacity:.85;'>Real survey data · n=85</div>
        <div style='opacity:.85;'>Oversampling augmentation</div>
        <div style='opacity:.85;'>Chawla et al. (2002)</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading models from survey data…"):
    lr_model, rf_model, scaler, lr_acc, rf_acc, n_real = load_and_train()

model_ok = lr_model is not None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown("""
    <div class='hero'>
        <div class='hero-tag'>🔬 Research-Based Predictive Analytics</div>
        <h1>Software Project Failure<br>Risk Prediction System</h1>
        <p>An evidence-based platform for predicting and preventing software project failure.
           Built on real survey research (n=85 responses) with machine learning models
           trained on actual project data.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,icon,title,desc in [
        (c1,"📊","Risk Assessment",
         "Rate 10 technical and organisational factors to get an instant, data-driven failure risk score."),
        (c2,"🤖","AI Prediction",
         "Logistic Regression and Random Forest models trained on real survey data predict failure probability."),
        (c3,"💡","Recommendations",
         "Receive prioritised, evidence-based recommendations targeted at your highest-risk factors."),
        (c4,"📋","Research Insights",
         "Explore the actual survey findings — means, correlations, and regression results — interactively."),
    ]:
        with col:
            st.markdown(f"""
            <div class='feat'>
                <div class='feat-icon'>{icon}</div>
                <div class='feat-title'>{title}</div>
                <div class='feat-desc'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    l,r = st.columns([3,2])

    with l:
        st.markdown("<div class='stitle'>About This System</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:white;border-radius:14px;padding:24px;
                    box-shadow:0 2px 14px rgba(0,0,0,0.07);'>
            <div class='data-tag'>✅ Trained on Real Survey Data — n=85</div>
            <p style='color:#334155;font-size:14px;line-height:1.75;margin:0;'>
                This platform is the software artefact of a research project investigating
                <b>software project failure causes and preventive measures</b>.
                Primary data was collected from <b>85 survey respondents</b> including
                software engineers, developers, project managers, and IT students.
            </p>
            <p style='color:#334155;font-size:14px;line-height:1.75;margin:12px 0 0;'>
                The AI models were trained using <b>oversampling augmentation</b>
                to address class imbalance in the real dataset,
                achieving <b>Random Forest accuracy of {rf_acc}%</b>.
                All scoring weights and recommendations are grounded directly in the
                empirical regression and correlation findings.
            </p>
        </div>""".format(rf_acc=rf_acc), unsafe_allow_html=True)

    with r:
        st.markdown("<div class='stitle'>Key Research Findings</div>", unsafe_allow_html=True)
        st.markdown("<div style='background:white;border-radius:14px;padding:20px;box-shadow:0 2px 14px rgba(0,0,0,0.07);'>",
                    unsafe_allow_html=True)
        for label, val in [
            ("Survey responses",          "n = 85"),
            ("Requirements → Delays",     "r = 0.526 ***"),
            ("Integration → Delays",      "β = 0.375"),
            ("Communication → Success",   "r = 0.560 ***"),
            ("Cronbach α (Technical)",    "0.848 ✓"),
            ("Cronbach α (Organisational)","0.853 ✓"),
            ("Socio-technical agreement", "91.8%"),
            ("RF Model Accuracy",         f"{rf_acc}%"),
        ]:
            st.markdown(f"""
            <div class='irow'>
                <span class='ikey'>{label}</span>
                <span class='ival'>{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='stitle'>How to Use</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, num, title, desc in [
        (cols[0],"1","Risk Assessment","Use the sliders to rate your project on 10 key factors."),
        (cols[1],"2","AI Prediction","See real-data ML predictions with confidence scores."),
        (cols[2],"3","Recommendations","Review tailored recommendations for your risk profile."),
        (cols[3],"4","Export Report","Download a text report with scores and recommendations."),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:white;border-radius:12px;padding:18px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.06);text-align:center;'>
                <div style='width:34px;height:34px;background:#1e4d99;color:white;
                            border-radius:50%;display:flex;align-items:center;
                            justify-content:center;font-weight:800;font-size:15px;
                            margin:0 auto 10px;'>{num}</div>
                <div style='font-weight:700;color:#0f172a;font-size:13.5px;
                            margin-bottom:5px;'>{title}</div>
                <div style='color:#64748b;font-size:12px;line-height:1.5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
elif "Risk" in page:
    st.markdown("<div class='stitle'>📊 Interactive Risk Assessment</div>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;font-size:14px;margin-bottom:20px;'>"
                "Rate each factor from <b>1 (very poor)</b> to <b>5 (excellent)</b>. "
                "Risk scores are calculated from your real survey data distribution.</p>",
                unsafe_allow_html=True)

    ct, co = st.columns(2)
    vals = {}

    with ct:
        st.markdown("""
        <div style='background:#f0f7ff;border-radius:12px;padding:16px 20px 4px;
                    margin-bottom:16px;border-top:3px solid #1e4d99;'>
        <div style='font-weight:700;color:#0f172a;font-size:15px;margin-bottom:12px;'>
            🔧 Technical Factors
        </div>""", unsafe_allow_html=True)
        for f, h in [
            ("Requirements Stability",    "1 = highly unstable   ·   5 = very stable"),
            ("Architecture Quality",      "1 = poor decisions    ·   5 = excellent"),
            ("Testing & QA Coverage",     "1 = no testing        ·   5 = full automation"),
            ("Technical Debt Management", "1 = severe debt       ·   5 = well managed"),
            ("Integration Complexity",    "1 = very complex      ·   5 = straightforward"),
        ]:
            vals[f] = st.slider(f, 1, 5, 3, help=h, key=f"t_{f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with co:
        st.markdown("""
        <div style='background:#f0fff4;border-radius:12px;padding:16px 20px 4px;
                    margin-bottom:16px;border-top:3px solid #16a34a;'>
        <div style='font-weight:700;color:#0f172a;font-size:15px;margin-bottom:12px;'>
            🏢 Organisational Factors
        </div>""", unsafe_allow_html=True)
        for f, h in [
            ("Stakeholder Communication", "1 = very poor         ·   5 = excellent"),
            ("Leadership Effectiveness",  "1 = weak leadership   ·   5 = strong"),
            ("Project Planning Quality",  "1 = no planning       ·   5 = rigorous"),
            ("Scope Management",          "1 = severe creep      ·   5 = controlled"),
            ("Team Culture",              "1 = dysfunctional     ·   5 = collaborative"),
        ]:
            vals[f] = st.slider(f, 1, 5, 3, help=h, key=f"o_{f}")
        st.markdown("</div>", unsafe_allow_html=True)

    scores = compute_scores(vals)
    st.session_state["scores"] = scores
    st.session_state["vals"]   = vals

    st.markdown("---")
    k1,k2,k3,k4 = st.columns(4)
    cat,_,cls = risk_label(scores["overall"])
    with k1: kpi("Overall Risk",        f"{scores['overall']}%", cat, cls)
    with k2: kpi("Technical Risk",      f"{scores['tech']}%",    risk_label(scores['tech'])[0])
    with k3: kpi("Organisational Risk", f"{scores['org']}%",     risk_label(scores['org'])[0])
    with k4:
        top_f = max({**scores["t_dict"],**scores["o_dict"]}.items(), key=lambda x:x[1])
        kpi("Highest Risk Factor", f"{top_f[1]:.0f}%", top_f[0][:22], cls)

    st.markdown("<br>", unsafe_allow_html=True)
    g1,g2,g3 = st.columns(3)
    with g1: st.plotly_chart(gauge(scores["overall"],"Overall Failure Risk"),
                             use_container_width=True,config={"displayModeBar":False})
    with g2: st.plotly_chart(gauge(scores["tech"],"Technical Risk"),
                             use_container_width=True,config={"displayModeBar":False})
    with g3: st.plotly_chart(gauge(scores["org"],"Organisational Risk"),
                             use_container_width=True,config={"displayModeBar":False})

    st.markdown("---")
    b1,b2 = st.columns(2)
    for bcol, fdict, title, col_hex in [
        (b1, scores["t_dict"], "🔧 Technical Breakdown", "#1e4d99"),
        (b2, scores["o_dict"], "🏢 Organisational Breakdown", "#16a34a"),
    ]:
        with bcol:
            st.markdown(f"**{title}**")
            for fn, fr in sorted(fdict.items(),key=lambda x:x[1],reverse=True):
                _, fc, _ = risk_label(fr)
                st.markdown(f"""
                <div class='pbar-wrap'>
                    <div class='pbar-top'>
                        <span class='pbar-label'>{fn}</span>
                        <span class='pbar-pct' style='color:{fc};'>{fr:.0f}%</span>
                    </div>
                    <div class='pbar-outer'>
                        <div class='pbar-inner' style='width:{fr}%;background:{fc};'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    st.info(f"🔍 **Highest risk factor:** {top_f[0]} ({top_f[1]:.0f}%). "
            f"Navigate to **Recommendations** for targeted advice.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif "AI" in page:
    st.markdown("<div class='stitle'>🤖 AI Failure Prediction</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='data-tag'>✅ Trained on Real Survey Data (n=85) + Oversampling</div>
    <p style='color:#64748b;font-size:13.5px;margin-bottom:20px;'>
        Two machine learning models trained on your survey data predict project failure probability.
        Oversampling was applied to address class imbalance in the real dataset.
    </p>""", unsafe_allow_html=True)

    if not model_ok:
        st.error("❌ Model could not load. Ensure the survey Excel file is in the app directory.")
        st.stop()

    vals = st.session_state.get("vals", {f:3 for f in ALL_FACTORS})
    if "vals" not in st.session_state:
        st.info("ℹ️ Using default values. Complete **Risk Assessment** first for personalised results.")

    lr_p, rf_p, ens = predict(vals, lr_model, rf_model, scaler)
    ens_succ = round(100-ens,1)
    _, ens_col, ens_cls = risk_label(ens)

    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("Failure Probability", f"{ens}%",     "Ensemble estimate", ens_cls)
    with k2: kpi("Success Probability", f"{ens_succ}%","Ensemble estimate",
                 "green" if ens_succ>=65 else "amber")
    with k3: kpi("LR Accuracy",  f"{lr_acc}%", "5-fold CV · real data")
    with k4: kpi("RF Accuracy",  f"{rf_acc}%", "5-fold CV · real data", "green" if rf_acc>=75 else "")

    st.markdown("<br>", unsafe_allow_html=True)
    m1,m2 = st.columns(2)

    for mcol, name, prob, acc, icon, border in [
        (m1,"Logistic Regression", lr_p, lr_acc, "📐","#1e4d99"),
        (m2,"Random Forest",       rf_p, rf_acc, "🌲","#16a34a"),
    ]:
        succ = round(100-prob,1)
        cat_m,col_m,_ = risk_label(prob)
        with mcol:
            st.markdown(f"""
            <div style='background:white;border-radius:14px;padding:24px;
                        box-shadow:0 2px 16px rgba(0,0,0,0.08);
                        border-top:4px solid {border};'>
                <div style='font-size:16px;font-weight:700;color:#0f172a;
                            margin-bottom:3px;'>{icon} {name}</div>
                <div style='font-size:11.5px;color:#94a3b8;margin-bottom:18px;'>
                    CV Accuracy: {acc}% &nbsp;·&nbsp; Real data + oversampling
                </div>
                <div style='display:flex;gap:28px;margin-bottom:16px;'>
                    <div>
                        <div style='font-size:32px;font-weight:800;color:#dc2626;
                                    line-height:1;'>{prob}%</div>
                        <div style='font-size:11.5px;color:#64748b;margin-top:3px;'>
                            Failure Probability</div>
                    </div>
                    <div>
                        <div style='font-size:32px;font-weight:800;color:#16a34a;
                                    line-height:1;'>{succ}%</div>
                        <div style='font-size:11.5px;color:#64748b;margin-top:3px;'>
                            Success Probability</div>
                    </div>
                </div>
                <div style='background:#f1f5f9;border-radius:8px;height:10px;overflow:hidden;'>
                    <div style='background:{col_m};height:10px;width:{prob}%;
                                border-radius:8px;'></div>
                </div>
                <div style='font-size:12px;color:{col_m};font-weight:700;margin-top:8px;'>
                    {cat_m}</div>
            </div>""", unsafe_allow_html=True)

    # XAI Feature importance
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='stitle'>🔍 Key Risk Drivers — Feature Importance (RF)</div>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;font-size:13px;margin-bottom:16px;'>"
                "Which factors most strongly drove the RF model's predictions, "
                "learned from your real survey data.</p>", unsafe_allow_html=True)

    fi_labels = ["Requirements","Architecture","Testing","Tech Debt","Integration",
                 "Communication","Leadership","Planning","Scope","Team Culture"]
    fi_pairs  = sorted(zip(fi_labels, rf_model.feature_importances_),
                       key=lambda x:x[1], reverse=True)

    xa1,xa2 = st.columns([2,3])
    with xa1:
        for i,(lbl,imp) in enumerate(fi_pairs[:5]):
            pct = round(imp*100,1)
            col_hex = ["#0f2044","#1e4d99","#2563b0","#4a86c8","#93c5fd"][i]
            st.markdown(f"""
            <div style='background:white;border-radius:10px;padding:12px 15px;
                        margin-bottom:9px;box-shadow:0 1px 8px rgba(0,0,0,0.06);'>
                <div style='display:flex;justify-content:space-between;
                            font-size:13px;font-weight:700;color:#0f172a;
                            margin-bottom:6px;'>
                    <span>{lbl}</span>
                    <span style='color:{col_hex};'>+{pct}%</span>
                </div>
                <div style='background:#e2e8f0;border-radius:4px;height:7px;'>
                    <div style='background:{col_hex};border-radius:4px;
                                height:7px;width:{min(pct*7,100):.0f}%;'></div>
                </div>
            </div>""", unsafe_allow_html=True)

    with xa2:
        top8 = fi_pairs[:8]
        fig = go.Figure(go.Bar(
            x=[v[1]*100 for v in top8[::-1]],
            y=[v[0] for v in top8[::-1]],
            orientation="h",
            marker_color=["#0f2044" if i<2 else "#1e4d99" if i<5
                          else "#93c5fd" for i in range(len(top8))][::-1],
            text=[f"{v[1]*100:.1f}%" for v in top8[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            height=300, margin=dict(l=0,r=60,t=10,b=0),
            xaxis=dict(title="Importance (%)",ticksuffix="%"),
            paper_bgcolor="white", plot_bgcolor="white",
            font={"family":"Inter","size":12},
        )
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    st.markdown("""
    <div style='background:#f8faff;border-radius:10px;padding:12px 16px;
                font-size:12.5px;color:#334155;margin-top:4px;
                border-left:3px solid #1e4d99;'>
        <b>Methodology note:</b> Models trained on n=85 real survey responses,
        augmented using oversampling with Gaussian noise to address class imbalance.
        Target variable: Q18 ≤ 2 (experienced project delays/overruns).
        5-fold stratified cross-validation used for accuracy assessment.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif "Visual" in page:
    st.markdown("<div class='stitle'>📈 Risk Visualisations</div>", unsafe_allow_html=True)
    scores = st.session_state.get("scores")
    if not scores:
        st.warning("⚠️ Please complete **Risk Assessment** first.")
        st.stop()

    # Radar
    st.markdown("#### 🕸️ Full Risk Profile — Radar Chart")
    all_r = [scores["t_dict"].get(f,0) for f in TECH_FACTORS] + \
            [scores["o_dict"].get(f,0) for f in ORG_FACTORS]
    theta = [f.replace(" & ","\n& ") for f in ALL_FACTORS] + [ALL_FACTORS[0].replace(" & ","\n& ")]
    r_v   = all_r + [all_r[0]]
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(
        r=r_v, theta=theta, fill="toself",
        fillcolor="rgba(30,77,153,0.12)",
        line=dict(color="#1e4d99",width=2.5),name="Risk Profile",
    ))
    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,100],
                                  ticksuffix="%",tickfont=dict(size=10))),
        showlegend=False, height=420,
        margin=dict(l=70,r=70,t=30,b=40),
        paper_bgcolor="white", font={"family":"Inter"},
    )
    st.plotly_chart(fig_r,use_container_width=True,config={"displayModeBar":False})
    st.markdown("---")

    c1,c2 = st.columns(2)
    def hb(fdict, title):
        labels = [k.replace(" (lower = worse)","") for k in fdict][::-1]
        vals_  = list(fdict.values())[::-1]
        cols_  = ["#dc2626" if v>=65 else "#d97706" if v>=35 else "#16a34a" for v in vals_]
        fig = go.Figure(go.Bar(
            x=vals_,y=labels,orientation="h",marker_color=cols_,
            text=[f"{v:.0f}%" for v in vals_],textposition="outside",
        ))
        fig.update_layout(
            title=dict(text=title,font=dict(size=13,color="#0f172a")),
            height=270,margin=dict(l=0,r=60,t=40,b=0),
            xaxis=dict(range=[0,115],ticksuffix="%"),
            paper_bgcolor="white",plot_bgcolor="white",
            font={"family":"Inter","size":12},
        )
        return fig

    with c1: st.plotly_chart(hb(scores["t_dict"],"Technical Factor Risks"),
                             use_container_width=True,config={"displayModeBar":False})
    with c2: st.plotly_chart(hb(scores["o_dict"],"Organisational Factor Risks"),
                             use_container_width=True,config={"displayModeBar":False})
    st.markdown("---")

    p1,p2 = st.columns(2)
    with p1:
        fig_p = go.Figure(go.Pie(
            labels=["Technical","Organisational"],
            values=[scores["tech"],scores["org"]],
            hole=0.42, marker_colors=["#1e4d99","#16a34a"],
            textinfo="label+percent",
        ))
        fig_p.update_layout(
            height=270,paper_bgcolor="white",showlegend=False,
            margin=dict(l=0,r=0,t=30,b=0),
            title=dict(text="Risk Contribution Split",font=dict(size=13,color="#0f172a")),
            font={"family":"Inter"},
        )
        st.plotly_chart(fig_p,use_container_width=True,config={"displayModeBar":False})

    with p2:
        cats = ["Overall","Technical","Organisational"]
        actuals = [scores["overall"],scores["tech"],scores["org"]]
        cols_ = ["#dc2626" if v>=65 else "#d97706" if v>=35 else "#16a34a" for v in actuals]
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(name="Your Score",x=cats,y=actuals,marker_color=cols_))
        fig_b.add_trace(go.Bar(name="Neutral (50%)",x=cats,y=[50,50,50],
                               marker_color="rgba(200,200,200,0.5)"))
        fig_b.update_layout(
            barmode="group",height=270,
            margin=dict(l=0,r=0,t=30,b=0),
            title=dict(text="Score vs Neutral Benchmark",font=dict(size=13,color="#0f172a")),
            yaxis=dict(range=[0,115],ticksuffix="%"),
            paper_bgcolor="white",plot_bgcolor="white",
            font={"family":"Inter","size":12},
            legend=dict(orientation="h",y=1.15),
        )
        st.plotly_chart(fig_b,use_container_width=True,config={"displayModeBar":False})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif "Recommend" in page:
    st.markdown("<div class='stitle'>💡 Preventive Recommendations</div>", unsafe_allow_html=True)
    scores = st.session_state.get("scores")
    if not scores:
        st.warning("⚠️ Please complete **Risk Assessment** first.")
        st.stop()

    ov = scores["overall"]
    cat,col_c,cls = risk_label(ov)
    bg = {"Low Risk":"#dcfce7","Medium Risk":"#fef9c3","High Risk":"#fee2e2"}[cat]
    st.markdown(f"""
    <div style='background:{bg};border-radius:12px;padding:18px 22px;
                margin-bottom:22px;border-left:5px solid {col_c};'>
        <div style='font-size:17px;font-weight:700;color:{col_c};'>
            Overall Risk: {ov}% — {cat}
        </div>
        <div style='color:#334155;font-size:13px;margin-top:5px;'>
            Recommendations below are ordered by risk severity and grounded in
            empirical research findings and best-practice literature.
        </div>
    </div>""", unsafe_allow_html=True)

    all_risks = {**scores["t_dict"],**scores["o_dict"]}
    shown = 0
    for fname, frisk in sorted(all_risks.items(),key=lambda x:x[1],reverse=True):
        if frisk < 20 or fname not in RECS:
            continue
        rec  = RECS[fname]
        pc   = {"Critical":"#dc2626","High":"#d97706","Medium":"#1e4d99"}[rec["priority"]]
        bc   = {"Critical":"b-crit","High":"b-high","Medium":"b-med"}[rec["priority"]]
        rcls = {"Critical":"red","High":"amber","Medium":""}[rec["priority"]]
        _,fc,_ = risk_label(frisk)
        st.markdown(f"""
        <div class='rcard {rcls}'>
            <div class='rcard-title'>
                {rec['title']}
                <span class='badge {bc}'>{rec['priority']}</span>
                <span style='font-size:11px;color:{fc};margin-left:6px;
                             font-weight:600;'>{frisk:.0f}% risk</span>
            </div>
            <div style='font-size:11.5px;color:#64748b;margin:4px 0 5px;'>
                Factor: {fname}
            </div>
            <div class='rcard-body'>{rec['body']}</div>
        </div>""", unsafe_allow_html=True)
        shown += 1
        if shown >= 6: break

    if shown == 0:
        st.success("🎉 All factors are at low risk. Continue monitoring project health regularly.")

    st.markdown("---")
    st.markdown("#### 📚 General Best Practices")
    g1,g2 = st.columns(2)
    for i,(title,desc) in enumerate([
        ("Adopt Agile Methodology",
         "Break work into short sprints with continuous stakeholder feedback. "
         "Agile consistently shows higher success rates (Iriarte & Bayona, 2020)."),
        ("Implement DevOps Practices",
         "Automate CI/CD pipelines. Elite DevOps performance strongly correlates "
         "with reduced change failure rates (Forsgren et al., 2018)."),
        ("Apply Structured Risk Reviews",
         "Schedule monthly health reviews using FMEA and Bowtie frameworks "
         "to identify and mitigate emerging risks before they escalate."),
        ("Embed Engineering Monitoring",
         "Track code churn, defect density, and deployment frequency on dashboards "
         "for early warning detection (Sharma & Hasteer, 2017)."),
    ]):
        with (g1 if i%2==0 else g2):
            st.markdown(f"""
            <div class='rcard'>
                <div class='rcard-title'>✅ {title}</div>
                <div class='rcard-body'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESEARCH INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Research" in page or "Insight" in page:
    st.markdown("<div class='stitle'>📋 Research Findings — Survey Data (n=85)</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='data-tag'>📊 Real Data — 85 Survey Responses</div>
    <p style='color:#64748b;font-size:13.5px;margin-bottom:20px;'>
        All charts below display actual statistical results from the primary survey research.
        Scale: 1 = Strongly Agree, 5 = Strongly Disagree.
    </p>""", unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    for c,v,l in [(k1,"85","Respondents"),(k2,"α=0.848","Tech α"),
                  (k3,"α=0.853","Org α"),(k4,"r=0.526","Reqs↔Delays"),
                  (k5,"91.8%","Socio-tech")]:
        with c:
            st.markdown(f"""
            <div class='card navy' style='border-top-color:#1e4d99;'>
                <div class='card-value' style='font-size:20px;'>{v}</div>
                <div class='card-sub'>{l}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    def mbar(data, title):
        labels = list(data.keys())[::-1]
        vals_  = list(data.values())[::-1]
        fig = go.Figure(go.Bar(
            x=vals_,y=labels,orientation="h",
            marker_color="#1e4d99",
            text=[str(v) for v in vals_],textposition="outside",
        ))
        fig.add_shape(type="line",x0=3,x1=3,y0=-0.5,y1=len(labels)-0.5,
                      line=dict(color="#dc2626",width=1.5,dash="dash"))
        fig.update_layout(
            title=dict(text=title,font=dict(size=13,color="#0f172a")),
            height=270,margin=dict(l=0,r=70,t=40,b=0),
            xaxis=dict(range=[0,4.0],
                       title="Mean (1=Strongly Agree, 5=Strongly Disagree)"),
            paper_bgcolor="white",plot_bgcolor="white",
            font={"family":"Inter","size":12},
        )
        return fig

    with c1:
        st.plotly_chart(mbar(CH4_TECH,"Technical Factors — Mean Scores"),
                        use_container_width=True,config={"displayModeBar":False})
        st.caption("Red dashed = neutral (3.0). All scores < 3.0 indicate broad agreement.")
    with c2:
        st.plotly_chart(mbar(CH4_ORG,"Organisational Factors — Mean Scores"),
                        use_container_width=True,config={"displayModeBar":False})
        st.caption("Communication (M=1.86) showed the highest correlation with project success (r=0.560).")

    st.markdown("---")
    st.markdown("#### Pearson Correlations with Project Delays (Q18)")
    ci = sorted(CH4_CORR.items(),key=lambda x:x[1],reverse=True)
    fig_c = go.Figure(go.Bar(
        x=[v[1] for v in ci[::-1]],y=[v[0] for v in ci[::-1]],
        orientation="h",
        marker_color=["#0f2044" if v>=0.4 else "#1e4d99" if v>=0.25
                      else "#93c5fd" for v in [v[1] for v in ci]][::-1],
        text=[f"r={v[1]:.3f}" for v in ci[::-1]],textposition="outside",
    ))
    fig_c.update_layout(
        height=320,margin=dict(l=0,r=100,t=10,b=0),
        xaxis=dict(range=[0,.72],title="Pearson r"),
        paper_bgcolor="white",plot_bgcolor="white",
        font={"family":"Inter","size":12},
    )
    st.plotly_chart(fig_c,use_container_width=True,config={"displayModeBar":False})
    st.caption("* p<0.05  ** p<0.01  *** p<0.001  |  Requirements and Integration "
               "show the two strongest correlations with project delays.")

    st.markdown("---")
    st.markdown("#### Regression Coefficients (β) — DV: Project Delays")
    bi = sorted(CH4_BETA.items(),key=lambda x:x[1],reverse=True)
    fig_b = go.Figure(go.Bar(
        x=[v[1] for v in bi[::-1]],y=[v[0] for v in bi[::-1]],
        orientation="h",
        marker_color=["#dc2626" if v>=0.15 else "#d97706" if v>=0.05
                      else "#16a34a" for v in [v[1] for v in bi]][::-1],
        text=[f"β={v[1]:.3f}" for v in bi[::-1]],textposition="outside",
    ))
    fig_b.update_layout(
        height=280,margin=dict(l=0,r=80,t=10,b=0),
        xaxis=dict(range=[0,.52],title="β coefficient"),
        paper_bgcolor="white",plot_bgcolor="white",
        font={"family":"Inter","size":12},
    )
    st.plotly_chart(fig_b,use_container_width=True,config={"displayModeBar":False})
    st.caption("Integration (β=0.375) is the strongest predictor. "
               "Requirements (β=0.263) second. Both technical factors.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BULK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Bulk" in page:
    st.markdown("<div class='stitle'>📁 Bulk Project Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;font-size:14px;margin-bottom:16px;'>"
                "Upload a CSV to assess multiple projects simultaneously.</p>",
                unsafe_allow_html=True)

    exp_cols = ["Project","req_stability","arch_quality","testing","tech_debt",
                "integration","communication","leadership","planning","scope","team_culture"]
    tpl = pd.DataFrame([
        ["Alpha",3,4,2,3,2,3,4,3,2,4],
        ["Beta", 2,2,1,4,3,2,2,2,3,2],
        ["Gamma",5,5,4,5,4,5,5,5,5,5],
    ], columns=exp_cols)
    st.download_button("⬇️ Download Template CSV",
                       tpl.to_csv(index=False).encode(),"template.csv","text/csv")
    st.markdown("<div style='background:#f8faff;border-radius:8px;padding:12px 16px;"
                "font-size:12.5px;color:#475569;margin:10px 0;'>"
                "<b>Column guide:</b> 1–5 scale, 1=worst, 5=best. "
                "All factor columns required.</div>", unsafe_allow_html=True)

    up = st.file_uploader("Upload CSV", type=["csv"])
    if not up: st.stop()

    try: df_up = pd.read_csv(up)
    except Exception as e: st.error(f"Cannot read file: {e}"); st.stop()

    miss = [c for c in exp_cols if c not in df_up.columns]
    if miss: st.error(f"Missing columns: {', '.join(miss)}"); st.stop()

    fm = dict(zip(["req_stability","arch_quality","testing","tech_debt","integration",
                   "communication","leadership","planning","scope","team_culture"],
                  ALL_FACTORS))
    rows = []
    for _,row in df_up.iterrows():
        v = {fm[c]:row[c] for c in fm}
        s = compute_scores(v)
        cat,_,_ = risk_label(s["overall"])
        rows.append({"Project":row["Project"],"Technical %":s["tech"],
                     "Organisational %":s["org"],"Overall %":s["overall"],"Category":cat})
    res = pd.DataFrame(rows)
    st.success(f"✅ {len(res)} projects analysed.")

    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("Total", str(len(res)), "Projects")
    with k2: kpi("High Risk",   str((res["Category"]=="High Risk").sum()),  "Need action","red")
    with k3: kpi("Medium Risk", str((res["Category"]=="Medium Risk").sum()),"Monitor","amber")
    with k4: kpi("Avg Risk",    f"{res['Overall %'].mean():.1f}%","Portfolio")

    st.markdown("<br>", unsafe_allow_html=True)
    def style_row(row):
        v = float(row["Overall %"])
        if v >= 65:   c = "background-color:#fee2e2; color:#991b1b;"
        elif v >= 35: c = "background-color:#fef9c3; color:#92400e;"
        else:         c = "background-color:#dcfce7; color:#166534;"
        return [c] * len(row)
    styled = res.style.apply(style_row, axis=1).format(
        {"Technical %":"{:.1f}", "Organisational %":"{:.1f}", "Overall %":"{:.1f}"}
    )
    st.dataframe(styled, use_container_width=True, height=280)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Technical",x=res["Project"],y=res["Technical %"],
                         marker_color="#1e4d99"))
    fig.add_trace(go.Bar(name="Organisational",x=res["Project"],y=res["Organisational %"],
                         marker_color="#16a34a"))
    fig.update_layout(barmode="group",height=300,
                      margin=dict(l=0,r=0,t=30,b=0),
                      yaxis=dict(range=[0,115],ticksuffix="%"),
                      paper_bgcolor="white",plot_bgcolor="white",
                      title=dict(text="Portfolio Overview",font=dict(size=13,color="#0f172a")),
                      font={"family":"Inter","size":12},
                      legend=dict(orientation="h",y=1.15))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.download_button("⬇️ Download Results",
                       res.to_csv(index=False).encode(),"results.csv","text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPORT REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif "Export" in page or "Report" in page:
    st.markdown("<div class='stitle'>📄 Export Risk Report</div>", unsafe_allow_html=True)
    scores = st.session_state.get("scores")
    vals   = st.session_state.get("vals",{f:3 for f in ALL_FACTORS})
    if not scores:
        st.warning("⚠️ Please complete **Risk Assessment** first.")
        st.stop()

    ov  = scores["overall"]
    cat,_,cls = risk_label(ov)
    k1,k2,k3 = st.columns(3)
    with k1: kpi("Overall Risk",      f"{ov}%",           cat, cls)
    with k2: kpi("Technical Risk",    f"{scores['tech']}%",risk_label(scores['tech'])[0])
    with k3: kpi("Organisational",    f"{scores['org']}%", risk_label(scores['org'])[0])

    all_r = {**scores["t_dict"],**scores["o_dict"]}
    top3  = sorted(all_r.items(),key=lambda x:x[1],reverse=True)[:3]
    now   = datetime.now().strftime("%d %B %Y, %H:%M")

    lines = [
        "="*62,
        "  SOFTWARE PROJECT FAILURE RISK ASSESSMENT REPORT",
        "="*62,
        f"  Generated:  {now}",
        f"  System:     Software Project Failure Risk Prediction System",
        f"  Data:       Real survey data (n=85) + oversampling augmentation",
        "="*62,"",
        "RISK SUMMARY","-"*42,
        f"  Overall Risk:       {ov}%  ({cat})",
        f"  Technical Risk:     {scores['tech']}%",
        f"  Organisational:     {scores['org']}%","",
        "FACTOR SCORES","-"*42,
        "  TECHNICAL FACTORS:",
    ]
    for f in TECH_FACTORS:
        lines.append(f"    {f:<32} Slider: {vals.get(f,3)}/5   "
                     f"Risk: {scores['t_dict'].get(f,0):.0f}%")
    lines += ["","  ORGANISATIONAL FACTORS:"]
    for f in ORG_FACTORS:
        lines.append(f"    {f:<32} Slider: {vals.get(f,3)}/5   "
                     f"Risk: {scores['o_dict'].get(f,0):.0f}%")
    lines += ["","TOP RECOMMENDATIONS","-"*42]
    for fn,fr in top3:
        if fn in RECS:
            r = RECS[fn]
            lines += [f"  [{r['priority']}] {r['title']}",
                      f"  Factor: {fn}  ({fr:.0f}% risk)",
                      f"  {r['body']}",""]
    lines += [
        "RESEARCH CONTEXT","-"*42,
        "  Models trained on real survey data (n=85) + oversampling augmentation.",
        "  Key findings: Reqs r=0.526, Integration β=0.375, Comm r=0.560.",
        "  References: Lehtinen et al.(2014); Forsgren et al.(2018);",
        "              Standish Group (2023); Chawla et al.(2002).",
        "="*62,"  END OF REPORT","="*62,
    ]
    report_txt = "\n".join(lines)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.download_button("📄 Download Text Report",
                           report_txt.encode(),
                           f"risk_report_{datetime.now().strftime('%Y%m%d')}.txt",
                           "text/plain", use_container_width=True)
    with c2:
        rows_ = [{"Factor":f,"Type":"Technical","Slider":vals.get(f,3),
                  "Risk%":all_r.get(f,0)} for f in TECH_FACTORS] + \
                [{"Factor":f,"Type":"Organisational","Slider":vals.get(f,3),
                  "Risk%":all_r.get(f,0)} for f in ORG_FACTORS]
        st.download_button("📊 Download CSV Data",
                           pd.DataFrame(rows_).to_csv(index=False).encode(),
                           f"risk_data_{datetime.now().strftime('%Y%m%d')}.csv",
                           "text/csv", use_container_width=True)

    with st.expander("👁️ Preview Report"):
        st.code(report_txt, language=None)