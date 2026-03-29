import streamlit as st
import os, re, json, hashlib, requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PyPDF2
import docx2txt
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HireAI – Resume Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── BASE ── */
html, body { background-color: #080c14 !important; }
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] { background-color: #080c14 !important; }
.stApp > header { background-color: transparent !important; }
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] { background-color: transparent !important; }
section[data-testid="stSidebar"] { background-color: #0d1626 !important; }
html, body, .stApp, p, span, div, label, li {
    font-family: 'DM Sans', sans-serif;
    color: #e8eaf0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.8rem 4rem !important; max-width: 1400px !important; }

/* ── ANIMATED BG ORBS ── */
body::before {
    content: '';
    position: fixed; top: -40%; left: -20%;
    width: 70vw; height: 70vw;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
    animation: drift 18s ease-in-out infinite alternate;
    pointer-events: none; z-index: 0;
}
body::after {
    content: '';
    position: fixed; bottom: -30%; right: -10%;
    width: 55vw; height: 55vw;
    background: radial-gradient(circle, rgba(139,92,246,0.07) 0%, transparent 70%);
    animation: drift2 22s ease-in-out infinite alternate;
    pointer-events: none; z-index: 0;
}
@keyframes drift  { to { transform: translate(6%, 8%); } }
@keyframes drift2 { to { transform: translate(-6%,-8%); } }

/* ── METRIC TILES ── */
[data-testid="metric-container"], [data-testid="stMetric"] {
    background: #0f1a2e !important;
    border: 1px solid rgba(56,189,248,0.22) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.5rem !important;
    transition: border-color .2s;
}
[data-testid="metric-container"]:hover { border-color: rgba(56,189,248,0.5) !important; }
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: .68rem !important; letter-spacing: .14em !important;
    text-transform: uppercase !important; color: #7dd3fc !important;
}
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.1rem !important; font-weight: 800 !important;
    color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
}

/* ── BUTTONS ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    letter-spacing: .04em !important; border-radius: 10px !important;
    background: linear-gradient(120deg,#38bdf8,#818cf8) !important;
    color: #080c14 !important; border: none !important;
    padding: .6rem 1.8rem !important; transition: all .2s !important;
}
.stButton > button:hover {
    opacity: .88 !important; transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(56,189,248,0.25) !important;
}
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid rgba(56,189,248,0.30) !important;
    color: #38bdf8 !important;
}

/* ── INPUTS ── */
input, [data-testid="stTextInput"] input {
    background: #0d1626 !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    -webkit-text-fill-color: #f1f5f9 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .85rem !important;
    caret-color: #38bdf8 !important;
}
input::placeholder { color: #475569 !important; -webkit-text-fill-color: #475569 !important; opacity: 1 !important; }
input:focus { border-color: rgba(56,189,248,0.5) !important; box-shadow: 0 0 0 2px rgba(56,189,248,0.08) !important; background: #0f1e35 !important; }
textarea, .stTextArea textarea, [data-testid="stTextArea"] textarea {
    background: #0d1626 !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    -webkit-text-fill-color: #f1f5f9 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .84rem !important;
    caret-color: #38bdf8 !important;
}
textarea::placeholder,
.stTextArea textarea::placeholder {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
    opacity: 1 !important;
}
textarea::-webkit-input-placeholder { color: #475569 !important; -webkit-text-fill-color: #475569 !important; opacity: 1 !important; }
textarea::-moz-placeholder           { color: #475569 !important; opacity: 1 !important; }
textarea:focus, .stTextArea textarea:focus {
    border-color: rgba(56,189,248,0.5) !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.08) !important;
    background: #0f1e35 !important;
}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(15,23,42,0.6) !important;
    border-radius: 12px !important; padding: .3rem !important;
    border: 1px solid rgba(56,189,248,0.1) !important;
    gap: .3rem !important;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'DM Mono', monospace !important; font-size: .78rem !important;
    letter-spacing: .08em !important; border-radius: 8px !important;
    color: #64748b !important; padding: .45rem 1.2rem !important;
    border: none !important; background: transparent !important;
    transition: all .2s !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(120deg,rgba(56,189,248,0.15),rgba(129,140,248,0.15)) !important;
    color: #38bdf8 !important;
    box-shadow: inset 0 0 0 1px rgba(56,189,248,0.25) !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: rgba(0,0,0,0.2) !important;
    border: 1px dashed rgba(56,189,248,0.35) !important;
    border-radius: 12px !important; padding: .4rem !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzoneInstructions"] button,
[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(120deg,#38bdf8,#818cf8) !important;
    color: #080c14 !important; border: none !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    border-radius: 8px !important; padding: .4rem 1.2rem !important;
}
[data-testid="stFileDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #94a3b8 !important; font-family: 'DM Mono', monospace !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important; overflow: hidden;
    border: 1px solid rgba(56,189,248,0.10) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    background: rgba(56,189,248,0.06) !important;
    border: 1px solid rgba(56,189,248,0.18) !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important; font-size: .8rem !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #38bdf8 !important; }
hr { border-color: rgba(56,189,248,0.08) !important; }

/* ── REUSABLE CLASSES ── */
.lbl {
    font-family: 'DM Mono', monospace;
    font-size: .65rem; letter-spacing: .18em;
    text-transform: uppercase; color: #475569; margin-bottom: .35rem;
}
.sec-h {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.3rem !important; font-weight: 700 !important;
    color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
    letter-spacing: -.01em; display: flex; align-items: center; gap: .6rem;
    margin: 2.5rem 0 1rem; padding: .4rem 0;
}
.sec-h::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg,rgba(56,189,248,.35),transparent);
    margin-left: .4rem;
}
.card {
    background: linear-gradient(160deg,rgba(15,23,42,0.95),rgba(15,23,42,0.8));
    border: 1px solid rgba(56,189,248,0.14); border-radius: 20px;
    padding: 2rem 1.8rem;
    box-shadow: 0 0 40px rgba(56,189,248,0.04), 0 20px 60px rgba(0,0,0,0.5);
    backdrop-filter: blur(20px);
}
.card-sm {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(56,189,248,0.11);
    border-radius: 14px; padding: 1.3rem 1.5rem;
    transition: border-color .2s, box-shadow .2s;
}
.card-sm:hover {
    border-color: rgba(56,189,248,0.28);
    box-shadow: 0 0 20px rgba(56,189,248,0.06);
}
.pill {
    display: inline-block;
    background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.18);
    border-radius: 999px; padding: .2rem .85rem;
    font-family: 'DM Mono', monospace; font-size: .72rem;
    color: #38bdf8; margin: .25rem .2rem; letter-spacing: .06em;
}
.badge-free { background:rgba(100,116,139,0.15);border:1px solid rgba(100,116,139,0.3);color:#94a3b8;border-radius:999px;padding:.15rem .7rem;font-family:'DM Mono',monospace;font-size:.68rem; }
.badge-pro  { background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3); color:#38bdf8;border-radius:999px;padding:.15rem .7rem;font-family:'DM Mono',monospace;font-size:.68rem; }
.badge-biz  { background:rgba(168,85,247,0.12); border:1px solid rgba(168,85,247,0.3); color:#c084fc;border-radius:999px;padding:.15rem .7rem;font-family:'DM Mono',monospace;font-size:.68rem; }
.usage-bg   { background:rgba(255,255,255,0.05);border-radius:999px;height:8px;overflow:hidden;margin:.3rem 0 .5rem; }
.usage-fill { height:100%;border-radius:999px;background:linear-gradient(90deg,#38bdf8,#818cf8); }
.usage-warn { background:linear-gradient(90deg,#f59e0b,#ef4444) !important; }
.sbar-bg    { background:rgba(255,255,255,0.05);border-radius:999px;height:6px;overflow:hidden; }
.sbar-fill  { height:100%;border-radius:999px; }
.price-card {
    background: rgba(15,23,42,0.9); border: 1px solid rgba(56,189,248,0.12);
    border-radius: 18px; padding: 1.8rem; text-align: center;
    transition: all .25s; position: relative; overflow: hidden;
}
.price-card.featured { border-color:rgba(56,189,248,0.45); box-shadow:0 0 40px rgba(56,189,248,0.10); }
.price-card.featured::before {
    content:'MOST POPULAR'; position:absolute; top:14px; right:-22px;
    background:linear-gradient(120deg,#38bdf8,#818cf8); color:#080c14;
    font-family:'DM Mono',monospace; font-size:.52rem; font-weight:700;
    letter-spacing:.1em; padding:.3rem 2.5rem; transform:rotate(35deg);
}
.price-card:hover { transform:translateY(-3px); border-color:rgba(56,189,248,.3); }
.price-amt { font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;color:#f1f5f9; }
.price-per { font-family:'DM Mono',monospace;font-size:.72rem;color:#64748b; }
.price-feat { font-family:'DM Mono',monospace;font-size:.75rem;color:#94a3b8;line-height:1.9;margin:1rem 0;text-align:left; }
.cand-card {
    background:rgba(15,23,42,0.85); border:1px solid rgba(56,189,248,0.10);
    border-radius:16px; padding:1.5rem 1.8rem; margin-bottom:1.2rem;
    backdrop-filter:blur(10px); transition:border-color .2s,box-shadow .2s;
}
.cand-card:hover { border-color:rgba(56,189,248,0.28); box-shadow:0 0 24px rgba(56,189,248,0.07); }
.q-block {
    background:rgba(15,23,42,0.8); border-left:3px solid #818cf8;
    border-radius:0 12px 12px 0; padding:1rem 1.4rem;
    font-family:'DM Mono',monospace; font-size:.8rem; color:#94a3b8; line-height:1.85;
}
.skills-block, .rec-block {
    background:rgba(0,0,0,0.2); border-radius:10px; padding:.9rem 1.1rem;
    font-family:'DM Mono',monospace; font-size:.78rem; color:#94a3b8; line-height:1.75; margin-top:.4rem;
}
.block-title {
    font-family:'DM Mono',monospace; font-size:.63rem; text-transform:uppercase;
    letter-spacing:.18em; color:#475569; margin-bottom:.35rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS & PLAN CONFIG
# ══════════════════════════════════════════════════════════════════════
USERS_FILE = "hireai_users.json"

PLAN_LIMITS = {"free": 3, "demo": 5, "pro": 50, "business": 999999}

PLAN_FEATURES = {
    "free":     {"interview_questions": False, "ai_recommendation": False},
    "demo":     {"interview_questions": False, "ai_recommendation": False},
    "pro":      {"interview_questions": True,  "ai_recommendation": True},
    "business": {"interview_questions": True,  "ai_recommendation": True},
}

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════
for _k, _v in {
    "authenticated": False, "user_email": "", "user_name": "",
    "plan": "free", "scans_used": 0, "razorpay_sub_id": "",
    "page": "auth",   # auth | dashboard | app
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════
# USER DB HELPERS
# ══════════════════════════════════════════════════════════════════════
def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()

def _load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def _save_users(u):
    with open(USERS_FILE, "w") as f:
        json.dump(u, f, indent=2)

def _seed_demo_account():
    """Always ensure demo account exists — called at app startup."""
    users = _load_users()
    # Always force demo account settings (resets plan & features on every restart)
    users["demo@hireai.com"] = {
        "name": "Demo User",
        "password": _hash("demo123"),
        "plan": "demo",       # 5 scans only — encourages real sign-up
        "scans_used": 0,       # reset on every app restart
        "joined": datetime.now().strftime("%Y-%m-%d"),
        "razorpay_sub_id": ""
    }
    _save_users(users)

# Seed demo account every time the app starts
_seed_demo_account()

def _valid_email(e): return bool(re.match(r"[^@]+@[^@]+\.[^@]+", e))

def _persist_scans():
    users = _load_users()
    e = st.session_state.user_email
    if e in users:
        users[e]["scans_used"] = st.session_state.scans_used
        _save_users(users)

def _upgrade_locally(plan):
    users = _load_users()
    e = st.session_state.user_email
    if e in users:
        users[e]["plan"] = plan
        _save_users(users)
    st.session_state.plan = plan

# ══════════════════════════════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════════════════════════════
def do_login(email, password):
    users = _load_users()
    e = email.lower().strip()
    if e not in users:       return False, "No account found with that email."
    if users[e]["password"] != _hash(password): return False, "Incorrect password."
    u = users[e]
    st.session_state.update({
        "authenticated": True, "user_email": e, "user_name": u["name"],
        "plan": u.get("plan","free"), "scans_used": u.get("scans_used",0),
        "razorpay_sub_id": u.get("razorpay_sub_id",""), "page": "app"
    })
    return True, "Welcome back!"

def do_signup(name, email, password):
    users = _load_users()
    e = email.lower().strip()
    if not name.strip():        return False, "Please enter your name."
    if not _valid_email(e):     return False, "Please enter a valid email."
    if len(password) < 6:       return False, "Password must be at least 6 characters."
    if e in users:              return False, "An account already exists with this email."
    users[e] = {
        "name": name.strip(), "password": _hash(password),
        "plan": "free", "scans_used": 0,
        "joined": datetime.now().strftime("%Y-%m-%d"), "razorpay_sub_id": ""
    }
    _save_users(users)
    st.session_state.update({
        "authenticated": True, "user_email": e, "user_name": name.strip(),
        "plan": "free", "scans_used": 0, "page": "app"
    })
    return True, "Account created!"

# ══════════════════════════════════════════════════════════════════════
# RAZORPAY HELPERS
# ══════════════════════════════════════════════════════════════════════

# Plan prices in INR (paise = INR × 100)
PLAN_PRICES_INR = {
    "pro":      {"amount": 1499_00, "name": "HireAI Pro",      "desc": "50 scans/month + AI features"},
    "business": {"amount": 3999_00, "name": "HireAI Business", "desc": "Unlimited scans + all features"},
}

def razorpay_keys_configured():
    """Check if Razorpay keys are properly set in secrets."""
    try:
        k = st.secrets["razorpay"]["key_id"]
        s = st.secrets["razorpay"]["key_secret"]
        return bool(k and s and not k.startswith("rzp_test_your"))
    except Exception:
        return False

def create_razorpay_order(plan: str, email: str):
    """Creates a Razorpay order and returns (order_id, amount, key_id, name)."""
    try:
        key_id     = st.secrets["razorpay"]["key_id"]
        key_secret = st.secrets["razorpay"]["key_secret"]
        price      = PLAN_PRICES_INR[plan]
        r = requests.post(
            "https://api.razorpay.com/v1/orders",
            auth=(key_id, key_secret),
            json={
                "amount":   price["amount"],
                "currency": "INR",
                "receipt":  f"{plan}_{email[:20]}",
                "notes":    {"user_email": email, "plan": plan},
            }
        )
        if r.status_code == 200:
            return r.json()["id"], price["amount"], key_id, price["name"]
        return None, None, None, None
    except Exception:
        return None, None, None, None

def verify_razorpay_payment(order_id, payment_id, signature):
    """Verifies Razorpay payment signature."""
    import hmac, hashlib
    try:
        key_secret = st.secrets["razorpay"]["key_secret"]
        msg        = f"{order_id}|{payment_id}"
        expected   = hmac.new(key_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)
    except Exception:
        return False

def show_razorpay_button(plan: str, email: str, name: str):
    """Opens Razorpay checkout via a user-clickable button inside an iframe."""
    # If Razorpay keys not set — instant demo upgrade
    if not razorpay_keys_configured():
        _upgrade_locally(plan)
        st.success(f"✅ Upgraded to {plan.title()} plan successfully!")
        st.rerun()
        return

    order_id, amount, key_id, plan_name = create_razorpay_order(plan, email)
    if not order_id:
        _upgrade_locally(plan)
        st.success(f"✅ Upgraded to {plan.title()} plan successfully!")
        st.rerun()
        return

    amount_inr = amount // 100
    app_url = st.secrets.get("app_url", "http://localhost:8501")

    import streamlit.components.v1 as components
    rzp_html = f"""
<!DOCTYPE html>
<html>
<head>
<script src="https://checkout.razorpay.com/v1/checkout.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:transparent; display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:120px; font-family:Arial,sans-serif; }}
  #pay-btn {{
    background: linear-gradient(120deg, #38bdf8, #818cf8);
    color: #080c14;
    border: none;
    border-radius: 12px;
    padding: 14px 40px;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    width: 100%;
    letter-spacing: .04em;
    transition: opacity .2s, transform .2s;
    box-shadow: 0 8px 24px rgba(56,189,248,0.35);
  }}
  #pay-btn:hover {{ opacity:.88; transform:translateY(-2px); }}
  #pay-btn:active {{ transform:translateY(0); }}
  #msg {{ font-size:12px; color:#64748b; margin-top:10px; text-align:center; display:none; }}
</style>
</head>
<body>
<button id="pay-btn" onclick="openRazorpay()">
  💳 Pay ₹{amount_inr:,} — Complete Upgrade to {plan_name}
</button>
<div id="msg">⏳ Opening secure payment window...</div>
<script>
function openRazorpay() {{
  document.getElementById('pay-btn').style.opacity = '0.6';
  document.getElementById('msg').style.display = 'block';
  var options = {{
    key:         '{key_id}',
    amount:      '{amount}',
    currency:    'INR',
    name:        'HireAI',
    description: '{plan_name}',
    order_id:    '{order_id}',
    prefill: {{
      name:  '{name}',
      email: '{email}'
    }},
    theme: {{ color: '#38bdf8' }},
    modal: {{
      ondismiss: function() {{
        document.getElementById('pay-btn').style.opacity = '1';
        document.getElementById('msg').style.display = 'none';
      }}
    }},
    handler: function(response) {{
      var url = '{app_url}'
        + '?rzp_order='   + response.razorpay_order_id
        + '&rzp_payment=' + response.razorpay_payment_id
        + '&rzp_sig='     + response.razorpay_signature
        + '&plan={plan}&email={email}';
      window.top.location.href = url;
    }}
  }};
  var rzp = new Razorpay(options);
  rzp.open();
}}
</script>
</body>
</html>
"""
    components.html(rzp_html, height=500, scrolling=False)

def handle_razorpay_callback():
    """Checks URL params after Razorpay redirect and upgrades plan."""
    params = st.query_params
    if "rzp_payment" in params:
        order_id   = params.get("rzp_order","")
        payment_id = params.get("rzp_payment","")
        signature  = params.get("rzp_sig","")
        plan       = params.get("plan","")
        email      = params.get("email","")
        if plan and email:
            # Verify signature (skip in demo if keys not set)
            valid = True
            try:
                valid = verify_razorpay_payment(order_id, payment_id, signature)
            except Exception:
                valid = True  # demo mode — trust it
            if valid:
                _upgrade_locally(plan)
                # save sub id to user record
                u = _load_users()
                if email in u:
                    u[email]["razorpay_sub_id"] = payment_id
                    _save_users(u)
                st.query_params.clear()
                st.success(f"🎉 Payment successful! You are now on the {plan.title()} plan.")
                st.rerun()
            else:
                st.error("⚠️ Payment verification failed. Contact support.")
                st.query_params.clear()

# ══════════════════════════════════════════════════════════════════════
# AI & ML HELPERS
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def call_ai(prompt: str) -> str:
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                     "Content-Type": "application/json"},
            json={"model":"openai/gpt-4o-mini",
                  "messages":[{"role":"user","content":prompt}],
                  "temperature":0.2}
        )
        return r.json()["choices"][0]["message"]["content"] if r.status_code==200 else "⚠️ AI unavailable."
    except Exception as ex:
        return f"⚠️ Error: {ex}"

def extract_text(file) -> str:
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        reader = PyPDF2.PdfReader(file)
        return "".join(p.extract_text() or "" for p in reader.pages)
    return docx2txt.process(file)

def compute_similarity(resume_texts, jd_text):
    model  = load_embed_model()
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    out    = []
    for name, text in resume_texts:
        emb   = model.encode(text, convert_to_tensor=True)
        score = round(util.cos_sim(jd_emb, emb).item() * 100, 1)
        out.append((name, text, score))
    return sorted(out, key=lambda x: x[2], reverse=True)

def extract_skills(text): return call_ai(f"""
Extract the top 8–10 professional skills from this resume.
Return ONLY a clean bullet list, no preamble.
Resume: {text[:2500]}
""")

def generate_questions(jd): return call_ai(f"""
Generate exactly 10 sharp, role-specific interview questions.
Return ONLY numbered questions.
Job Description: {jd}
""")

def generate_recommendation(jd, resume, score): return call_ai(f"""
You are a senior hiring manager. Analyse this candidate concisely.
Match Score: {score}%
Job Description: {jd[:1200]}
Resume: {resume[:1800]}
Return three short sections:
★ STRENGTHS (2–3 bullets)
⚠ GAPS (2–3 bullets)
✅ RECOMMENDATION (1 sentence: Hire / Maybe / Skip + reason)
""")

def score_color(s):
    if s >= 70: return "#22c55e"
    if s >= 45: return "#f59e0b"
    return "#ef4444"

# ══════════════════════════════════════════════════════════════════════
# SHARED NAV BAR
# ══════════════════════════════════════════════════════════════════════
def render_nav(active="app"):
    plan   = st.session_state.plan
    used   = st.session_state.scans_used
    limit  = PLAN_LIMITS.get(plan,3)
    remain = "∞" if limit >= 999999 else str(max(limit-used,0))
    badge  = {"free":"badge-free","pro":"badge-pro","business":"badge-biz"}.get(plan,"badge-free")

    nl, nc, nr = st.columns([1.2, 1, 1])
    with nl:
        st.markdown("""
        <span style="font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:800;
                     background:linear-gradient(120deg,#38bdf8,#818cf8);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     background-clip:text">⚡ HireAI</span>
        """, unsafe_allow_html=True)
    with nc:
        st.markdown(f"""
        <div style="text-align:center;padding:.5rem 0">
            <span class="{badge}">{plan.upper()}</span>
            <span style="font-family:'DM Mono',monospace;font-size:.68rem;
                         color:#475569;margin-left:.5rem">{remain} scans left</span>
        </div>
        """, unsafe_allow_html=True)
    with nr:
        b1, b2 = st.columns(2, gap="small")
        with b1:
            lbl = "📊 Dashboard" if active=="app" else "🚀 Screener"
            if st.button(lbl, key="nav_toggle", use_container_width=True):
                st.session_state.page = "dashboard" if active=="app" else "app"
                st.rerun()
        with b2:
            if st.button("🚪 Sign Out", key="nav_out", use_container_width=True):
                for k in ["authenticated","user_email","user_name","plan","scans_used","page"]:
                    st.session_state[k] = False if k=="authenticated" else ("auth" if k=="page" else "")
                st.rerun()
    st.markdown("<hr style='margin:.6rem 0 1.8rem'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 1: AUTH
# ══════════════════════════════════════════════════════════════════════
def page_auth():
    st.markdown("""
    <div style="text-align:center;padding:3.5rem 0 2rem">
        <p style="font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.28em;
                  text-transform:uppercase;color:#38bdf8;margin-bottom:.8rem">
            ⚡ AI-Powered Hiring Platform
        </p>
        <h1 style="font-family:'Syne',sans-serif;font-size:clamp(2.4rem,5vw,4rem);
                   font-weight:800;letter-spacing:-.03em;
                   background:linear-gradient(120deg,#38bdf8,#818cf8,#c084fc);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   background-clip:text;margin:0 0 .6rem">
            HireAI
        </h1>
        <p style="font-family:'DM Sans',sans-serif;font-size:1.05rem;color:#64748b;margin:0">
            Screen smarter. Hire faster. Start free.
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        tab_in, tab_up = st.tabs(["  Sign In  ", "  Create Account  "])

        # ── LOGIN ──────────────────────────────────────────────
        with tab_in:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="lbl">Email Address</p>', unsafe_allow_html=True)
            li_email = st.text_input("E", key="li_e", label_visibility="collapsed",
                                     placeholder="you@company.com")
            st.markdown('<p class="lbl">Password</p>', unsafe_allow_html=True)
            li_pass = st.text_input("P", key="li_p", type="password",
                                    label_visibility="collapsed", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In  →", use_container_width=True, key="btn_li"):
                if li_email and li_pass:
                    ok, msg = do_login(li_email, li_pass)
                    if ok:   st.success(msg); st.rerun()
                    else:    st.error(msg)
                else:
                    st.warning("Please fill in all fields.")

            st.markdown("""
            <p style="font-family:'DM Mono',monospace;font-size:.7rem;color:#334155;
                      text-align:center;margin-top:.9rem">
                Demo → <span style="color:#38bdf8">demo@hireai.com</span> /
                <span style="color:#38bdf8">demo123</span>
                <span style="color:#f59e0b;margin-left:.4rem">(5 free scans)</span>
            </p>
            """, unsafe_allow_html=True)

        # ── SIGNUP ─────────────────────────────────────────────
        with tab_up:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="lbl">Full Name</p>', unsafe_allow_html=True)
            su_name = st.text_input("N", key="su_n", label_visibility="collapsed",
                                    placeholder="Jane Smith")
            st.markdown('<p class="lbl">Email Address</p>', unsafe_allow_html=True)
            su_email = st.text_input("E2", key="su_e", label_visibility="collapsed",
                                     placeholder="you@company.com")
            st.markdown('<p class="lbl">Password</p>', unsafe_allow_html=True)
            su_pass = st.text_input("P2", key="su_p", type="password",
                                    label_visibility="collapsed", placeholder="Min 6 characters")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Free Account  →", use_container_width=True, key="btn_su"):
                ok, msg = do_signup(su_name, su_email, su_pass)
                if ok:   st.success(msg); st.rerun()
                else:    st.error(msg)
            st.markdown("""
            <p style="font-family:'DM Mono',monospace;font-size:.68rem;color:#334155;
                      text-align:center;margin-top:.8rem">
                Free plan · No credit card · 3 scans/month
            </p>
            """, unsafe_allow_html=True)

    # Feature pills
    st.markdown("""
    <div style="text-align:center;margin-top:2.5rem">
        <span class="pill">⚡ Semantic AI Matching</span>
        <span class="pill">🔍 Skill Extraction</span>
        <span class="pill">💬 Interview Questions</span>
        <span class="pill">📋 Hire Recommendations</span>
        <span class="pill">📥 CSV Export</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 2: DASHBOARD
# ══════════════════════════════════════════════════════════════════════
def page_dashboard():
    handle_razorpay_callback()   # check if returning from Razorpay payment
    render_nav("dashboard")

    plan   = st.session_state.plan
    used   = st.session_state.scans_used
    limit  = PLAN_LIMITS.get(plan, 3)
    name   = st.session_state.user_name
    email  = st.session_state.user_email
    pct    = min(int((used/limit)*100),100) if limit < 999999 else 3
    badge  = {"free":"badge-free","pro":"badge-pro","business":"badge-biz"}.get(plan,"badge-free")
    warn   = "usage-warn" if pct >= 80 else ""

    # Welcome
    st.markdown(f"""
    <div style="margin-bottom:1.5rem">
        <p style="font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.2em;
                  text-transform:uppercase;color:#64748b;margin:0">Welcome back</p>
        <h2 style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
                   letter-spacing:-.02em;color:#f1f5f9;margin:.2rem 0 .4rem">{name}</h2>
        <span class="{badge}">{plan.upper()} PLAN</span>
        <span style="font-family:'DM Mono',monospace;font-size:.68rem;color:#475569;
                     margin-left:.7rem">{email}</span>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Scans Used",      used)
    k2.metric("Scans Remaining", "∞" if limit>=999999 else max(limit-used,0))
    k3.metric("Monthly Limit",   "∞" if limit>=999999 else limit)
    k4.metric("Current Plan",    plan.title())

    # Usage bar
    st.markdown('<div class="sec-h">📊 Monthly Usage</div>', unsafe_allow_html=True)
    if limit >= 999999:
        st.markdown("""<p style="font-family:'DM Mono',monospace;font-size:.8rem;color:#22c55e">
            ✦ Unlimited scans — Business plan active</p>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="margin-bottom:.4rem;display:flex;justify-content:space-between">
            <span style="font-family:'DM Mono',monospace;font-size:.78rem;color:#94a3b8">
                {used} of {limit} scans used this month</span>
            <span style="font-family:'DM Mono',monospace;font-size:.78rem;
                         color:{'#ef4444' if pct>=80 else '#38bdf8'}">{pct}%</span>
        </div>
        <div class="usage-bg"><div class="usage-fill {warn}" style="width:{pct}%"></div></div>
        """, unsafe_allow_html=True)
        if pct >= 100: st.error("🚫 Monthly limit reached. Upgrade to continue screening.")
        elif pct >= 80: st.warning(f"⚠️ Almost at your limit — {limit-used} scans remaining.")

    # Account info
    st.markdown('<div class="sec-h">👤 Account Details</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(f"""<div class="card-sm">
            <p class="lbl">Full Name</p>
            <p style="font-family:'DM Sans',sans-serif;font-size:1rem;color:#e2e8f0;margin:0">{name}</p>
        </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown(f"""<div class="card-sm">
            <p class="lbl">Email Address</p>
            <p style="font-family:'DM Mono',monospace;font-size:.88rem;color:#e2e8f0;margin:0">{email}</p>
        </div>""", unsafe_allow_html=True)

    # Quick actions
    st.markdown("<br>", unsafe_allow_html=True)
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        if st.button("🚀 Start Screening", use_container_width=True, key="d_goapp"):
            st.session_state.page = "app"; st.rerun()
    with qa2:
        if st.button("🔄 Reset Usage (Demo)", use_container_width=True, key="d_reset"):
            users = _load_users()
            if email in users: users[email]["scans_used"] = 0; _save_users(users)
            st.session_state.scans_used = 0
            st.success("Usage reset!"); st.rerun()
    with qa3:
        if st.button("🚪 Sign Out", use_container_width=True, key="d_out"):
            for k in ["authenticated","user_email","user_name","plan","scans_used","page"]:
                st.session_state[k] = False if k=="authenticated" else ("auth" if k=="page" else "")
            st.rerun()

    # ── PRICING ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-h">💳 Plans & Billing</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.8rem">
        <p style="font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.2em;
                  text-transform:uppercase;color:#38bdf8;margin:0 0 .4rem">Simple Pricing</p>
        <h3 style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
                   color:#f1f5f9;margin:0">Choose your plan</h3>
    </div>
    """, unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3, gap="medium")

    # FREE
    with pc1:
        cur = plan == "free"
        st.markdown(f"""<div class="price-card" style="border-color:{'rgba(100,116,139,0.4)' if cur else 'rgba(56,189,248,0.1)'}">
            <p style="font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:#64748b">Free</p>
            <div class="price-amt">$0</div><div class="price-per">/ month</div>
            <div class="price-feat">✦ 3 resume scans/month<br>✦ AI skill extraction<br>✦ Semantic matching<br>✦ CSV export<br>✗ Interview questions<br>✗ AI recommendations</div>
        </div>""", unsafe_allow_html=True)
        if cur: st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:.7rem;color:#64748b;text-align:center;margin-top:.5rem">✓ Current Plan</p>', unsafe_allow_html=True)

    # PRO
    with pc2:
        cur = plan == "pro"
        featured_cls = "" if cur else "featured"
        st.markdown(f"""<div class="price-card {featured_cls}" style="border-color:{'rgba(56,189,248,0.5)' if cur else ''}">
            <p style="font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:#38bdf8">Pro</p>
            <div class="price-amt">₹1,499</div><div class="price-per">/ month</div>
            <div class="price-feat">✦ 50 resume scans/month<br>✦ AI skill extraction<br>✦ Semantic matching<br>✦ CSV export<br>✦ Interview questions<br>✦ AI recommendations</div>
        </div>""", unsafe_allow_html=True)
        if cur:
            st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:.7rem;color:#38bdf8;text-align:center;margin-top:.5rem">✓ Current Plan</p>', unsafe_allow_html=True)
        elif plan in ("free","demo"):
            if st.button("💳 Upgrade to Pro →", key="up_pro", use_container_width=True):
                show_razorpay_button("pro", email, st.session_state.user_name)

    # BUSINESS
    with pc3:
        cur = plan == "business"
        st.markdown(f"""<div class="price-card" style="border-color:{'rgba(168,85,247,0.5)' if cur else 'rgba(56,189,248,0.1)'}">
            <p style="font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:#c084fc">Business</p>
            <div class="price-amt">₹3,999</div><div class="price-per">/ month</div>
            <div class="price-feat">✦ Unlimited scans<br>✦ AI skill extraction<br>✦ Semantic matching<br>✦ CSV export<br>✦ Interview questions<br>✦ AI recommendations<br>✦ Priority support</div>
        </div>""", unsafe_allow_html=True)
        if cur:
            st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:.7rem;color:#c084fc;text-align:center;margin-top:.5rem">✓ Current Plan</p>', unsafe_allow_html=True)
        elif plan != "business":
            if st.button("💳 Upgrade to Business →", key="up_biz", use_container_width=True):
                show_razorpay_button("business", email, st.session_state.user_name)

    # Footer
    st.markdown("""
    <div style="text-align:center;margin-top:4rem;padding-top:2rem;
                border-top:1px solid rgba(56,189,248,0.07)">
        <span style="font-family:'DM Mono',monospace;font-size:.65rem;
                     letter-spacing:.14em;color:#1e293b">
            HIREAI · AI-POWERED RESUME INTELLIGENCE · BUILT WITH ♥
        </span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 3: MAIN SCREENING APP
# ══════════════════════════════════════════════════════════════════════
def page_app():
    render_nav("app")

    plan  = st.session_state.plan
    used  = st.session_state.scans_used
    limit = PLAN_LIMITS.get(plan, 3)
    name  = st.session_state.user_name
    feats = PLAN_FEATURES.get(plan, {})
    can_scan = limit >= 999999 or used < limit

    # ── HERO + UPLOAD PANEL ───────────────────────────────────────────
    col_hero, col_panel = st.columns([1.15, 1], gap="large")

    with col_hero:
        badge = {"free":"badge-free","pro":"badge-pro","business":"badge-biz"}.get(plan,"badge-free")
        st.markdown(f"""
        <div style="padding:1.5rem 0 1rem">
            <p style="font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.22em;
                      text-transform:uppercase;color:#38bdf8;margin:0 0 .8rem">
                ⚡ Powered by AI · Built for modern hiring
            </p>
            <h1 style="font-family:'Syne',sans-serif;font-size:clamp(2rem,4vw,3.2rem);
                       font-weight:800;letter-spacing:-.03em;line-height:1.1;
                       color:#f1f5f9;margin:0 0 .7rem">
                Screen smarter.<br>
                <span style="background:linear-gradient(120deg,#38bdf8,#818cf8,#c084fc);
                             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                             background-clip:text">Hire faster.</span>
            </h1>
            <p style="font-family:'DM Sans',sans-serif;font-size:.95rem;color:#64748b;
                      line-height:1.65;max-width:480px;margin-bottom:1.4rem">
                Hello <strong style="color:#94a3b8">{name}</strong> — upload resumes and a job
                description to rank every candidate with semantic AI matching.
            </p>
            <span class="pill">⚡ Semantic Matching</span>
            <span class="pill">🔍 Skill Extraction</span>
            <span class="pill">📥 CSV Export</span>
            {"<span class='pill'>💬 Interview Questions</span>" if feats.get("interview_questions") else ""}
            {"<span class='pill'>📋 AI Recommendations</span>" if feats.get("ai_recommendation") else ""}
        </div>
        """, unsafe_allow_html=True)

    with col_panel:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="lbl">📂 Resume Upload (PDF / DOCX)</p>', unsafe_allow_html=True)
        resume_files = st.file_uploader("Resumes", type=["pdf","docx"],
                                        accept_multiple_files=True, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="lbl">📝 Job Description</p>', unsafe_allow_html=True)
        jd_input = st.text_area("JD", height=180, label_visibility="collapsed",
                                 placeholder="Paste full job description here…")
        st.markdown("<br>", unsafe_allow_html=True)

        if not can_scan:
            st.error("🚫 Monthly scan limit reached. Upgrade your plan.")
            if st.button("💳 View Plans", use_container_width=True, key="gate_plan"):
                st.session_state.page = "dashboard"; st.rerun()
            analyze = False
        else:
            left_txt = "" if limit>=999999 else f" ({limit-used} scans left)"
            analyze = st.button(f"⚡  Analyze Candidates{left_txt}", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Free plan notice
    if plan == "free":
        st.markdown("""
        <div style="background:rgba(56,189,248,0.05);border:1px solid rgba(56,189,248,0.14);
                    border-radius:10px;padding:.8rem 1.2rem;margin:.6rem 0;
                    font-family:'DM Mono',monospace;font-size:.75rem;color:#64748b">
            ℹ️ Free plan includes basic matching & skill extraction.
            <strong style="color:#38bdf8">Upgrade to Pro ($19/mo)</strong>
            to unlock interview questions & AI hiring recommendations.
        </div>
        """, unsafe_allow_html=True)

    # ── RESULTS ──────────────────────────────────────────────────────
    if analyze:
        if not resume_files or not jd_input.strip():
            st.warning("⚠️  Please upload at least one resume and provide a job description.")
            st.stop()

        with st.spinner("🔍  Running semantic AI analysis…"):
            resume_texts = [(f.name, extract_text(f)) for f in resume_files]
            results      = compute_similarity(resume_texts, jd_input)
            questions    = generate_questions(jd_input) if feats.get("interview_questions") else None

        # increment usage
        st.session_state.scans_used += 1
        _persist_scans()

        scores    = [r[2] for r in results]
        avg_score = round(float(np.mean(scores)), 1)
        top_score = max(scores)

        # KPIs
        st.markdown('<div class="sec-h">📊 Screening Dashboard</div>', unsafe_allow_html=True)
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Resumes Uploaded",  len(resume_files))
        k2.metric("Candidates Ranked", len(results))
        k3.metric("Top Match Score",   f"{top_score}%")
        k4.metric("Avg Match Score",   f"{avg_score}%")

        # Top candidate banner
        top = results[0]
        col = score_color(top[2])
        st.markdown(f"""
        <div style="background:linear-gradient(120deg,rgba(56,189,248,0.09),rgba(129,140,248,0.09));
                    border:1px solid rgba(56,189,248,0.22);border-radius:14px;
                    padding:1.2rem 1.6rem;display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem">
            <div style="font-size:2rem">🏆</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:{col}">{top[0]}</div>
                <div style="font-family:'DM Mono',monospace;font-size:.75rem;color:#64748b">
                    Best overall match · {top[2]}% similarity
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Score chart
        st.markdown('<div class="sec-h">📈 Match Score Comparison</div>', unsafe_allow_html=True)
        names  = [r[0] for r in results]
        values = [r[2] for r in results]
        bcolors = [score_color(v) for v in values]
        fig, ax = plt.subplots(figsize=(9, max(3, len(names)*0.6)))
        fig.patch.set_facecolor("#080c14"); ax.set_facecolor("#0d1626")
        bars = ax.barh(names, values, color=bcolors, height=0.55, zorder=3)
        ax.set_xlim(0,100)
        ax.set_xlabel("Match Score (%)", color="#64748b", fontsize=9, fontfamily="monospace")
        ax.tick_params(colors="#94a3b8", labelsize=9); ax.spines[:].set_visible(False)
        ax.xaxis.grid(True, color=(1,1,1,0.04), zorder=0); ax.set_axisbelow(True)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width()+.8, bar.get_y()+bar.get_height()/2,
                    f"{val}%", va="center", ha="left", color="#94a3b8", fontsize=8, fontfamily="monospace")
        ax.set_title("Candidate Similarity Scores", color="#f1f5f9", fontsize=10, fontfamily="monospace", pad=12)
        plt.tight_layout(); st.pyplot(fig)

        # Ranking table
        st.markdown('<div class="sec-h">🗂 Candidate Ranking</div>', unsafe_allow_html=True)
        df = pd.DataFrame([(i+1,r[0],f"{r[2]}%") for i,r in enumerate(results)],
                          columns=["Rank","Candidate","Match Score"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("📥 Export CSV", df.to_csv(index=False).encode("utf-8"),
                           "candidate_ranking.csv","text/csv")

        # Interview questions (Pro+)
        if questions:
            st.markdown("""<div class="sec-h">💬 Interview Questions
                <span style="font-family:'DM Mono',monospace;font-size:.6rem;color:#38bdf8;
                             background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);
                             border-radius:4px;padding:.08rem .4rem">PRO</span>
            </div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="q-block">{questions.replace(chr(10),"<br>")}</div>',
                        unsafe_allow_html=True)
        elif plan == "free":
            st.markdown("""
            <div style="background:rgba(56,189,248,0.04);border:1px dashed rgba(56,189,248,0.18);
                        border-radius:12px;padding:1rem 1.4rem;text-align:center;
                        font-family:'DM Mono',monospace;font-size:.8rem;color:#475569;margin:1.5rem 0">
                🔒 Interview questions — available on <strong style="color:#38bdf8">Pro & Business</strong> plans
            </div>
            """, unsafe_allow_html=True)

        # Candidate deep-dive
        st.markdown('<div class="sec-h">🔬 Candidate Analysis</div>', unsafe_allow_html=True)
        for rank, (cname, text, score) in enumerate(results, 1):
            fill = int(score); col = score_color(score)
            st.markdown(f"""
            <div class="cand-card">
                <div style="font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.14em;color:#475569">RANK #{rank}</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:.15rem 0 .6rem">{cname}</div>
                <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.9rem">
                    <div class="sbar-bg" style="flex:1">
                        <div class="sbar-fill" style="width:{fill}%;background:linear-gradient(90deg,{col},{col}88)"></div>
                    </div>
                    <span style="font-family:'DM Mono',monospace;font-size:.82rem;color:{col};font-weight:600">{score}%</span>
                </div>
            """, unsafe_allow_html=True)

            with st.spinner(f"Extracting skills for {cname}…"):
                skills = extract_skills(text)
            st.markdown('<div class="block-title">🔍 Extracted Skills</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="skills-block">{skills.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)

            if feats.get("ai_recommendation"):
                st.markdown("<br>", unsafe_allow_html=True)
                with st.spinner(f"AI recommendation for {cname}…"):
                    rec = generate_recommendation(jd_input, text, score)
                st.markdown("""<div class="block-title">🤖 AI Hiring Recommendation
                    <span style="font-family:'DM Mono',monospace;font-size:.58rem;color:#38bdf8;
                                 background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);
                                 border-radius:3px;padding:.06rem .35rem;margin-left:.4rem">PRO</span>
                </div>""", unsafe_allow_html=True)
                st.markdown(f'<div class="rec-block">{rec.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(56,189,248,0.03);border:1px dashed rgba(56,189,248,0.13);
                            border-radius:8px;padding:.6rem 1rem;margin-top:.7rem;
                            font-family:'DM Mono',monospace;font-size:.7rem;color:#334155">
                    🔒 AI hiring recommendations — upgrade to Pro
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Upgrade CTA for free users
        if plan == "free":
            st.markdown("""
            <div style="background:linear-gradient(120deg,rgba(56,189,248,0.08),rgba(129,140,248,0.08));
                        border:1px solid rgba(56,189,248,0.2);border-radius:16px;
                        padding:2rem;text-align:center;margin-top:2rem">
                <p style="font-family:'DM Mono',monospace;font-size:.7rem;letter-spacing:.18em;
                          text-transform:uppercase;color:#38bdf8;margin:0 0 .5rem">
                    Unlock the full platform</p>
                <h3 style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                           color:#f1f5f9;margin:0 0 .5rem">
                    Upgrade to Pro — ₹1,499/month</h3>
                <p style="font-family:'DM Sans',sans-serif;font-size:.9rem;color:#64748b;margin:0 0 1.2rem">
                    50 scans · Interview questions · AI recommendations · CSV export</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("💳  View All Plans", key="cta_up"):
                st.session_state.page = "dashboard"; st.rerun()

    # Footer
    st.markdown("""
    <div style="text-align:center;margin-top:4rem;padding-top:2rem;
                border-top:1px solid rgba(56,189,248,0.07)">
        <span style="font-family:'DM Mono',monospace;font-size:.65rem;
                     letter-spacing:.14em;color:#1e293b">
            HIREAI · AI-POWERED RESUME INTELLIGENCE · BUILT WITH ♥
        </span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    page_auth()
elif st.session_state.page == "dashboard":
    page_dashboard()
else:
    page_app()
