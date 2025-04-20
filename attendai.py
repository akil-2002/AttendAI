#!/usr/bin/env python
"""
AttendAI – single‑file demo
Predict NHS outpatient 'Did‑Not‑Attend' risk and send personalised reminders.

RUN:
    pip install -r requirements.txt       # see REQS block below
    python attendai.py                    # trains + prints draft messages
    python attendai.py --api              # starts FastAPI on :8000
"""

# ---------- REQS (copy to requirements.txt if desired) -----------------------
# pandas
# scikit-learn
# openai>=1.14.0
# langchain
# twilio
# fastapi
# uvicorn
# python-dotenv
# ---------------------------------------------------------------------------

import os, io, argparse, textwrap, sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None  # Twilio optional

try:
    from fastapi import FastAPI
    import uvicorn
except ImportError:
    FastAPI = None
    uvicorn = None

# ------------------ ENV ------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TWILIO_SID     = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM    = os.getenv("TWILIO_FROM_NUMBER", "")

openai.api_key = OPENAI_API_KEY
twilio_client  = TwilioClient(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and TWILIO_TOKEN else None

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "attendai_model.joblib"

# ------------------ SAMPLE DATA (small CSV in-memory) ------------------------
SAMPLE_CSV = """\
patient_id,patient_name,phone,appointment_date,appointment_time,clinic,previous_dna_count,age,imd_decile,travel_minutes,appointment_hour,dna
1001,John Smith,+447500111222,2025-05-02,09:30,Cardiology,2,54,3,55,9,0
1002,Mary Jones,+447500333444,2025-05-03,14:15,Cardiology,0,38,7,20,14,1
1003,Imran Patel,+447500555666,2025-05-03,10:45,Cardiology,1,60,4,35,10,0
"""

FEATURES = [
    "previous_dna_count",
    "age",
    "imd_decile",
    "travel_minutes",
    "appointment_hour",
]

# ------------------ MODEL ----------------------------------------------------
def train_model(df: pd.DataFrame, save_to: Path = MODEL_PATH) -> Pipeline:
    X = df[FEATURES]
    y = df["dna"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)
    print("[MODEL] Accuracy:", pipe.score(X_test, y_test))
    joblib.dump(pipe, save_to)
    return pipe

def load_or_train_model() -> Pipeline:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    print("[MODEL] No saved model found; training a new one...")
    df = pd.read_csv(io.StringIO(SAMPLE_CSV))
    return train_model(df)

MODEL = load_or_train_model()

# ------------------ MESSAGING ------------------------------------------------
CHAT_TMPL = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are NHS AttendAI assistant. Draft a concise SMS/WhatsApp reminder. "
         "Use a friendly tone, include date/time and department, offer a one‑tap link to reschedule."),
        ("human", "{content}")
    ]
)
CHAT = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo") if OPENAI_API_KEY else None

def draft_message(appt: Dict) -> str:
    if not CHAT:
        return textwrap.dedent(f"""\
            Hi {appt['patient_name']}, this is a reminder of your {appt['appointment_time']} {appt['clinic']} appointment on {appt['appointment_date']}.
            Reply 'R' to rebook or call us if you can’t attend. Thanks!""")
    content = (
        f"Patient name: {appt['patient_name']}\n"
        f"Appointment: {appt['appointment_date']} at {appt['appointment_time']}\n"
        f"Department: {appt['clinic']}\n"
    )
    msg = CHAT(CHAT_TMPL.format_prompt(content=content).to_messages())
    return msg.content.strip()

def send_sms(to_phone: str, body: str) -> str:
    if not twilio_client:
        print("[SMS MOCK]", body)
        return "mock-sid"
    message = twilio_client.messages.create(body=body, from_=TWILIO_FROM, to=to_phone)
    print(f"[SMS] Sent → {to_phone}")
    return message.sid

# ------------------ PIPELINE -------------------------------------------------
def fetch_appointments() -> pd.DataFrame:
    # Demo pulls from embedded CSV; swap with FHIR query in production.
    return pd.read_csv(io.StringIO(SAMPLE_CSV))

def score_risk(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    probs = MODEL.predict_proba(df[FEATURES])[:, 1]
    df = df.copy()
    df["risk"] = probs
    return df[df["risk"] >= threshold]

def process(send=False):
    df = fetch_appointments()
    high_risk = score_risk(df)
    print(f"[PIPELINE] High‑risk appointments: {len(high_risk)}")
    for _, row in high_risk.iterrows():
        msg = draft_message(row)
        sid = send_sms(row["phone"], msg) if send else "not‑sent"
        print(f"{row['patient_name']} → {sid}")

# ------------------ FASTAPI --------------------------------------------------
def start_api():
    if not FastAPI:
        print("fastapi / uvicorn not installed. `pip install fastapi uvicorn`")
        sys.exit(1)
    app = FastAPI()

    @app.get("/high_risk")
    def get_high_risk():
        df = fetch_appointments()
        hr = score_risk(df)[
            ["patient_id", "patient_name", "appointment_date", "appointment_time",
             "clinic", "phone", "risk"]
        ]
        return hr.to_dict("records")

    uvicorn.run(app, host="0.0.0.0", port=8000)

# ------------------ CLI ------------------------------------------------------
def cli():
    parser = argparse.ArgumentParser(description="AttendAI single‑file demo")
    parser.add_argument("--send", action="store_true", help="actually send SMS (needs Twilio creds)")
    parser.add_argument("--api", action="store_true", help="start FastAPI micro‑service")
    args = parser.parse_args()

    if args.api:
        start_api()
    else:
        process(send=args.send)

if __name__ == "__main__":
    cli()
