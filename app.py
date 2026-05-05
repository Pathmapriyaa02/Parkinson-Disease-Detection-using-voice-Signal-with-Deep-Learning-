import os, io, csv, pickle, shutil, time
import numpy as np
import librosa
import joblib
from flask import (Flask, render_template, request, redirect,
                   session, send_file, jsonify, flash)
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from werkzeug.utils import secure_filename
from database import *
from feature_extractor import extract_dataset_features

# ──────────────────────────────────────────────────────────────
# FFMPEG
# Browser recordings are usually .webm. Pydub needs ffmpeg/ffprobe
# to convert them to wav before librosa can analyse them.
# ──────────────────────────────────────────────────────────────
def configure_ffmpeg():
    candidates = [
        os.environ.get("FFMPEG_BIN"),
        os.path.join(os.path.dirname(__file__), "ffmpeg", "bin"),
        os.path.join(os.path.dirname(__file__), "ffmpeg"),
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
    ]

    for folder in candidates:
        if not folder:
            continue
        ffmpeg_exe = os.path.join(folder, "ffmpeg.exe")
        ffprobe_exe = os.path.join(folder, "ffprobe.exe")
        if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
            AudioSegment.converter = ffmpeg_exe
            AudioSegment.ffmpeg = ffmpeg_exe
            AudioSegment.ffprobe = ffprobe_exe
            os.environ["PATH"] += os.pathsep + folder
            return True

    ffmpeg_exe = shutil.which("ffmpeg")
    ffprobe_exe = shutil.which("ffprobe")
    if ffmpeg_exe and ffprobe_exe:
        AudioSegment.converter = ffmpeg_exe
        AudioSegment.ffmpeg = ffmpeg_exe
        AudioSegment.ffprobe = ffprobe_exe
        return True

    return False

FFMPEG_READY = configure_ffmpeg()

# ──────────────────────────────────────────────────────────────
# APP CONFIG
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "neuro_voice_secret_2025"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_AUDIO_SECONDS = 8

# ──────────────────────────────────────────────────────────────
# CNN-BiLSTM MODEL  (voice input only)
# ──────────────────────────────────────────────────────────────
model     = load_model("saved_model/cnn_bilstm_autoencoder.keras", compile=False)
threshold = joblib.load("saved_model/threshold.pkl")

# ──────────────────────────────────────────────────────────────
# NUMERIC CLASSIFIER  (19 features: Jitter, Shimmer, ZCR, HNR,
#                      RPDE, PPE, MFCC0–MFCC12)
# ──────────────────────────────────────────────────────────────
import pandas as _pd

_MDIR = os.path.join(os.path.dirname(__file__), 'dataset_model')

with open(os.path.join(_MDIR, 'clf_numeric.pkl'),    'rb') as _f: _clf_num    = pickle.load(_f)
with open(os.path.join(_MDIR, 'scaler_numeric.pkl'), 'rb') as _f: _scaler_num = pickle.load(_f)
with open(os.path.join(_MDIR, 'feats_numeric.pkl'),  'rb') as _f: _feats_num  = pickle.load(_f)
_ref_num = _pd.read_csv(os.path.join(_MDIR, 'ref_numeric.csv'), index_col=0)

NUMERIC_FEATURES = _feats_num   # 19 features in order

_VOICE_CLF_PATH = os.path.join(_MDIR, "voice_classifier.pkl")
_VOICE_SCALER_PATH = os.path.join(_MDIR, "voice_scaler.pkl")
_VOICE_FEATS_PATH = os.path.join(_MDIR, "voice_features.pkl")
_voice_clf = _voice_scaler = _voice_features = None
if os.path.exists(_VOICE_CLF_PATH) and os.path.exists(_VOICE_SCALER_PATH) and os.path.exists(_VOICE_FEATS_PATH):
    with open(_VOICE_CLF_PATH, "rb") as _f: _voice_clf = pickle.load(_f)
    with open(_VOICE_SCALER_PATH, "rb") as _f: _voice_scaler = pickle.load(_f)
    with open(_VOICE_FEATS_PATH, "rb") as _f: _voice_features = pickle.load(_f)

init_db()

# ──────────────────────────────────────────────────────────────
# AUDIO HELPERS
# ──────────────────────────────────────────────────────────────
def boost_volume(y, target_dBFS=-3.0):
    rms = np.sqrt(np.mean(y ** 2))
    if rms == 0:
        return y
    return np.clip(y * (10 ** ((target_dBFS - 20*np.log10(rms)) / 20.0)), -1.0, 1.0)

def reduce_noise(y, sr, noise_dur=0.3):
    ns = int(noise_dur * sr)
    if len(y) <= ns:
        return y
    profile = np.mean(np.abs(librosa.stft(y[:ns])), axis=1, keepdims=True)
    S, ph   = librosa.magphase(librosa.stft(y))
    return librosa.istft(np.maximum(S - profile * 1.5, 0) * ph)

def convert_to_wav(path):
    try:
        if os.path.getsize(path) == 0:
            return None
        if not FFMPEG_READY:
            print("Convert error: ffmpeg/ffprobe not found. Install FFmpeg or set FFMPEG_BIN.")
            return None
        out = path.rsplit(".", 1)[0] + ".wav"
        audio = AudioSegment.from_file(path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio = audio[:MAX_AUDIO_SECONDS * 1000]
        audio.export(out, format="wav")
        return out if os.path.exists(out) and os.path.getsize(out) > 0 else None
    except Exception as e:
        print("Convert error:", e); return None

def preprocess_audio(path):
    try:
        y, sr = librosa.load(path, sr=16000, mono=True, duration=MAX_AUDIO_SECONDS)
        if y is None or len(y) == 0:
            return None, None
        y = boost_volume(y)
        try: y = reduce_noise(y, 16000)
        except: pass
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) < 1000:
            return None, None
        if np.max(np.abs(y)) > 0:
            y = librosa.util.normalize(y)
        return y, 16000
    except Exception as e:
        print("Preprocess error:", e); return None, None

# ──────────────────────────────────────────────────────────────
# CNN-BiLSTM FEATURE EXTRACTION  (40 MFCC × 176 frames)
# ──────────────────────────────────────────────────────────────
def extract_cnn_features(path):
    try:
        y, sr = preprocess_audio(path)
        if y is None or len(y) < 2000:
            return None
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T   # (frames, 40)
        if feat.shape[0] > 176:
            feat = feat[:176]
        else:
            feat = np.pad(feat, ((0, 176 - feat.shape[0]), (0, 0)))
        return feat
    except Exception as e:
        print("CNN feature error:", e); return None

# ──────────────────────────────────────────────────────────────
# AUDIO QUALITY METRICS  (for quality badge only)
# ──────────────────────────────────────────────────────────────
def audio_quality_metrics(path):
    try:
        y, sr  = librosa.load(path, sr=16000)
        y      = boost_volume(y)
        zcr    = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        energy = float(np.mean(librosa.feature.rms(y=y)))
        sp     = np.mean(y**2)
        nf     = np.percentile(np.abs(y), 10) ** 2 + 1e-10
        snr_db = float(10 * np.log10(sp / nf))
        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return dict(zcr=round(zcr,4), energy=round(energy,4),
                    snr_db=round(snr_db,2),
                    mfcc_mean=round(float(np.mean(mfcc)),4))
    except:
        return dict(zcr=None, energy=None, snr_db=None, mfcc_mean=None)

# ──────────────────────────────────────────────────────────────
# CNN-BiLSTM RISK LOGIC
# ──────────────────────────────────────────────────────────────
def cnn_risk(mse, thr):
    if mse <= thr:
        return "Normal", "Normal"
    ratio = mse / thr
    risk  = "High Risk" if ratio > 2.0 else ("Moderate Risk" if ratio > 1.5 else "Low Risk")
    return "Abnormal", risk

def normalize_risk_label(risk):
    if risk in ("High", "High Risk"):
        return "High Risk"
    if risk in ("Moderate", "Moderate Risk"):
        return "Moderate Risk"
    if risk in ("Low", "Low Risk"):
        return "Low Risk"
    return "Normal"

def risk_points(risk):
    risk = normalize_risk_label(risk)
    if risk == "High Risk":
        return 3
    if risk == "Moderate Risk":
        return 2
    if risk == "Low Risk":
        return 1
    return 0

def combine_voice_results(cnn_status, cnn_risk_level, ds_result):
    """
    Final voice verdict uses both model views:
    1. CNN-BiLSTM reconstruction anomaly score
    2. Dataset classifier over extracted biomarkers such as Jitter, Shimmer,
       HNR, RPDE, PPE and MFCC values.
    """
    if ds_result:
        p_pd = float(ds_result["proba_pd"])
        if ds_result.get("model_source") == "Supervised Voice Classifier":
            if p_pd >= 0.50:
                return "Abnormal", normalize_risk_label(ds_result["risk_level"])
            return "Normal", "Normal"
        if p_pd >= 0.50:
            return "Abnormal", normalize_risk_label(ds_result["risk_level"])
        if cnn_status == "Abnormal":
            return "Abnormal", cnn_risk_level
        return "Normal", "Normal"

    return cnn_status, cnn_risk_level

def build_voice_explanation(status, risk, cnn_mse, ds_result, ds_features, qm):
    reasons = []
    if status == "Normal":
        if ds_result and ds_result.get("proba_pd") is not None:
            reasons.append(f"Dataset voice classifier gave a low Parkinson probability ({float(ds_result['proba_pd']) * 100:.1f}%).")
        if cnn_mse is not None:
            reasons.append("CNN-BiLSTM reconstruction error stayed within the learned normal range.")
        if ds_features:
            hnr = ds_features.get("HNR")
            jitter = ds_features.get("Jitter")
            shimmer = ds_features.get("Shimmer")
            if hnr is not None:
                reasons.append(f"HNR value ({hnr:.3f}) did not strongly indicate noisy or unstable voice.")
            if jitter is not None and shimmer is not None:
                reasons.append(f"Jitter ({jitter:.4f}) and Shimmer ({shimmer:.4f}) were not high enough to push the sample into Parkinson risk.")
        if qm and qm.get("snr_db") is not None:
            reasons.append(f"Audio quality was usable with SNR {qm['snr_db']} dB.")
        return reasons or ["The extracted voice pattern matched the healthy/normal side of the trained model."]

    if status == "Abnormal":
        if ds_result and ds_result.get("proba_pd") is not None:
            reasons.append(f"Dataset voice classifier gave Parkinson probability {float(ds_result['proba_pd']) * 100:.1f}%, which maps to {risk}.")
        if cnn_mse is not None:
            reasons.append("CNN-BiLSTM reconstruction error was above the learned normal voice threshold.")
        if ds_features:
            hnr = ds_features.get("HNR")
            jitter = ds_features.get("Jitter")
            shimmer = ds_features.get("Shimmer")
            ppe = ds_features.get("PPE")
            if jitter is not None:
                reasons.append(f"Jitter value was {jitter:.4f}, showing cycle-to-cycle pitch variation.")
            if shimmer is not None:
                reasons.append(f"Shimmer value was {shimmer:.4f}, showing amplitude variation.")
            if hnr is not None:
                reasons.append(f"HNR value was {hnr:.3f}, reflecting voice noise/hoarseness information.")
            if ppe is not None:
                reasons.append(f"PPE value was {ppe:.4f}, reflecting pitch-period irregularity.")
        return reasons or ["The extracted voice pattern was closer to Parkinson training samples than normal samples."]

    return ["This file could not be fully analysed, so the model cannot give a reliable reason."]

# ──────────────────────────────────────────────────────────────
# NUMERIC CLASSIFIER PREDICTION
# ──────────────────────────────────────────────────────────────
def predict_numeric_features(feat_dict):
    """
    feat_dict: {feature_name: float}  — exactly NUMERIC_FEATURES keys
    Returns:   dict with label, risk_level, proba_pd, proba_norm,
                         feature_comparison list
    """
    vec    = np.array([feat_dict[f] for f in NUMERIC_FEATURES], dtype=np.float32).reshape(1, -1)
    vec_sc = _scaler_num.transform(vec)
    proba  = _clf_num.predict_proba(vec_sc)[0]
    cls    = list(_clf_num.classes_)
    p_pd   = float(proba[cls.index(1)])
    p_nm   = float(proba[cls.index(0)])

    if   p_pd >= 0.75: risk = "High"
    elif p_pd >= 0.55: risk = "Moderate"
    elif p_pd >= 0.40: risk = "Low"
    else:              risk = "Normal"

    label = "Parkinson's Detected" if p_pd >= 0.50 else "Normal (Healthy)"

    comparison = []
    for feat in NUMERIC_FEATURES:
        val       = feat_dict[feat]
        pd_mean   = float(_ref_num.loc[feat, 'pd_mean'])
        nm_mean   = float(_ref_num.loc[feat, 'norm_mean'])
        pd_std    = float(_ref_num.loc[feat, 'pd_std'])   + 1e-10
        nm_std    = float(_ref_num.loc[feat, 'norm_std']) + 1e-10
        z_pd      = abs(val - pd_mean) / pd_std
        z_nm      = abs(val - nm_mean) / nm_std
        comparison.append(dict(
            name      = feat,
            value     = round(val, 5),
            pd_mean   = round(pd_mean, 4),
            norm_mean = round(nm_mean, 4),
            closer_to = 'PD' if z_pd < z_nm else 'Normal',
            z_pd      = round(z_pd, 3),
            z_norm    = round(z_nm, 3),
        ))

    return dict(label=label, risk_level=risk,
                proba_pd=round(p_pd,4), proba_norm=round(p_nm,4),
                feature_comparison=comparison)

def predict_voice_classifier(feat_dict):
    if _voice_clf is None or _voice_scaler is None or _voice_features is None:
        return None

    vec = np.array([feat_dict[f] for f in _voice_features], dtype=np.float32).reshape(1, -1)
    vec_sc = _voice_scaler.transform(vec)
    proba = _voice_clf.predict_proba(vec_sc)[0]
    cls = list(_voice_clf.classes_)
    p_pd = float(proba[cls.index(1)])
    p_nm = float(proba[cls.index(0)])

    if p_pd >= 0.75:
        risk = "High"
    elif p_pd >= 0.55:
        risk = "Moderate"
    elif p_pd >= 0.40:
        risk = "Low"
    else:
        risk = "Normal"

    label = "Parkinson's Detected" if p_pd >= 0.50 else "Normal (Healthy)"
    return dict(
        label=label,
        risk_level=risk,
        proba_pd=round(p_pd, 4),
        proba_norm=round(p_nm, 4),
        model_source="Supervised Voice Classifier",
    )

# ──────────────────────────────────────────────────────────────
# PDF REPORT
# ──────────────────────────────────────────────────────────────
def generate_pdf_report(data, username):
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image)
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, datetime

    fname = f"Report_{username}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    fpath = os.path.join(os.getcwd(), fname)
    doc   = SimpleDocTemplate(fpath, rightMargin=40, leftMargin=40, topMargin=30, bottomMargin=30)
    st    = getSampleStyleSheet()
    el    = []

    hs = ParagraphStyle("H", fontSize=18, alignment=1, spaceAfter=4,
                         textColor=colors.HexColor("#0a6ebd"), fontName="Helvetica-Bold")
    el.append(Paragraph("NeuroVoice Clinical Report", hs))
    el.append(Paragraph("Parkinson's Disease Voice Analysis", st["Normal"]))
    el.append(Spacer(1, 12))

    now = datetime.datetime.now()
    meta = [[Paragraph("<b>Patient</b>", st["Normal"]),  username],
            [Paragraph("<b>Date</b>",    st["Normal"]),  now.strftime("%Y-%m-%d %H:%M")],
            [Paragraph("<b>Report ID</b>",st["Normal"]), f"RPT-{now.strftime('%Y%m%d%H%M%S')}"]]
    t = Table(meta, colWidths=[130,300])
    t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.6,colors.grey),
                            ("BACKGROUND",(0,0),(0,-1),colors.HexColor("#e8f0fe"))]))
    el.append(t); el.append(Spacer(1,14))

    total    = len(data)
    abnormal = sum(1 for d in data if "Abnormal" in str(d[2]) or "Parkinson" in str(d[2]))
    normal   = total - abnormal

    el.append(Paragraph("<b>Summary</b>", st["Heading2"]))
    sm = [["Total","Normal","Abnormal","Detection Rate"],
          [total, normal, abnormal,
           f"{round(abnormal/total*100,1)}%" if total else "0%"]]
    t2 = Table(sm, colWidths=[120]*4)
    t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.6,colors.grey),
                             ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0a6ebd")),
                             ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                             ("ALIGN",(0,0),(-1,-1),"CENTER"),
                             ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")]))
    el.append(t2); el.append(Spacer(1,14))

    el.append(Paragraph("<b>Detailed Results</b>", st["Heading2"]))
    hdr  = ["File","MSE / PD Prob","Result","Risk","ZCR","Energy","SNR(dB)"]
    rows = [hdr]
    for row in data:
        rows.append([row[0][:24], round(row[1],5), row[2],
                     row[4] if len(row)>4 else "-",
                     row[5] if len(row)>5 else "-",
                     row[6] if len(row)>6 else "-",
                     row[8] if len(row)>8 else "-"])
    t3 = Table(rows, colWidths=[100,70,80,70,50,55,55])
    rs = [("GRID",(0,0),(-1,-1),.5,colors.grey),
          ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0a6ebd")),
          ("TEXTCOLOR",(0,0),(-1,0),colors.white),
          ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
          ("FONTSIZE",(0,0),(-1,-1),7),("ALIGN",(1,0),(-1,-1),"CENTER")]
    t3.setStyle(TableStyle(rs))
    el.append(t3); el.append(Spacer(1,14))

    if data:
        fnames = [d[0][:14] for d in data]
        mses   = [d[1] for d in data]
        cbars  = ["#ef4444" if ("Abnormal" in str(d[2]) or "Parkinson" in str(d[2])) else "#10b981" for d in data]
        fig, ax = plt.subplots(figsize=(7,3))
        ax.bar(range(len(fnames)), mses, color=cbars)
        thr_val = float(threshold)
        ax.axhline(y=thr_val, color="#f59e0b", linestyle="--",
                   linewidth=1.5, label=f"CNN Threshold ({round(thr_val,4)})")
        ax.set_xticks(range(len(fnames)))
        ax.set_xticklabels(fnames, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Score"); ax.set_title("Score per Sample")
        ax.legend(fontsize=8); plt.tight_layout()
        gp = "report_graph.png"; plt.savefig(gp, dpi=120); plt.close()
        el.append(Paragraph("<b>Score Analysis</b>", st["Heading2"]))
        el.append(Image(gp, width=440, height=180)); el.append(Spacer(1,14))

    if abnormal == 0:
        msg = "<font color='green'><b>All samples within normal limits.</b></font>"
        rec = "Continue routine health monitoring."
    elif abnormal < total:
        msg = "<font color='orange'><b>Partial abnormalities detected.</b></font>"
        rec = "Consult a neurologist for further assessment."
    else:
        msg = "<font color='red'><b>Significant Parkinsonian biomarkers detected.</b></font>"
        rec = "Immediate neurological consultation strongly recommended."

    el.append(Paragraph("<b>Clinical Interpretation</b>", st["Heading2"]))
    el.append(Paragraph(msg, st["Normal"])); el.append(Spacer(1,8))
    el.append(Paragraph(f"<b>Recommendation:</b> {rec}", st["Normal"]))
    el.append(Spacer(1,16))
    el.append(Paragraph(
        "<i>This report is AI-generated and does not constitute a final diagnosis. "
        "Always consult a qualified healthcare professional.</i>", st["Italic"]))
    doc.build(el)
    return fpath

# ════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ════════════════════════════════════════════════════════════════
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        ok = register_user(request.form["username"], request.form["password"],
                           request.form.get("email",""),
                           int(request.form.get("age",0) or 0),
                           request.form.get("gender",""))
        if not ok:
            flash("Username already exists.", "danger")
            return render_template("register.html")
        flash("Account created! Please log in.", "success")
        return redirect("/")
    return render_template("register.html")

@app.route("/login", methods=["POST"])
def do_login():
    user = validate_user(request.form["username"], request.form["password"])
    if user:
        session["user_id"] = user[0]
        flash("Login successful! Welcome to NeuroVoice.", "success")
        return redirect("/home")
    flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear(); return redirect("/")

# ════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════
@app.route("/home")
def home():
    if "user_id" not in session: return redirect("/")
    s = get_stats(session["user_id"])
    return render_template("dashboard.html",
        total_tests=s["total"], normal=s["normal"],
        abnormal=s["abnormal"], avg_mse=s["avg_mse"])

@app.route("/voice")
def voice_page():
    if "user_id" not in session: return redirect("/")
    return render_template("voice.html")

@app.route("/numeric")
def numeric_page():
    if "user_id" not in session: return redirect("/")
    return render_template("numeric.html", features=NUMERIC_FEATURES)

@app.route("/symptoms", methods=["GET", "POST"])
def symptoms_page():
    if "user_id" not in session: return redirect("/")
    symptoms = [
        ("tremor", "Resting tremor"),
        ("slowness", "Slowness of movement"),
        ("stiffness", "Muscle stiffness"),
        ("balance", "Balance difficulty"),
        ("voice", "Soft or unclear voice"),
        ("writing", "Small handwriting"),
        ("smell", "Reduced smell"),
        ("sleep", "Sleep movement problems"),
    ]
    result = None
    voice_summary = session.get("last_voice_result")
    values = {key: 0 for key, _ in symptoms}
    if request.method == "POST":
        values = {key: int(request.form.get(key, 0) or 0) for key, _ in symptoms}
        score = sum(values.values())
        if score >= 20:
            level = "High"
            advice = "Book a neurologist consultation soon and carry your voice-analysis report/history."
        elif score >= 11:
            level = "Moderate"
            advice = "Monitor symptoms, repeat voice screening, and consider a medical check-up if symptoms persist."
        elif score >= 5:
            level = "Low"
            advice = "Continue observation and repeat screening if voice or movement symptoms increase."
        else:
            level = "Minimal"
            advice = "Maintain routine health monitoring."

        combined_level = level
        combined_advice = advice
        if voice_summary:
            v_points = risk_points(voice_summary.get("risk", "Normal"))
            s_points = 3 if score >= 20 else (2 if score >= 11 else (1 if score >= 5 else 0))
            combined_points = max(v_points, s_points)
            if combined_points >= 3:
                combined_level = "High"
                combined_advice = "Voice and/or symptom findings suggest high concern. Consult a neurologist soon and carry the report."
            elif combined_points == 2:
                combined_level = "Moderate"
                combined_advice = "Repeat voice screening and consider a medical check-up if symptoms continue."
            elif combined_points == 1:
                combined_level = "Low"
                combined_advice = "Monitor symptoms and repeat screening after some days."
            else:
                combined_level = "Minimal"
                combined_advice = "Voice and symptom inputs are within low concern. Continue routine monitoring."

        result = dict(
            score=score, max_score=len(symptoms) * 4,
            level=level, advice=advice,
            combined_level=combined_level,
            combined_advice=combined_advice,
        )
    return render_template("symptoms.html", symptoms=symptoms, values=values,
                           result=result, voice_summary=voice_summary)

@app.route("/about_model")
def about_model():
    if "user_id" not in session: return redirect("/")
    return render_template("about_model.html")

# ════════════════════════════════════════════════════════════════
# ROUTE 1 — VOICE INPUT → CNN-BiLSTM ONLY
# ════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session: return redirect("/")

    files   = request.files.getlist("file")
    results = []

    for file in files:
        filename = "audio"
        try:
            if not file.filename:
                continue

            original_filename = secure_filename(os.path.basename(file.filename)) or "audio.webm"
            name, ext = os.path.splitext(original_filename)
            filename  = f"{name}_{int(time.time() * 1000)}{ext or '.webm'}"
            fpath     = os.path.join(UPLOAD_FOLDER, filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            file.save(fpath)

            if os.path.getsize(fpath) == 0:
                results.append(_err(filename, "Empty file")); continue

            # Convert webm → wav
            if filename.lower().endswith(".webm"):
                new = convert_to_wav(fpath)
                if not new:
                    results.append(_err(filename, "WebM conversion failed. Install FFmpeg or upload WAV/MP3.")); continue
                fpath = new

            # ── CNN-BiLSTM prediction ─────────────────────────────
            cnn_mse, cnn_status, cnn_risk_level = None, "Error", "-"
            feat = extract_cnn_features(fpath)

            if feat is not None:
                try:
                    feat_in = np.expand_dims(feat, 0)
                    recon   = model.predict(feat_in, verbose=0)
                    if recon.shape != feat_in.shape:
                        recon = np.reshape(recon, feat_in.shape)
                    cnn_mse = float(np.mean(np.square(feat_in - recon)))
                    cnn_status, cnn_risk_level = cnn_risk(cnn_mse, threshold)
                except Exception as e:
                    print("CNN predict error:", e); cnn_status = "Model Error"
            else:
                cnn_status = "Poor Audio / Retry"

            # ── Dataset biomarker classifier from raw voice ────────
            ds_result = None
            ds_features = extract_dataset_features(fpath)
            if ds_features:
                try:
                    ds_result = predict_voice_classifier(ds_features) or predict_numeric_features(ds_features)
                except Exception as e:
                    print("Voice dataset classifier error:", e)

            status, risk = combine_voice_results(cnn_status, cnn_risk_level, ds_result)

            # ── Audio quality metrics ─────────────────────────────
            qm = audio_quality_metrics(fpath)
            explanation = build_voice_explanation(status, risk, cnn_mse, ds_result, ds_features, qm)

            # ── Save to DB ────────────────────────────────────────
            lbl = f"Abnormal ({risk})" if status == "Abnormal" else status
            if cnn_mse is not None:
                insert_result(
                    session["user_id"], filename, cnn_mse, lbl,
                    risk_level = risk,
                    zcr        = qm["zcr"]      or 0,
                    energy     = qm["energy"]   or 0,
                    mfcc_mean  = qm["mfcc_mean"]or 0,
                    snr_db     = qm["snr_db"]   or 0,
                ds_label   = ds_result["label"] if ds_result else "",
                ds_pd_prob = ds_result["proba_pd"] if ds_result else 0.0,
                )

            results.append(dict(
                filename = filename,
                cnn_mse  = round(cnn_mse, 6) if cnn_mse is not None else "N/A",
                status   = status,
                risk     = risk,
                cnn_status = cnn_status,
                cnn_risk   = cnn_risk_level,
                ds_label   = ds_result["label"] if ds_result else "N/A",
                ds_pd_prob = ds_result["proba_pd"] if ds_result else None,
                ds_risk    = normalize_risk_label(ds_result["risk_level"]) if ds_result else "N/A",
                ds_source  = ds_result.get("model_source", "Dataset Classifier") if ds_result else "N/A",
                features   = ds_features,
                explanation = explanation,
                zcr      = qm["zcr"],
                energy   = qm["energy"],
                snr_db   = qm["snr_db"],
            ))
        except Exception as e:
            print("Voice route error:", e)
            results.append(_err(filename, "Processing failed. Try a shorter WAV recording."))

    normal_c   = sum(1 for r in results if r["status"] == "Normal")
    abnormal_c = sum(1 for r in results if r["status"] == "Abnormal")
    failed_c   = len(results) - normal_c - abnormal_c

    analysed_results = [r for r in results if r["status"] in ("Normal", "Abnormal")]
    if analysed_results:
        strongest = max(analysed_results, key=lambda r: risk_points(r.get("risk", "Normal")))
        session["last_voice_result"] = {
            "filename": strongest["filename"],
            "status": strongest["status"],
            "risk": strongest["risk"],
            "pd_probability": strongest.get("ds_pd_prob"),
            "source": strongest.get("ds_source"),
        }

    return render_template("result_voice.html",
        results   = results,
        threshold = round(float(threshold), 6),
        total     = len(results),
        normal    = normal_c,
        abnormal  = abnormal_c,
        failed    = failed_c,
    )

def _err(filename, reason):
    return dict(filename=filename, cnn_mse="N/A",
                status=reason, risk="-",
                cnn_status=reason, cnn_risk="-",
                ds_label="N/A", ds_pd_prob=None,
                ds_risk="N/A", ds_source="N/A",
                features=None,
                explanation=[reason],
                zcr=None, energy=None, snr_db=None)

# ════════════════════════════════════════════════════════════════
# ROUTE 2 — NUMERIC INPUT → DATASET CLASSIFIER ONLY
#           19 features: Jitter, Shimmer, ZCR, HNR, RPDE, PPE,
#                        MFCC0–MFCC12
# ════════════════════════════════════════════════════════════════
@app.route("/predict_numeric", methods=["POST"])
def predict_numeric():
    if "user_id" not in session: return redirect("/")

    feat_dict, missing = {}, []
    for col in NUMERIC_FEATURES:
        val = request.form.get(col, "").strip()
        if val == "":
            missing.append(col)
        else:
            try:   feat_dict[col] = float(val)
            except: missing.append(col)

    if missing:
        flash(f"Missing values for: {', '.join(missing)}", "danger")
        return redirect("/numeric")

    result = predict_numeric_features(feat_dict)

    # Save to DB (use proba_pd as the "mse" column for storage)
    insert_result(
        session["user_id"], "numeric_input",
        result["proba_pd"], result["label"],
        risk_level = result["risk_level"],
        zcr        = feat_dict.get("ZCR", 0),
        energy     = 0,
        mfcc_mean  = feat_dict.get("MFCC0", 0),
        snr_db     = feat_dict.get("HNR", 0),
        ds_label   = result["label"],
        ds_pd_prob = result["proba_pd"],
    )

    return render_template("result_numeric.html",
        result   = result,
        features = feat_dict,
    )

# ════════════════════════════════════════════════════════════════
# HISTORY / DELETE
# ════════════════════════════════════════════════════════════════
@app.route("/history")
def history():
    if "user_id" not in session: return redirect("/")
    return render_template("history.html", data=get_user_results_with_ids(session["user_id"]))

@app.route("/delete/<int:rid>", methods=["POST"])
def delete_record(rid):
    if "user_id" not in session: return redirect("/")
    delete_result(rid, session["user_id"])
    flash("Record deleted.", "success")
    return redirect("/history")

# ════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════
@app.route("/export_csv")
def export_csv():
    if "user_id" not in session: return redirect("/")
    data   = get_user_results(session["user_id"])
    output = io.StringIO()
    w      = csv.writer(output)
    w.writerow(["Filename","MSE/Prob","Prediction","Timestamp","Risk",
                "ZCR","Energy","MFCC Mean","SNR(dB)","DS Label","PD Prob"])
    for row in data: w.writerow(row)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()),
                     mimetype="text/csv", as_attachment=True,
                     download_name="neurovoice_history.csv")

@app.route("/download_report")
def download_report():
    if "user_id" not in session: return redirect("/")
    pdf = generate_pdf_report(
        get_user_results(session["user_id"]),
        get_username(session["user_id"]))
    return send_file(pdf, as_attachment=True)

# ════════════════════════════════════════════════════════════════
# API — AUDIO QUALITY CHECK
# ════════════════════════════════════════════════════════════════
@app.route("/api/check_audio", methods=["POST"])
def check_audio():
    if "file" not in request.files:
        return jsonify({"ok": False})
    tmp = os.path.join(UPLOAD_FOLDER, "_check.wav")
    request.files["file"].save(tmp)
    qm  = audio_quality_metrics(tmp)
    snr = qm.get("snr_db") or 0
    return jsonify({"ok": True, "snr_db": snr, "energy": qm.get("energy",0),
                    "quality": "Good" if snr>15 else ("Fair" if snr>5 else "Poor")})

# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, threaded=False)
