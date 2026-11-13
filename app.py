
import streamlit as st
import streamlit.components.v1 as components
import sqlite3
import hashlib
import secrets
import hmac
import io
import json
import urllib.parse
import base64
import os

from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw
import requests

# Optional libraries
try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# =====================================================
# CONFIGURATION
# =====================================================

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "plant_guardian.db"

st.set_page_config(
    page_title="Plant Guardian AI",
    page_icon="🌿",
    layout="wide"
)


# =====================================================
# THEME
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg:#071a14;
    --panel:#10271e;
    --green:#38d996;
    --mint:#a4f3ca;
}

html, body, [data-testid="stAppViewContainer"] {
    background:linear-gradient(135deg,#06150f,#0a2118,#102c20);
    color:#effff5;
    font-family:'DM Sans',sans-serif;
}

[data-testid="stHeader"] {
    background:rgba(4,18,12,.85);
}

[data-testid="stSidebar"] {
    background:#091d15;
    border-right:1px solid #204b36;
}

[data-testid="stSidebar"] * {
    color:#eafff1 !important;
}

h1,h2,h3 {
    color:#effff5 !important;
}

p,label {
    color:#d8eee1;
}

div[data-testid="stMetric"] {
    background:#10271e;
    border:1px solid #28583f;
    padding:16px;
    border-radius:16px;
}

.stButton>button,
.stDownloadButton>button {
    background:linear-gradient(90deg,#1c9e68,#36d995);
    color:#04170e;
    border:0;
    border-radius:12px;
    font-weight:800;
    min-height:42px;
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background:#10271e !important;
    color:#effff5 !important;
    border-radius:10px !important;
}

[data-testid="stFileUploader"] {
    background:#10271e;
    border:1px dashed #3b9467;
    border-radius:14px;
    padding:8px;
}

.pg-hero {
    padding:28px;
    border-radius:22px;
    background:linear-gradient(115deg,rgba(24,105,69,.72),rgba(13,39,28,.95));
    border:1px solid #347b55;
    margin-bottom:18px;
}

.pg-card {
    background:linear-gradient(145deg,#143426,#0d241a);
    border:1px solid #28583f;
    padding:18px;
    border-radius:18px;
    min-height:112px;
}

.pg-muted {
    color:#a9c5b6;
    font-size:13px;
}

footer {
    visibility:hidden;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# DATABASE
# =====================================================

def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:

        con.execute("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            salt TEXT,
            password_hash TEXT
        )
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS plants(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            name TEXT,
            category TEXT,
            size_cm REAL,
            created_at TEXT,
            photo BLOB
        )
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id INTEGER,
            log_date TEXT,
            height_cm REAL,
            notes TEXT
        )
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS reminders(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id INTEGER,
            due_date TEXT,
            message TEXT,
            completed INTEGER DEFAULT 0
        )
        """)

        def columns(table):
            return {
                row[1]
                for row in con.execute(
                    f"PRAGMA table_info({table})"
                )
            }

        user_cols = columns("users")

        for col, typ in {
            "username": "TEXT",
            "salt": "TEXT",
            "password_hash": "TEXT"
        }.items():
            if col not in user_cols:
                con.execute(
                    f"ALTER TABLE users ADD COLUMN {col} {typ}"
                )

        plant_cols = columns("plants")

        for col, typ in {
            "username": "TEXT",
            "name": "TEXT",
            "category": "TEXT",
            "size_cm": "REAL",
            "created_at": "TEXT",
            "photo": "BLOB"
        }.items():
            if col not in plant_cols:
                con.execute(
                    f"ALTER TABLE plants ADD COLUMN {col} {typ}"
                )

        plant_cols = columns("plants")

        if "crop" in plant_cols:
            con.execute("""
                UPDATE plants
                SET category=crop
                WHERE category IS NULL
            """)

        if "added" in plant_cols:
            con.execute("""
                UPDATE plants
                SET created_at=added
                WHERE created_at IS NULL
            """)

        if "user_id" in plant_cols and "id" in user_cols:
            con.execute("""
                UPDATE plants
                SET username=(
                    SELECT username FROM users
                    WHERE users.id=plants.user_id
                )
                WHERE username IS NULL
            """)

        con.execute("""
            UPDATE plants
            SET created_at=?
            WHERE created_at IS NULL
        """, (datetime.now().isoformat(),))

        con.execute("""
            UPDATE plants
            SET category='Other'
            WHERE category IS NULL
        """)

        reminder_cols = columns("reminders")

        for col, typ in {
            "due_date": "TEXT",
            "message": "TEXT",
            "completed": "INTEGER DEFAULT 0"
        }.items():
            if col not in reminder_cols:
                con.execute(
                    f"ALTER TABLE reminders ADD COLUMN {col} {typ}"
                )

        reminder_cols = columns("reminders")

        if "task" in reminder_cols:
            con.execute("""
                UPDATE reminders
                SET message=task
                WHERE message IS NULL
            """)

        if "done" in reminder_cols:
            con.execute("""
                UPDATE reminders
                SET completed=done
            """)

        con.execute("""
            UPDATE reminders
            SET message='Plant care reminder'
            WHERE message IS NULL
        """)

        con.execute("""
            UPDATE reminders
            SET due_date=?
            WHERE due_date IS NULL
        """, (date.today().isoformat(),))


init_db()


# =====================================================
# ACCOUNT HELPERS
# =====================================================

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        180000
    ).hex()


def register_user(username, password):
    salt = secrets.token_hex(16)

    try:
        with db() as con:
            con.execute(
                """
                INSERT INTO users(username,salt,password_hash)
                VALUES(?,?,?)
                """,
                (username, salt, hash_password(password, salt))
            )
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username, password):
    with db() as con:
        row = con.execute(
            """
            SELECT salt,password_hash
            FROM users
            WHERE username=?
            """,
            (username,)
        ).fetchone()

    return bool(
        row and hmac.compare_digest(
            hash_password(password, row["salt"]),
            row["password_hash"]
        )
    )


# =====================================================
# CONSTANTS
# =====================================================

MOISTURE_OPTIONS = [
    "Bone Dry / Oven Dry",
    "Dry",
    "Slightly Moist / Damp",
    "Moist",
    "Very Moist",
    "Wet / Soggy",
    "Saturated / Waterlogged"
]

MOISTURE_TAMIL = {
    "Bone Dry / Oven Dry": "முற்றிலும் வறண்ட மண்",
    "Dry": "வறண்ட மண்",
    "Slightly Moist / Damp": "சற்று ஈரமான மண்",
    "Moist": "ஈரப்பதமான மண்",
    "Very Moist": "அதிக ஈரப்பதமான மண்",
    "Wet / Soggy": "நனைந்த / சேறான மண்",
    "Saturated / Waterlogged": "நீர் தேங்கிய மண்"
}

PLANT_CATEGORIES = [
    "Flowering plant",
    "Vegetable",
    "Fruit plant",
    "Herb",
    "Indoor plant",
    "Outdoor plant",
    "Tree",
    "Cactus / Succulent",
    "Cereal / Crop",
    "Other"
]

COMMON_PLANTS = [
    "Tomato", "Potato", "Brinjal / Eggplant",
    "Chilli", "Okra", "Cucumber", "Watermelon",
    "Papaya", "Mango", "Banana", "Guava",
    "Lemon", "Rose", "Hibiscus", "Jasmine",
    "Tulsi", "Mint", "Coriander", "Aloe vera",
    "Money plant", "Snake plant", "Areca palm",
    "Peace lily", "Sunflower", "Marigold",
    "Rice", "Wheat", "Maize", "Cotton",
    "Neem", "Coconut", "Jackfruit",
    "Drumstick tree", "Bougainvillea", "Orchid", "Fern"
]


# =====================================================
# IMAGE HELPERS
# =====================================================

def safe_image(upload):
    if upload is None:
        return None

    try:
        im = Image.open(upload).convert("RGB")

        if im.width < 80 or im.height < 80:
            return None

        im.thumbnail((1800, 1800))
        return im

    except Exception:
        return None


def image_looks_like_plant(im):
    if im is None:
        return False

    try:
        import numpy as np

        small = im.resize((80, 80))
        arr = np.asarray(small).astype("float32")

        green = (
            (arr[:, :, 1] > arr[:, :, 0] * 1.08)
            & (arr[:, :, 1] > arr[:, :, 2] * 1.03)
            & (arr[:, :, 1] > 45)
        )

        return green.mean() > 0.035

    except Exception:
        return False


# =====================================================
# AI MODEL
# =====================================================

@st.cache_resource(show_spinner="Loading plant AI model...")
def load_disease_model():

    if pipeline is None:
        return None

    return pipeline(
        "image-classification",
        model="neghani/plantvillage-mobilenetv2"
    )


def disease_scan(im):

    model = load_disease_model()

    if model is None:
        return None, "Transformers is not installed."

    try:
        preds = model(im, top_k=3)
        return preds, None

    except Exception as e:
        return None, f"Model could not run: {e}"


def plant_id_model(im):

    preds, err = disease_scan(im)

    if err:
        return None, err

    labels = [
        p["label"].split("___")[0].replace("_", " ")
        for p in preds
    ]

    return [
        {"label": label, "score": p["score"]}
        for label, p in zip(labels, preds)
    ], None


# =====================================================
# GOOGLE VISION API
# =====================================================

def get_vision_api_key():

    try:
        key = (
            st.secrets.get("GOOGLE_VISION_API_KEY", "")
            or st.secrets.get("GOOGLE_CLOUD_VISION_API_KEY", "")
        )
    except Exception:
        key = ""

    return (
        key
        or os.environ.get("GOOGLE_VISION_API_KEY", "")
        or os.environ.get("GOOGLE_CLOUD_VISION_API_KEY", "")
    )


def google_vision_web_detection(im):

    api_key = get_vision_api_key()

    if not api_key:
        return None, "Google Vision API key is not configured."

    try:
        buff = io.BytesIO()
        im.save(buff, format="JPEG", quality=88)

        encoded = base64.b64encode(
            buff.getvalue()
        ).decode("ascii")

        endpoint = "https://vision.googleapis.com/v1/images:annotate"

        payload = {
            "requests": [{
                "image": {"content": encoded},
                "features": [{
                    "type": "WEB_DETECTION",
                    "maxResults": 10
                }]
            }]
        }

        response = requests.post(
            endpoint,
            params={"key": api_key},
            json=payload,
            timeout=35
        )

        response.raise_for_status()

        data = response.json()["responses"][0]

        if "error" in data:
            return None, data["error"].get(
                "message",
                "Vision API returned an error."
            )

        return data.get("webDetection", {}), None

    except requests.RequestException as exc:
        return None, f"Google Vision request failed: {exc}"

    except Exception as exc:
        return None, f"Could not read Google Vision results: {exc}"


def show_web_detection_results(web):

    if not web:
        st.info("No web matches were returned.")
        return

    guesses = web.get("bestGuessLabels", [])
    entities = web.get("webEntities", [])
    pages = web.get("pagesWithMatchingImages", [])
    images = web.get("visuallySimilarImages", [])

    st.markdown("#### Google Vision best-guess labels")

    if guesses:
        st.write(
            " • ".join(
                x.get("label", "")
                for x in guesses
                if x.get("label")
            )
        )
    else:
        st.caption("No best-guess label returned.")

    st.markdown("#### Matching web entities")

    shown_entities = [
        x for x in entities
        if x.get("description")
    ]

    for item in shown_entities[:8]:
        score = item.get("score")
        suffix = (
            f" — score {score:.2f}"
            if isinstance(score, (int, float))
            else ""
        )
        st.write(f"• {item['description']}{suffix}")

    st.markdown("#### Pages containing matching images")

    for page in pages[:8]:
        url = page.get("url", "")
        title = page.get("pageTitle") or url

        if url:
            st.markdown(f"- [{title}]({url})")

    st.markdown("#### Visually similar images")

    if images:
        cols = st.columns(min(4, len(images[:8])))

        for index, item in enumerate(images[:8]):
            url = item.get("url")

            if url:
                with cols[index % len(cols)]:
                    st.image(url, use_container_width=True)
                    st.caption("Similar image")
    else:
        st.caption("No similar-image URLs returned.")


# =====================================================
# OTHER HELPERS
# =====================================================

def google_search_url(query):
    return (
        "https://www.google.com/search?tbm=isch&q="
        + urllib.parse.quote(query)
    )


def watering_advice(name, size, moisture, temp, rain, humidity):

    base_days = 3 if size < 30 else 4 if size < 100 else 5

    if moisture in ["Wet / Soggy", "Saturated / Waterlogged"]:
        return (
            "Do not water now",
            "Soil is already too wet. Check drainage."
        )

    if moisture == "Very Moist":
        return (
            "Wait and recheck tomorrow",
            "Soil still contains substantial moisture."
        )

    if moisture == "Moist":
        return (
            "Recheck in 1–2 days",
            "Water when the upper soil becomes slightly dry."
        )

    if rain >= 60:
        return (
            "Delay watering and recheck after rain",
            "Rain is likely. Avoid unnecessary watering."
        )

    if moisture in ["Bone Dry / Oven Dry", "Dry"]:
        return (
            "Water soon",
            "The selected soil condition is dry."
        )

    if temp >= 34 and humidity < 40:
        return (
            "Check soil again today",
            "Hot, dry air can increase water loss."
        )

    return (
        f"Recheck in about {base_days} days",
        "Check the soil before watering."
    )


def annotate_suspected_area(im):

    out = im.copy()
    w, h = out.size

    draw = ImageDraw.Draw(out, "RGBA")

    step = max(35, min(w, h) // 7)

    for x in range(0, w, step):
        draw.line(
            (x, 0, x, h),
            fill=(40, 145, 255, 150),
            width=2
        )

    for y in range(0, h, step):
        draw.line(
            (0, y, w, y),
            fill=(40, 145, 255, 150),
            width=2
        )

    cx, cy = w // 2, h // 2
    r = max(28, min(w, h) // 9)

    draw.ellipse(
        (cx-r, cy-r, cx+r, cy+r),
        outline=(255, 190, 45, 255),
        width=5
    )

    return out


def growth_blueprint(im, name, label, scale=1.0):

    W, H = 700, 430

    canvas = Image.new("RGB", (W, H), (8, 31, 22))
    d = ImageDraw.Draw(canvas)

    d.rounded_rectangle(
        (18, 18, W-18, H-18),
        radius=24,
        outline=(55, 180, 120),
        width=3
    )

    d.text(
        (42, 35),
        f"{name} | {label}",
        fill=(170, 245, 205)
    )

    base = 350

    d.line(
        (80, base, 620, base),
        fill=(120, 180, 140),
        width=4
    )

    stem_top = int(base - 145 * scale)

    d.line(
        (350, base, 350, stem_top),
        fill=(70, 205, 120),
        width=10
    )

    for yy, side in [
        (stem_top+35, -1),
        (stem_top+75, 1),
        (stem_top+115, -1)
    ]:
        x = 350 + side * 75

        d.ellipse(
            (
                min(x, 350)-55,
                yy-22,
                max(x, 350)+55,
                yy+22
            ),
            fill=(42, 150, 88),
            outline=(135, 235, 160),
            width=3
        )

    d.text(
        (42, 385),
        "Illustrative growth concept",
        fill=(190, 215, 200)
    )

    return canvas


# =====================================================
# PDF
# =====================================================

def make_pdf(plant, result, care):

    if not REPORTLAB_OK:
        return None

    buff = io.BytesIO()

    doc = SimpleDocTemplate(
        buff,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "Plant Guardian AI - Plant Health Report",
            styles["Title"]
        ),
        Spacer(1, 14),
        Paragraph(
            f"<b>Plant:</b> {plant}",
            styles["BodyText"]
        ),
        Spacer(1, 8),
        Paragraph(
            f"<b>AI scan:</b> {result}",
            styles["BodyText"]
        ),
        Spacer(1, 8),
        Paragraph(
            f"<b>Care guidance:</b> {care}",
            styles["BodyText"]
        ),
        Spacer(1, 12),
        Paragraph(
            "This report is informational, not a professional diagnosis.",
            styles["Italic"]
        )
    ]

    doc.build(story)

    buff.seek(0)
    return buff.getvalue()


# =====================================================
# PLANT DATABASE FUNCTIONS
# =====================================================

def get_user_plants(username):

    with db() as con:
        return con.execute(
            """
            SELECT *
            FROM plants
            WHERE username=?
            ORDER BY id DESC
            """,
            (username,)
        ).fetchall()


def save_plant_record(username, name, category, size_cm, photo):

    now = datetime.now().isoformat()

    with db() as con:

        cols = {
            row[1]
            for row in con.execute(
                "PRAGMA table_info(plants)"
            )
        }

        values = {
            "username": username,
            "name": name,
            "category": category,
            "size_cm": size_cm,
            "created_at": now,
            "photo": photo,
            "crop": category,
            "added": now,
            "last_watered": date.today().isoformat(),
            "water_interval": 3,
            "notes": ""
        }

        if "user_id" in cols:

            user_cols = {
                row[1]
                for row in con.execute(
                    "PRAGMA table_info(users)"
                )
            }

            if "id" in user_cols:

                row = con.execute(
                    """
                    SELECT id FROM users
                    WHERE username=?
                    LIMIT 1
                    """,
                    (username,)
                ).fetchone()

                values["user_id"] = row["id"] if row else None

        insert_cols = [
            c for c in values
            if c in cols
        ]

        sql = (
            f"INSERT INTO plants ({','.join(insert_cols)}) "
            f"VALUES ({','.join(['?'] * len(insert_cols))})"
        )

        con.execute(
            sql,
            [values[c] for c in insert_cols]
        )


# =====================================================
# PUBLIC WORKSPACE
# =====================================================

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None


st.markdown("""
<div class="pg-hero">
<h1>🌿 Plant Guardian AI</h1>
<p>Botanical Intelligence • Understand. Protect. Grow.</p>
<span class="pg-muted">AI-assisted plant care workspace</span>
</div>
""", unsafe_allow_html=True)


# Public workspace uses the first existing profile
with db() as con:

    existing_user = con.execute(
        "SELECT username FROM users LIMIT 1"
    ).fetchone()

username = (
    existing_user["username"]
    if existing_user
    else "Guest"
)

plants = get_user_plants(username)


# =====================================================
# NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("## 🌿 Plant Guardian")
    st.caption("Public workspace")

    page = st.radio(
        "Workspace",
        [
            "Dashboard",
            "AI Disease Detection",
            "AI Plant Identification",
            "Intelligent Water Predictor",
            "AI Growth Prediction",
            "Plant Digital Twin",
            "Personal Workspace",
            "Plant Health PDF"
        ]
    )

    st.markdown("---")
    st.caption("Botanical Intelligence • v2")


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("Your Plant Dashboard")

    c1, c2, c3 = st.columns(3)

    c1.metric("Saved plants", len(plants))

    with db() as con:

        due = con.execute(
            """
            SELECT COUNT(*) n
            FROM reminders r
            JOIN plants p ON p.id=r.plant_id
            WHERE p.username=?
            AND r.completed=0
            AND r.due_date<=?
            """,
            (username, date.today().isoformat())
        ).fetchone()["n"]

    c2.metric("Care reminders due", due)

    c3.metric(
        "Today",
        date.today().strftime("%d %b %Y")
    )

    st.markdown("### Quick actions")

    cols = st.columns(3)

    quick = [
        ("🔬", "Disease Detection", "Screen a plant or leaf photo"),
        ("💧", "Water Predictor", "Check soil and weather"),
        ("📈", "Growth Prediction", "View growth milestones")
    ]

    for col, (emoji, label, desc) in zip(cols, quick):

        with col:

            st.markdown(
                f"""
                <div class="pg-card">
                <h2>{emoji}</h2>
                <b>{label}</b>
                <p class="pg-muted">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Your collection")

    if plants:

        st.dataframe(
            pd.DataFrame([
                {
                    "Plant": p["name"],
                    "Category": p["category"],
                    "Size (cm)": p["size_cm"],
                    "Added": p["created_at"][:10]
                }
                for p in plants
            ]),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "Your collection is empty. "
            "Open Personal Workspace to add your first plant."
        )


# =====================================================
# DISEASE DETECTION
# =====================================================

elif page == "AI Disease Detection":

    st.title("🔬 AI Disease Detection")

    mode = st.radio(
        "What do you want to analyse?",
        ["Analyse Plant", "Analyse Leaf"],
        horizontal=True
    )

    file = st.file_uploader(
        "Upload a clear plant or leaf photo",
        type=["jpg", "jpeg", "png"],
        key="disease_img"
    )

    if file:

        im = safe_image(file)

        if im is None:

            st.error(
                "Image could not be read. Upload a clear JPG or PNG."
            )

        else:

            st.image(
                im,
                caption="Uploaded image",
                use_container_width=True
            )

            if not image_looks_like_plant(im):

                st.error(
                    "Please upload a plant or leaf photo."
                )

            elif st.button(
                "Analyse image",
                type="primary"
            ):

                with st.spinner("Analysing image..."):
                    preds, err = disease_scan(im)

                if err:

                    st.error(err)

                elif preds:

                    best = preds[0]

                    label = (
                        best["label"]
                        .replace("___", " — ")
                        .replace("_", " ")
                    )

                    st.session_state.last_scan = {
                        "plant": label.split(" — ")[0],
                        "disease": label,
                        "confidence": best["score"]
                    }

                    st.success("AI screening complete")

                    left, right = st.columns(2)

                    with left:

                        st.image(
                            annotate_suspected_area(im),
                            caption=(
                                "Inspection guide only. "
                                "Not verified disease localization."
                            ),
                            use_container_width=True
                        )

                    with right:

                        st.subheader(label)

                        st.metric(
                            "Model confidence",
                            f"{best['score'] * 100:.1f}%"
                        )

                        st.caption(
                            "The model predicts a class for the whole image."
                        )

                        st.markdown("**Other model predictions**")

                        for x in preds[1:]:

                            other_label = (
                                x["label"]
                                .replace("___", " — ")
                                .replace("_", " ")
                            )

                            st.write(
                                f"• {other_label} — "
                                f"{x['score'] * 100:.1f}%"
                            )

                    st.markdown("### Prevention and care")

                    st.write(
                        "Avoid overwatering, improve airflow, "
                        "and check reliable local agricultural guidance."
                    )

                    st.warning(
                        "This is an AI screening result. "
                        "Confirm suspected disease with an agricultural expert."
                    )

                    q = f"{label} plant disease symptoms prevention"

                    st.link_button(
                        "Cross-check disease information on Google",
                        "https://www.google.com/search?q="
                        + urllib.parse.quote(q)
                    )

                    st.link_button(
                        "Compare reference images",
                        google_search_url(q)
                    )

                    if st.button(
                        "Find visual web matches for this photo",
                        key="vision_disease_search"
                    ):

                        with st.spinner(
                            "Checking visual matches on the web..."
                        ):

                            web_results, web_error = (
                                google_vision_web_detection(im)
                            )

                        if web_error:

                            st.warning(web_error)

                        else:

                            show_web_detection_results(
                                web_results
                            )

    else:

        st.info(
            "Choose Analyse Plant or Analyse Leaf, "
            "then upload a clear photo."
        )


# =====================================================
# PLANT IDENTIFICATION
# =====================================================

elif page == "AI Plant Identification":

    st.title("🌼 AI Plant Identification")

    st.write(
        "Upload a whole-plant photo and/or a leaf photo. "
        "The model is limited to supported classes."
    )

    full = st.file_uploader(
        "Whole plant photo (optional)",
        type=["jpg", "jpeg", "png"],
        key="id_full"
    )

    leaf = st.file_uploader(
        "Leaf photo (optional)",
        type=["jpg", "jpeg", "png"],
        key="id_leaf"
    )

    if st.button(
        "Identify plant",
        type="primary"
    ):

        chosen = (
            safe_image(full)
            if full
            else safe_image(leaf)
            if leaf
            else None
        )

        if chosen is None:

            st.error(
                "Upload at least one clear plant or leaf photo."
            )

        elif not image_looks_like_plant(chosen):

            st.error(
                "Please upload a plant or leaf photo."
            )

        else:

            st.image(
                chosen,
                caption="Uploaded plant image",
                use_container_width=True
            )

            with st.spinner(
                "Checking supported plant classes..."
            ):

                preds, err = plant_id_model(chosen)

            if err:

                st.error(err)

            elif preds:

                st.success("Possible identification")

                for p in preds:

                    st.write(
                        f"**{p['label']}** — "
                        f"{p['score'] * 100:.1f}% model score"
                    )

                st.caption(
                    "This is not a universal plant-name database. "
                    "Cross-check the result."
                )

                st.link_button(
                    "Search plant name and reference images",
                    google_search_url(
                        "identify this plant leaf "
                        + preds[0]["label"]
                    )
                )

                if st.button(
                    "Find matching webpages and similar images",
                    key="vision_plant_search"
                ):

                    with st.spinner(
                        "Checking visual matches on the web..."
                    ):

                        web_results, web_error = (
                            google_vision_web_detection(chosen)
                        )

                    if web_error:

                        st.warning(web_error)

                    else:

                        show_web_detection_results(
                            web_results
                        )

    st.markdown("### Search plant names")

    query = st.text_input(
        "Search common plant names",
        placeholder="Example: jasmine, mango, Ocimum tenuiflorum"
    )

    matches = [
        x for x in COMMON_PLANTS
        if query.lower() in x.lower()
    ] if query else COMMON_PLANTS

    st.write(
        ", ".join(matches[:30])
        if matches
        else "No local starter-list match."
    )

    if query:

        st.link_button(
            "Search the wider plant world on Google",
            "https://www.google.com/search?q="
            + urllib.parse.quote(
                query + " plant scientific name"
            )
        )


# =====================================================
# WATER PREDICTOR
# =====================================================

elif page == "Intelligent Water Predictor":

    st.title("💧 Intelligent Water Predictor")

    st.write(
        "A combined decision aid using plant type, "
        "size, soil condition and weather."
    )

    x1, x2 = st.columns(2)

    with x1:

        plant_name = st.text_input(
            "Plant name",
            placeholder="Search or type a plant name"
        )

        category = st.selectbox(
            "Plant category",
            PLANT_CATEGORIES
        )

        size = st.number_input(
            "Approximate plant height (cm)",
            min_value=1.0,
            max_value=2000.0,
            value=35.0
        )

        pphoto = st.file_uploader(
            "Plant photo (optional)",
            type=["jpg", "jpeg", "png"],
            key="water_plant"
        )

        lphoto = st.file_uploader(
            "Leaf photo (optional)",
            type=["jpg", "jpeg", "png"],
            key="water_leaf"
        )

    with x2:

        soilphoto = st.file_uploader(
            "Soil photo (optional)",
            type=["jpg", "jpeg", "png"],
            key="water_soil"
        )

        moisture = st.selectbox(
            "Soil moisture",
            MOISTURE_OPTIONS,
            index=2,
            format_func=lambda x: (
                f"{x} — {MOISTURE_TAMIL[x]}"
            )
        )

        temp = st.number_input(
            "Weather temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=30.0
        )

        humidity = st.slider(
            "Air humidity (%)",
            0, 100, 60
        )

        rain = st.slider(
            "Rain chance (%)",
            0, 100, 20
        )

    if soilphoto:

        sim = safe_image(soilphoto)

        if sim:

            st.image(
                sim,
                caption=(
                    "Soil photo received. "
                    "Photo cannot accurately measure moisture."
                ),
                width=240
            )

    if st.button(
        "Predict watering need",
        type="primary"
    ):

        if not plant_name.strip():

            st.error("Enter a plant name first.")

        else:

            answer, reason = watering_advice(
                plant_name,
                size,
                moisture,
                temp,
                rain,
                humidity
            )

            st.markdown(
                f"""
                <div class="pg-card">
                <h2>💧 {answer}</h2>
                <p>{reason}</p>
                <p class="pg-muted">
                Plant: {plant_name} • Category: {category}
                • Height: {size:.0f} cm
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "This is a rule-based estimate. "
                "Check the soil before watering."
            )

            due = date.today() + timedelta(
                days=1 if "soon" in answer.lower()
                else 2 if "tomorrow" in answer.lower()
                else 3
            )

            st.write(
                f"Suggested next check: **{due.strftime('%d %B %Y')}**"
            )

            if plants:

                chosen = st.selectbox(
                    "Save reminder to an existing plant",
                    options=[p["id"] for p in plants],
                    format_func=lambda i: next(
                        p["name"]
                        for p in plants
                        if p["id"] == i
                    )
                )

                if st.button("Save care reminder"):

                    with db() as con:

                        con.execute(
                            """
                            INSERT INTO reminders(
                                plant_id,due_date,message,completed
                            )
                            VALUES(?,?,?,0)
                            """,
                            (
                                chosen,
                                due.isoformat(),
                                answer
                            )
                        )

                    st.success(
                        "Reminder saved in your workspace."
                    )


# =====================================================
# GROWTH PREDICTION
# =====================================================

elif page == "AI Growth Prediction":

    st.title("📈 AI Growth Prediction")

    st.write(
        "Create a visual growth-planning blueprint."
    )

    name = st.text_input(
        "Plant name",
        placeholder="Example: tomato"
    )

    imgfile = st.file_uploader(
        "Plant photo (optional)",
        type=["jpg", "jpeg", "png"],
        key="growth_img"
    )

    g1, g2 = st.columns(2)

    with g1:

        gsoil = st.selectbox(
            "Soil moisture",
            MOISTURE_OPTIONS,
            format_func=lambda x: (
                f"{x} — {MOISTURE_TAMIL[x]}"
            )
        )

        gtemp = st.number_input(
            "Average weather temperature (°C)",
            value=28.0
        )

        gsun = st.selectbox(
            "Daily sunlight",
            ["Low", "Moderate", "Bright", "Very bright"]
        )

    with g2:

        gsize = st.number_input(
            "Current plant height (cm)",
            min_value=1.0,
            max_value=2000.0,
            value=20.0
        )

        gweather = st.selectbox(
            "Weather outlook",
            [
                "Mostly sunny",
                "Partly cloudy",
                "Rainy",
                "Hot and dry",
                "Cool"
            ]
        )

        gnotes = st.text_input(
            "Extra notes (optional)"
        )

    if imgfile:

        gim = safe_image(imgfile)

        if gim:

            st.image(
                gim,
                caption="Current plant reference",
                width=260
            )

            if not image_looks_like_plant(gim):

                st.error(
                    "Please upload a plant or leaf photo."
                )

    if st.button(
        "Generate growth blueprint",
        type="primary"
    ):

        if not name.strip():

            st.error("Enter a plant name.")

        elif imgfile and (
            safe_image(imgfile) is None
            or not image_looks_like_plant(safe_image(imgfile))
        ):

            st.error(
                "Please upload a valid plant or leaf photo."
            )

        else:

            moisture_factor = {
                "Bone Dry / Oven Dry": 0.25,
                "Dry": 0.45,
                "Slightly Moist / Damp": 0.8,
                "Moist": 1.0,
                "Very Moist": 0.8,
                "Wet / Soggy": 0.4,
                "Saturated / Waterlogged": 0.2
            }[gsoil]

            weather_factor = (
                0.8
                if gweather in [
                    "Hot and dry",
                    "Cool",
                    "Rainy"
                ]
                else 1.0
            )

            st.markdown("### Growth blueprint")

            cards = st.columns(3)

            for col, label, scale in zip(
                cards,
                ["Tomorrow", "3 days later", "1 week later"],
                [1.01, 1.04, 1.09]
            ):

                with col:

                    st.image(
                        growth_blueprint(
                            None,
                            name,
                            label,
                            scale
                        ),
                        use_container_width=True
                    )

                    st.markdown(f"**{label}**")

            st.info(
                "Condition readiness indicator: "
                f"{moisture_factor * weather_factor * 100:.0f}/100"
            )

            st.caption(
                "Illustrative planning indicator, "
                "not a biological growth forecast."
            )


# =====================================================
# DIGITAL TWIN
# =====================================================

elif page == "Plant Digital Twin":

    st.title("🪴 Plant Digital Twin")

    if not plants:

        st.info(
            "Add a plant in Personal Workspace "
            "to create its Digital Twin."
        )

    else:

        selected = st.selectbox(
            "Choose your plant",
            options=[p["id"] for p in plants],
            format_func=lambda i: next(
                p["name"]
                for p in plants
                if p["id"] == i
            )
        )

        plant = next(
            p for p in plants
            if p["id"] == selected
        )

        st.markdown(f"### {plant['name']}")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Category",
            plant["category"] or "Not set"
        )

        c2.metric(
            "Recorded height",
            f"{plant['size_cm'] or 0:g} cm"
        )

        c3.metric(
            "Added",
            plant["created_at"][:10]
        )

        if plant["photo"]:

            st.image(
                Image.open(
                    io.BytesIO(plant["photo"])
                ),
                caption="Plant reference photo",
                width=350
            )

        with db() as con:

            logs = con.execute(
                """
                SELECT log_date,height_cm,notes
                FROM logs
                WHERE plant_id=?
                ORDER BY log_date
                """,
                (selected,)
            ).fetchall()

        if logs:

            df = pd.DataFrame(
                [dict(x) for x in logs]
            )

            st.plotly_chart(
                px.line(
                    df,
                    x="log_date",
                    y="height_cm",
                    markers=True,
                    title="Recorded growth history"
                ),
                use_container_width=True
            )

            st.dataframe(
                df,
                use_container_width=True
            )

        else:

            st.info(
                "No growth measurements yet. "
                "Add measurements in Personal Workspace."
            )

        st.caption(
            "Digital Twin is a digital record of your plant's "
            "photos, measurements and care history."
        )


# =====================================================
# PERSONAL WORKSPACE
# =====================================================

elif page == "Personal Workspace":

    st.title("👤 Personal Workspace")

    st.markdown("### Add a plant")

    with st.form("add_plant_form"):

        pname = st.text_input("Plant name")

        pcat = st.selectbox(
            "Plant category",
            PLANT_CATEGORIES
        )

        psize = st.number_input(
            "Current height (cm)",
            min_value=0.0,
            max_value=2000.0,
            value=10.0
        )

        pimg = st.file_uploader(
            "Plant photo (optional)",
            type=["jpg", "jpeg", "png"],
            key="collection_photo"
        )

        submitted = st.form_submit_button(
            "Save plant"
        )

        if submitted:

            if not pname.strip():

                st.error("Enter a plant name.")

            else:

                raw = (
                    pimg.getvalue()
                    if pimg
                    else None
                )

                if pimg:

                    check = safe_image(pimg)

                    if (
                        check is None
                        or not image_looks_like_plant(check)
                    ):

                        st.error(
                            "Please upload a valid plant photo."
                        )

                        raw = None

                    else:

                        save_plant_record(
                            username,
                            pname.strip(),
                            pcat,
                            psize,
                            raw
                        )

                        st.success("Plant saved.")
                        st.rerun()

                else:

                    save_plant_record(
                        username,
                        pname.strip(),
                        pcat,
                        psize,
                        None
                    )

                    st.success("Plant saved.")
                    st.rerun()

    st.markdown("### Search plant directory")

    search = st.text_input(
        "Find a plant",
        placeholder="Type a common or scientific name"
    )

    found = [
        x for x in COMMON_PLANTS
        if search.lower() in x.lower()
    ] if search else COMMON_PLANTS

    st.write(
        ", ".join(found)
        if found
        else "No match in the starter list."
    )

    if search:

        st.link_button(
            "Search more plant names online",
            "https://www.google.com/search?q="
            + urllib.parse.quote(
                search + " plant species"
            )
        )

    st.markdown("### Record growth")

    if plants:

        pid = st.selectbox(
            "Plant",
            options=[p["id"] for p in plants],
            format_func=lambda i: next(
                p["name"]
                for p in plants
                if p["id"] == i
            )
        )

        h = st.number_input(
            "Measured height (cm)",
            min_value=0.0,
            max_value=2000.0,
            value=10.0,
            key="height_log"
        )

        note = st.text_input("Observation")

        if st.button("Save growth measurement"):

            with db() as con:

                con.execute(
                    """
                    INSERT INTO logs(
                        plant_id,log_date,height_cm,notes
                    )
                    VALUES(?,?,?,?)
                    """,
                    (
                        pid,
                        date.today().isoformat(),
                        h,
                        note
                    )
                )

                con.execute(
                    """
                    UPDATE plants
                    SET size_cm=?
                    WHERE id=?
                    """,
                    (h, pid)
                )

            st.success("Growth record saved.")
            st.rerun()

        st.markdown("### Existing plants")

        for p in plants:

            with st.expander(
                f"{p['name']} — {p['category']}"
            ):

                if p["photo"]:

                    st.image(
                        Image.open(
                            io.BytesIO(p["photo"])
                        ),
                        width=180
                    )

                st.write(
                    f"Height: {p['size_cm']} cm"
                )

    else:

        st.info("No plants saved yet.")


# =====================================================
# PDF REPORT
# =====================================================

elif page == "Plant Health PDF":

    st.title("📄 Plant Health Report")

    if not REPORTLAB_OK:

        st.error(
            "PDF library missing. Install reportlab."
        )

    else:

        pname = st.text_input(
            "Plant name for report"
        )

        diagnosis = st.text_input(
            "Latest diagnosis / observation",
            value="Not assessed"
        )

        care = st.text_area(
            "Care notes",
            value=(
                "Check soil moisture before watering. "
                "Provide suitable light and airflow."
            )
        )

        if st.button("Create PDF report"):

            pdf = make_pdf(
                pname or "Unspecified plant",
                diagnosis,
                care
            )

            if pdf:

                st.download_button(
                    "Download plant health PDF",
                    data=pdf,
                    file_name="plant_health_report.pdf",
                    mime="application/pdf"
                )

                st.success("Report created.")


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:25px; color:#a4f3ca;">
<h3>🌿 Plant Guardian AI</h3>
<p>Created by <b>G. Sujan & V. Sharan</b></p>
<p>Class 11A</p>
<p>Dasar Higher Secondary School</p>
<p>Tiruninravur</p>
<p style="font-size:11px;">
AI-Powered Plant Care Project • 2026
</p>
</div>
""", unsafe_allow_html=True)
