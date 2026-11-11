
import streamlit as st
import streamlit.components.v1 as components
import timm
import torch
import requests
import sqlite3
import plotly.express as px
from transformers import pipeline
from PIL import Image
from datetime import date, datetime, timedelta

# =====================================================
# PAGE CONFIGURATION AND THEME
# =====================================================

st.set_page_config(
    page_title="Plant Guardian AI",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f1f8f3, #e6f2e9);
}
[data-testid="stSidebar"] {
    background: #123b2d;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
.main-title {
    color: #145c3a;
    font-size: 42px;
    font-weight: 800;
}
.sub-title {
    color: #567767;
    font-size: 17px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #dceadd;
}
.stButton button {
    background: #176b43;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}
.stButton button:hover {
    background: #0e5031;
    color: white;
}
section[data-testid="stMain"] .block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🌿 PLANT GUARDIAN AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Intelligent Plant Identification • Disease Detection • Smart Care</div>',
    unsafe_allow_html=True
)

# =====================================================
# DATABASE
# =====================================================

DB = "plant_guardian.db"

def get_db():
    return sqlite3.connect(DB, check_same_thread=False)

def init_db():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS plants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            crop TEXT NOT NULL,
            added TEXT NOT NULL,
            last_watered TEXT NOT NULL,
            water_interval INTEGER NOT NULL,
            notes TEXT DEFAULT ''
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS growth (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id INTEGER NOT NULL,
            record_date TEXT NOT NULL,
            height REAL NOT NULL,
            notes TEXT DEFAULT ''
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id INTEGER NOT NULL,
            task TEXT NOT NULL,
            due_date TEXT NOT NULL,
            completed INTEGER DEFAULT 0
        )
        """)

init_db()

def get_plants():
    with get_db() as conn:
        return conn.execute(
            "SELECT id, name, crop, added, last_watered, water_interval, notes FROM plants ORDER BY id DESC"
        ).fetchall()

def get_plant(plant_id):
    with get_db() as conn:
        return conn.execute(
            "SELECT id, name, crop, added, last_watered, water_interval, notes FROM plants WHERE id=?",
            (plant_id,)
        ).fetchone()

# =====================================================
# PLANT KNOWLEDGE BASE
# =====================================================

plant_profiles = {
    "Tomato": {
        "water": 2,
        "sun": "6–8 hours of direct sunlight",
        "soil": "Well-drained, fertile soil",
        "fertilizer": "Use a balanced vegetable fertiliser according to its label. During fruiting, use a suitable potassium-containing fertiliser if soil needs it."
    },
    "Chilli": {
        "water": 2,
        "sun": "6–8 hours of sunlight",
        "soil": "Well-drained loamy soil",
        "fertilizer": "Use a balanced vegetable fertiliser in small label-directed doses. Avoid excess nitrogen."
    },
    "Brinjal": {
        "water": 2,
        "sun": "6–8 hours of sunlight",
        "soil": "Fertile, well-drained soil",
        "fertilizer": "Use a balanced vegetable fertiliser according to soil needs and product instructions."
    },
    "Watermelon": {
        "water": 2,
        "sun": "6–8 hours of full sunlight",
        "soil": "Sandy loam with good drainage",
        "fertilizer": "Use a crop-appropriate balanced fertiliser based on soil test and growth stage."
    },
    "Papaya": {
        "water": 3,
        "sun": "Full sunlight",
        "soil": "Deep, well-drained soil",
        "fertilizer": "Papaya needs balanced nutrition. Use a locally recommended papaya fertiliser schedule based on plant age and soil test."
    },
    "Mango": {
        "water": 5,
        "sun": "Full sunlight",
        "soil": "Deep, well-drained soil",
        "fertilizer": "Use a tree-age and soil-test-based nutrient plan. Avoid applying fertiliser against the trunk."
    },
    "Banana": {
        "water": 2,
        "sun": "Full sunlight",
        "soil": "Moist but well-drained fertile soil",
        "fertilizer": "Banana has high nutrient demand. Follow local soil-test-based recommendations, especially for potassium."
    },
    "Rice": {
        "water": 2,
        "sun": "Full sunlight",
        "soil": "Suitable paddy soil",
        "fertilizer": "Follow local agricultural department recommendations and soil testing."
    },
    "Potato": {
        "water": 3,
        "sun": "6 hours or more",
        "soil": "Loose, well-drained soil",
        "fertilizer": "Use a potato-specific nutrient plan based on soil testing."
    },
    "Grape": {
        "water": 3,
        "sun": "Full sunlight",
        "soil": "Well-drained soil",
        "fertilizer": "Use balanced nutrients based on vine age and soil test."
    },
    "Other": {
        "water": 3,
        "sun": "Depends on species",
        "soil": "Identify the plant before selecting a soil mix",
        "fertilizer": "Identify the species and check soil nutrients before choosing fertiliser."
    }
}

# =====================================================
# AI MODEL LOADERS
# =====================================================

@st.cache_resource(show_spinner=False)
def load_ai_model():
    return pipeline(
        "image-classification",
        model="neghani/plantvillage-mobilenetv2",
        revision="v1.0"
    )

@st.cache_resource(show_spinner=False)
def load_plant_id_model():
    model = timm.create_model(
        "hf-hub:ixrbhii/leaf-crop-detector-16-convnext",
        pretrained=True
    ).eval()

    config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**config)
    labels = model.pretrained_cfg["label_names"]

    return model, transform, labels

def scan_disease(image):
    classifier = load_ai_model()
    return classifier(image, top_k=3)

def identify_plant(image):
    model, transform, labels = load_plant_id_model()
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        probabilities = model(tensor).softmax(dim=1)[0]

    indices = torch.argsort(
        probabilities, descending=True
    )[:3].tolist()

    return [
        (str(labels[i]), probabilities[i].item())
        for i in indices
    ]

def disease_details(label):
    text = label.lower().replace("___", " ").replace("_", " ")
    if "healthy" in text:
        return (
            "No disease pattern was recognised by the model.",
            "Continue regular monitoring, proper watering and good airflow.",
            "Do not apply pesticides unless a problem is confirmed."
        )
    if "early blight" in text:
        return (
            "Possible early blight, a fungal disease that can cause leaf spots.",
            "Remove severely affected leaves, avoid overhead watering and improve airflow.",
            "Confirm the disease before using a fungicide. Follow the product label."
        )
    if "late blight" in text:
        return (
            "Possible late blight, which can spread quickly in favourable conditions.",
            "Separate affected plants where practical, remove infected plant material safely and seek local agricultural advice.",
            "Do not compost suspected infected material. Use only locally approved treatment after confirmation."
        )
    if "black rot" in text:
        return (
            "Possible black rot, a disease that can cause dark leaf lesions.",
            "Remove affected material, keep foliage dry and improve plant spacing.",
            "Check the crop-specific diagnosis before selecting treatment."
        )
    if "powdery mildew" in text:
        return (
            "Possible powdery mildew, a fungal disease that can form a pale powdery coating.",
            "Improve airflow, avoid overcrowding and remove badly affected leaves.",
            "Use crop-approved treatment only when the disease is confirmed."
        )
    return (
        "The model has detected a possible plant disease category.",
        "Inspect both sides of the leaf, check nearby plants and remove severely affected leaves if appropriate.",
        "A local plant expert should confirm the diagnosis before treatment."
    )

# =====================================================
# WEATHER SERVICE
# =====================================================

def get_weather(city):
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo = requests.get(
        geo_url,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=12
    )
    geo.raise_for_status()
    places = geo.json().get("results", [])

    if not places:
        return None

    place = places[0]
    forecast = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "daily": "precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 1
        },
        timeout=12
    )
    forecast.raise_for_status()

    return place, forecast.json()

# =====================================================
# AUTOMATIC WATERING CALCULATION
# =====================================================

def watering_due(plant):
    last = date.fromisoformat(plant[4])
    interval = int(plant[5])
    return last + timedelta(days=interval)

def browser_notification(reminders):
    safe_items = []
    for item in reminders:
        safe_items.append({
            "plant": item[0],
            "task": item[1],
            "date": item[2]
        })

    components.html("""
    <div style="font-family:Arial;padding:8px">
      <button id="enable" style="background:#176b43;color:white;border:0;padding:10px 16px;border-radius:8px">
        Enable browser notifications
      </button>
      <p id="status" style="font-size:12px;color:#456b50"></p>
    </div>
    <script>
    const due = """ + str(safe_items).replace("'", '"') + """;
    const button = document.getElementById("enable");
    const status = document.getElementById("status");

    button.onclick = async () => {
      if (!("Notification" in window)) {
        status.textContent = "This browser does not support notifications.";
        return;
      }
      const permission = await Notification.requestPermission();
      status.textContent = "Notification permission: " + permission;
      if (permission === "granted") {
        due.forEach(item => {
          new Notification("Plant Guardian AI", {
            body: item.plant + ": " + item.task + " is due today."
          });
        });
      }
    };
    </script>
    """, height=85)

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("## 🌿 PLANT GUARDIAN")
st.sidebar.caption("SMART GARDEN MANAGEMENT")

pages = [
    "Dashboard",
    "AI Disease Scanner",
    "Plant Identification",
    "Disease Information",
    "Plant Doctor",
    "Smart Watering Guide",
    "Fertilizer Recommendation",
    "Weather-Based Care",
    "Plant Growth Tracker",
    "Smart Reminders",
    "AI Pest Assistant",
    "Soil Health Assistant",
    "Nutrient Deficiency Assistant",
    "Sunlight Advisor",
    "My Plant Collection"
]

page = st.sidebar.radio("NAVIGATION", pages)

st.sidebar.markdown("---")
st.sidebar.caption("Plant Guardian AI • 2026")

plants = get_plants()

# =====================================================
# 1. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.header("🌱 Garden Overview")
    st.write("Monitor your plants and keep your garden healthy.")

    today = date.today()
    due_plants = [
        p for p in plants
        if watering_due(p) <= today
    ]

    a, b, c, d = st.columns(4)
    a.metric("My Plants", len(plants))
    b.metric("Watering Due", len(due_plants))
    c.metric("Growth Records", len(get_db().execute(
        "SELECT id FROM growth"
    ).fetchall()))
    d.metric("Today's Date", today.strftime("%d %b"))

    st.divider()
    st.subheader("💧 Plant Care Status")

    if due_plants:
        for p in due_plants:
            st.warning(
                f"{p[1]} ({p[2]}) may need a watering check. "
                f"Calculated due date: {watering_due(p)}"
            )
    else:
        st.success("No registered plants are currently due for watering.")

    st.subheader("🌿 Your Garden")
    if plants:
        cols = st.columns(3)
        for i, p in enumerate(plants):
            with cols[i % 3]:
                st.markdown(f"### 🌱 {p[1]}")
                st.write(f"Type: {p[2]}")
                st.write(f"Next watering check: {watering_due(p)}")
    else:
        st.info("Add your first plant in My Plant Collection.")

# =====================================================
# 2. AI DISEASE SCANNER
# =====================================================

elif page == "AI Disease Scanner":

    st.header("🔬 AI Plant Disease Scanner")
    st.write("Upload a clear leaf image to get possible disease predictions.")

    file = st.file_uploader(
        "Upload leaf photo",
        type=["jpg", "jpeg", "png"],
        key="disease_file"
    )

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, width=360)

        if st.button("Analyse Leaf", type="primary"):
            try:
                with st.spinner("AI is analysing the leaf..."):
                    results = scan_disease(image)

                for result in results:
                    label = result["label"]
                    score = float(result["score"])
                    st.write(f"**{label.replace('___', ' – ').replace('_', ' ')}**")
                    st.progress(score)
                    st.write(f"Model score: {score * 100:.2f}%")

                st.warning(
                    "Model scores are not verified diagnostic probabilities. "
                    "The model supports a limited set of crop and disease classes."
                )
            except Exception as e:
                st.error("Disease scanning failed.")
                st.code(str(e))

# =====================================================
# 3. PLANT IDENTIFICATION
# =====================================================

elif page == "Plant Identification":

    st.header("🌿 AI Plant Identification")
    st.write("Estimate the crop category from a clear leaf photograph.")

    file = st.file_uploader(
        "Upload a leaf or plant photo",
        type=["jpg", "jpeg", "png"],
        key="identify_file"
    )

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, width=360)

        if st.button("Identify Plant", type="primary"):
            try:
                with st.spinner("Identifying plant..."):
                    results = identify_plant(image)

                st.subheader("Possible matches")
                for name, score in results:
                    st.write(f"**{name}**")
                    st.progress(score)
                    st.caption(f"Relative model score: {score * 100:.2f}%")

                st.info(
                    "This model recognises 16 crop categories, not every plant "
                    "species worldwide. Similar plants may be confused. "
                    "Compare leaf shape, veins, edges, arrangement and flowers "
                    "before accepting an identification."
                )
            except Exception as e:
                st.error("Plant identification failed.")
                st.code(str(e))

# =====================================================
# 4. DISEASE INFORMATION
# =====================================================

elif page == "Disease Information":

    st.header("📚 Disease Information from Leaf Photo")

    file = st.file_uploader(
        "Upload affected leaf photo",
        type=["jpg", "jpeg", "png"],
        key="disease_info_file"
    )

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, width=350)

        if st.button("Detect and Explain Disease", type="primary"):
            try:
                with st.spinner("Checking leaf..."):
                    results = scan_disease(image)

                label = results[0]["label"]
                score = float(results[0]["score"])
                description, care, prevention = disease_details(label)

                st.subheader(label.replace("___", " – ").replace("_", " "))
                st.metric("Top model score", f"{score * 100:.2f}%")
                st.write(description)

                st.markdown("### What you should do")
                st.write(care)

                st.markdown("### Prevention")
                st.write(prevention)

                st.markdown("### Other possible matches")
                for result in results[1:]:
                    st.write(
                        f"{result['label']} – {result['score'] * 100:.2f}%"
                    )

                st.warning(
                    "A model score is not the same as diagnostic certainty. "
                    "Confirm with an agricultural expert before treatment."
                )
            except Exception as e:
                st.error("Analysis failed.")
                st.code(str(e))

# =====================================================
# 5. PLANT DOCTOR
# =====================================================

elif page == "Plant Doctor":

    st.header("🩺 AI Plant Doctor")
    st.write("Upload a plant photo for a structured preliminary assessment.")

    file = st.file_uploader(
        "Upload plant or leaf photo",
        type=["jpg", "jpeg", "png"],
        key="doctor_file"
    )

    crop = st.selectbox(
        "Select crop if known",
        ["Unknown"] + list(plant_profiles.keys())
    )

    symptoms = st.multiselect(
        "What else do you observe?",
        [
            "Yellow leaves",
            "Brown spots",
            "White powder",
            "Wilting",
            "Holes in leaves",
            "Curling leaves",
            "Slow growth",
            "No visible symptoms"
        ]
    )

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, width=350)

        if st.button("Ask Plant Doctor", type="primary"):
            try:
                with st.spinner("Preparing plant assessment..."):
                    results = scan_disease(image)

                label = results[0]["label"]
                score = float(results[0]["score"])
                description, care, prevention = disease_details(label)

                st.subheader("Preliminary Assessment")
                st.write(f"**AI model result:** {label}")
                st.write(f"**Model score:** {score * 100:.2f}%")
                st.write(description)

                st.markdown("### Step-by-step care plan")
                st.write("1. Inspect the upper and lower sides of the leaves.")
                st.write("2. Check soil moisture and confirm that the pot drains properly.")
                st.write("3. Keep affected plants separated if a contagious disease is suspected.")
                st.write("4. Remove severely affected plant material using clean tools, where appropriate.")
                st.write("5. Avoid applying multiple chemicals without confirming the cause.")
                st.write("6. Monitor new leaf growth over the next several days.")

                st.markdown("### Suggested action")
                st.write(care)

                st.markdown("### Prevention")
                st.write(prevention)

                st.markdown("### Your observations")
                st.write(", ".join(symptoms) if symptoms else "No additional symptoms selected.")

                if crop != "Unknown":
                    st.write(f"Selected crop: {crop}")

                st.warning(
                    "The disease model is not a universal plant doctor. "
                    "Its result may be wrong for unsupported plants or conditions. "
                    "Seek local agricultural advice for serious damage."
                )
            except Exception as e:
                st.error("Plant Doctor could not complete the assessment.")
                st.code(str(e))

# =====================================================
# 6. SMART WATERING GUIDE
# =====================================================

elif page == "Smart Watering Guide":

    st.header("💧 Smart Watering Guide")
    st.write("Get plant-specific watering guidance.")

    crop = st.selectbox("Choose plant", list(plant_profiles.keys()))
    soil = st.selectbox(
        "Soil moisture",
        ["Dry", "Slightly moist", "Wet"]
    )
    condition = st.selectbox(
        "Weather",
        ["Hot", "Normal", "Rainy", "Very humid"]
    )

    if st.button("Generate Watering Advice", type="primary"):

        if soil == "Wet":
            st.success("Do not water now. Check drainage and let excess water drain.")
            st.write("Check the soil again before the next watering.")
        elif soil == "Dry":
            st.success("Water slowly around the root zone until the soil is evenly moist.")
            st.write("Avoid leaving the pot standing in collected water.")
        else:
            st.info("The soil is slightly moist. Wait and check again before watering.")

        if condition == "Rainy":
            st.write("Rain may supply water. Reduce or skip watering if the soil is already moist.")
        elif condition == "Hot":
            st.write("Check moisture more often during hot weather. Water in the cooler morning hours.")
        elif condition == "Very humid":
            st.write("Allow the soil surface to dry as appropriate and avoid overwatering.")
        else:
            st.write("Follow soil moisture rather than a fixed calendar alone.")

        st.write(f"General plant sunlight need: {plant_profiles[crop]['sun']}")
        st.caption("These are general starting points. Soil, pot size and plant age change watering needs.")

# =====================================================
# 7. FERTILIZER RECOMMENDATION
# =====================================================

elif page == "Fertilizer Recommendation":

    st.header("🌾 Plant Fertilizer Advisor")
    st.write("Get a cautious recommendation based on crop and growth stage.")

    crop = st.selectbox("Select crop", list(plant_profiles.keys()))
    stage = st.selectbox(
        "Growth stage",
        ["Seedling", "Vegetative", "Flowering", "Fruiting"]
    )
    soil_test = st.selectbox(
        "Soil test status",
        ["Not tested", "Low nutrients", "Balanced", "High nutrients"]
    )

    file = st.file_uploader(
        "Optional plant photo",
        type=["jpg", "jpeg", "png"],
        key="fertilizer_photo"
    )

    if file:
        st.image(Image.open(file).convert("RGB"), width=300)

    if st.button("Get Fertilizer Recommendation", type="primary"):

        st.subheader(f"Guidance for {crop}")
        st.write(plant_profiles[crop]["fertilizer"])

        stage_advice = {
            "Seedling": "Young plants are sensitive. Avoid strong or concentrated fertiliser.",
            "Vegetative": "Support healthy leaf and stem growth with balanced nutrition.",
            "Flowering": "Avoid excessive nitrogen, which may encourage leaves over flowers.",
            "Fruiting": "Maintain balanced nutrition and check potassium needs."
        }

        st.write(f"**Growth stage:** {stage}")
        st.write(stage_advice[stage])

        if soil_test == "Low nutrients":
            st.info("Use a soil-test-based nutrient correction recommended for this crop.")
        elif soil_test == "High nutrients":
            st.warning("Do not add more fertiliser until the soil test is reviewed.")
        elif soil_test == "Balanced":
            st.success("Maintain the current nutrient plan and monitor plant growth.")
        else:
            st.info("A soil test is recommended before deciding the fertiliser quantity.")

        st.warning(
            "A photograph cannot measure soil nutrients or determine an exact "
            "fertiliser dose. Follow the product label and local crop recommendations. "
            "Do not guess grams or spoonfuls from a photo."
        )

# =====================================================
# 8. WEATHER-BASED PLANT CARE
# =====================================================

elif page == "Weather-Based Care":

    st.header("🌦️ Live Weather-Based Plant Care")

    city = st.text_input("Enter your city", placeholder="Example: Chennai")
    crop = st.selectbox("Select your plant", list(plant_profiles.keys()))

    if st.button("Get Live Weather Advice", type="primary"):

        if not city.strip():
            st.error("Enter a city name.")
        else:
            try:
                with st.spinner("Checking live weather..."):
                    result = get_weather(city.strip())

                if result is None:
                    st.error("City not found. Check the spelling.")
                else:
                    place, data = result
                    current = data.get("current", {})
                    daily = data.get("daily", {})

                    temp = current.get("temperature_2m")
                    humidity = current.get("relative_humidity_2m")
                    rain = current.get("precipitation")
                    rain_chance = (
                        daily.get("precipitation_probability_max", [None])[0]
                    )

                    st.subheader(f"Weather in {place['name']}, {place.get('country', '')}")

                    x, y, z = st.columns(3)
                    x.metric("Temperature", f"{temp} °C")
                    y.metric("Humidity", f"{humidity}%")
                    z.metric("Rain now", f"{rain} mm")

                    st.markdown("### Plant-specific advice")

                    if rain_chance is not None and rain_chance >= 60:
                        st.write("1. Rain is likely. Check the soil before watering.")
                    elif temp is not None and temp >= 32:
                        st.write("1. Hot conditions: check soil moisture more frequently.")
                    else:
                        st.write("1. Check soil moisture before deciding to water.")

                    if humidity is not None and humidity >= 80:
                        st.write("2. Humidity is high. Improve airflow and avoid wet leaves.")
                    else:
                        st.write("2. Maintain good spacing and airflow around the plant.")

                    st.write(f"3. General sunlight requirement: {plant_profiles[crop]['sun']}.")
                    st.write(f"4. Soil preference: {plant_profiles[crop]['soil']}.")

                    st.caption("Weather information is supplied by Open-Meteo. Forecasts can change.")

            except Exception as e:
                st.error("Could not retrieve live weather. Check your internet connection.")
                st.caption(str(e))

# =====================================================
# 9. PLANT GROWTH TRACKER
# =====================================================

elif page == "Plant Growth Tracker":

    st.header("📈 Plant Growth Tracker")

    if not plants:
        st.info("Add a plant in My Plant Collection before recording growth.")
    else:
        options = {f"{p[1]} ({p[2]}) – ID {p[0]}": p[0] for p in plants}
        selected = st.selectbox("Choose plant", list(options.keys()))
        plant_id = options[selected]

        with st.form("growth_form"):
            height = st.number_input(
                "Measured plant height (cm)",
                min_value=0.1,
                max_value=10000.0,
                value=10.0
            )
            record_date = st.date_input("Measurement date", value=date.today())
            note = st.text_area("Optional observations")
            save = st.form_submit_button("Save Growth Record")

            if save:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO growth (plant_id, record_date, height, notes) VALUES (?, ?, ?, ?)",
                        (plant_id, str(record_date), height, note)
                    )
                st.success("Growth record saved.")

        with get_db() as conn:
            records = conn.execute(
                "SELECT record_date, height, notes FROM growth WHERE plant_id=? ORDER BY record_date",
                (plant_id,)
            ).fetchall()

        if records:
            st.subheader("Growth History")
            st.dataframe(
                [
                    {"Date": r[0], "Height (cm)": r[1], "Notes": r[2]}
                    for r in records
                ],
                use_container_width=True
            )

            if len(records) >= 2:
                chart = px.line(
                    x=[r[0] for r in records],
                    y=[r[1] for r in records],
                    markers=True,
                    labels={"x": "Date", "y": "Height (cm)"},
                    title="Plant Height Progress"
                )
                st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("No growth measurements recorded yet.")

# =====================================================
# 10. SMART REMINDERS
# =====================================================

elif page == "Smart Reminders":

    st.header("⏰ Automatic Plant Care Reminders")
    st.write("Watering due dates are calculated from the last watering date and the selected interval.")

    if not plants:
        st.info("Add plants in My Plant Collection first.")
    else:
        due_today = []

        for p in plants:
            due = watering_due(p)
            days_left = (due - date.today()).days

            if days_left <= 0:
                due_today.append((p[1], "Watering check", str(due)))

            st.markdown(f"### 🌱 {p[1]}")
            st.write(f"Next watering check: **{due}**")

            if days_left < 0:
                st.error(f"Watering check overdue by {abs(days_left)} day(s). Check the soil.")
            elif days_left == 0:
                st.warning("Watering check due today. Check the soil before watering.")
            else:
                st.success(f"Next check in {days_left} day(s).")

            if st.button(f"I watered {p[1]}", key=f"water_{p[0]}"):
                with get_db() as conn:
                    conn.execute(
                        "UPDATE plants SET last_watered=? WHERE id=?",
                        (str(date.today()), p[0])
                    )
                st.success("Last watering date updated.")
                st.rerun()

        st.divider()
        st.subheader("Browser Notifications")
        st.write("Allow notifications to receive alerts for plants due today.")
        if due_today:
            browser_notification(due_today)
        else:
            st.info("No watering checks are due today.")

        st.caption(
            "This prototype checks due dates when the website is opened. "
            "It does not send notifications after the browser is closed."
        )

# =====================================================
# 11. AI PEST ASSISTANT
# =====================================================

elif page == "AI Pest Assistant":

    st.header("🐛 Plant Pest Assistant")

    file = st.file_uploader(
        "Upload a photo of the affected leaf",
        type=["jpg", "jpeg", "png"],
        key="pest_photo"
    )

    pest_symptoms = st.multiselect(
        "Select visible signs",
        [
            "Small holes",
            "Sticky surface",
            "Visible insects",
            "Webbing",
            "Leaf curling",
            "Yellow patches",
            "Chewed edges"
        ]
    )

    if file:
        st.image(Image.open(file).convert("RGB"), width=350)

    if st.button("Assess Pest Symptoms", type="primary"):

        if not pest_symptoms:
            st.warning("Select at least one visible sign.")
        else:
            st.subheader("Possible causes")

            if "Sticky surface" in pest_symptoms or "Leaf curling" in pest_symptoms:
                st.write("Possible sap-feeding insects such as aphids or whiteflies.")
            if "Webbing" in pest_symptoms:
                st.write("Possible spider mites. Inspect the underside of leaves.")
            if "Small holes" in pest_symptoms or "Chewed edges" in pest_symptoms:
                st.write("Possible caterpillars, beetles or other chewing insects.")
            if "Visible insects" in pest_symptoms:
                st.write("Photograph the insect clearly for identification by an expert.")

            st.markdown("### Suggested actions")
            st.write("1. Inspect both sides of leaves and nearby stems.")
            st.write("2. Remove heavily infested leaves where appropriate.")
            st.write("3. Avoid unnecessary pesticide use.")
            st.write("4. Seek local agricultural advice if infestation spreads.")

            st.warning("This is symptom-based guidance, not an AI-confirmed pest identification.")

# =====================================================
# 12. SOIL HEALTH ASSISTANT
# =====================================================

elif page == "Soil Health Assistant":

    st.header("🪴 Soil Health Assistant")

    soil_type = st.selectbox(
        "Soil type",
        ["Sandy", "Clay", "Loamy", "Silty", "Unknown"]
    )
    drainage = st.selectbox(
        "Drainage condition",
        ["Fast", "Good", "Poor", "Unknown"]
    )
    ph = st.number_input(
        "Soil pH (if known)",
        min_value=0.0,
        max_value=14.0,
        value=7.0
    )
    moisture = st.selectbox(
        "Soil moisture",
        ["Dry", "Moist", "Waterlogged"]
    )

    if st.button("Analyse Soil Details", type="primary"):

        st.subheader("Soil Guidance")

        advice = {
            "Sandy": "Sandy soil drains quickly. Organic matter can improve water retention.",
            "Clay": "Clay soil may hold excess water. Improve structure and drainage carefully.",
            "Loamy": "Loamy soil is often suitable for many garden plants.",
            "Silty": "Silty soil can retain moisture. Avoid compaction.",
            "Unknown": "A simple soil texture test or local soil lab can help identify soil type."
        }

        st.write(advice[soil_type])

        if drainage == "Poor":
            st.warning("Improve drainage before increasing watering.")
        if moisture == "Waterlogged":
            st.warning("Avoid watering and check drainage and root health.")

        if ph < 5.5:
            st.info("The entered pH is acidic. Check crop tolerance and seek soil-test advice.")
        elif ph > 7.5:
            st.info("The entered pH is alkaline. Check nutrient availability with a soil test.")
        else:
            st.success("The entered pH is within a commonly suitable range for many plants, but crop needs vary.")

        st.caption("This is an educational assessment, not a laboratory soil test.")

# =====================================================
# 13. NUTRIENT DEFICIENCY ASSISTANT
# =====================================================

elif page == "Nutrient Deficiency Assistant":

    st.header("🍃 Nutrient Deficiency Assistant")

    file = st.file_uploader(
        "Upload leaf photo",
        type=["jpg", "jpeg", "png"],
        key="nutrient_photo"
    )

    symptom = st.selectbox(
        "Main leaf symptom",
        [
            "Older leaves yellow",
            "New leaves yellow",
            "Yellowing between veins",
            "Brown leaf edges",
            "Purple or reddish leaves",
            "Poor growth"
        ]
    )

    if file:
        st.image(Image.open(file).convert("RGB"), width=350)

    if st.button("Check Possible Deficiency", type="primary"):

        guidance = {
            "Older leaves yellow":
                "Possible nitrogen deficiency, but overwatering and root problems can also cause this.",
            "New leaves yellow":
                "Possible iron or other micronutrient availability issue. Soil pH may be involved.",
            "Yellowing between veins":
                "Possible magnesium or iron-related issue depending on leaf age and crop.",
            "Brown leaf edges":
                "Possible potassium issue, salt stress or irregular watering.",
            "Purple or reddish leaves":
                "Possible phosphorus availability issue or temperature stress.",
            "Poor growth":
                "Possible nutrient imbalance, root restriction, unsuitable light or watering."
        }

        st.info(guidance[symptom])
        st.write("Recommended next steps:")
        st.write("1. Check soil moisture and root drainage.")
        st.write("2. Compare symptoms on older and newer leaves.")
        st.write("3. Check soil pH and obtain a soil test.")
        st.write("4. Correct nutrients only after confirming the likely cause.")

        st.warning(
            "Leaf colour alone cannot confirm a nutrient deficiency. "
            "This feature is symptom-based, not a trained deficiency-detection model."
        )

# =====================================================
# 14. SUNLIGHT ADVISOR
# =====================================================

elif page == "Sunlight Advisor":

    st.header("☀️ Plant Sunlight Advisor")

    crop = st.selectbox("Choose your plant", list(plant_profiles.keys()))
    hours = st.slider("Daily sunlight received (hours)", 0, 12, 5)

    if st.button("Check Sunlight Suitability", type="primary"):

        st.write(f"**General sunlight requirement:** {plant_profiles[crop]['sun']}")

        if hours >= 6:
            st.success("Your plant receives a substantial amount of sunlight. Check whether the species tolerates direct midday sun.")
        elif hours >= 3:
            st.info("This may be partial sunlight. Check whether this crop needs full sun.")
        else:
            st.warning("Sunlight may be limited for this crop. Consider a brighter location if suitable.")

        st.write("Observe leaf condition and growth over time before changing the plant's location.")

# =====================================================
# 15. MY PLANT COLLECTION
# =====================================================

elif page == "My Plant Collection":

    st.header("🌿 My Plant Collection")
    st.write("Save your plants and manage their care records.")

    with st.form("add_plant_form"):

        name = st.text_input("Plant nickname")
        crop = st.selectbox("Plant type", list(plant_profiles.keys()))
        interval = st.number_input(
            "Watering check interval (days)",
            min_value=1,
            max_value=30,
            value=plant_profiles[crop]["water"]
        )
        last_watered = st.date_input(
            "Last watered date",
            value=date.today()
        )
        notes = st.text_area("Plant notes")

        submitted = st.form_submit_button("Add Plant")

        if submitted:
            if name.strip():
                with get_db() as conn:
                    conn.execute(
                        """INSERT INTO plants
                        (name, crop, added, last_watered, water_interval, notes)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            name.strip(),
                            crop,
                            str(date.today()),
                            str(last_watered),
                            int(interval),
                            notes
                        )
                    )
                st.success("Plant added to your collection.")
                st.rerun()
            else:
                st.error("Enter a plant nickname.")

    st.divider()
    st.subheader("Saved Plants")

    plants = get_plants()

    if plants:
        for p in plants:
            with st.expander(f"🌱 {p[1]} – {p[2]}"):
                st.write(f"Added: {p[3]}")
                st.write(f"Last watered: {p[4]}")
                st.write(f"Watering check interval: {p[5]} days")
                st.write(f"Next watering check: {watering_due(p)}")
                st.write(f"Notes: {p[6]}")

                if st.button("Delete Plant", key=f"delete_{p[0]}"):
                    with get_db() as conn:
                        conn.execute("DELETE FROM growth WHERE plant_id=?", (p[0],))
                        conn.execute("DELETE FROM reminders WHERE plant_id=?", (p[0],))
                        conn.execute("DELETE FROM plants WHERE id=?", (p[0],))
                    st.success("Plant deleted.")
                    st.rerun()
    else:
        st.info("Your collection is empty. Add your first plant above.")

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="text-align:center;color:#527563;padding:15px">
🌿 <b>Plant Guardian AI</b><br>
Smart Plant Care and Garden Management System<br>
<small>AI predictions are estimates. Verify important plant health decisions with a qualified agricultural expert.</small>
</div>
""", unsafe_allow_html=True)
