import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Prediksi Harga Rumah Tangerang",
    page_icon="🏠",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #F7F8FC; }
    [data-testid="stSidebar"] { background: #1A1F36; }

    /* Semua teks di sidebar */
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }

    /* Label field */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #94A3B8 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
    }

    /* Input number & select box background + text */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea {
        background: #2D3452 !important;
        color: #F1F5F9 !important;
        border: 1px solid #3D4572 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }

    /* Selectbox dropdown wrapper */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #2D3452 !important;
        border: 1px solid #3D4572 !important;
        border-radius: 8px !important;
        color: #F1F5F9 !important;
    }

    /* Teks value di dalam selectbox */
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #F1F5F9 !important;
    }

    /* Tombol +/- number input */
    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        background: #3D4572 !important;
        color: #E2E8F0 !important;
        border: none !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
        background: #6366F1 !important;
    }

    /* Placeholder text */
    [data-testid="stSidebar"] input::placeholder { color: #64748B !important; }

    /* Divider */
    [data-testid="stSidebar"] hr { border-color: #2D3452 !important; }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.07);
        border-left: 4px solid #6366F1;
        margin-bottom: 1rem;
    }
    .metric-label { color: #64748B; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
    .metric-value { color: #1A1F36; font-size: 1.7rem; font-weight: 700; line-height: 1.2; }
    .metric-sub { color: #94A3B8; font-size: 0.82rem; margin-top: 2px; }
    .section-title { color: #1A1F36; font-size: 1.05rem; font-weight: 700; margin: 1.5rem 0 0.7rem; }
    .predict-btn > button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        color: white !important; border-radius: 12px !important;
        font-weight: 600 !important; font-size: 1rem !important;
        padding: 0.65rem 2rem !important; border: none !important;
        width: 100%; margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    district_mean_price = joblib.load('district_mean_price.pkl')
    return model, district_mean_price

model, district_mean_price = load_artifacts()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Input Properti")
    st.markdown("---")

    district = st.selectbox("District", district_mean_price.index)
    facilities = st.number_input("Jumlah Fasilitas", 0, 20, 5)
    bedrooms = st.number_input("Bedrooms", 0, 10, 3)
    bathrooms = st.number_input("Bathrooms", 0, 10, 2)
    land_size = st.number_input("Land Size (m²)", 30, 1000, 120)
    building_size = st.number_input("Building Size (m²)", 30, 1000, 90)
    carports = st.number_input("Carports", 0, 5, 1)
    electricity = st.number_input("Electricity (VA)", 900, 10000, 2200)
    maid_bedrooms = st.number_input("Maid Bedrooms", 0, 3, 0)
    maid_bathrooms = st.number_input("Maid Bathrooms", 0, 3, 0)
    floors = st.number_input("Floors", 1, 5, 1)
    property_condition = st.selectbox(
        "Kondisi Properti",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: ["Butuh Renovasi","Bagus","Bagus Sekali","Sudah Renovasi","Baru"][x]
    )
    garages = st.number_input("Garages", 0, 5, 0)
    furnishing = st.selectbox(
        "Furnishing",
        options=[0, 1, 2],
        format_func=lambda x: ["Unfurnished","Semi Furnished","Furnished"][x]
    )

    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Prediksi Harga", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Main area header ─────────────────────────────────────────────────────────
st.markdown("# 🏠 Prediksi Harga Rumah Tangerang")
st.markdown("Estimasi harga properti berbasis machine learning dengan analisis visual lengkap.")
st.markdown("---")

# ── Session state: simpan hasil prediksi agar tidak hilang saat re-run ────────
if predict_clicked:
    district_encoded = district_mean_price[district]
    input_data = np.array([[
        facilities, bedrooms, bathrooms, land_size, building_size,
        carports, electricity, maid_bedrooms, maid_bathrooms,
        floors, property_condition, garages, furnishing, district_encoded
    ]])
    st.session_state["prediction"] = model.predict(input_data)[0]
    st.session_state["inputs"] = {
        "district": district, "facilities": facilities, "bedrooms": bedrooms,
        "bathrooms": bathrooms, "land_size": land_size, "building_size": building_size,
        "carports": carports, "electricity": electricity, "maid_bedrooms": maid_bedrooms,
        "maid_bathrooms": maid_bathrooms, "floors": floors,
        "property_condition": property_condition, "garages": garages,
        "furnishing": furnishing, "district_encoded": float(district_encoded)
    }
    st.session_state["ai_analysis"] = None  # reset analisis lama

if "prediction" not in st.session_state:
    st.info("👈 Isi detail properti di sidebar, lalu klik **Prediksi Harga**.")
    st.stop()

# Ambil dari session state
prediction    = st.session_state["prediction"]
inp           = st.session_state["inputs"]
district      = inp["district"]
facilities    = inp["facilities"]
bedrooms      = inp["bedrooms"]
bathrooms     = inp["bathrooms"]
land_size     = inp["land_size"]
building_size = inp["building_size"]
carports      = inp["carports"]
electricity   = inp["electricity"]
maid_bedrooms = inp["maid_bedrooms"]
maid_bathrooms= inp["maid_bathrooms"]
floors        = inp["floors"]
property_condition = inp["property_condition"]
garages       = inp["garages"]
furnishing    = inp["furnishing"]
district_encoded   = inp["district_encoded"]

# Helper prices
price_B        = prediction / 1e9
price_per_land = prediction / land_size if land_size > 0 else 0
price_per_bldg = prediction / building_size if building_size > 0 else 0

# ── Row 1: Metric cards ───────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Estimasi Harga</div>
        <div class="metric-value">Rp {price_B:.2f} M</div>
        <div class="metric-sub">Rp {prediction:,.0f}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card" style="border-color:#10B981">
        <div class="metric-label">Harga / m² Tanah</div>
        <div class="metric-value">Rp {price_per_land/1e6:.1f} Jt</div>
        <div class="metric-sub">per meter persegi lahan</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card" style="border-color:#F59E0B">
        <div class="metric-label">Harga / m² Bangunan</div>
        <div class="metric-value">Rp {price_per_bldg/1e6:.1f} Jt</div>
        <div class="metric-sub">per meter persegi bangunan</div>
    </div>""", unsafe_allow_html=True)

# ── Row 2: Gauge + Radar ──────────────────────────────────────────────────────
st.markdown('<p class="section-title">📊 Visualisasi Harga & Profil Properti</p>', unsafe_allow_html=True)
col_g, col_r = st.columns(2)

with col_g:
    # Price gauge
    min_p, max_p = 300e6, 10e9
    gauge_val = min(max(prediction, min_p), max_p)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction / 1e9,
        number={"suffix": " M", "valueformat": ".2f", "font": {"size": 28, "color": "#1A1F36"}},
        delta={"reference": district_mean_price.mean() / 1e9, "valueformat": ".2f",
               "suffix": " M", "relative": False},
        title={"text": "Estimasi Harga (Milyar Rp)", "font": {"size": 13, "color": "#64748B"}},
        gauge={
            "axis": {"range": [min_p/1e9, max_p/1e9], "tickformat": ".1f",
                     "ticksuffix": "M", "tickfont": {"size": 10}},
            "bar": {"color": "#6366F1", "thickness": 0.3},
            "bgcolor": "#F1F5F9",
            "steps": [
                {"range": [0.3, 2], "color": "#DCFCE7"},
                {"range": [2, 5], "color": "#FEF9C3"},
                {"range": [5, 10], "color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": "#6366F1", "width": 3},
                "thickness": 0.75,
                "value": prediction / 1e9
            }
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20),
                            paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_r:
    # Radar / spider chart
    cond_label = ["Butuh Renovasi","Bagus","Bagus Sekali","Sudah Renovasi","Baru"][property_condition]
    furnish_label = ["Unfurnished","Semi Furnished","Furnished"][furnishing]

    categories = ["Kamar Tidur","Kamar Mandi","Lahan","Bangunan","Fasilitas","Lantai"]
    # normalize 0–10 scale
    vals = [
        min(bedrooms / 10 * 10, 10),
        min(bathrooms / 10 * 10, 10),
        min(land_size / 1000 * 10, 10),
        min(building_size / 1000 * 10, 10),
        min(facilities / 20 * 10, 10),
        min(floors / 5 * 10, 10),
    ]
    fig_radar = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(99,102,241,0.15)',
        line=dict(color='#6366F1', width=2),
        marker=dict(color='#6366F1', size=6)
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        title=dict(text="Profil Fitur Properti (skor 0–10)", font=dict(size=13, color="#64748B"), x=0.5),
        height=280,
        margin=dict(t=50, b=10, l=40, r=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ── Row 3: District comparison + Feature bar ──────────────────────────────────
st.markdown('<p class="section-title">📍 Perbandingan Harga & Kontribusi Fitur</p>', unsafe_allow_html=True)
col_d, col_f = st.columns(2)

with col_d:
    # Harga rata-rata tiap distrik vs prediksi
    df_dist = pd.DataFrame({
        "District": district_mean_price.index,
        "Harga Rata-Rata (M)": district_mean_price.values / 1e9
    }).sort_values("Harga Rata-Rata (M)", ascending=True)

    colors = ["#6366F1" if d == district else "#CBD5E1" for d in df_dist["District"]]
    fig_dist = go.Figure(go.Bar(
        x=df_dist["Harga Rata-Rata (M)"],
        y=df_dist["District"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}M" for v in df_dist["Harga Rata-Rata (M)"]],
        textposition="outside",
        textfont=dict(size=9)
    ))
    fig_dist.add_vline(
        x=prediction / 1e9,
        line_dash="dot", line_color="#F59E0B", line_width=2,
        annotation_text=f"Prediksi: {price_B:.1f}M",
        annotation_font_size=10,
        annotation_font_color="#F59E0B"
    )
    fig_dist.update_layout(
        title=dict(text="Rata-rata Harga per Distrik", font=dict(size=13, color="#64748B")),
        xaxis_title="Milyar Rp",
        height=max(300, len(district_mean_price) * 28),
        margin=dict(t=40, b=30, l=10, r=60),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9")
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_f:
    # Sensitivitas fitur: naikan 1 satuan tiap fitur, hitung delta harga
    base_input = [
        facilities, bedrooms, bathrooms, land_size, building_size,
        carports, electricity, maid_bedrooms, maid_bathrooms,
        floors, property_condition, garages, furnishing, district_encoded
    ]
    feat_names = [
        "Fasilitas","Bedrooms","Bathrooms","Land Size (m²)",
        "Building Size (m²)","Carports","Electricity (VA)","Maid BR",
        "Maid Bath","Floors","Kondisi Properti","Garages","Furnishing","—"
    ]
    # Kenaikan 1 satuan yang realistis per fitur
    steps = [1, 1, 1, 10, 10, 1, 100, 1, 1, 1, 1, 1, 1, 0]

    deltas = []
    for i, step in enumerate(steps):
        if step == 0:
            deltas.append(0)
            continue
        modified = base_input.copy()
        modified[i] = modified[i] + step
        new_pred = model.predict(np.array([modified]))[0]
        deltas.append((new_pred - prediction) / 1e6)  # dalam Juta Rp

    df_sens = pd.DataFrame({
        "Fitur": feat_names[:-1],  # exclude district
        "Delta (Juta Rp)": deltas[:-1]
    }).sort_values("Delta (Juta Rp)", ascending=True)

    bar_colors = ["#10B981" if v >= 0 else "#EF4444" for v in df_sens["Delta (Juta Rp)"]]
    fig_feat = go.Figure(go.Bar(
        x=df_sens["Delta (Juta Rp)"],
        y=df_sens["Fitur"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"+{v:.1f}Jt" if v >= 0 else f"{v:.1f}Jt" for v in df_sens["Delta (Juta Rp)"]],
        textposition="outside",
        textfont=dict(size=9)
    ))
    fig_feat.add_vline(x=0, line_color="#94A3B8", line_width=1)
    fig_feat.update_layout(
        title=dict(
            text="Dampak Kenaikan 1 Satuan per Fitur terhadap Harga",
            font=dict(size=12, color="#64748B")
        ),
        xaxis_title="Perubahan Harga (Juta Rp)",
        height=max(320, len(df_sens) * 24 + 80),
        margin=dict(t=50, b=30, l=10, r=70),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(gridcolor="#F1F5F9", zeroline=False),
        yaxis=dict(gridcolor="#F1F5F9")
    )
    st.plotly_chart(fig_feat, use_container_width=True)

# ── Row 4: Price range simulation ─────────────────────────────────────────────
st.markdown('<p class="section-title">📈 Simulasi Sensitivitas Harga vs Luas Tanah</p>', unsafe_allow_html=True)

land_range = np.arange(30, 1001, 10)
prices_sim = []
for ls in land_range:
    inp = np.array([[facilities, bedrooms, bathrooms, ls, building_size,
                     carports, electricity, maid_bedrooms, maid_bathrooms,
                     floors, property_condition, garages, furnishing, district_encoded]])
    prices_sim.append(model.predict(inp)[0] / 1e9)

fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(
    x=land_range, y=prices_sim,
    mode="lines", line=dict(color="#6366F1", width=2.5),
    fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
    name="Estimasi Harga"
))
fig_sim.add_trace(go.Scatter(
    x=[land_size], y=[prediction / 1e9],
    mode="markers",
    marker=dict(color="#F59E0B", size=12, symbol="circle",
                line=dict(color="white", width=2)),
    name=f"Properti Ini ({land_size}m²)"
))
fig_sim.update_layout(
    xaxis_title="Luas Tanah (m²)",
    yaxis_title="Estimasi Harga (Milyar Rp)",
    height=300,
    margin=dict(t=20, b=40, l=50, r=20),
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor="#F1F5F9"),
    yaxis=dict(gridcolor="#F1F5F9"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_sim, use_container_width=True)

# ── Row 5: Analisis LLM (Groq) ───────────────────────────────────────────────
st.markdown('<p class="section-title">🤖 Analisis AI — Interpretasi Hasil Prediksi</p>', unsafe_allow_html=True)

# Siapkan ringkasan sensitivitas untuk konteks LLM
top_positive = df_sens[df_sens["Delta (Juta Rp)"] > 0].sort_values("Delta (Juta Rp)", ascending=False).head(3)
top_negative = df_sens[df_sens["Delta (Juta Rp)"] < 0].sort_values("Delta (Juta Rp)").head(3)

sens_pos_text = ", ".join([f"{r['Fitur']} (+{r['Delta (Juta Rp)']:.1f} Jt)" for _, r in top_positive.iterrows()])
sens_neg_text = ", ".join([f"{r['Fitur']} ({r['Delta (Juta Rp)']:.1f} Jt)" for _, r in top_negative.iterrows()])

kondisi_label = ["Butuh Renovasi","Bagus","Bagus Sekali","Sudah Renovasi","Baru"][property_condition]
furnish_label = ["Unfurnished","Semi Furnished","Furnished"][furnishing]
dist_avg = district_mean_price[district] / 1e9
dist_rank = int((district_mean_price.rank(ascending=False)[district]))
dist_total = len(district_mean_price)

prompt_context = f"""
Kamu adalah analis properti berpengalaman di pasar Tangerang, Indonesia.
Berikan analisis singkat, jelas, dan actionable dalam Bahasa Indonesia.
Gunakan format dengan beberapa paragraf pendek (bukan bullet point).

DATA PROPERTI:
- Distrik: {district} (rata-rata harga distrik: Rp {dist_avg:.2f} Milyar, peringkat {dist_rank} dari {dist_total} distrik)
- Luas Tanah: {land_size} m², Luas Bangunan: {building_size} m²
- Kamar Tidur: {bedrooms}, Kamar Mandi: {bathrooms}
- Lantai: {floors}, Garasi: {garages}, Carport: {carports}
- Kondisi: {kondisi_label}, Furnishing: {furnish_label}
- Fasilitas: {facilities}, Listrik: {electricity} VA
- Kamar Pembantu: {maid_bedrooms}, Kamar Mandi Pembantu: {maid_bathrooms}

HASIL PREDIKSI MODEL:
- Estimasi Harga: Rp {price_B:.2f} Milyar (Rp {prediction:,.0f})
- Harga per m² Tanah: Rp {price_per_land/1e6:.1f} Juta
- Harga per m² Bangunan: Rp {price_per_bldg/1e6:.1f} Juta
- Selisih vs rata-rata distrik: Rp {(prediction - district_mean_price[district])/1e6:.1f} Juta

SENSITIVITAS FITUR (fitur paling berpengaruh menaikkan harga): {sens_pos_text}
SENSITIVITAS FITUR (fitur yang justru menurunkan harga jika ditambah): {sens_neg_text}

Tolong berikan analisis yang mencakup:
1. Penilaian singkat apakah harga ini wajar untuk distrik {district}
2. Keunggulan dan kelemahan properti ini berdasarkan data
3. Saran konkret untuk meningkatkan nilai properti
4. Peringatan atau hal yang perlu diperhatikan pembeli/penjual
"""

groq_api_key = st.secrets.get("GROQ_API_KEY", "")

col_btn, col_model = st.columns([2, 1])
with col_model:
    groq_model = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
        label_visibility="collapsed"
    )
with col_btn:
    analyze_clicked = st.button("✨ Generate Analisis AI", use_container_width=True)

if analyze_clicked:
    if not groq_api_key:
        st.warning("⚠️ GROQ_API_KEY belum diset. Tambahkan di Settings → Secrets.")
    else:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)

            with st.spinner("🧠 AI sedang menganalisis properti ini..."):
                response = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Kamu adalah analis properti senior yang ahli di pasar perumahan Tangerang dan Jabodetabek. Berikan analisis yang tajam, jujur, dan praktis."
                        },
                        {
                            "role": "user",
                            "content": prompt_context
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                st.session_state["ai_analysis"] = {
                    "text": response.choices[0].message.content,
                    "model": groq_model,
                    "tokens": response.usage.total_tokens
                }

        except ImportError:
            st.error("Package `groq` belum terinstall. Tambahkan `groq` di packages.txt")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Tampilkan hasil analisis dari session_state (tetap ada meski re-run)
if st.session_state.get("ai_analysis"):
    result = st.session_state["ai_analysis"]
    st.markdown("""
    <div style="background:white; border-radius:16px; padding:1.6rem 2rem;
                box-shadow:0 1px 4px rgba(0,0,0,.07); border-left:4px solid #8B5CF6;
                margin-top:0.5rem; line-height:1.8; color:#1A1F36;">
    """ + result["text"].replace("\n", "<br>") + """
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Dianalisis oleh: {result['model']} via Groq · {result['tokens']} tokens digunakan")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model prediksi berbasis data properti Tangerang. Hasil bersifat estimasi dan dapat berbeda dengan harga pasar aktual.")