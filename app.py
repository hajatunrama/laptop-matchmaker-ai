import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Iron Tetrad AI | The Honest Matchmaker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
div[data-testid="metric-container"] {
    background-color: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 12px 20px;
    border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


@st.cache_resource
def load_pure_brain():
    with open("weights.json", "r") as f:
        w_dict = json.load(f)
    W1, b1 = np.array(w_dict["w1"]), np.array(w_dict["b1"])
    W2, b2 = np.array(w_dict["w2"]), np.array(w_dict["b2"])
    W3, b3 = np.array(w_dict["w3"]), np.array(w_dict["b3"])
    scaler = joblib.load("scaler_laptop.pkl")
    return (W1, b1, W2, b2, W3, b3), scaler


(W1, b1, W2, b2, W3, b3), scaler = load_pure_brain()


def ai_predict(x_input):
    a1 = relu(np.dot(x_input, W1) + b1)
    a2 = relu(np.dot(a1, W2) + b2)
    return softmax(np.dot(a2, W3) + b3)[0]


@st.cache_data
def load_katalog():
    return pd.read_csv("katalog_laptop_nyata.csv")


df_katalog = load_katalog()

if "Terakhir_Diperbarui" in df_katalog.columns:
    waktu_otomatis = df_katalog["Terakhir_Diperbarui"].iloc[0]
else:
    waktu_otomatis = "Statis (Menunggu sinkronisasi Cron-Daemon)"


def hitung_topsis(df_subset, bobot_vektor):
    harga = df_subset["Harga_Juta"].values.astype(float)
    gaming = df_subset["skor_gaming"].values.astype(float)
    editing = df_subset["skor_editing"].values.astype(float)

    berat = (
        df_subset["Berat"]
        .astype(str)
        .str.replace(" kg", "")
        .astype(float)
        .values
    )
    baterai = (
        df_subset["Baterai"]
        .astype(str)
        .str.replace(" Wh", "")
        .astype(float)
        .values
    )

    matriks = np.column_stack([harga, gaming, editing, berat, baterai])

    pembagi = np.sqrt((matriks**2).sum(axis=0))
    pembagi[pembagi == 0] = 1.0
    R = matriks / pembagi

    V = R * bobot_vektor

    A_plus = np.array(
        [
            V[:, 0].min(),
            V[:, 1].max(),
            V[:, 2].max(),
            V[:, 3].min(),
            V[:, 4].max(),
        ]
    )
    A_minus = np.array(
        [
            V[:, 0].max(),
            V[:, 1].min(),
            V[:, 2].min(),
            V[:, 3].max(),
            V[:, 4].min(),
        ]
    )

    S_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
    S_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))

    skor_topsis = S_minus / (S_plus + S_minus)
    skor_topsis[np.isnan(skor_topsis)] = 0.5
    return skor_topsis


def get_radar_unit_scale(row):
    cpu = float(row["skor_editing"])
    gpu = float(row["skor_gaming"])

    wh = float(str(row["Baterai"]).replace(" Wh", ""))
    bat = float(np.clip((wh - 35.0) / 10.0, 1.0, 5.0))

    kg = float(str(row["Berat"]).replace(" kg", ""))
    rng = float(np.clip(5.0 - ((kg - 1.1) * 2.0), 1.0, 5.0))

    hrg = float(row["Harga_Juta"])
    mrh = float(np.clip(5.0 - ((hrg - 5.0) / 6.5), 1.0, 5.0))

    return [cpu, gpu, bat, rng, mrh]


def buat_plot_radar(unit_vals, target_vals, nama_unit):
    kategori = [
        "Kuat Ngedit / Banyak Tab",
        "Lancar Main Game / 3D",
        "Aman Tanpa Colokan",
        "Pundak Gak Pegal",
        "Dompet Tetap Aman",
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=target_vals + [target_vals[0]],
            theta=kategori + [kategori[0]],
            fill="toself",
            fillcolor="rgba(0, 230, 255, 0.18)",
            line=dict(color="#00e6ff", width=2, dash="dot"),
            name="Ekspektasi Anda",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=unit_vals + [unit_vals[0]],
            theta=kategori + [kategori[0]],
            fill="toself",
            fillcolor="rgba(255, 43, 115, 0.45)",
            line=dict(color="#ff2b73", width=3),
            name=f"Unit: {nama_unit}",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                showticklabels=False,
                gridcolor="rgba(255,255,255,0.15)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.15)",
                tickfont=dict(size=13, color="white"),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(color="white"),
        ),
        margin=dict(l=35, r=35, t=35, b=35),
    )
    return fig


with st.sidebar:
    st.title("⚖️ KONTROL TETRAD")
    st.caption("Kalibrasi Hukum Fisika")
    st.markdown("---")

    prinsip = st.radio(
        "Pilih Realitas Pengorbanan:",
        [
            "🧱 **1. CONCRETE BEAST** (Kencang & Murah | Korban: Berat & Boros)",
            "🪶 **2. NOMAD CLOUD** (Super Ringan | Korban: Gak bisa game berat)",
            "👑 **3. EXECUTIVE PARADOX** (Kencang & Tipis | Korban: Harga mahal)",
            "☕ **4. RATIONAL SPARTAN** (Murah & Awet | Korban: Gengsi & Performa)",
        ],
        index=0,
    )

    st.write("")
    budget_maks = st.slider(
        "💰 Batas Anggaran (Juta IDR)",
        min_value=4.5,
        max_value=35.0,
        value=14.5,
        step=0.5,
    )

    st.write("")
    tombol_eksekusi = st.button(
        "⚡ JALANKAN AUDIT AI", type="primary", use_container_width=True
    )

st.title("🖥️ IRON TETRAD AI: THE VISUAL EPIPHANY")
st.caption(
    f"🟢 **Autonomous DB:** Harga berpatokan pada Live Forex USD/IDR (Update: `{waktu_otomatis}`)"
)
st.markdown("---")

if tombol_eksekusi:

    if "CONCRETE" in prinsip:
        vektor_bobot = np.array([0.35, 0.30, 0.25, 0.05, 0.05])
        input_ann = np.array([[budget_maks, 4.5, 4.0, 1.5, 2.0]])
        radar_target = [4.5, 4.5, 1.5, 1.5, 4.0]
        target_teks = "Raw Computational Power (Kinerja Mentah)"
        hukuman_teks = (
            "Bodi setebal batako + Adaptor bata. Pundak dijamin miring."
        )
    elif "NOMAD" in prinsip:
        vektor_bobot = np.array([0.25, 0.05, 0.20, 0.25, 0.25])
        input_ann = np.array([[budget_maks, 1.5, 3.0, 4.8, 4.5]])
        radar_target = [2.5, 1.5, 4.5, 4.8, 4.0]
        target_teks = "Ekstrem Portabilitas & Otonomi Daya"
        hukuman_teks = "Prosesor U-Series hemat daya. Haram dipaksa render 3D."
    elif "EXECUTIVE" in prinsip:
        vektor_bobot = np.array([0.05, 0.30, 0.30, 0.20, 0.15])
        input_ann = np.array([[budget_maks, 4.0, 4.5, 4.5, 4.0]])
        radar_target = [4.5, 4.0, 4.0, 4.5, 1.5]
        target_teks = "No-Compromise Flagship Ultra-portable"
        hukuman_teks = (
            "Pajak Premium. Anda membayar 35% ekstra demi sasis tipis."
        )
    else:  # SPARTAN
        vektor_bobot = np.array([0.40, 0.0, 0.20, 0.20, 0.20])
        input_ann = np.array([[budget_maks, 1.0, 2.5, 3.5, 4.0]])
        radar_target = [2.5, 1.0, 4.0, 3.5, 4.8]
        target_teks = "Efisiensi Nilai Ekonomis Mutlak"
        hukuman_teks = (
            "Material plastik Polycarbonate. Layar standar pudar 45% NTSC."
        )

    input_scaled = scaler.transform(input_ann)
    prediksi = ai_predict(input_scaled)
    pemenang_idx = np.argmax(prediksi)
    daftar_kategori = [
        "Entry Level / Pelajar",
        "Ultrabook Tipis",
        "High Performance Gaming",
        "Heavy Workstation",
    ]

    df_mampu = df_katalog[df_katalog["Harga_Juta"] <= budget_maks].copy()

    if len(df_mampu) == 0:
        st.error(
            "❌ Anggaran Anda terlalu rendah untuk menebus hukum fisika ini."
        )
    else:
        df_mampu["TOPSIS_Score"] = hitung_topsis(df_mampu, vektor_bobot)
        rekomendasi = (
            df_mampu.sort_values("TOPSIS_Score", ascending=False).head(3)
        )
        pemenang_utama = rekomendasi.iloc[0]

        col_grafik, col_pemenang = st.columns([4.8, 5.2], gap="large")

        with col_grafik:
            st.subheader("🕸️ The Radar of Truth")
            radar_vals_unit = get_radar_unit_scale(pemenang_utama)
            fig_radar = buat_plot_radar(
                radar_vals_unit, radar_target, pemenang_utama["Tipe"]
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            selisih_radar = np.array(radar_vals_unit) - np.array(radar_target)
            kategori_human = [
                "Kinerja Multitasking",
                "Performa Grafis/3D",
                "Daya Tahan Baterai",
                "Kenyamanan Pundak",
                "Sisa Tabungan",
            ]

            area_tekor = [
                kategori_human[i] for i in range(5) if selisih_radar[i] < -0.7
            ]
            area_mubazir = [
                kategori_human[i] for i in range(5) if selisih_radar[i] > 1.2
            ]

            with st.container(border=True):
                st.markdown("#### 🗣️ Terjemahan AI untuk Orang Awam:")
                if area_tekor:
                    st.error(
                        f"⚠️ **Anda Tekor di Sektor:** `{', '.join(area_tekor)}`  \n*(Garis merah kempes ke dalam, artinya laptop ini tidak sanggup memenuhi target ideal Anda di area tersebut)*."
                    )
                else:
                    st.success(
                        "✨ **Kecocokan Harmonis:** Seluruh kapasitas fisik laptop berhasil memenuhi ekspektasi Anda."
                    )

                if area_mubazir:
                    st.info(
                        f"💡 **Potensi Mubazir:** Sektor `{', '.join(area_mubazir)}` terlalu melimpah. *(Anda membayar performa ekstra yang sebenarnya tidak Anda butuhkan)*."
                    )

            st.caption(
                f"**Audit Klasifikasi Saraf (ANN):** Terdeteksi sebagai kelas `{daftar_kategori[pemenang_idx]}` *(Confidence: {prediksi[pemenang_idx]*100:.1f}%)*"
            )

        with col_pemenang:
            st.subheader("👑 KANDIDAT MUTLAK (RANK 1)")

            with st.container(border=True):
                st.markdown(
                    f"### {pemenang_utama['Merk']} {pemenang_utama['Tipe']}"
                )
                st.write(
                    f"**Koefisien Ekuilibrium TOPSIS:** `{pemenang_utama['TOPSIS_Score']*100:.2f}%`"
                )
                st.write("")

                m1, m2, m3 = st.columns(3)
                sisa_duit = budget_maks - pemenang_utama["Harga_Juta"]
                m1.metric(
                    "Market Value",
                    f"Rp {pemenang_utama['Harga_Juta']} Jt",
                    delta=f"Surplus Rp {sisa_duit:.1f} Jt" if sisa_duit > 0 else "Pas Budget",
                )
                m2.metric(
                    "Beban Fisik",
                    f"{pemenang_utama['Berat']}",
                    delta="+0.65kg Adaptor" if pemenang_utama["skor_gaming"] >= 3.5 else "+0.2kg Type-C",
                    delta_color="off",
                )
                m3.metric(
                    "Kapasitas Sel",
                    f"{pemenang_utama['Baterai']}",
                    delta="Boros (TDP Tinggi)" if pemenang_utama["skor_gaming"] >= 3.5 else "Awet Seharian",
                    delta_color="off",
                )

                st.markdown("---")
                c_spek1, c_spek2 = st.columns(2)
                with c_spek1:
                    st.write(f"⚡ **CPU:** `{pemenang_utama['CPU']}`")
                    st.write(
                        f"💾 **RAM/SSD:** `{pemenang_utama['RAM']} | {pemenang_utama['Storage']}`"
                    )
                with c_spek2:
                    st.write(f"🎮 **GPU:** `{pemenang_utama['GPU']}`")
                    st.write(f"🖥️ **Display:** `{pemenang_utama['Layar']}`")

            st.success(f"🎯 **Objektif:** {target_teks}")
            st.warning(f"⚠️ **Hukuman Fisika:** {hukuman_teks}")

        st.markdown("---")
        st.subheader("🥈 & 🥉 ALTERNATIF KOMPROMI LAINNYA:")

        col_r2, col_r3 = st.columns(2, gap="medium")

        if len(rekomendasi) > 1:
            r2 = rekomendasi.iloc[1]
            with col_r2:
                with st.container(border=True):
                    st.markdown(f"#### 🥈 {r2['Merk']} {r2['Tipe']}")
                    st.caption(
                        f"Harga: **Rp {r2['Harga_Juta']} Jt** | Skor TOPSIS: `{r2['TOPSIS_Score']*100:.1f}%`"
                    )
                    st.write(
                        f"▪️ **CPU/GPU:** {r2['CPU']} | {r2['GPU'].split()[0]}"
                    )
                    st.write(
                        f"▪️ **Fisik:** {r2['Berat']} | {r2['Layar'].split()[0]}"
                    )

        if len(rekomendasi) > 2:
            r3 = rekomendasi.iloc[2]
            with col_r3:
                with st.container(border=True):
                    st.markdown(f"#### 🥉 {r3['Merk']} {r3['Tipe']}")
                    st.caption(
                        f"Harga: **Rp {r3['Harga_Juta']} Jt** | Skor TOPSIS: `{r3['TOPSIS_Score']*100:.1f}%`"
                    )
                    st.write(
                        f"▪️ **CPU/GPU:** {r3['CPU']} | {r3['GPU'].split()[0]}"
                    )
                    st.write(
                        f"▪️ **Fisik:** {r3['Berat']} | {r3['Layar'].split()[0]}"
                    )

else:
    st.info(
        "👈 Silakan pilih Realitas Pengorbanan di panel sebelah kiri, lalu klik **Jalankan Audit AI**."
    )
