%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
from datetime import date

st.set_page_config(page_title="EDA Ecommerce", layout="wide")
st.title("Exploratory Data Analysis of Online Retail Dataset")

# ---------- helpers ----------
@st.cache_data(show_spinner=False)
def load_csv(path_or_file):
    return pd.read_csv(path_or_file, encoding_errors="ignore")

def apply_outlier_filter(df, cols):
    clean = df.copy()
    for c in cols:
        if c not in clean.columns:
            continue
        s = pd.to_numeric(clean[c], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        clean = clean[(s >= lo) & (s <= hi)]
    return clean

def annotate_bars(ax, fmt="{:.0f}"):
    for c in ax.containers:
        ax.bar_label(c, fmt=fmt, padding=3)

sns.set_style("whitegrid")

# ============ Data Input ============

df = load_csv("ecommerce.csv")

# ---------- CLEAN & LOCK (tanpa outlier) ----------
df = df.copy()

if "Country" in df.columns:
    df["Country"] = df["Country"].replace("Unspecified", "United Kingdom")

# Numerik & revenue
if "Quantity" in df.columns:
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
if "UnitPrice" in df.columns:
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
if "Revenue" not in df.columns:
    if {"UnitPrice", "Quantity"}.issubset(df.columns):
        df["Revenue"] = df["UnitPrice"] * df["Quantity"]
    else:
        st.error("Need UnitPrice & Quantity to compute Revenue.")
        st.stop()

# Datetime fields
if "InvoiceDate" in df.columns:
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Month"] = df["InvoiceDate"].dt.month
    df["Year"] = df["InvoiceDate"].dt.year
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["DayOfWeek_Name"] = df["InvoiceDate"].dt.day_name()
    month_map = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    df["Month_Name"] = df["Month"].map(month_map)

# Description hygiene
if "Description" in df.columns:
    df = df[df["Description"].notna()].copy()
    df["Description"] = df["Description"].astype(str).str.strip()

# LOCK: remove returns/cancellations
if "Quantity" in df.columns:
    df = df[df["Quantity"] > 0]
if "InvoiceNo" in df.columns:
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# LOCK: remove outliers (IQR) globally on Quantity & UnitPrice
cols_for_outlier = [c for c in ["Quantity","UnitPrice"] if c in df.columns]
df_locked = apply_outlier_filter(df, cols_for_outlier) if cols_for_outlier else df.copy()

# ============ Sidebar: FILTERS ============
with st.sidebar:
    st.header("Filters")

    # Country filter
    countries = sorted(df_locked["Country"].dropna().astype(str).unique()) \
                if "Country" in df_locked.columns else []
    country_sel = st.multiselect("Country (kosongkan = semua)", countries, default=[])

    # Date range (opsional) — inisialisasi aman
    d_from, d_to = None, None
    if "InvoiceDate" in df_locked.columns:
        min_dt = df_locked["InvoiceDate"].min()
        max_dt = df_locked["InvoiceDate"].max()
        if pd.notna(min_dt) and pd.notna(max_dt):
            min_d, max_d = min_dt.date(), max_dt.date()
            dr = st.date_input("Date range", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)
            if isinstance(dr, tuple) and len(dr) == 2:
                d_from, d_to = dr

# ============ Apply filters ke VIEW (grafik) ============
df_view = df_locked.copy()
if country_sel:
    df_view = df_view[df_view["Country"].astype(str).isin(country_sel)]

if (d_from is not None) and (d_to is not None) and ("InvoiceDate" in df_view.columns):
    mask = (df_view["InvoiceDate"].dt.date >= d_from) & (df_view["InvoiceDate"].dt.date <= d_to)
    df_view = df_view[mask]

st.header("About Me")

left, right = st.columns([3, 1], vertical_alignment="center")

with left:
    st.markdown("""
    ### Femy Rahma Fitria
    *Data Science Enthusiast*

    A Fisheries graduate driven by a passion for lifelong learning and development,
    especially in data management, and currently sharpening my skills through an online data course.
    """)
    st.markdown("**Contact**")
    st.markdown("🔗 **LinkedIn:** linkedin.com/in/femyfitria")
    st.markdown("📧 **Email:** femyrahmaf@gmail.com")

with right:
    ILLUSTRATION_EMOJI = "👩‍💻"
    st.markdown(f"""
    <style>
      .about-emoji {{
        text-align:center;
        line-height:1;
        /* besar & responsif: min 96px, max 200px, skala sesuai lebar */
        font-size: clamp(96px, 12vw, 220px);
        /* opsional: sedikit bayangan agar pop-out */
        filter: drop-shadow(0 2px 4px rgba(0,0,0,.12));
      }}
    </style>
    <div class="about-emoji">{ILLUSTRATION_EMOJI}</div>
    """, unsafe_allow_html=True)


st.header("Data Understanding")
# ---------- Dataset Information ----------
st.subheader("Dataset Information")
st.markdown(
      "Data e-commerce yang berisi semua transaksi yang terjadi antara 01/12/2010 hingga 09/12/2011 untuk online retail yang terdaftar dan berbasis di Inggris Raya."
)
st.markdown(
    "🧾 Rows : 4.870")
st.markdown(
    "🧱 Columns : 8")

st.subheader("Dataset Column Description")

data = [
    {"Kolom":"🧾InvoiceNo","Tipe":"string","Deskripsi":"Nomor faktur unik untuk setiap transaksi."},
    {"Kolom":"🏷️ StockCode","Tipe":"string","Deskripsi":"Kode unik untuk tiap produk."},
    {"Kolom":"📝 Description","Tipe":"string","Deskripsi":"Nama/uraian produk."},
    {"Kolom":"👤 CustomerID","Tipe":"string","Deskripsi":"ID pelanggan yang bertransaksi."},
    {"Kolom":"🌍 Country","Tipe":"string","Deskripsi":"Negara asal pelanggan."},
    {"Kolom":"📅 InvoiceDate","Tipe":"datetime","Deskripsi":"Tanggal & waktu transaksi."},
    {"Kolom":"📦 Quantity","Tipe":"int","Deskripsi":"Jumlah produk dalam satu transaksi."},
    {"Kolom":"💲 UnitPrice","Tipe":"float","Deskripsi":"Harga per unit produk."},
    {"Kolom":"💰 Revenue","Tipe":"float","Deskripsi":"Total pendapatan per baris (UnitPrice × Quantity)."},
    {"Kolom":"🗓️ Month_Name","Tipe":"string","Deskripsi":"Nama bulan hasil turunan dari InvoiceDate."},
    {"Kolom":"📆 Year","Tipe":"int","Deskripsi":"Tahun hasil turunan dari InvoiceDate."},
    {"Kolom":"⏰ Hour","Tipe":"int (0–23)","Deskripsi":"Jam transaksi (0–23) dari InvoiceDate."},
    {"Kolom":"🗓️ DayOfWeek_Name","Tipe":"string","Deskripsi":"Nama hari dalam seminggu dari InvoiceDate."},
]

schema_df = pd.DataFrame(data, columns=["Kolom","Tipe","Deskripsi"])
st.dataframe(schema_df, use_container_width=True, hide_index=True)


# ---------- Overview ----------
st.subheader("Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df_locked):,}")
c2.metric("Unique Products", f"{df_locked['Description'].nunique():,}" if 'Description' in df_locked else "–")
c3.metric("Countries", f"{df_locked['Country'].nunique():,}" if 'Country' in df_locked else "–")
c4.metric("Total Revenue", f"{df_locked['Revenue'].sum():,.2f}")
with st.expander("Sample rows (locked dataset)"):
    st.dataframe(df_locked.head(20))

st.header("Business Insight")
# ============ 1) Revenue by Country ============
st.subheader("Country mana dengan pendapatan tertinggi dan terendah?")
if "Country" in df_view.columns:
    top5_view = (df_view.groupby("Country", as_index=False)["Revenue"].sum()
                           .sort_values("Revenue", ascending=False).head(5))

    # override style hanya untuk plot ini (tanpa grid)
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.barplot(
            y="Country", x="Revenue", data=top5_view, ax=ax,
            palette=sns.color_palette("mako", n_colors=len(top5_view))
        )

        # hilangkan grid & rapikan tampilan
        ax.grid(False)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#E5E7EB")

        ax.set_title("Top 5 Sales Performance per Country")
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # 1,000 format
        ax.margins(x=0.02)

        annotate_bars(ax, fmt="{:,.0f}")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown(
        "Pendapatan tertinggi diperoleh di Negara United Kingdom dengan jumlah "
        "pendapatan 44,942."
    )
else:
    st.info("Column 'Country' not found.")

# ============ 2) Top Products by Quantity ============
st.subheader("Produk kategori apa yang paling diminati oleh konsumen berdasarkan banyaknya pembelian?")
if {"Description","Quantity"}.issubset(df_view.columns):
    top_product_view = (
        df_view.groupby("Description", as_index=False)["Quantity"].sum()
               .sort_values("Quantity", ascending=False).head(10)
    )

    # plot tanpa grid + palette mako
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            y="Description", x="Quantity",
            data=top_product_view, ax=ax,
            palette=sns.color_palette("mako", n_colors=len(top_product_view))
        )

        ax.grid(False)  # matikan grid
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#E5E7EB")
        ax.set_title("Top Sales Product")
        ax.set_ylabel("")  # rapikan label Y
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # format ribuan
        ax.margins(x=0.02)

        annotate_bars(ax, fmt="{:.0f}")
        plt.tight_layout()
        st.pyplot(fig)

    # Insight dari seluruh data (locked)
    top_all = (df_locked.groupby("Description", as_index=False)["Quantity"].sum()
                        .sort_values("Quantity", ascending=False).head(10))
    top1 = top_all.iloc[0]
    share_top10 = top_all["Quantity"].sum() / df_locked["Quantity"].sum()

    st.markdown(
        "Produk dengan penjualan tertinggi yaitu 60 TEATIME FAIRY CAKE CASES yang "
        "terjual hingga 249 buah. Barang lain yang menjadi top sales yaitu peralatan "
        "dapur/baking (cake cases, jelly moulds, jam set), dekorasi dan seasonal items "
        "(bunting, Christmas ornament, glass T-light), dan produk serbaguna/hadiah "
        "(wallets, jumbo bags)."
    )
else:
    st.info("Columns 'Description'/'Quantity' not found.")

# ============ 3) Transactions per Hour ============
st.subheader("Kapan pelanggan paling banyak melakukan transaksi?")
if "Hour" in df_view.columns:
    trx_hour = df_view.groupby("Hour").size().reindex(range(24), fill_value=0)

    with sns.axes_style("white"):  # override whitegrid → tanpa grid
        fig, ax = plt.subplots(figsize=(10, 4))

        line_color = sns.color_palette("mako", 6)[4]  # ambil shade mako
        ax.plot(
            trx_hour.index, trx_hour.values,
            marker="o", linewidth=2.2, markersize=5,
            color=line_color
        )

        # hilangkan grid & rapikan spines
        ax.grid(False)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#E5E7EB")

        ax.set_title("Transactions per Hour")
        ax.set_xlabel("Hour"); ax.set_ylabel("Count")
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.margins(x=0.01)

        # label tiap titik (tetap seperti punyamu)
        ymax = trx_hour.values.max()
        for x, y in zip(trx_hour.index, trx_hour.values):
            ax.text(x, y + ymax*0.02, f"{int(y)}", ha="center", fontsize=8)

        st.pyplot(fig)

    # Insight
    st.markdown(
      "Pelanggan paling banyak melakukan transaksi pada pukul 12.00 dengan jumlah"
      "transaksi 771 dan paling sedikit pada pukul 07.00 dengan jumlah transaksi 2."
      "Transaksi mulai meningkat pada pukul 08:00 dan menurun tajam setelah pukul 15:00."
      "Aktivitas sangat rendah terjadi di atas pukul 18:00.")

# ============ 4) Monthly revenue trend (2011) ============
st.subheader("Bagaimana tren revenue dalam 1 tahun? Pada bulan apa didapatkan revenue tertinggi dan terendah?")
if set(["Year","Month","Month_Name"]).issubset(df_view.columns):
    y2011_view = df_view[df_view["Year"] == 2011]
    if not y2011_view.empty:
        year_revenue = (
            y2011_view.groupby(["Month","Month_Name"], as_index=False)["Revenue"].sum()
                      .sort_values("Month")
        )

        with sns.axes_style("white"):  # tanpa grid
            fig, ax = plt.subplots(figsize=(10, 4))
            order = year_revenue["Month_Name"].tolist()

            sns.barplot(
                x="Month_Name", y="Revenue", data=year_revenue, ax=ax,
                order=order,
                palette=sns.color_palette("mako", n_colors=len(year_revenue))
            )

            # hilangkan grid & rapikan spines
            ax.grid(False)
            for s in ["top", "right", "left"]:
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color("#E5E7EB")

            ax.set_title("Total Revenue per Month — 2011")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

            annotate_bars(ax, fmt="{:,.0f}")
            plt.tight_layout()
            st.pyplot(fig)

        # Insight
        st.markdown(
        "Revenue cenderung meningkat stabil sepanjang tahun, terutama sejak"
        "September hingga November. Dapat diketahui total pendapatan terbesar"
        "didapat pada Bulan November yaitu 8056, sedangkan total pendapatan"
        "terkecil didapat pada Bulan Desember sebesar 1914. Bulan Desember"
        "terjadi penurunan tajam dimungkinkan karena ketersediaan data yang"
        "belum lengkap.")

    else:
        st.info("No rows for 2011 under current filters.")
else:
    st.info("Date columns needed for monthly trend are missing.")

# ============ Viz 5: November Drill-down ============
st.subheader("Apa yang terjadi pada Bulan November 2011?")
if "Month_Name" in df_locked.columns:
    nov = df_locked[df_locked["Month_Name"] == "November"]
    if not nov.empty and {"Description","Quantity"}.issubset(nov.columns):
        prod_nov = (
            nov.groupby("Description", as_index=False)["Quantity"].sum()
               .sort_values("Quantity", ascending=False).head(10)
        )

        with sns.axes_style("white"):  # tanpa grid
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                y="Description", x="Quantity", data=prod_nov, ax=ax,
                palette=sns.color_palette("mako", n_colors=len(prod_nov))
            )

            # bersihkan grid & spines
            ax.grid(False)
            for s in ["top", "right", "left"]:
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color("#E5E7EB")

            ax.set_title("Top Products — November")
            ax.set_ylabel("")  # rapikan label Y
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # format ribuan

            annotate_bars(ax, fmt="{:.0f}")
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("No November rows after filtering or missing needed columns.")

# Insight
    st.markdown(
      "Event besar atau peningkatan musiman berpengaruh, seperti Black Friday"
      "dan Cyber Monday. Banyak toko online memberikan diskon dan promo menarik"
      "yang mendorong konsumen untuk berbelanja. Beberapa produk dengan jumlah"
      "penjualan tertinggi di bulan tersebut merupakan barang-barang natal,"
      "seperti WOODEN HEART CHRISTMAS SCANDINAVIAN dan 6 GIFT TAGS VINTAGE"
      "CHRISTMAS yang menunjukkan pelanggan mulai membeli barang-barang untuk"
      "mempersiapkan natal. Produk dengan pembelian terbanyak pada Bulan November"
      "yaitu WOODEN HEART CHRISTMAS SCANDINAVIAN dengan jumlah 75 buah.")


# ============ 6) Correlation ============
st.subheader("Bagaimana korelasi antara Quantity, Revenue, dan Unit Price?")
num_cols = [c for c in ["Quantity","UnitPrice","Revenue"] if c in df_view.columns]
if len(num_cols) >= 2:
    corr = df_view[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

    # Insight
    st.markdown(
    "Korelasi antara Quantity dan Revenue (0.53) Terdapat korelasi positif yang"
    "cukup kuat, menunjukkan semakin banyak jumlah barang yang dibeli (Quantity),"
    "semakin besar pendapatan (Revenue). Korelasi antara Quantity dan UnitPrice"
    "(-0.34) Terdapat korelasi negatif yang lemah, menunjukkan ketika jumlah barang"
    "yang dibeli meningkat, harga unit cenderung sedikit menurun. Korelasi antara"
    "UnitPrice dan Revenue (0.35) Korelasi positif yang lemah menunjukkan harga"
    "unit yang lebih tinggi berkontribusi pada peningkatan pendapatan.")

else:
    st.info("Not enough numerical columns for correlation.")

st.header("Business Recommendation")

# --- Ikon (inline SVG, aman offline) ---
icon_discount = """
<svg viewBox="0 0 24 24" fill="none"><circle cx="8" cy="8" r="3" stroke="#2563EB" stroke-width="2"/>
<circle cx="16" cy="16" r="3" stroke="#2563EB" stroke-width="2"/>
<path d="M7 17L17 7" stroke="#2563EB" stroke-width="2" stroke-linecap="round"/></svg>
"""
icon_calendar = """
<svg viewBox="0 0 24 24" fill="none"><rect x="3" y="4" width="18" height="17" rx="2" stroke="#0EA5E9" stroke-width="2"/>
<path d="M3 9h18" stroke="#0EA5E9" stroke-width="2"/><path d="M8 2v4M16 2v4" stroke="#0EA5E9" stroke-width="2" stroke-linecap="round"/>
<circle cx="16" cy="14" r="2.2" fill="#0EA5E9"/></svg>
"""
icon_stock = """
<svg viewBox="0 0 24 24" fill="none"><rect x="3" y="7" width="18" height="12" rx="2" stroke="#F59E0B" stroke-width="2"/>
<path d="M3 12h18M12 7v12" stroke="#F59E0B" stroke-width="2"/></svg>
"""
icon_cx = """
<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="3.5" stroke="#10B981" stroke-width="2"/>
<path d="M3 20c2.2-3 5.3-4.5 9-4.5S18.8 17 21 20" stroke="#10B981" stroke-width="2" stroke-linecap="round"/></svg>
"""

# --- Styles untuk kartu ---
st.markdown("""
<style>
.rec-card{background:#fff;border:1px solid #e9eef4;border-radius:16px;padding:18px;
          box-shadow:0 2px 10px rgba(0,0,0,.05);height:100%}
.rec-icon{display:flex;justify-content:center;align-items:center;background:#f6f8ff;
          width:56px;height:56px;border-radius:999px;margin-bottom:10px}
.rec-icon svg{width:28px;height:28px}
.rec-title{font-weight:700;margin:2px 0 6px 0}
.rec-text{color:#475569;font-size:.95rem;line-height:1.45}
</style>
""", unsafe_allow_html=True)

def card(icon_svg, title, lines):
    bullets = "".join([f"<li>{l}</li>" for l in lines])
    st.markdown(f"""
    <div class="rec-card">
      <div class="rec-icon">{icon_svg}</div>
      <div class="rec-title">{title}</div>
      <div class="rec-text"><ul>{bullets}</ul></div>
    </div>
    """, unsafe_allow_html=True)

# --- grid 2x2: 2 kolom × 2 baris ---
r1c1, r1c2 = st.columns(2)
with r1c1:
    card(icon_discount, "Optimalkan Diskon & Promosi", [
        "Jadwalkan promo/flash sale saat puncak ±12:00 (jam makan siang).",
        "Berikan penawaran eksklusif/time-limited di jam tersebut.",
        "Kirim email marketing / push notification menjelang jam puncak."
    ])
with r1c2:
    card(icon_calendar, "Musiman, Event Besar & Stok", [
        "Maksimalkan kampanye November (Black Friday, Cyber Monday).",
        "Promo/bundle bertema Natal & liburan untuk menarik lebih banyak konsumen.",
        "Perkirakan permintaan musiman agar stok aman di puncak penjualan."
    ])

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
r2c1, r2c2 = st.columns(2)
with r2c1:
    card(icon_stock, "Dorong Pembelian Jumlah Besar", [
        "Diskon grosir / volume pricing & penawaran bundle.",
        "Tampilkan harga bertingkat untuk mendorong pembelian lebih banyak.",
        "Targetkan segmen yang berpotensi beli banyak sekaligus."
    ])
with r2c2:
    card(icon_cx, "Tingkatkan Layanan & UX", [
        "Rekomendasi produk yang dipersonalisasi dari pola belanja.",
        "Respons cepat (live chat) & pengiriman tepat waktu.",
        "Perkuat pengalaman pasca-beli (tracking, retur mudah)."
    ])

st.divider()
st.success("Thank You.")
