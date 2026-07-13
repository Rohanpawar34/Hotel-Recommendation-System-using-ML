import streamlit as st
import pandas as pd
import plotly.express as px
from recommender import recommend_hotels,similar_hotels,df
import os
import requests
# ==========================
# LOAD REVIEWS file 
# ==========================
if os.path.exists("reviews.csv"):
    reviews_df = pd.read_csv("reviews.csv")

    reviews_df["Hotel_Name"] = (reviews_df["Hotel_Name"].astype(str).str.strip().str.lower())
    reviews_df["Review"] = (reviews_df["Review"].astype(str))

else:
    reviews_df = pd.DataFrame(columns=["Hotel_Name", "Review"])

from clustering import cluster_hotels
# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Hotel Recommendation System",
    page_icon="🏩",
    layout="wide"
)
# =====================================
# GLOBAL DARK THEME 
# =====================================
st.markdown(
    """
    <style>
      :root{
        --bg0:#0b1220;
        --bg1:#0f172a;
        --card:rgba(255,255,255,.06);
        --card2:rgba(255,255,255,.09);
        --text:#e5e7eb;
        --muted:rgba(229,231,235,.75);
        --border:rgba(255,255,255,.12);
        --accent:#F4C26B;
        --green:#22C55E;
      }

      /* Background */
      .stApp{ background: radial-gradient(1200px 600px at 10% 10%, rgba(244,194,107,.10), transparent 60%), var(--bg0) !important; }
      section.main{ background: transparent !important; }

      /* Sidebar */
      section[data-testid="stSidebar"]{ background: var(--bg1) !important; border-right:1px solid var(--border); }
      section[data-testid="stSidebar"] *{ color: var(--text) !important; }

      /* Text */
      .stMarkdown, .stText, .stLabel, label{ color: var(--text) !important; }

      /* Containers */
      [data-testid="stContainer"], .block-container{ background: transparent !important; }

      /* Inputs */
      .stSelectbox > div, .stMultiSelect > div, .stTextInput > div, .stSlider > div, .stNumberInput > div,
      textarea, input, .stTextArea textarea{
        background: rgba(255,255,255,.06) !important;
        color: var(--text) !important;
      }
      .stTextInput input, .stNumberInput input, textarea{
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
      }

      /* Buttons */
      button[kind="primary"], .stButton > button{
        background: linear-gradient(135deg, rgba(244,194,107,.28), rgba(244,194,107,.10)) !important;
        border:1px solid rgba(244,194,107,.35) !important;
        color: var(--text) !important;
        border-radius: 12px !important;
      }
      button[kind="primary"]:hover, .stButton > button:hover{
        border:1px solid rgba(244,194,107,.55) !important;
      }

      /* Metrics */
      [data-testid="stMetric"]{
        background: var(--card) !important;
        border:1px solid rgba(255,255,255,.10) !important;
        border-radius: 15px !important;
        box-shadow: none !important;
      }
      [data-testid="stMetric"] .metric-value{ color: var(--accent) !important; }

      /* Divider */
      hr, .stDivider{ border-color: rgba(255,255,255,.12) !important; }

      /* Plotly */
      .js-plotly-plot{ border-radius: 15px; overflow: hidden; }

      /* Chips (multiselect) */
      .stMultiSelect [data-baseweb="chip"]{
        background: rgba(255,255,255,.06) !important;
        border: 1px solid rgba(255,255,255,.10) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
#======================================
# HOTEL IMAGES
#======================================
IMAGE_DIR = "Pictures_Hotels"
HOTEL_IMAGE_MAP = {
    "The Leela Inn Mumbai": "Grand_hotel.jpg",
    "Taj Suites Pune": "Taj_Hotel.jpeg",
    "Courtyard Inn Ahmedabad": "the-residency-towers.jpg",
    "Trident Grand Pune": "Luxury_hotel.jpg",
    "The Leela Grand Chennai": "Orchid.jpg",
    "Westin Suites Pune": "hotel_2.jpg",
    "Vivanta Royal Pune": "Heritage_hotel.webp",
    "Fortune Resort Delhi": "fernando-alvarez-rodriguez-M7GddPqJowg-unsplash.jpg",
    "Vivanta Suites Pune": "Grand_hotel.jpg",
    "Vivanta Suites Goa": "Taj_Hotel.jpeg",
    "Ramada Grand Jaipur": "the-residency-towers.jpg",
    "ITC Plaza Chennai": "Luxury_hotel.jpg",
    "Trident Inn Pune": "Orchid.jpg",
}

IMAGE_FILES = [
    "fernando-alvarez-rodriguez-M7GddPqJowg-unsplash.jpg",
    "Grand_hotel.jpg",
    "Heritage_hotel.webp",
    "hotel_2.jpg",
    "Luxury_hotel.jpg",
    "Orchid.jpg",
    "Taj_Hotel.jpeg",
    "the-residency-towers.jpg",
]

def get_hotel_image(hotel_name: str) -> str:
    """Return an absolute-ish path usable by Streamlit for a given hotel."""
    if pd.isna(hotel_name):
        return f"{IMAGE_DIR}/{IMAGE_FILES[0]}"

    hotel_name = str(hotel_name).strip()

    if hotel_name in HOTEL_IMAGE_MAP:
        return f"{IMAGE_DIR}/{HOTEL_IMAGE_MAP[hotel_name]}"

    # Fallback: deterministic choice per hotel name.
    idx = abs(hash(hotel_name)) % len(IMAGE_FILES)
    return f"{IMAGE_DIR}/{IMAGE_FILES[idx]}"

st.markdown("""
<style>

/* Hero Section */
.hero {
    background: linear-gradient(
        135deg,
        rgba(244,194,107,0.15),
        rgba(255,255,255,0.05)
    );
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    border: 1px solid rgba(244,194,107,0.3);
    margin-bottom: 25px;
}

/* Cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
}

/* Charts */
.js-plotly-plot {
    border-radius: 15px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)
# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>

/* HOTEL CARD */

.hotel-card{
background:#1E293B;
border-radius:18px;
padding:15px;
border:1px solid rgba(255,255,255,.08);
box-shadow:0 10px 25px rgba(0,0,0,.35);
height:450px;
display:flex;
flex-direction:column;
justify-content:space-between;
transition:.3s;
margin-bottom:20px;

}

.hotel-card:hover{
transform:translateY(-8px);
border:1px solid #F4C26B;
box-shadow:0 18px 40px rgba(244,194,107,.20);

}

/* IMAGE */
.hotel-img{
width:100%;
height:230px;
object-fit:cover;
border-radius:12px;
margin-bottom:12px;
}

/* HOTEL NAME */
.hotel-title{
font-size:22px;
font-weight:700;
color:#F4C26B;
margin-top:8px;
}

/* INFO */
.hotel-info{
color:#E5E7EB;
font-size:15px;
line-height:1.8;
}

/* PRICE */
.price{
background:#16A34A;
padding:8px 15px;
border-radius:8px;
display:inline-block;
font-weight:bold;
color:white;
margin-top:8px;
}

/* REVIEW */
.review{
background:#111827;
padding:10px;
border-left:5px solid #22C55E;
border-radius:8px;
margin-top:8px;
font-size:14px;
color:white;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HERO SECTION
# =====================================
st.markdown("""
<div style="
padding:25px;
border-radius:18px;
background:rgba(255,255,255,0.05);
margin-bottom:20px;
text-align:center;
">
<h1 style="color:#f4c26b;">
🏨 Find the Best Hotels Based on Your Preferences
</h1>

<h4>
Discover Personalized Hotel Recommendations
            Based on Your Budget, Rating, Amenities, and Sentiment Analysis of Reviews.
</h4>

</div>
""", unsafe_allow_html=True)
# =====================================
# SIDEBAR
# =====================================

st.sidebar.markdown(
    "<div class='sidebar-title'>Find Your Perfect Hotel</div>",
    unsafe_allow_html=True
)
st.markdown("""
<style>
.sidebar-title {
    color: #d4af37;   /* gold */
    font-size: 24px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


city = st.sidebar.selectbox(
    "Select City",
    sorted(df["City"].unique())
)

budget = st.sidebar.slider(
    "Maximum Budget",
    int(df["Price"].min()),
    int(df["Price"].max()),
    int(df["Price"].median())
)

rating = st.sidebar.slider("Minimum Rating",1.0,5.0,4.0)

amenities = st.sidebar.multiselect("Amenities",
    ["Wifi","Pool","Gym","Spa","Parking","Restaurant"]
)
# =====================================
# HOTEL RECOMMENDATIONS
# =====================================

if st.sidebar.button("🏨 Recommend Hotels"):

    hotels = recommend_hotels(city,budget,rating,amenities)

    st.subheader("⭐ Recommended Hotels")

    if hotels.empty:
        st.warning("No hotels found.")

    else:

        cols = st.columns(2)

        for idx, (_, row) in enumerate(hotels.head(10).iterrows()):


            with cols[idx % 2]:

                with st.container(border=True):

                    # ---------------- IMAGE ---------------- #

                    st.image(
                        get_hotel_image(row["Hotel_Name"]),
                        use_container_width=True
                    )

                    st.markdown(
                        f"""
<h3 style='color:#F4C26B;text-align:center;margin-top:10px;'>
{row['Hotel_Name']}
</h3>
""",
                        unsafe_allow_html=True,
                    )

                    # ---------------- INFO ---------------- #

                    c1, c2 = st.columns(2)

                    with c1:
                        st.metric(
                            "⭐ Rating",
                            f"{row['Rating']}"
                        )

                    with c2:
                        st.metric(
                            "💰 Price",
                            f"₹{int(row['Price'])}"
                        )
                    # ---------------- CATEGORY ---------------- #

                    if row["Price"] <= 3000:

                        st.success("💚 Budget Hotel")

                    elif row["Price"] <= 6000:

                        st.info("🟦 Standard Hotel")

                    else:
                        st.warning("🟧 Luxury Hotel")
                    # ---------------- AMENITIES ---------------- #

                    st.markdown("#### 🛎 Amenities")

                    amenities_list = (
                        str(row["Amenities"])
                        .split(",")
                    )

                    for amenity in amenities_list[:5]:

                        st.write(f" {amenity.strip()}")
# =====================================================
# SIMILAR HOTELS
# =====================================================
st.divider()
st.header("🔍 Similar Hotels")
selected = st.selectbox("Select Hotel", df["Hotel_Name"].unique())

if st.button("Find Similar"):

    sim = similar_hotels(selected)

    if sim.empty:
        st.warning("No similar hotels found")
    else:
        cols = st.columns(2)

        for i, (_, r) in enumerate(sim.head(10).iterrows()):
            with cols[i % 2]:
                with st.container(border=True):

                    st.markdown(f"### 🏨 {r['Hotel_Name']}")
                    st.write(f"📍 {r['City']}")
                    st.write(f"⭐ {r['Rating']}")

                    price_val = float(r["Price"])

                    st.write(f"💰 ₹{int(price_val):,}")

                    # =========================
                    # CATEGORY BUCKETS
                    # =========================
                    if price_val >= 20000:
                        st.warning(f"₹{int(price_val):,} • Luxury")
                    elif price_val >= 10000:
                        st.info(f"₹{int(price_val):,} • Standard")
                    else:
                        st.success(f"₹{int(price_val):,} • Economic")

## Hotel Booking 
booking_file = "Bookings.csv"
booking_columns = [
    "Hotel_Name",
    "City",
    "Name",
    "Phone",
    "Check_in",
    "Check_out",
    "Guests",
    "Nights",
    "Price_per_night",
    "Total_Price",
]
if not os.path.exists(booking_file):
    pd.DataFrame(columns=booking_columns).to_csv(booking_file, index=False)

st.divider()
st.subheader("Enter the Details to Book Your Hotel")
with st.form("Booking_Detail", clear_on_submit=True):

    hotel_choice = st.selectbox(
        "Select Hotel",
        df["Hotel_Name"].tolist(),
    )

    name = st.text_input("Full Name")
    phone = st.text_input("Phone")
    check_in = st.date_input("Check-in")
    check_out = st.date_input("Check-out")
    guests = st.number_input("Guests", min_value=1, max_value=20, value=1, step=1)

    submit = st.form_submit_button("Confirm Booking")

    if submit:
            if not name.strip():
                st.error("Please enter your name.")
                st.stop()
            if not phone.strip():
                st.error("Please enter your phone number.")
                st.stop()
            if not (check_in and check_out):
                st.error("Please select check-in and check-out dates.")
                st.stop()

            nights = (pd.Timestamp(check_out) - pd.Timestamp(check_in)).days
            if nights <= 0:
                st.error("Check-out must be after check-in.")
                st.stop()

            hotel_match = df[df["Hotel_Name"] == hotel_choice].iloc[0]
            price_per_night = float(hotel_match["Price"])
            total_price = nights * price_per_night
            city_name = str(hotel_match["City"]).strip()

            booking_row = {
                "Hotel_Name": str(hotel_choice).strip(),
                "City": city_name,
                "Name": name.strip(),
                "Phone": phone.strip(),
                "Check_in": str(check_in),
                "Check_out": str(check_out),
                "Guests": int(guests),
                "Nights": int(nights),
                "Price_per_night": float(price_per_night),
                "Total_Price": float(total_price),
            }

            existing = pd.read_csv(booking_file)
            updated = pd.concat([existing, pd.DataFrame([booking_row])], ignore_index=True)
            updated.to_csv(booking_file, index=False)

            st.success("Booking confirmed| Thank you for choosing our service!")


# =====================================
# Hotel ANALYTICS DASHBOARD
# =====================================
    st.divider()
    st.header("Hotel Analytics Dashboard")
# =====================================
# KPI SECTION
# =====================================
st.markdown("""

<style>

.kpi-card{

    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    height: 120px;

    display:flex;
    flex-direction:column;
    justify-content:center;

    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

.kpi-title{
    color:#D1D5DB;
    font-size:22px;
    font-weight:700;
    margin-bottom:12px;
    line-height:1.3;
}

.kpi-value{
    color:#F4C26B;
    font-size:34px;
    font-weight:700;
}

</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Hotels Available</div>
        <div class="kpi-value">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)


pool_hotels = df["Amenities"].str.contains("Pool", case=False, na=False).sum()

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Hotels Offering Pool</div>
        <div class="kpi-value">{pool_hotels}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">No of Cities Covered</div>
        <div class="kpi-value">{df['City'].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Average Rating of Hotels </div>
        <div class="kpi-value">{round(df['Rating'].mean(),1)}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
col1, col2 = st.columns(2)
# -------------------------------------
# Hotels Availability by City
# -------------------------------------

with col1:
    city_count = ( df["City"].value_counts().reset_index())
    city_count.columns = ["City", "Hotels"]

    fig1 = px.bar(
        city_count.sort_values("Hotels"),
        x="Hotels",
        y="City",
        orientation="h",
        title=" Hotels Available by City",
    )

    fig1.update_traces(marker_color="#4A90E2")

    fig1.update_layout(
        height=450,
        template="plotly_dark",
        paper_bgcolor="rgba(11,18,32,0)",
        plot_bgcolor="rgba(11,18,32,0)",
        margin=dict(l=20,r=20,t=55,b=10),


        xaxis_title="",
        yaxis_title="",

        title_font=dict(
            size=22,
            color="White"
        ),

        xaxis=dict(
            tickfont=dict(
                size=15,
                family="Arial Black",
                color="white"
            ),
            title_font=dict(
                size=16,
                color="white"
            )
        ),

        yaxis=dict(
            tickfont=dict(
                size=15,
                family="Arial Black",
                color="white"
            ),
            title_font=dict(
                size=16,
                color="white"
            )
        )
    )

    st.plotly_chart(fig1, width='stretch')
# -------------------------------------
# Average Price by City
# -------------------------------------
with col2:

    avg_price = (df.groupby("City")["Price"] .mean().reset_index())
    fig2 = px.bar(avg_price,x="City",y="Price",title="💰 Average Price by City")

    fig2.update_traces(
        marker_color="#4CAF50",
        textposition="none"   # hide labels
    )

    fig2.update_layout(
        height=450,
        template="plotly_dark",
        paper_bgcolor="rgba(11,18,32,0)",
        plot_bgcolor="rgba(11,18,32,0)",
        margin=dict(l=20,r=20,t=55,b=10),


        xaxis_title="",
        yaxis_title="",

        title_font=dict(
            size=22,
            color="white",
            family="Arial Black"
        ),

        xaxis=dict(
            tickfont=dict(
                size=15,
                color="white",
                family="Arial Black"
            ),
            title_font=dict(
                size=15,
                color="white",
                family="Arial Black"
            )
        ),

        yaxis=dict(
            tickfont=dict(
                size=15,
                color="white",
                family="Arial Black"
            ),
            title_font=dict(
                size=15,
                color="white",
                family="Arial Black"
            )
        )
    )

    st.plotly_chart(fig2,width='stretch')

col3, col4 = st.columns(2)
# -------------------------------------
# Hotel Rating by Sections
# -------------------------------------
with col3:
    category_df = df.copy()
    category_df["Category"] = pd.cut(
        category_df["Price"],
        bins=[0, 3000, 6000, float("inf")],
        labels=["Economic","Standard","Luxury"]
    )

    section_rating = (
        category_df
        .groupby("Category")["Rating"]
        .mean()
        .reset_index()
    )

    fig3 = px.bar(
        section_rating,
        x="Category",
        y="Rating",
        text=section_rating["Rating"].round(1),
        color="Category",
        title="⭐ Hotel Rating by Sections",
        color_discrete_map={
            "Economic": "#4CAF50",
            "Standard": "#E67E22",
            "Luxury": "#D4AF37"
        }
    )

    fig3.update_traces(textposition="outside")

    fig3.update_layout(
        height=400,
        template="plotly_dark",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Hotel Category",
        yaxis_title="Average Rating"
    )

    st.plotly_chart(fig3,width='stretch')
# -------------------------------------
# Top Rated Hotels
# -------------------------------------
with col4:

    top_hotels = (
        df.sort_values(
            "Rating",
            ascending=False
        )
        .head(10)
    )

    fig4 = px.bar(
        top_hotels,
        x="Rating",
        y="Hotel_Name",
        orientation="h",
        text="Rating",
        title="🏆 Top Rated Hotels"
    )

    fig4.update_traces(
        marker_color="#F4C26B"
    )

    fig4.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Rating",
        yaxis_title=""
    )
    st.plotly_chart(fig4,width="stretch")
# =====================================
# ABOUT THE SYSTEM
# =====================================
st.markdown("""
<div style="
background:rgba(255,255,255,0.05);
padding:35px 45px;
border-radius:18px;
border:1px solid rgba(244,194,107,0.25);
margin-bottom:30px;
">

<h2 style="
color:#F4C26B;
text-align:center;
font-size:36px;
font-family:'Segoe UI', Arial, sans-serif;
font-weight:bold;
margin-bottom:25px;
">
 About the Hotel Recommendation System
</h2>

<p style="
font-family:'Segoe UI', Arial, sans-serif;
font-size:16px;
font-weight:400;
color:#E5E7EB;
line-height:1.9;
text-align:justify;
letter-spacing:0.3px;
">

The <b>Hotel Recommendation System</b> is an intelligent web application developed using <b>Python</b> and <b>Streamlit</b>. It is designed to help users discover hotels that best match their travel preferences. Users can search hotels by selecting their preferred city, budget, minimum rating, and desired amenities. The recommendation engine analyzes hotel information and suggests the most suitable hotels based on the selected criteria. The application provides a simple, fast, and user-friendly interface for searching hotels.

In addition to personalized recommendations, the system offers a <b>Similar Hotels</b> feature that recommends hotels with comparable characteristics. This allows users to explore more accommodation options without performing multiple searches. The application also includes an interactive analytics dashboard that presents useful insights through attractive charts and visualizations. Users can analyze hotel availability across different cities, compare average hotel prices, and view rating distributions. The dashboard also highlights the top-rated hotels available in the dataset.

The system incorporates <b>sentiment analysis</b> of customer reviews to understand guest opinions and improve recommendation quality. These review insights help users make informed booking decisions based on previous customer experiences. The application features a responsive interface with a modern dark theme, ensuring an engaging user experience. By combining recommendation techniques, review analysis, and interactive data visualization, the Hotel Recommendation System simplifies hotel selection and helps users choose the best accommodation according to their needs.

</p>

<hr style="border:1px solid rgba(255,255,255,0.12);margin:30px 0;">

<h3 style="
color:#F4C26B;
font-size:28px;
font-family:'Segoe UI', Arial, sans-serif;
font-weight:600;
margin-bottom:20px;
">
Key Features
</h3>

<table style="
width:100%;
border-collapse:separate;
border-spacing:15px 12px;
font-family:'Segoe UI', Arial, sans-serif;
font-size:16px;
color:#E5E7EB;
">

<tr>
<td> Personalized Hotel Recommendations</td>
<td> Similar Hotel Suggestions</td>
<td> Budget-Based Filtering</td>
<td> Rating-Based Filtering</td>
</tr>

<tr>
<td> Amenities-Based Filtering</td>
<td> City-Wise Hotel Search</td>
<td> Interactive Analytics Dashboard</td>
<td> Average Price Analysis by City</td>
</tr>

<tr>
<td> Top-Rated Hotels Display</td>
<td> Customer Review Sentiment Analysis</td>
<td> Interactive Data Visualization</td>
</tr>

</table>

</div>
""", unsafe_allow_html=True)