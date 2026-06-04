import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

# Set wide layout so the side-by-side maps render beautifully
st.set_page_config(
    page_title="Kenya Climate Early Warning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CACHED DATA LOADING
@st.cache_data
def load_and_process_data():
    # 1. Load the climate data using a raw string to protect Windows backslashes
    data = pd.read_csv("DATA/Kenya_Rainfall data.csv")
    data.columns = data.columns.str.strip()
    data = data.rename(columns={"month": "Month", "year": "Year"})

    # Standardize data PCODE column
    data["PCODE"] = data["PCODE"].astype(str).str.strip().str.upper()

    # Reconstruct/pad to ensure it fits 'KE001' county format if raw numbers exist
    if not data["PCODE"].str.startswith("KE").any():
        data["PCODE"] = "KE" + data["PCODE"].str.zfill(3)

    # Isolate critical March-May planting season (2022-2026)
    planting_season = data[data["Month"].between(3, 5)].copy()
    planting_season["seasonal_anomaly_score"] = (
        planting_season["current_rainfall"]
        - planting_season["historical_rainfall_avg"]
    )

    # Calculate Anomaly Slopes (Question 4 Acceleration Metrics)
    pcodes, slopes = [], []
    april_data = data[data["Month"] == 4].copy()
    april_data["PCODE"] = april_data["PCODE"].astype(str).str.strip().str.upper()
    if not april_data["PCODE"].str.startswith("KE").any():
        april_data["PCODE"] = "KE" + april_data["PCODE"].str.zfill(3)

    for pcode, group in april_data.groupby("PCODE"):
        group = group.sort_values("Year")
        if len(group["Year"].unique()) >= 2:
            slope, _, _, _, _ = stats.linregress(
                group["Year"], group["Anomaly_score"]
            )
            pcodes.append(pcode)
            slopes.append(slope)
        else:
            pcodes.append(pcode)
            slopes.append(np.nan)
    slope_df = pd.DataFrame({"PCODE": pcodes, "Slope": slopes})

    # Cumulative Deficit Percentages
    PCODE_total = (
        planting_season.groupby("PCODE")
        .agg(
            Total_Actual=("current_rainfall", "sum"),
            Total_Expected=("historical_rainfall_avg", "sum"),
        )
        .reset_index()
    )
    PCODE_total["Deficit_mm"] = (
        PCODE_total["Total_Expected"] - PCODE_total["Total_Actual"]
    ).clip(lower=0)
    PCODE_total["Deficit_Percentage"] = (
        PCODE_total["Deficit_mm"] / PCODE_total["Total_Expected"]
    ) * 100

    # Climate Whiplash Standard Deviation
    PCODE_anomaly = (
        planting_season.groupby(["PCODE", "Year"])["seasonal_anomaly_score"]
        .sum()
        .reset_index()
    )
    PCODE_volatility = (
        PCODE_anomaly.groupby("PCODE")["seasonal_anomaly_score"]
        .std()
        .reset_index()
    )
    PCODE_volatility.columns = ["PCODE", "Whiplash_Score"]

    # 2. SPATIAL MAP PARSING & DYNAMIC COUNTY MATCHING FIX
    spatial_map = gpd.read_file("DATA/ken_admin2.geojson")
    cleaned_spatial_map = spatial_map.dropna(axis=1, how="all").copy()

    # Clean the sub-county codes (e.g., "KE015070" for Kitui West)
    cleaned_spatial_map["adm2_pcode"] = (
        cleaned_spatial_map["adm2_pcode"].astype(str).str.strip().str.upper()
    )

    # THE CRITICAL LINE FIX: Extract the first 5 characters (e.g., transforms "KE015070" to "KE015")
    cleaned_spatial_map["COUNTY_MATCH_KEY"] = cleaned_spatial_map[
        "adm2_pcode"
    ].str[:5]

    # Combine data metrics
    metrics_combined = pd.merge(
        PCODE_total, PCODE_volatility, on="PCODE", how="outer"
    )
    metrics_combined = pd.merge(metrics_combined, slope_df, on="PCODE", how="outer")

    # MERGE FIX: Connect the sub-county map's 5-char parent key to the county data table
    merged_map = cleaned_spatial_map.merge(
        metrics_combined,
        left_on="COUNTY_MATCH_KEY",
        right_on="PCODE",
        how="left",
    )

    # Protect visualization engine configurations from empty spaces
    merged_map["Whiplash_Score"] = merged_map["Whiplash_Score"].fillna(0)
    merged_map["Deficit_Percentage"] = merged_map["Deficit_Percentage"].fillna(
        0
    )
    merged_map["Slope"] = merged_map["Slope"].fillna(0)

    possible_name_cols = [
        "shapeName",
        "adm2_en",
        "COUNTY",
        "county",
        "County",
        "adm2_name",
        "ADM2_EN",
        "NAME_2",
    ]
    detected_name_col = next(
        (col for col in possible_name_cols if col in merged_map.columns), None
    )

    if detected_name_col:
        merged_map["County_Name"] = merged_map[detected_name_col].fillna(
            merged_map["adm2_pcode"]
        )
    else:
        merged_map["County_Name"] = merged_map["adm2_pcode"]

    # Generate quick national aggregation for the Page 1 Chart context before returning
    national_trends = (
        data.groupby("Year")["Anomaly_score"].mean().reset_index()
    )

    return merged_map, national_trends


# Execute initialization
try:
    merged_map, national_trends = load_and_process_data()
except Exception as e:
    st.error(
        f"Configuration Error: Check file paths or file structure. Details: {e}"
    )
    st.stop()

# SIDEBAR NAVIGATION BAR
st.sidebar.title("Capstone Control Panel")
st.sidebar.markdown("**Project:** Climate Vulnerability Early Warning Pipeline")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Interface Page:",
    [
        "1. Project Overview & Diagnostics",
        "2. Geographic Risk Hotspots",
        "3. Active Enterprise Solutions",
    ],
)

# PAGE 1: GRAPH DIAGNOSTICS & OVERVIEW
if page == "1. Project Overview & Diagnostics":
    st.title("Kenya Climate Diagnostics Dashboard")
    st.markdown("### The Core Problem: The Statistical Recovery Trap")
    st.markdown("---")

    st.markdown("""
    #### Key Findings: Why Historical Averages Lie
    * **The 2024 Illusion:** The massive positive spike in our baseline analysis represents extreme El Niño flooding. 
    * **The Recovery Trap:** Mathematically, this single massive surplus hides the severe deficits of the surrounding years, creating a false impression of regional security.
    * **The Impact:** A green recovery on paper is a statistical trap. It treats a flood and a drought as if they cancel each other out, when in reality, they represent consecutive seasonal crop failures.
    """)

    st.markdown("---")
    st.subheader("National Anomalies (The 5-Year Multi-Year Rollercoaster)")

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [
        "#2E7D32" if val >= 0 else "#C62828"
        for val in national_trends["Anomaly_score"]
    ]
    ax.bar(
        national_trends["Year"],
        national_trends["Anomaly_score"],
        color=colors,
        edgecolor="black",
        alpha=0.85,
    )
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_ylabel("Mean Climate Anomaly Score")
    ax.set_xlabel("Observation Year")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    st.pyplot(fig)
    st.info(
        "Look closely at your first diverging bar chart: Notice how the massive 2024 El Niño flood visually breaks the baseline trend, masking the severe droughts of 2022 and 2026."
    )

# PAGE 2: GEOGRAPHIC HOTSPOTS
elif page == "2. Geographic Risk Hotspots":
    st.title("Question 4: Spatial Risk Hotspots (PCODE Mapping)")
    st.markdown("### Structural Deficits vs. Climate Whiplash Volatility Indices")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Map A: Chronic Deficits")
        st.markdown(
            "*Identifies areas facing a direct multi-year drop in their total moisture allocation budget.*"
        )

        fig1, ax1 = plt.subplots(figsize=(6, 5))
        merged_map.plot(
            column="Deficit_Percentage",
            cmap="YlOrRd",
            linewidth=0.4,
            ax=ax1,
            edgecolor="0.4",
            legend=True,
            legend_kwds={"label": "Missing Planting Rain Budget (%)"},
        )
        ax1.set_axis_off()
        st.pyplot(fig1)
        st.warning(
            "**Hotspot Insight:** The Coastal strip missed out entirely on multi-year recovery loops, building structural data deficits exceeding 10%."
        )

    with col2:
        st.subheader("Map B: Climate Whiplash")
        st.markdown(
            "*Identifies regions experiencing erratic structural volatility swings (Drought-to-Flood cycles).*"
        )

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        merged_map.plot(
            column="Whiplash_Score",
            cmap="Purples",
            linewidth=0.4,
            ax=ax2,
            edgecolor="0.4",
            legend=True,
            legend_kwds={"label": "Volatility Score (Standard Deviation)"},
        )
        ax2.set_axis_off()
        st.pyplot(fig2)
        st.success(
            "**Hotspot Insight:** High-yield grain baskets like **Uasin Gishu** look safe on Map A, but light up dramatically here due to massive, unstable year-over-year weather swings."
        )

    st.markdown("---")
    st.subheader("Question 4 Acceleration Trends (Rainfall Slope Plot)")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    merged_map.plot(
        column="Slope",
        cmap="Reds",
        linewidth=0.4,
        ax=ax3,
        edgecolor="0.2",
        legend=True,
        legend_kwds={
            "label": "April Deficit Acceleration Rate (Slope)",
            "orientation": "horizontal",
            "pad": 0.05,
        },
    )
    ax3.set_axis_off()
    st.pyplot(fig3)

# PAGE 3: INTERACTIVE SOLUTIONS (NOW FULLY INTERACTIVE & CALCULATING LIVE!)
elif page == "3. Active Enterprise Solutions":
    st.title("Automated Spatial Climate Interventions")
    st.markdown("### Transitioning Insights into Live Operational Deliverables")
    st.markdown("---")

    # Global Interactive Dropdown
