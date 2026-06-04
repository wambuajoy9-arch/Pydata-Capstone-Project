import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Kenya Climate Pipeline", layout="wide")

# ── Refresh button (OUTSIDE cache, BEFORE data load) ──
if st.sidebar.button(" Refresh Data"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data(ttl=3600)
def load_and_process_data():
    data = pd.read_csv("DATA/Kenya_Rainfall data.csv")
    data.columns = data.columns.str.strip()
    data = data.rename(columns={'month': 'Month', 'year': 'Year'})
    data["PCODE"] = data["PCODE"].astype(str).str.strip().str.upper()

    # ── DEBUG: return raw year/month range so we can verify ──
    year_range = sorted(data["Year"].unique())
    month_range = sorted(data["Month"].unique())

    # Use actual data range instead of hardcoding 2022-2026
    min_year = int(data["Year"].min())
    max_year = int(data["Year"].max())

    planting_season = data[
        data["Month"].between(3, 5) &
        data["Year"].between(min_year, max_year)
    ].copy()

    planting_season["seasonal_anomaly_score"] = (
        planting_season["current_rainfall"] - planting_season["historical_rainfall_avg"]
    )

    # Cumulative Deficit
    PCODE_total = planting_season.groupby("PCODE").agg(
        Total_Actual=("current_rainfall", "sum"),
        Total_Expected=("historical_rainfall_avg", "sum")
    ).reset_index()
    PCODE_total["Deficit_mm"] = (PCODE_total["Total_Expected"] - PCODE_total["Total_Actual"]).clip(lower=0)
    PCODE_total["Deficit_Percentage"] = (PCODE_total["Deficit_mm"] / PCODE_total["Total_Expected"]) * 100

    # Climate Whiplash
    PCODE_anomaly = planting_season.groupby(["PCODE", "Year"])["seasonal_anomaly_score"].sum().reset_index()
    PCODE_volatility = PCODE_anomaly.groupby("PCODE")["seasonal_anomaly_score"].std().reset_index()
    PCODE_volatility.columns = ["PCODE", "Whiplash_Score"]

    # Yearly anomaly trend per PCODE (for sparklines)
    yearly_trend = PCODE_anomaly.copy()

    # Spatial join
    spatial_map = gpd.read_file("DATA/ken_admin2.geojson")
    cleaned_spatial_map = spatial_map.dropna(axis=1, how="all")
    cleaned_spatial_map["adm2_pcode"] = cleaned_spatial_map["adm2_pcode"].astype(str).str.strip().str.upper()

    PCODE_total["PCODE"] = PCODE_total["PCODE"].astype(str).str.strip().str.upper()
    PCODE_volatility["PCODE"] = PCODE_volatility["PCODE"].astype(str).str.strip().str.upper()

    metrics_combined = pd.merge(PCODE_total, PCODE_volatility, on="PCODE", how="outer")
    merged_map = cleaned_spatial_map.merge(metrics_combined, left_on="adm2_pcode", right_on="PCODE", how="left")

    merged_map["Whiplash_Score"] = merged_map["Whiplash_Score"].fillna(0)
    merged_map["Deficit_Percentage"] = merged_map["Deficit_Percentage"].fillna(0)

    possible_name_cols = ["shapeName", "adm2_en", "COUNTY", "county", "County", "adm2_name", "ADM2_EN", "NAME_2"]
    detected_name_col = next((col for col in possible_name_cols if col in merged_map.columns), None)
    if detected_name_col:
        merged_map["County_Name"] = merged_map[detected_name_col].fillna(merged_map["adm2_pcode"])
    else:
        merged_map["County_Name"] = merged_map["adm2_pcode"]

    return merged_map, year_range, month_range, yearly_trend

# ── Load data ──
try:
    merged_map, year_range, month_range, yearly_trend = load_and_process_data()
except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# ── Sidebar navigation ──
st.sidebar.title("Capstone Control Panel")
st.sidebar.markdown("**Project:** Climate Vulnerability Early Warning Pipeline")
st.sidebar.markdown("---")

# Data health check in sidebar
with st.sidebar.expander("Data Health Check"):
    st.write("**Years in dataset:**", year_range)
    st.write("**Months in dataset:**", month_range)
    st.write("**Rows in merged map:**", len(merged_map))
    non_zero = (merged_map["Deficit_Percentage"] > 0).sum()
    st.write(f"**PCODEs with non-zero deficit:** {non_zero}/{len(merged_map)}")

page = st.sidebar.radio(
    "Select Interface Page:",
    ["1. Project Overview & Diagnostics",
     "2. Geographic Risk Hotspots",
     "3. Active Enterprise Solutions"]
)


# PAGE 1: OVERVIEW

if page == "1. Project Overview & Diagnostics":
    st.title("Kenya Climate Risk Dashboard")
    st.markdown("### The Statistical Recovery Trap")
    st.markdown("---")

    st.markdown("""
    #### Key Findings: Why Historical Averages Lie
    - **The 2024 Illusion:** The massive positive spike represents extreme El Niño flooding.
    - **The Recovery Trap:** This single surplus hides the severe deficits of surrounding years.
    - **The Impact:** A flood and a drought do not cancel out — they represent *consecutive crop failures*.
    """)

    st.markdown("---")
    st.subheader("National Anomaly Trend — Mar–May Planting Season")

    # Build national diverging bar chart from yearly_trend
    national_trend = yearly_trend.groupby("Year")["seasonal_anomaly_score"].mean().reset_index()
    national_trend.columns = ["Year", "Avg_Anomaly"]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#d32f2f" if v < 0 else "#1976d2" for v in national_trend["Avg_Anomaly"]]
    bars = ax.bar(national_trend["Year"].astype(str), national_trend["Avg_Anomaly"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Average Seasonal Rainfall Anomaly (Mar–May)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Anomaly (mm vs. historical avg)")
    ax.set_xlabel("Year")

    for bar, val in zip(bars, national_trend["Avg_Anomaly"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (2 if val >= 0 else -8),
                f"{val:.0f}", ha="center", va="bottom", fontsize=8, color="black")

    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")
    st.pyplot(fig)
    st.info("Notice how a single massive surplus year visually dominates, masking drought severity in adjacent years.")


# PAGE 2: SPATIAL HOTSPOT MAPS

elif page == "2. Geographic Risk Hotspots":
    st.title("Spatial Risk Hotspots — PCODE Mapping")
    st.markdown("### Structural Deficits vs. Climate Whiplash Volatility")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Map A: Chronic Deficits")
        st.markdown("*Multi-year cumulative rainfall shortfall as % of expected.*")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        merged_map.plot(column="Deficit_Percentage", cmap="YlOrRd", linewidth=0.4,
                        ax=ax1, edgecolor="0.4", legend=True,
                        legend_kwds={"label": "Missing Rain Budget (%)", "shrink": 0.7})
        ax1.set_axis_off()
        ax1.set_title("Cumulative Deficit %", fontsize=11, fontweight="bold")
        st.pyplot(fig1)
        st.warning("**Hotspot:** Coastal strip shows structural deficits exceeding 10% across planting seasons.")

    with col2:
        st.subheader("Map B: Climate Whiplash")
        st.markdown("*Year-over-year volatility — drought-to-flood swing intensity.*")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        merged_map.plot(column="Whiplash_Score", cmap="Purples", linewidth=0.4,
                        ax=ax2, edgecolor="0.4", legend=True,
                        legend_kwds={"label": "Volatility Score (Std Dev)", "shrink": 0.7})
        ax2.set_axis_off()
        ax2.set_title("Climate Whiplash Score", fontsize=11, fontweight="bold")
        st.pyplot(fig2)
        st.success("**Hotspot:** Uasin Gishu looks safe on Map A but lights up here — massive unstable year-over-year swings.")

    st.markdown("---")
    st.subheader("Interactive County Drilldown")

    # Interactive: click a county to see its anomaly time series
    county_list = sorted(merged_map["County_Name"].dropna().unique())
    selected_county = st.selectbox("Select a county to inspect its anomaly trend:", county_list)

    selected_pcode = merged_map[merged_map["County_Name"] == selected_county]["adm2_pcode"].values[0]
    county_trend = yearly_trend[yearly_trend["PCODE"] == selected_pcode].sort_values("Year")

    if county_trend.empty:
        st.warning(f"No planting season anomaly data found for **{selected_county}** (PCODE: {selected_pcode}). This county may not match any rainfall record.")
    else:
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        bar_colors = ["#d32f2f" if v < 0 else "#2e7d32" for v in county_trend["seasonal_anomaly_score"]]
        ax3.bar(county_trend["Year"].astype(str), county_trend["seasonal_anomaly_score"],
                color=bar_colors, edgecolor="white", linewidth=0.5)
        ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax3.set_title(f"Mar–May Seasonal Anomaly — {selected_county} ({selected_pcode})",
                      fontsize=12, fontweight="bold")
        ax3.set_ylabel("Anomaly (mm)")
        ax3.set_xlabel("Year")
        fig3.patch.set_facecolor("#f9f9f9")
        ax3.set_facecolor("#f9f9f9")
        st.pyplot(fig3)

        # Highlight map showing selected county
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        merged_map.plot(color="#e0e0e0", linewidth=0.3, ax=ax4, edgecolor="0.5")
        merged_map[merged_map["adm2_pcode"] == selected_pcode].plot(
            color="#e53935", linewidth=1.5, ax=ax4, edgecolor="black"
        )
        ax4.set_axis_off()
        ax4.set_title(f"Location: {selected_county}", fontsize=10)
        st.pyplot(fig4)


# PAGE 3: ENTERPRISE SOLUTIONS

elif page == "3. Active Enterprise Solutions":
    st.title("Automated Spatial Climate Interventions")
    st.markdown("### Transitioning Insights into Live Operational Deliverables")
    st.markdown("---")

    county_list = sorted(merged_map["County_Name"].dropna().unique())
    selected_name = st.selectbox("Step 1: Select Target Region for Live Evaluation:", county_list)

    county_row = merged_map[merged_map["County_Name"] == selected_name]
    target_pcode = county_row["adm2_pcode"].values[0]
    target_deficit = county_row["Deficit_Percentage"].values[0]
    target_whiplash = county_row["Whiplash_Score"].values[0]

    tab1, tab2 = st.tabs(["Solution 1: Risk Analytics Engine", "Solution 2: Last-Mile Alert Dispatcher"])

    with tab1:
        st.subheader("Dynamic Machine-Readable Risk Pipeline")

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Target PCODE", target_pcode)
        m_col2.metric("Cumulative Deficit", f"{target_deficit:.1f}%",
                      delta="⚠ Critical" if target_deficit > 8 else "✅ Stable",
                      delta_color="inverse" if target_deficit > 8 else "normal")
        m_col3.metric("Volatility Score", f"{target_whiplash:.1f}",
                      delta="⚠ High" if target_whiplash > 250 else "✅ Stable",
                      delta_color="inverse" if target_whiplash > 250 else "normal")

        st.markdown("#### Simulated API Live Output Payload:")
        json_output = {
            "PCODE": str(target_pcode),
            "Geographic_Name": str(selected_name),
            "Deficit_Percentage": round(float(target_deficit), 2),
            "Volatility_Index": round(float(target_whiplash), 2),
            "System_Status": "CRITICAL_ACTION" if (target_deficit > 8.0 or target_whiplash > 250) else "STABLE"
        }
        st.json(json_output)

        # Mini anomaly chart for selected county
        county_trend = yearly_trend[yearly_trend["PCODE"] == target_pcode].sort_values("Year")
        if not county_trend.empty:
            st.markdown("#### Anomaly History:")
            fig5, ax5 = plt.subplots(figsize=(8, 2.5))
            ax5.plot(county_trend["Year"], county_trend["seasonal_anomaly_score"],
                     marker="o", color="#1565c0", linewidth=2)
            ax5.axhline(0, color="red", linewidth=0.8, linestyle="--")
            ax5.fill_between(county_trend["Year"], county_trend["seasonal_anomaly_score"], 0,
                             where=county_trend["seasonal_anomaly_score"] < 0,
                             alpha=0.3, color="red", label="Deficit")
            ax5.fill_between(county_trend["Year"], county_trend["seasonal_anomaly_score"], 0,
                             where=county_trend["seasonal_anomaly_score"] >= 0,
                             alpha=0.3, color="green", label="Surplus")
            ax5.legend(fontsize=8)
            ax5.set_xlabel("Year")
            ax5.set_ylabel("Anomaly (mm)")
            fig5.patch.set_facecolor("#f9f9f9")
            ax5.set_facecolor("#f9f9f9")
            st.pyplot(fig5)

    with tab2:
        st.subheader("Interactive Alert Formulation & Outbox")

        custom_threshold = st.slider(
            "Adjust Critical Deficit Warning Cutoff (%)",
            min_value=1.0, max_value=15.0, value=8.0, step=0.5
        )
        st.markdown("---")

        if st.button("Execute System Diagnostic & Run Dispatch Pipeline"):
            with st.spinner("Processing geospatial layer and assembling SMS payload..."):
                if target_deficit >= custom_threshold:
                    st.error(
                        f"**SMS QUEUED (Priority: Critical) → {selected_name} Extension Officers:**\n\n"
                        f"*EMERGENCY ALERT: PCODE {target_pcode} structural deficit has reached "
                        f"{target_deficit:.1f}%, exceeding your threshold of {custom_threshold}%. "
                        f"Discontinue regular planting guidelines. Deploy emergency irrigation assistance.*"
                    )
                elif target_whiplash > 250.0:
                    st.warning(
                        f"**SMS QUEUED (Priority: High Volatility) → {selected_name} Cooperatives:**\n\n"
                        f"*ALERT: High volatility index ({target_whiplash:.1f}) detected. "
                        f"Advise farming pools to pause planting or swap to short-cycle adaptive seeds.*"
                    )
                else:
                    st.success(
                        f"**SMS QUEUED (Priority: Normal) → {selected_name} Field Hubs:**\n\n"
                        f"*STATUS: PCODE {target_pcode} parameters within safe bounds. "
                        f"Proceed with standard seasonal baseline advice.*"
                    )
            st.toast("Alert compilation completed!", icon="✅")

