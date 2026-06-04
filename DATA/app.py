import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
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
    data = data.rename(columns={'month':'Month','year':'Year'})
    
    # Standardize data PCODE column
    data["PCODE"] = data["PCODE"].astype(str).str.strip().str.upper()
    
    # Isolate critical March-May planting season (2022-2026)
    planting_season = data[data["Month"].between(3, 5)].copy()
    planting_season["seasonal_anomaly_score"] = planting_season["current_rainfall"] - planting_season["historical_rainfall_avg"]
    
    # Cumulative Deficit Percentages
    PCODE_total = planting_season.groupby("PCODE").agg(
        Total_Actual=("current_rainfall", "sum"),
        Total_Expected=("historical_rainfall_avg", "sum")
    ).reset_index()
    PCODE_total["Deficit_mm"] = (PCODE_total["Total_Expected"] - PCODE_total["Total_Actual"]).clip(lower=0)
    PCODE_total["Deficit_Percentage"] = (PCODE_total["Deficit_mm"] / PCODE_total["Total_Expected"]) * 100
    
    # Climate Whiplash Standard Deviation (Fixed Variable & Column Mismatches)
    PCODE_anomaly = planting_season.groupby(["PCODE", "Year"])["seasonal_anomaly_score"].sum().reset_index()
    PCODE_volatility = PCODE_anomaly.groupby("PCODE")["seasonal_anomaly_score"].std().reset_index()
    PCODE_volatility.columns = ["PCODE", "Whiplash_Score"]
    
    # Load your shapefile boundary map using a raw string
    spatial_map = gpd.read_file("DATA/ken_admin2.geojson")
    
    # Drops columns where all rows are completely empty
    cleaned_spatial_map = spatial_map.dropna(axis=1, how="all")
    
    # Clean the PCODE columns to ensure a perfect string match
    cleaned_spatial_map['adm2_pcode'] = (cleaned_spatial_map['adm2_pcode'].astype(str).str.strip().str.upper())
    
    # Merge metrics cleanly onto spatial map rows
    merged_map = cleaned_spatial_map.merge(PCODE_volatility, left_on="adm2_pcode", right_on="PCODE", how="left")
    merged_map = merged_map.merge(PCODE_total, left_on="adm2_pcode", right_on="PCODE", how="left")
    
    # Clean fillna targets to protect visualization engines from breaking
    merged_map["Whiplash_Score"] = merged_map["Whiplash_Score"].fillna(0)
    merged_map["Deficit_Percentage"] = merged_map["Deficit_Percentage"].fillna(0)

    possible_name_cols = ["shapeName", "adm2_en", "COUNTY", "county", "County", "adm2_name", "ADM2_EN", "NAME_2"]
    

    detected_name_col = next((col for col in possible_name_cols if col in merged_map.columns), None)
    
    if detected_name_col:
        merged_map["County_Name"] = merged_map[detected_name_col].fillna(merged_map["adm2_pcode"])
    else:
        # Emergency backup: if no name column is found, use the PCODE as the dropdown label
        merged_map["County_Name"] = merged_map["adm2_pcode"]
    
    return merged_map

# Execute and isolate loading errors on dashboard screen cleanly
try:
    merged_map = load_and_process_data()
except Exception as e:
    st.error(f"Configuration Error: Check file paths or file structure. Details: {e}")
    st.stop()


# SIDEBAR NAVIGATION BAR 
st.sidebar.title("Capstone Control Panel")
st.sidebar.markdown("**Project:** Climate Vulnerability Early Warning Pipeline")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Interface Page:",
    ["1. Project Overview & Diagnostics", "2. Geographic Risk Hotspots", "3. Active Enterprise Solutions"]
)

# PAGE 1: GRAPH DIAGNOSTICS & OVERVIEW (Clean single-chart layout)

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
    
    # PLACEHOLDER: Your code will dynamically generate or attach your diverging bar chart right below this line
    st.info("Look closely at your first diverging bar chart: Notice how the massive 2024 El Niño flood visually breaks the baseline trend, masking the severe droughts of 2022 and 2026.")
# PAGE 2: GEOGRAPHIC HOTSPOTS

elif page == "2. Geographic Risk Hotspots":
    st.title("Question 4: Spatial Risk Hotspots (PCODE Mapping)")
    st.markdown("### Structural Deficits vs. Climate Whiplash Volatility Indices")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Map A: Chronic Deficits")
        st.markdown("*Identifies areas facing a direct multi-year drop in their total moisture allocation budget.*")
        
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        merged_map.plot(
            column="Deficit_Percentage", cmap="YlOrRd", linewidth=0.4, 
            ax=ax1, edgecolor="0.4", legend=True,
            legend_kwds={"label": "Missing Planting Rain Budget (%)"}
        )
        ax1.set_axis_off()
        st.pyplot(fig1)
        st.warning("**Hotspot Insight:** The Coastal strip missed out entirely on multi-year recovery loops, building structural data deficits exceeding 10%.")

    with col2:
        st.subheader("Map B: Climate Whiplash")
        st.markdown("*Identifies regions experiencing erratic structural volatility swings (Drought-to-Flood cycles).*")
        
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        merged_map.plot(
            column="Whiplash_Score", cmap="Purples", linewidth=0.4, 
            ax=ax2, edgecolor="0.4", legend=True,
            legend_kwds={"label": "Volatility Score (Standard Deviation)"}
        )
        ax2.set_axis_off()
        st.pyplot(fig2)
        st.success("**Hotspot Insight:** High-yield grain baskets like **Uasin Gishu** look safe on Map A, but light up dramatically here due to massive, unstable year-over-year weather swings.")


# PAGE 3: INTERACTIVE SOLUTIONS

elif page == "3. Active Enterprise Solutions":
    st.title("Automated Spatial Climate Interventions")
    st.markdown("### Transitioning Insights into Live Operational Deliverables")
    st.markdown("---")
    
    # Global Interactive Dropdown
    county_list = sorted(merged_map["County_Name"].unique())
    selected_name = st.selectbox("Step 1: Select Target Region for Live Evaluation:", county_list)
    
    # Extract structural metrics live based on drop-down choice
    county_row = merged_map[merged_map["County_Name"] == selected_name]
    target_pcode = county_row["adm2_pcode"].values[0]
    target_deficit = county_row["Deficit_Percentage"].values[0]
    target_whiplash = county_row["Whiplash_Score"].values[0]
    
    # Separate operations cleanly into Solution 1 and Solution 2
    tab1, tab2 = st.tabs(["Solution 1: Interactive Risk Analytics Engine", "📱 Solution 2: Last-Mile Alert Dispatcher"])
    
    with tab1:
        st.subheader("Solution 1: Dynamic Machine-Readable Risk Pipeline")
        st.markdown("This backend processing block runs automated evaluations directly on PCODE spatial rows, bypassing misleading aggregate numbers.")
        
        # Display Live Metric Cards
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Target PCODE Identifier", target_pcode)
        m_col2.metric("Computed Cumulative Deficit", f"{target_deficit:.1f}%")
        m_col3.metric("Computed Volatility Score", f"{target_whiplash:.1f}")
        
        # Interactive JSON API Simulation output
        st.markdown("#### Simulated API Live Output Payload:")
        json_output = {
            "PCODE": str(target_pcode),
            "Geographic_Name": str(selected_name),
            "Deficit_Percentage": round(float(target_deficit), 2),
            "Volatility_Index": round(float(target_whiplash), 2),
            "System_Status": "CRITICAL_ACTION" if (target_deficit > 8.0 or target_whiplash > 250) else "STABLE"
        }
        st.json(json_output)

    with tab2:
        st.subheader("Solution 2: Interactive Alert Formulation & Outbox")
        st.markdown("This frontend deployment interface lets system administrators simulate thresholds and push alerts directly to local field agents.")
        
        # Slider interaction parameter
        custom_threshold = st.slider(
            "Adjust Critical Deficit Warning Cutoff Point (%)",
            min_value=1.0, max_value=15.0, value=8.0, step=0.5
        )
        
        st.markdown("---")
        
        # Simulation button execution step
        if st.button("Execute System Diagnostic & Run Dispatch Pipeline"):
            with st.spinner("Processing geospatial layer coordinates and assembling SMS payload..."):
                
                 # Rule Evaluation Logic Loop               
                if target_deficit >= custom_threshold:
                    st.error(f"**SMS QUEUED (Priority: Critical) -> Route to {selected_name} Ext. Officers:**\n\n*'EMERGENCY ALERT: PCODE {target_pcode} structural deficit has reached {target_deficit:.1f}%, breaking your custom safety threshold of {custom_threshold}%. Discontinue regular planting guidelines. Immediately deploy regional economic cushions and emergency irrigation assistance.'*")
                elif target_whiplash > 250.0:
                    st.warning(f"**SMS QUEUED (Priority: High Volatility) -> Route to {selected_name} Cooperatives:**\n\n*'ALERT: High weather volatility index ({target_whiplash:.1f}) detected. Active seasonal whiplash and planting false-starts present. Advise farming pools to pause immediate planting cycles or swap into short-cycle adaptive grain seeds.'*")
                else:
                    st.success(f"**SMS QUEUED (Priority: Normal) -> Route to {selected_name} Field Hubs:**\n\n*'STATUS: PCODE {target_pcode} parameters are within safe bounds for this seasonal iteration. Proceed with standard baseline regional advice.'*")
            
            st.toast("Alert compilation process completed successfully!", icon="✅")