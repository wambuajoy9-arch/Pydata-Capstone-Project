import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def load_and_process_data():
    # 1. Load the climate data using a raw string to protect Windows backslashes
    data = pd.read_csv("DATA/Kenya_Rainfall data.csv") 
    
    data.columns = data.columns.str.strip()
    data = data.rename(columns={'month':'Month','year':'Year'})
    
    # Standardize data PCODE column
    data["PCODE"] = data["PCODE"].astype(str).str.strip().str.upper()
    
    # Isolate critical March-May planting season (2022-2026)
    planting_season = data[data["Month"].between(3, 5) & data["Year"].between(2022, 2026)].copy()
    planting_season["seasonal_anomaly_score"] = planting_season["current_rainfall"] - planting_season["historical_rainfall_avg"]
    
    # Cumulative Deficit Percentages
    PCODE_total = planting_season.groupby("PCODE").agg(
        Total_Actual=("current_rainfall", "sum"),
        Total_Expected=("historical_rainfall_avg", "sum")
    ).reset_index()
    PCODE_total["Deficit_mm"] = (PCODE_total["Total_Expected"] - PCODE_total["Total_Actual"]).clip(lower=0)
    PCODE_total["Deficit_Percentage"] = (PCODE_total["Deficit_mm"] / PCODE_total["Total_Expected"]) * 100
    
    # Climate Whiplash Standard Deviation
    PCODE_anomaly = planting_season.groupby(["PCODE", "Year"])["seasonal_anomaly_score"].sum().reset_index()
    PCODE_volatility = PCODE_anomaly.groupby("PCODE")["seasonal_anomaly_score"].std().reset_index()
    PCODE_volatility.columns = ["PCODE", "Whiplash_Score"]
    
    # Load your shapefile boundary map
    spatial_map = gpd.read_file("DATA/ken_admin2.geojson")
    cleaned_spatial_map = spatial_map.dropna(axis=1, how="all")
    
    # Clean the PCODE columns to ensure a perfect string match
    cleaned_spatial_map['adm2_pcode'] = (cleaned_spatial_map['adm2_pcode'].astype(str).str.strip().str.upper())
    
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
    
    return merged_map

# Execute data engine
try:
    merged_map = load_and_process_data()
except Exception as e:
    st.error(f"Configuration Error: Check file paths or file structure. Details: {e}")
    st.stop()

# MAIN INTERFACE
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

# Solution Tabs
tab1, tab2 = st.tabs(["Solution 1: Interactive Risk Analytics Engine", "Solution 2: Last-Mile Alert Dispatcher"])

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