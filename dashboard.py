import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Enhanced page config with custom theme
st.set_page_config(
    page_title="Belden Innovate X — AI Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #0066cc;
        --danger: #dc3545;
        --warning: #ffc107;
        --success: #28a745;
        --dark: #1a1a2e;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    .metric-critical { border-left-color: #dc3545; }
    .metric-warning { border-left-color: #ffc107; }
    .metric-success { border-left-color: #28a745; }
    
    /* Section headers */
    .section-header {
        color: #1a1a2e;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-critical {
        background: #dc354520;
        color: #dc3545;
    }
    
    .status-normal {
        background: #28a74520;
        color: #28a745;
    }
    
    /* Tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Loaders
# -----------------------------------------------------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return None

@st.cache_data
def load_text(path):
    try:
        with open(path, "r") as f:
            return f.read()
    except:
        return None

# Filenames
ANALYZED = "belden_network_logs_analyzed.csv"
HEALTH = "ai_health_report.csv"
SUMMARY = "analysis_summary.txt"
EXEC_SUMMARY = "executive_summary.txt"
ASSET_SUMMARY = "dashboard_asset_summary.csv"

df = load_csv(ANALYZED)
ai_report = load_csv(HEALTH)
analysis_summary = load_text(SUMMARY)
executive_summary = load_text(EXEC_SUMMARY)
asset_summary = load_csv(ASSET_SUMMARY)

# -----------------------------------------------------------
# SAMPLE FALLBACK
# -----------------------------------------------------------
if df is None:
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="H")
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Asset_ID": [f"ASSET_{i%40:03d}" for i in range(n)],
        "Asset_Model": np.random.choice(["Hirschmann-AX","EAGLE-40","BAT867","OpEdge-8D"], n),
        "Asset_Type": np.random.choice(["Switch","Firewall","WLAN_AP","Gateway"], n),
        "L1_Numeric_Val": np.random.randint(0,50,n),
        "L2_Error_Frames": np.random.randint(0,80,n),
        "MAC_Table_Count": np.random.randint(100,1200,n),
        "Link_Flaps": np.random.randint(0,8,n),
        "RSSI": np.random.randint(-95,-40,n),
        "SNR": np.random.randint(5,40,n),
        "DPI_Latency_ms": np.random.randint(0,200,n),
        "Protocol_Conversion_Time": np.random.randint(0,300,n),
        "L3_Latency_ms": np.random.randint(10,150,n),
        "L7_App_Response_ms": np.random.randint(20,350,n),
        "Env_Temp_C": np.random.normal(40,8,n),
        "RF_Noise_dBm": np.random.normal(-70,7,n)
    })
    df["Anomaly_Score"] = np.random.randn(n)
    df["AI_Risk_Assessment"] = np.where(df["Anomaly_Score"] < -1.3, "CRITICAL_RISK", "Stable")
    df["Severity"] = np.where(df["AI_Risk_Assessment"]=="CRITICAL_RISK","CRITICAL","NORMAL")
    df["Root_Cause"] = np.where(df["Severity"]=="CRITICAL","PHYSICAL: Heat-Induced Cable Failure","None")
    df["Confidence"] = np.where(df["Severity"]=="CRITICAL","HIGH (95%)","N/A")

if ai_report is None:
    ai_report = df[df["Severity"]=="CRITICAL"].copy()

if asset_summary is None:
    asset_summary = df.groupby("Asset_Model").agg(
        Issues=("Severity", lambda x: (x=="CRITICAL").sum()),
        Total=("Asset_ID","count")
    ).reset_index()
    asset_summary["Health_Percentage"] = 100*(1 - asset_summary["Issues"]/asset_summary["Total"])

# -----------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("---")

if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    min_ts, max_ts = df["Timestamp"].min(), df["Timestamp"].max()
    date_range = st.sidebar.date_input("📅 Date Range", (min_ts, max_ts))
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df = df[(df["Timestamp"] >= start) & (df["Timestamp"] < end)]

models = st.sidebar.multiselect(
    "🔧 Asset Models",
    sorted(df["Asset_Model"].unique()),
    sorted(df["Asset_Model"].unique())
)

atypes = st.sidebar.multiselect(
    "📊 Asset Types",
    sorted(df["Asset_Type"].unique()),
    sorted(df["Asset_Type"].unique())
)

sevs = st.sidebar.multiselect(
    "⚠ Severity Levels",
    sorted(df["Severity"].unique()),
    sorted(df["Severity"].unique())
)

df = df[df["Asset_Model"].isin(models)]
df = df[df["Asset_Type"].isin(atypes)]
df = df[df["Severity"].isin(sevs)]

# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🔮 Belden Innovate X</h1>
    <p>Advanced AI-Powered Network Health Dashboard</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# KPIs
# -----------------------------------------------------------
total_assets = df["Asset_ID"].nunique()
critical_count = (df["Severity"]=="CRITICAL").sum()
detection_rate = (critical_count/total_assets)*100 if total_assets>0 else 0

model_accuracy = None
if analysis_summary and "Model Accuracy" in analysis_summary:
    for line in analysis_summary.splitlines():
        if "Model Accuracy" in line:
            model_accuracy = line.split(":")[-1].strip()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_assets}</div>
        <div class="metric-label">Total Assets</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card metric-critical">
        <div class="metric-value">{critical_count}</div>
        <div class="metric-label">Critical Risks</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card metric-warning">
        <div class="metric-value">{detection_rate:.1f}%</div>
        <div class="metric-label">Detection Rate</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card metric-success">
        <div class="metric-value">{model_accuracy if model_accuracy else "N/A"}</div>
        <div class="metric-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------
# TABS
# -----------------------------------------------------------
tab_overview, tab_ts, tab_rc, tab_topo, tab_model, tab_exec = st.tabs([
    "📊 Overview",
    "📈 Time Series",
    "🔍 Root Causes",
    "🌐 Topology",
    "🤖 Model Insights",
    "📋 Executive"
])

# -----------------------------------------------------------
# TAB 1: Overview
# -----------------------------------------------------------
with tab_overview:
    st.markdown('<h2 class="section-header">System Overview</h2>', unsafe_allow_html=True)

    c1, c2 = st.columns([2,1])
    
    with c1:
        st.subheader("Issues per Asset Model")
        fig = px.bar(
            asset_summary.sort_values("Issues", ascending=False),
            x="Asset_Model",
            y="Issues",
            title="Issues by Asset Model",
            color="Issues",
            color_continuous_scale=["#28a745", "#ffc107", "#dc3545"]
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Severity Distribution")
        sev_counts = df["Severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity","Count"]
        fig2 = px.pie(
            sev_counts,
            names="Severity",
            values="Count",
            color="Severity",
            color_discrete_map={"CRITICAL": "#dc3545", "NORMAL": "#28a745"}
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Top 10 Root Causes")
        rc = df["Root_Cause"].value_counts().head(10).reset_index()
        rc.columns = ["Root_Cause","Count"]
        st.dataframe(rc, hide_index=True, use_container_width=True)

        st.subheader("Executive Snippet")
        if executive_summary:
            st.markdown(f'<div class="info-box">{executive_summary[:300]}...</div>', unsafe_allow_html=True)
        elif analysis_summary:
            snippet = "\n".join(analysis_summary.splitlines()[:5])
            st.markdown(f'<div class="info-box">{snippet}</div>', unsafe_allow_html=True)
        else:
            st.info("No executive summary available.")

# -----------------------------------------------------------
# TAB 2: Time Series
# -----------------------------------------------------------
with tab_ts:
    st.markdown('<h2 class="section-header">Time Series & Anomaly Detection</h2>', unsafe_allow_html=True)

    if "Timestamp" not in df.columns:
        st.warning("⚠ No Timestamp column available.")
    else:
        metric = st.selectbox("Select Metric", [
            "Env_Temp_C","L3_Latency_ms","Anomaly_Score",
            "L1_Numeric_Val","L2_Error_Frames","RF_Noise_dBm"
        ])

        df_sorted = df.sort_values("Timestamp")

        fig_ts = px.line(
            df_sorted,
            x="Timestamp",
            y=metric,
            color="Asset_Model",
            title=f"{metric} Over Time"
        )

        anomalies = df_sorted[df_sorted["Severity"]=="CRITICAL"]
        fig_ts.add_trace(go.Scatter(
            x=anomalies["Timestamp"],
            y=anomalies[metric],
            mode="markers",
            marker=dict(color="#dc3545", size=10, symbol="x"),
            name="🚨 Anomaly"
        ))

        fig_ts.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            font=dict(family="Inter, sans-serif")
        )

        st.plotly_chart(fig_ts, use_container_width=True)

# -----------------------------------------------------------
# TAB 3: Root Causes
# -----------------------------------------------------------
with tab_rc:
    st.markdown('<h2 class="section-header">Root Cause Analytics</h2>', unsafe_allow_html=True)

    rc = df["Root_Cause"].value_counts().reset_index()
    rc.columns = ["Root_Cause","Count"]
    
    fig = px.bar(
        rc,
        x="Root_Cause",
        y="Count",
        title="Distribution of Root Causes",
        color="Count",
        color_continuous_scale="Reds"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45,
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Additional insights
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Unique Root Causes", len(rc))
    with col2:
        st.metric("Most Common Issue", rc.iloc[0]["Root_Cause"] if not rc.empty else "N/A")

# -----------------------------------------------------------
# TAB 4: Topology
# -----------------------------------------------------------
with tab_topo:
    st.markdown('<h2 class="section-header">Network Topology & Dependencies</h2>', unsafe_allow_html=True)

    if "uplink" not in df.columns:
        st.info("ℹ No 'uplink' column found. Add topology data to enable graph view.")
    else:
        edges = df[["Asset_ID","uplink"]].dropna()
        edges = edges[edges["uplink"]!="Unknown"]

        G = nx.DiGraph()
        for _, row in edges.iterrows():
            G.add_edge(row["uplink"], row["Asset_ID"])

        pos = nx.spring_layout(G, seed=7, k=0.5)
        
        edge_x = []
        edge_y = []
        for e in G.edges():
            x0,y0 = pos[e[0]]
            x1,y1 = pos[e[1]]
            edge_x += [x0,x1,None]
            edge_y += [y0,y1,None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(color="#ddd", width=2),
            hoverinfo='none'
        )

        node_x = []
        node_y = []
        node_color = []
        node_text = []
        labels = []

        for n in G.nodes():
            x,y = pos[n]
            node_x.append(x)
            node_y.append(y)
            labels.append(n)

            nd = df[df["Asset_ID"]==n]
            if not nd.empty and (nd["Severity"]=="CRITICAL").any():
                node_color.append("#dc3545")
                node_text.append(f"{n}<br>Status: CRITICAL")
            else:
                node_color.append("#28a745")
                node_text.append(f"{n}<br>Status: Normal")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                size=16,
                color=node_color,
                line=dict(width=2, color="white")
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="Asset Dependency Graph",
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# TAB 5: Model Insights
# -----------------------------------------------------------
with tab_model:
    st.markdown('<h2 class="section-header">AI Model Performance & Correlations</h2>', unsafe_allow_html=True)

    df2 = df.copy()
    df2["is_anomaly"] = (df2["Severity"]=="CRITICAL").astype(int)

    numeric = df2.select_dtypes(include=[np.number]).columns.tolist()
    if "is_anomaly" in numeric:
        numeric.remove("is_anomaly")

    corr = df2[numeric + ["is_anomaly"]].corr()["is_anomaly"].drop("is_anomaly")
    corr = corr.abs().sort_values(ascending=False)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Top Predictive Features")
        corr_df = corr.head(15).reset_index()
        corr_df.columns = ["Feature", "Correlation"]
        corr_df["Correlation"] = corr_df["Correlation"].round(3)
        st.dataframe(corr_df, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Feature Correlation Matrix")
        fig = px.imshow(
            df2[numeric].corr(),
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=9)
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# TAB 6: Executive
# -----------------------------------------------------------
with tab_exec:
    st.markdown('<h2 class="section-header">Executive Summary & Reports</h2>', unsafe_allow_html=True)

    if executive_summary:
        st.markdown(f'<div class="info-box"><pre>{executive_summary}</pre></div>', unsafe_allow_html=True)
    elif analysis_summary:
        st.markdown(f'<div class="info-box"><pre>{analysis_summary}</pre></div>', unsafe_allow_html=True)
    else:
        st.info("ℹ No executive summary found.")

    st.markdown("---")
    
    st.subheader("Critical Assets Report")
    st.dataframe(
        ai_report.head(200),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📥 Download AI Health Report",
            data=ai_report.to_csv(index=False),
            file_name="ai_health_report.csv",
            mime="text/csv"
        )

    with col2:
        st.download_button(
            "📥 Download Full Dataset",
            data=df.to_csv(index=False),
            file_name="belden_network_logs_analyzed.csv",
            mime="text/csv"
        )

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem 0;">
    <p style="margin: 0;">💡 Ensure pipeline outputs are in the same folder as dashboard.py</p>
    <p style="margin: 0; font-size: 0.9rem;">Run using: <code>streamlit run dashboard.py</code></p>
</div>
""", unsafe_allow_html=True)
