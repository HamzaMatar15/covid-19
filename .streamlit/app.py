import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 Global Analytics",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2rem; }
    .block-container { padding-top: 1.5rem; }
    [data-testid="stSidebar"] { background-color: #0e1117; }
    .section-title {
        font-size: 1.1rem; font-weight: 600;
        color: #e0e0e0; margin-bottom: 0.2rem;
    }
    .section-sub {
        font-size: 0.8rem; color: #888; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PASSWORD GATE
# ══════════════════════════════════════════════════════════════════════════════
APP_PASSWORD = "covid2026"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("## 🦠 COVID-19 Global Analytics")
        st.markdown("##### Healthcare Analytics Dashboard — MSBA382")
        st.markdown("---")
        pwd = st.text_input("Enter dashboard password", type="password", placeholder="Password")
        if st.button("Login", use_container_width=True):
            if pwd == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        st.markdown("<br><small style='color:#555'>Consultant tool — authorized access only</small>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("owid-covid-data.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Remove aggregate rows (continents / world summaries)
    exclude = ["World", "High income", "Upper middle income", "Lower middle income",
               "Low income", "European Union", "Africa", "Asia", "Europe",
               "North America", "Oceania", "South America", "International"]
    df = df[~df["location"].isin(exclude)]
    df = df[df["continent"].notna()]

    # Case fatality rate per country (latest snapshot)
    latest = df.sort_values("date").groupby("location").last().reset_index()
    latest["cfr"] = (latest["total_deaths"] / latest["total_cases"] * 100).round(2)

    return df, latest

df, latest = load_data()

CONTINENTS = sorted(df["continent"].dropna().unique().tolist())
MIN_DATE = df["date"].min().date()
MAX_DATE = df["date"].max().date()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🦠 COVID-19")
    st.markdown("#### Global Analytics Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview", "Geographic Map", "Age & Risk Factors",
         "Vaccination Analysis", "Predictive Model"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Global Filters**")

    selected_continents = st.multiselect(
        "Continents",
        CONTINENTS,
        default=CONTINENTS
    )

    date_range = st.date_input(
        "Date range",
        value=(MIN_DATE, MAX_DATE),
        min_value=MIN_DATE,
        max_value=MAX_DATE
    )

    st.markdown("---")
    st.markdown("<small style='color:#555'>Data: Our World in Data<br>MSBA382 — Healthcare Analytics<br>AUB Suliman S. Olayan School of Business</small>", unsafe_allow_html=True)

# ── Apply filters ──────────────────────────────────────────────────────────────
if len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(MIN_DATE), pd.Timestamp(MAX_DATE)

mask = (
    df["continent"].isin(selected_continents) &
    (df["date"] >= start_date) &
    (df["date"] <= end_date)
)
filtered_df = df[mask]
filtered_latest = latest[latest["continent"].isin(selected_continents)]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("## Overview")
    st.markdown("<p class='section-sub'>Global COVID-19 summary — cases, deaths, and trends over time</p>", unsafe_allow_html=True)

    # KPI Cards
    total_cases  = filtered_latest["total_cases"].sum()
    total_deaths = filtered_latest["total_deaths"].sum()
    total_vacc   = filtered_latest["people_vaccinated"].sum() if "people_vaccinated" in filtered_latest.columns else 0
    global_cfr   = (total_deaths / total_cases * 100) if total_cases > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Confirmed Cases",  f"{total_cases/1e6:.1f}M")
    k2.metric("Total Deaths",           f"{total_deaths/1e6:.2f}M")
    k3.metric("People Vaccinated",      f"{total_vacc/1e6:.1f}M")
    k4.metric("Case Fatality Rate",     f"{global_cfr:.2f}%")

    st.markdown("---")

    # Time series — new cases smoothed
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<p class='section-title'>Daily new cases over time (7-day smoothed)</p>", unsafe_allow_html=True)
        metric_choice = st.selectbox(
            "Metric",
            ["new_cases_smoothed", "new_deaths_smoothed"],
            format_func=lambda x: "New Cases (smoothed)" if x == "new_cases_smoothed" else "New Deaths (smoothed)"
        )
        ts = (
            filtered_df.groupby("date")[metric_choice]
            .sum().reset_index()
            .rename(columns={metric_choice: "value"})
        )
        fig_ts = px.area(
            ts, x="date", y="value",
            color_discrete_sequence=["#4a90d9"],
            labels={"value": "Count", "date": "Date"}
        )
        fig_ts.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified"
        )
        fig_ts.update_xaxes(showgrid=False)
        fig_ts.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig_ts, use_container_width=True)

    with col2:
        st.markdown("<p class='section-title'>Cases by continent</p>", unsafe_allow_html=True)
        cont_data = (
            filtered_latest.groupby("continent")["total_cases"]
            .sum().reset_index()
            .sort_values("total_cases", ascending=False)
        )
        fig_pie = px.pie(
            cont_data, names="continent", values="total_cases",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig_pie.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Top 10 countries
    st.markdown("<p class='section-title'>Top 10 countries by total cases</p>", unsafe_allow_html=True)
    top10_metric = st.selectbox("Rank by", ["total_cases", "total_deaths", "cfr"],
                                format_func=lambda x: {"total_cases":"Total Cases","total_deaths":"Total Deaths","cfr":"Case Fatality Rate (%)"}[x])
    top10 = filtered_latest.nlargest(10, top10_metric)[["location", "continent", top10_metric]].dropna()
    fig_bar = px.bar(
        top10.sort_values(top10_metric), x=top10_metric, y="location",
        orientation="h", color="continent",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={top10_metric: top10_metric.replace("_"," ").title(), "location": ""}
    )
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0), showlegend=True
    )
    fig_bar.update_xaxes(gridcolor="#333")
    fig_bar.update_yaxes(showgrid=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: GEOGRAPHIC MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Geographic Map":
    st.markdown("## Geographic Map")
    st.markdown("<p class='section-sub'>Country-level distribution of COVID-19 burden worldwide</p>", unsafe_allow_html=True)

    map_metric = st.selectbox(
        "Color map by",
        ["total_cases_per_million", "total_deaths_per_million", "cfr"],
        format_func=lambda x: {
            "total_cases_per_million": "Total Cases per Million",
            "total_deaths_per_million": "Total Deaths per Million",
            "cfr": "Case Fatality Rate (%)"
        }[x]
    )

    map_data = latest[["location", "iso_code", map_metric]].dropna()
    fig_map = px.choropleth(
        map_data,
        locations="iso_code",
        color=map_metric,
        hover_name="location",
        color_continuous_scale="Reds",
        labels={map_metric: map_metric.replace("_"," ").title()}
    )
    fig_map.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0),
        geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False,
                 showcoastlines=True, coastlinecolor="#333",
                 showland=True, landcolor="#1a1a2e",
                 showocean=True, oceancolor="#0d0d1a")
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<p class='section-title'>Top 15 worst-affected countries</p>", unsafe_allow_html=True)
        top15 = filtered_latest.nlargest(15, map_metric)[["location", map_metric]].dropna()
        fig_t15 = px.bar(
            top15.sort_values(map_metric), x=map_metric, y="location",
            orientation="h", color=map_metric,
            color_continuous_scale="Reds",
            labels={map_metric: map_metric.replace("_"," ").title(), "location": ""}
        )
        fig_t15.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0), showlegend=False
        )
        fig_t15.update_xaxes(gridcolor="#333")
        fig_t15.update_yaxes(showgrid=False)
        st.plotly_chart(fig_t15, use_container_width=True)

    with col2:
        st.markdown("<p class='section-title'>Continental summary</p>", unsafe_allow_html=True)
        cont_sum = (
            filtered_latest.groupby("continent")[["total_cases","total_deaths"]]
            .sum().reset_index()
        )
        cont_sum["cfr"] = (cont_sum["total_deaths"] / cont_sum["total_cases"] * 100).round(2)
        fig_cont = px.bar(
            cont_sum, x="continent", y="total_cases",
            color="continent", color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"total_cases": "Total Cases", "continent": ""}
        )
        fig_cont.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0), showlegend=False
        )
        fig_cont.update_xaxes(showgrid=False)
        fig_cont.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig_cont, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: AGE & RISK FACTORS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Age & Risk Factors":
    st.markdown("## Age & Risk Factors")
    st.markdown("<p class='section-sub'>How population age structure and pre-existing conditions relate to COVID-19 outcomes</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<p class='section-title'>Median age vs. case fatality rate</p>", unsafe_allow_html=True)
        scatter_data = filtered_latest[["location","continent","median_age","cfr","total_cases"]].dropna()
        fig_sc = px.scatter(
            scatter_data, x="median_age", y="cfr",
            color="continent", hover_name="location",
            size="total_cases", size_max=40,
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"median_age": "Median Age", "cfr": "Case Fatality Rate (%)"}
        )
        fig_sc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0)
        )
        fig_sc.update_xaxes(gridcolor="#333")
        fig_sc.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        st.markdown("<p class='section-title'>Population aged 65+ vs. death rate per million</p>", unsafe_allow_html=True)
        aged_data = filtered_latest[["location","continent","aged_65_older","total_deaths_per_million"]].dropna()
        fig_aged = px.scatter(
            aged_data, x="aged_65_older", y="total_deaths_per_million",
            color="continent", hover_name="location",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"aged_65_older": "% Population Aged 65+",
                    "total_deaths_per_million": "Deaths per Million"}
        )
        fig_aged.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0)
        )
        fig_aged.update_xaxes(gridcolor="#333")
        fig_aged.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig_aged, use_container_width=True)

    st.markdown("---")

    # Risk factor comparison
    st.markdown("<p class='section-title'>Risk factor comparison by continent</p>", unsafe_allow_html=True)
    risk_metric = st.selectbox(
        "Select risk factor",
        ["diabetes_prevalence", "cardiovasc_death_rate", "hospital_beds_per_thousand",
         "male_smokers", "female_smokers", "handwashing_facilities"],
        format_func=lambda x: {
            "diabetes_prevalence": "Diabetes Prevalence (%)",
            "cardiovasc_death_rate": "Cardiovascular Death Rate",
            "hospital_beds_per_thousand": "Hospital Beds per 1,000",
            "male_smokers": "Male Smokers (%)",
            "female_smokers": "Female Smokers (%)",
            "handwashing_facilities": "Handwashing Facilities (%)"
        }[x]
    )
    risk_data = (
        filtered_latest.groupby("continent")[risk_metric]
        .mean().reset_index().dropna()
        .sort_values(risk_metric, ascending=False)
    )
    fig_risk = px.bar(
        risk_data, x="continent", y=risk_metric,
        color="continent", color_discrete_sequence=px.colors.qualitative.Set2,
        labels={risk_metric: risk_metric.replace("_"," ").title(), "continent": ""}
    )
    fig_risk.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0), showlegend=False
    )
    fig_risk.update_xaxes(showgrid=False)
    fig_risk.update_yaxes(gridcolor="#333")
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")

    # Gender: male vs female smokers
    st.markdown("<p class='section-title'>Male vs. female smokers by continent</p>", unsafe_allow_html=True)
    smoke_data = (
        filtered_latest.groupby("continent")[["male_smokers","female_smokers"]]
        .mean().reset_index().dropna()
    )
    smoke_melt = smoke_data.melt(id_vars="continent", var_name="Gender", value_name="Smokers (%)")
    smoke_melt["Gender"] = smoke_melt["Gender"].map({"male_smokers":"Male","female_smokers":"Female"})
    fig_smoke = px.bar(
        smoke_melt, x="continent", y="Smokers (%)", color="Gender",
        barmode="group",
        color_discrete_map={"Male":"#4a90d9","Female":"#e07b9a"},
        labels={"continent": ""}
    )
    fig_smoke.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0)
    )
    fig_smoke.update_xaxes(showgrid=False)
    fig_smoke.update_yaxes(gridcolor="#333")
    st.plotly_chart(fig_smoke, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: VACCINATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Vaccination Analysis":
    st.markdown("## Vaccination Analysis")
    st.markdown("<p class='section-sub'>Vaccine rollout progress and its relationship with COVID-19 outcomes</p>", unsafe_allow_html=True)

    VACC_COL = "people_vaccinated_per_hundred"
    has_vacc = VACC_COL in filtered_df.columns and filtered_df[VACC_COL].notna().sum() > 0

    if not has_vacc:
        st.info(
            "Vaccination data is not available in this dataset. "
            "The OWID dataset you downloaded covers up to September 2020 — "
            "vaccines were not yet distributed at that time. "
            "The charts below use the available vaccination-related columns instead."
        )

        st.markdown("---")

        # Fallback: show total_tests as proxy for healthcare response
        st.markdown("<p class='section-title'>Testing effort over time (proxy for healthcare response)</p>", unsafe_allow_html=True)

        available_countries = sorted(
            filtered_df[filtered_df["new_tests_smoothed"].notna()]["location"].unique()
        )
        default_countries = ["United States", "United Kingdom", "France", "Brazil", "India"]
        default_countries = [c for c in default_countries if c in available_countries][:5]

        selected_countries = st.multiselect(
            "Select countries to compare",
            available_countries,
            default=default_countries if default_countries else available_countries[:5]
        )

        if selected_countries:
            test_ts = filtered_df[
                filtered_df["location"].isin(selected_countries) &
                filtered_df["new_tests_smoothed"].notna()
            ][["date", "location", "new_tests_smoothed"]]

            fig_test = px.line(
                test_ts, x="date", y="new_tests_smoothed",
                color="location",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"new_tests_smoothed": "New Tests (7-day avg)", "date": "Date", "location": "Country"}
            )
            fig_test.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified"
            )
            fig_test.update_xaxes(showgrid=False)
            fig_test.update_yaxes(gridcolor="#333")
            st.plotly_chart(fig_test, use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<p class='section-title'>Tests per case by continent (positivity proxy)</p>", unsafe_allow_html=True)
            tpc = (
                filtered_latest.groupby("continent")["tests_per_case"]
                .mean().reset_index().dropna()
                .sort_values("tests_per_case", ascending=False)
            )
            fig_tpc = px.bar(
                tpc, x="continent", y="tests_per_case",
                color="continent",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"tests_per_case": "Tests per Case", "continent": ""}
            )
            fig_tpc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0), showlegend=False
            )
            fig_tpc.update_xaxes(showgrid=False)
            fig_tpc.update_yaxes(gridcolor="#333")
            st.plotly_chart(fig_tpc, use_container_width=True)

        with col2:
            st.markdown("<p class='section-title'>Positive rate by continent</p>", unsafe_allow_html=True)
            pr = (
                filtered_latest.groupby("continent")["positive_rate"]
                .mean().reset_index().dropna()
                .sort_values("positive_rate", ascending=False)
            )
            fig_pr = px.bar(
                pr, x="continent", y="positive_rate",
                color="continent",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"positive_rate": "Positive Rate (avg)", "continent": ""}
            )
            fig_pr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0), showlegend=False
            )
            fig_pr.update_xaxes(showgrid=False)
            fig_pr.update_yaxes(gridcolor="#333")
            st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("---")

        st.markdown("<p class='section-title'>Hospital beds per 1,000 people vs. death rate</p>", unsafe_allow_html=True)
        hb_data = filtered_latest[["location","continent","hospital_beds_per_thousand","total_deaths_per_million"]].dropna()
        fig_hb = px.scatter(
            hb_data, x="hospital_beds_per_thousand", y="total_deaths_per_million",
            color="continent", hover_name="location",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"hospital_beds_per_thousand": "Hospital Beds per 1,000",
                    "total_deaths_per_million": "Deaths per Million"}
        )
        fig_hb.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0)
        )
        fig_hb.update_xaxes(gridcolor="#333")
        fig_hb.update_yaxes(gridcolor="#333")
        st.plotly_chart(fig_hb, use_container_width=True)

    else:
        # Full vaccination analysis (for datasets that include vaccine data)
        available_countries = sorted(
            filtered_df[filtered_df[VACC_COL].notna()]["location"].unique()
        )
        default_countries = ["United States", "United Kingdom", "France", "Brazil", "India"]
        default_countries = [c for c in default_countries if c in available_countries][:5]

        selected_countries = st.multiselect(
            "Select countries to compare",
            available_countries,
            default=default_countries
        )

        if selected_countries:
            st.markdown("<p class='section-title'>Vaccination rollout over time</p>", unsafe_allow_html=True)
            vacc_ts = filtered_df[
                filtered_df["location"].isin(selected_countries) &
                filtered_df[VACC_COL].notna()
            ][["date", "location", VACC_COL]]

            fig_vacc = px.line(
                vacc_ts, x="date", y=VACC_COL,
                color="location",
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={VACC_COL: "People Vaccinated (%)", "date": "Date", "location": "Country"}
            )
            fig_vacc.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified"
            )
            fig_vacc.update_xaxes(showgrid=False)
            fig_vacc.update_yaxes(gridcolor="#333")
            st.plotly_chart(fig_vacc, use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<p class='section-title'>Vaccination rate vs. death rate per million</p>", unsafe_allow_html=True)
            vd_data = filtered_latest[
                ["location", "continent", VACC_COL, "total_deaths_per_million"]
            ].dropna()
            fig_vd = px.scatter(
                vd_data, x=VACC_COL, y="total_deaths_per_million",
                color="continent", hover_name="location",
                color_discrete_sequence=px.colors.qualitative.Set2,
                trendline="ols",
                labels={VACC_COL: "People Vaccinated (%)",
                        "total_deaths_per_million": "Deaths per Million"}
            )
            fig_vd.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0)
            )
            fig_vd.update_xaxes(gridcolor="#333")
            fig_vd.update_yaxes(gridcolor="#333")
            st.plotly_chart(fig_vd, use_container_width=True)

        with col2:
            st.markdown("<p class='section-title'>Top 10 most vaccinated countries</p>", unsafe_allow_html=True)
            top_vacc = (
                filtered_latest[["location", VACC_COL]]
                .dropna()
                .nlargest(10, VACC_COL)
            )
            fig_tv = px.bar(
                top_vacc.sort_values(VACC_COL),
                x=VACC_COL, y="location",
                orientation="h",
                color=VACC_COL,
                color_continuous_scale="Greens",
                labels={VACC_COL: "People Vaccinated (%)", "location": ""}
            )
            fig_tv.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False, coloraxis_showscale=False
            )
            fig_tv.update_xaxes(gridcolor="#333")
            fig_tv.update_yaxes(showgrid=False)
            st.plotly_chart(fig_tv, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: PREDICTIVE MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predictive Model":
    st.markdown("## Predictive Model")
    st.markdown("<p class='section-sub'>Linear regression to predict Case Fatality Rate (CFR) from country-level health indicators</p>", unsafe_allow_html=True)

    FEATURES = {
        "median_age": "Median Age",
        "aged_65_older": "% Population Aged 65+",
        "diabetes_prevalence": "Diabetes Prevalence (%)",
        "hospital_beds_per_thousand": "Hospital Beds per 1,000",
        "cardiovasc_death_rate": "Cardiovascular Death Rate",
        "human_development_index": "Human Development Index"
    }

    selected_features = st.multiselect(
        "Select predictor variables",
        list(FEATURES.keys()),
        default=list(FEATURES.keys())[:4],
        format_func=lambda x: FEATURES[x]
    )

    if len(selected_features) < 1:
        st.warning("Please select at least one predictor variable.")
    else:
        model_data = latest[selected_features + ["cfr", "location"]].dropna()

        if len(model_data) < 10:
            st.warning("Not enough data to train the model. Try adding more features or adjusting filters.")
        else:
            X = model_data[selected_features].values
            y = model_data["cfr"].values

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{r2:.3f}", help="How well the model explains variation in CFR (1.0 = perfect)")
            col2.metric("Countries used", str(len(model_data)))
            col3.metric("Predictors", str(len(selected_features)))

            st.markdown("---")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("<p class='section-title'>Predicted vs. actual CFR</p>", unsafe_allow_html=True)
                pred_df = pd.DataFrame({
                    "location": model_data["location"].values,
                    "Actual CFR (%)": y,
                    "Predicted CFR (%)": y_pred.round(3)
                })
                fig_pred = px.scatter(
                    pred_df, x="Actual CFR (%)", y="Predicted CFR (%)",
                    hover_name="location",
                    color_discrete_sequence=["#4a90d9"]
                )
                max_val = max(y.max(), y_pred.max()) * 1.1
                fig_pred.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                                   line=dict(color="#888", dash="dash"))
                fig_pred.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0)
                )
                fig_pred.update_xaxes(gridcolor="#333")
                fig_pred.update_yaxes(gridcolor="#333")
                st.plotly_chart(fig_pred, use_container_width=True)

            with col_b:
                st.markdown("<p class='section-title'>Feature importance (coefficients)</p>", unsafe_allow_html=True)
                coef_df = pd.DataFrame({
                    "Feature": [FEATURES[f] for f in selected_features],
                    "Coefficient": model.coef_
                }).sort_values("Coefficient")
                fig_coef = px.bar(
                    coef_df, x="Coefficient", y="Feature",
                    orientation="h",
                    color="Coefficient",
                    color_continuous_scale="RdBu",
                    labels={"Feature": ""}
                )
                fig_coef.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc", margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False
                )
                fig_coef.update_xaxes(gridcolor="#333")
                fig_coef.update_yaxes(showgrid=False)
                st.plotly_chart(fig_coef, use_container_width=True)

            st.markdown("---")

            # Interactive predictor
            st.markdown("<p class='section-title'>Live CFR predictor — adjust inputs to get a prediction</p>", unsafe_allow_html=True)
            st.markdown("<p class='section-sub'>Simulate a country's predicted case fatality rate based on its health indicators</p>", unsafe_allow_html=True)

            input_vals = []
            cols = st.columns(len(selected_features))
            for i, feat in enumerate(selected_features):
                col_min = float(model_data[feat].min())
                col_max = float(model_data[feat].max())
                col_mean = float(model_data[feat].mean())
                val = cols[i].slider(
                    FEATURES[feat],
                    min_value=round(col_min, 2),
                    max_value=round(col_max, 2),
                    value=round(col_mean, 2),
                    step=round((col_max - col_min) / 100, 3)
                )
                input_vals.append(val)

            predicted_cfr = model.predict([input_vals])[0]
            st.markdown(f"<br>", unsafe_allow_html=True)
            res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
            with res_col2:
                st.metric(
                    label="Predicted Case Fatality Rate",
                    value=f"{max(0, predicted_cfr):.2f}%",
                    help="Model prediction based on selected input values"
                )