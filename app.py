#code_content = '''
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import plotly.express as px

# ---------------------------------------------------------
# 1. PAGE CONFIG & PATH SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

# ---------------------------------------------------------
# 2. LOAD CHAMPION MODELS & DATASET
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    clf_joblib = os.path.join(BASE_DIR, "champion_classifier.joblib")
    reg_joblib = os.path.join(BASE_DIR, "champion_regressor.joblib")

    if os.path.exists(clf_joblib) and os.path.exists(reg_joblib):
        return joblib.load(clf_joblib), joblib.load(reg_joblib)

    client = MlflowClient()
    clf_ver = client.get_model_version_by_alias("RealEstate_GoodInvestment_Classifier", "champion")
    reg_ver = client.get_model_version_by_alias("RealEstate_FuturePrice5Y_Regressor", "champion")

    def resolve_model_path(model_ver):
        src = model_ver.source.replace("file:///", "").replace("file://", "")
        if os.path.exists(os.path.join(src, "MLmodel")):
            return src
        mlruns_dir = os.path.join(BASE_DIR, "mlruns")
        target_id = getattr(model_ver, "model_id", None) or model_ver.run_id
        for root, dirs, files in os.walk(mlruns_dir):
            if "MLmodel" in files and (target_id in root or model_ver.run_id in root):
                return root
        return f"models:/{model_ver.name}/{model_ver.version}"

    clf = mlflow.sklearn.load_model(resolve_model_path(clf_ver))
    reg = mlflow.sklearn.load_model(resolve_model_path(reg_ver))
    return clf, reg


@st.cache_data
def load_data():
    candidate_files = [
        "india_housing_prices.csv",
        "Cleaned_dataset.csv",
        "engineered_data.csv",
        "ML_file.csv"
    ]
    df = None
    for fname in candidate_files:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            break

    if df is None:
        return None

    for col in ["Parking_Space", "Security"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    cat_cols = [
        "State", "City", "Locality", "Property_Type", "Furnished_Status",
        "Facing", "Owner_Type", "Availability_Status", "Amenities",
        "Public_Transport_Accessibility"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown" if col != "Amenities" else "None").astype(str)

    return df


try:
    clf_model, reg_model = load_models()
    df = load_data()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Ensure 'champion_classifier.joblib' and 'champion_regressor.joblib' are in the same folder as 'app.py'.")
    st.stop()

if df is None:
    st.warning("Dataset CSV not found automatically. Please upload your housing dataset CSV below.")
    uploaded_file = st.file_uploader("Upload Housing CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR: DYNAMIC CASCADING LOCATION FILTERS + FORM
# ---------------------------------------------------------
st.sidebar.header("📍 Location Selection")
st.sidebar.caption("Select State first to dynamically filter City and Locality.")

state_options = sorted(df["State"].dropna().unique())
state = st.sidebar.selectbox("State", state_options, key="selected_state")

city_options = sorted(df[df["State"] == state]["City"].dropna().unique())
city = st.sidebar.selectbox("City", city_options, key="selected_city")

locality_options = sorted(df[(df["State"] == state) & (df["City"] == city)]["Locality"].dropna().unique())
locality = st.sidebar.selectbox("Locality", locality_options, key="selected_locality")

st.sidebar.divider()
st.sidebar.header("🏡 Property Specifications")

with st.sidebar.form("prediction_form"):
    prop_type = st.selectbox("Property Type", sorted(df["Property_Type"].dropna().unique()))
    bhk = st.slider("BHK", min_value=1, max_value=10, value=3)
    size_sqft = st.number_input("Size (in SqFt)", min_value=200, max_value=20000, value=1400, step=50)
    current_price = st.number_input("Current Asking Price (in Lakhs)", min_value=5.0, max_value=5000.0, value=120.0, step=5.0)

    col_a, col_b = st.columns(2)
    with col_a:
        year_built = st.number_input("Year Built", min_value=1970, max_value=2026, value=2018)
        floor_no = st.number_input("Floor No", min_value=0, max_value=80, value=4)
        nearby_schools = st.slider("Nearby Schools", 0, 15, 4)
        parking = st.selectbox("Parking Space", ["Yes", "No"])
    with col_b:
        age = 2026 - year_built
        st.metric("Calculated Age", f"{age} yrs")
        total_floors = st.number_input("Total Floors", min_value=1, max_value=80, value=12)
        nearby_hospitals = st.slider("Nearby Hospitals", 0, 15, 3)
        security = st.selectbox("24/7 Security", ["Yes", "No"])

    furnished = st.selectbox("Furnished Status", sorted(df["Furnished_Status"].dropna().unique()))
    transport = st.selectbox("Public Transport Access", sorted(df["Public_Transport_Accessibility"].dropna().unique()))
    facing = st.selectbox("Facing", sorted(df["Facing"].dropna().unique()))
    owner = st.selectbox("Owner Type", sorted(df["Owner_Type"].dropna().unique()))
    availability = st.selectbox("Availability Status", sorted(df["Availability_Status"].dropna().unique()))
    amenities = st.selectbox("Primary Amenities", sorted(df["Amenities"].dropna().unique()))

    submit_btn = st.form_submit_button("🔍 Analyze Investment", use_container_width=True)

# ---------------------------------------------------------
# 4. MAIN DASHBOARD HEADER & TABS
# ---------------------------------------------------------
st.title("🏢 Real Estate Investment Advisor")
st.markdown("Evaluate property investment potential, forecast 5-year appreciation, and explore local market trends powered by **MLflow**.")

tab1, tab2, tab3 = st.tabs([
    "🎯 Investment Prediction & Explainability",
    "🔎 Property Explorer & Filters",
    "📊 Location Heatmaps & Market Trends"
])

# ---------------------------------------------------------
# TAB 1: PREDICTION, CONFIDENCE & FEATURE IMPORTANCE
# ---------------------------------------------------------
with tab1:
    raw_input = {
        "Unnamed: 0": 0,
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": prop_type,
        "BHK": float(bhk),
        "Size_in_SqFt": float(size_sqft),
        "Price_in_Lakhs": float(current_price),
        "Price_per_SqFt": round((float(current_price) * 100000) / float(size_sqft), 2),
        "Year_Built": float(year_built),
        "Furnished_Status": furnished,
        "Floor_No": float(floor_no),
        "Total_Floors": float(total_floors),
        "Age_of_Property": float(age),
        "Nearby_Schools": float(nearby_schools),
        "Nearby_Hospitals": float(nearby_hospitals),
        "Public_Transport_Accessibility": transport,
        "Parking_Space": 1.0 if parking == "Yes" else 0.0,
        "Security": 1.0 if security == "Yes" else 0.0,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner,
        "Availability_Status": availability,
    }
    input_df = pd.DataFrame([raw_input])

    clf_features = list(clf_model.feature_names_in_)
    reg_features = list(reg_model.feature_names_in_)
    for col in set(clf_features + reg_features):
        if col not in input_df.columns:
            input_df[col] = 0

    clf_pred = int(clf_model.predict(input_df[clf_features])[0])
    reg_pred = float(reg_model.predict(input_df[reg_features])[0])


    clf_pred = int(clf_model.predict(input_df[clf_features])[0])
    reg_pred = float(reg_model.predict(input_df[reg_features])[0])


    if hasattr(clf_model, "predict_proba"):
        probs = clf_model.predict_proba(input_df[clf_features])[0]
        raw_prob = float(probs[1])
    else:
        raw_prob = 0.90 if clf_pred == 1 else 0.15

    city_data = df[df["City"] == city]
    if not city_data.empty:
        city_med_ppsqft = ((city_data["Price_in_Lakhs"] * 100000) / city_data["Size_in_SqFt"]).median()
    else:
        city_med_ppsqft = ((df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"]).median()
    user_ppsqft = float(input_df["Price_per_SqFt"].iloc[0])

    price_ratio = city_med_ppsqft / max(user_ppsqft, 1.0)
    price_score = float(np.clip(price_ratio * 0.65, 0.1, 1.0))

    transit_map = {"High": 1.0, "Medium": 0.65, "Low": 0.3}
    transit_score = transit_map.get(transport, 0.5)
    infra_score = min((nearby_schools + nearby_hospitals) / 16.0, 1.0)

    amenity_score = (
        (0.3 if parking == "Yes" else 0.0)
        + (0.3 if security == "Yes" else 0.0)
        + (0.2 if age <= 10 else 0.05)
        + (0.2 if availability == "Ready to Move" else 0.05)
    )

    blended_score = (
        0.40 * raw_prob
        + 0.25 * price_score
        + 0.15 * transit_score
        + 0.10 * infra_score
        + 0.10 * amenity_score
    ) * 100

    confidence = blended_score if clf_pred == 1 else (100.0 - blended_score)
    confidence = float(np.clip(confidence, 52.0, 96.8))

    price_diff = reg_pred - current_price
    pct_gain = (price_diff / current_price) * 100

    st.subheader(f"📋 Investment Verdict for {locality}, {city} ({state})")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        if clf_pred == 1:
            st.success("✅ **GOOD INVESTMENT**")
        else:
            st.error("⚠️ **NOT RECOMMENDED**")
        st.caption("Classification Verdict")

    with m2:
        st.metric("Investment Confidence", f"{confidence:.1f}%", delta=f"City Med: ₹{city_med_ppsqft:,.0f}/SqFt")
        st.caption("Model + Market Benchmark Score")

    with m3:
        st.metric(
            "Estimated Price (5 Years)",
            f"₹ {reg_pred:,.2f} L",
            delta=f"+₹ {price_diff:,.2f} L"
        )
        st.caption("Regression Forecast")

    with m4:
        st.metric("Projected 5Y Growth", f"{pct_gain:+.1f}%")
        st.caption(f"Base Price: ₹ {current_price:,.2f} L")

    st.divider()

    st.subheader("🧠 Model Explainability: Top Feature Importances")
    fi_col1, fi_col2 = st.columns(2)

    def plot_pipeline_importance(pipeline, title, color):
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        clean_names = [name.split("__")[-1] for name in feature_names]

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_).ravel()
        else:
            return None

        fi_df = (
            pd.DataFrame({"Feature": clean_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .head(10)
            .sort_values("Importance", ascending=True)
        )
        fig = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=title,
            color_discrete_sequence=[color],
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    with fi_col1:
        fig_clf = plot_pipeline_importance(clf_model, "Top Drivers: Good Investment Classifier", "#2E86C1")
        if fig_clf:
            st.plotly_chart(fig_clf, use_container_width=True)
        else:
            st.info("Feature importance chart not supported for this classifier type.")

    with fi_col2:
        fig_reg = plot_pipeline_importance(reg_model, "Top Drivers: 5-Year Price Regressor", "#28B463")
        if fig_reg:
            st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.info("Feature importance chart not supported for this regressor type.")

# ---------------------------------------------------------
# TAB 2: FILTER EXISTING PROPERTIES BY AREA, PRICE, BHK
# ---------------------------------------------------------
with tab2:
    st.subheader("🔎 Filter & Compare Market Listings")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        filter_cities = st.multiselect("Filter by City", sorted(df["City"].unique()), default=[city])
    with f2:
        filter_bhk = st.multiselect("Filter by BHK", sorted(df["BHK"].unique()), default=[bhk])
    with f3:
        min_p, max_p = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
        price_range = st.slider("Price Range (Lakhs)", min_p, max_p, (min_p, min(max_p, 500.0)))
    with f4:
        min_s, max_s = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())
        size_range = st.slider("Area Range (SqFt)", min_s, max_s, (min_s, min(max_s, 5000)))

    filtered_df = df[
        (df["City"].isin(filter_cities if filter_cities else df["City"].unique()))
        & (df["BHK"].isin(filter_bhk if filter_bhk else df["BHK"].unique()))
        & (df["Price_in_Lakhs"].between(price_range[0], price_range[1]))
        & (df["Size_in_SqFt"].between(size_range[0], size_range[1]))
    ]

    st.write(f"Showing **{len(filtered_df):,}** matching properties out of **{len(df):,}** total records.")
    display_cols = [
        "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
        "Price_in_Lakhs", "Price_per_SqFt", "Furnished_Status",
        "Public_Transport_Accessibility", "Availability_Status"
    ]
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(filtered_df[display_cols].head(500), use_container_width=True)

# ---------------------------------------------------------
# TAB 3: VISUAL INSIGHTS (HEATMAPS & TREND CHARTS)
# ---------------------------------------------------------
with tab3:
    st.subheader("📊 Location Heatmaps & Market Trends")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        top_cities = df["City"].value_counts().head(12).index
        heatmap_data = (
            df[df["City"].isin(top_cities)]
            .pivot_table(
                index="City",
                columns="Property_Type",
                values="Price_per_SqFt",
                aggfunc="mean"
            )
            .round(0)
        )
        fig_heat = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Location Heatmap: Avg Price/SqFt (Top 12 Cities vs. Property Type)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with chart_col2:
        trend_df = (
            df.groupby("Year_Built", as_index=False)["Price_in_Lakhs"]
            .mean()
            .sort_values("Year_Built")
        )
        fig_trend = px.line(
            trend_df,
            x="Year_Built",
            y="Price_in_Lakhs",
            markers=True,
            title="Price Trend by Construction Year (Avg Price in Lakhs)"
        )
        fig_trend.update_traces(line_color="#1F618D", line_width=2.5)
        st.plotly_chart(fig_trend, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        city_df = df[df["City"] == city]
        top_localities = (
            city_df.groupby("Locality", as_index=False)["Price_in_Lakhs"]
            .median()
            .sort_values("Price_in_Lakhs", ascending=False)
            .head(10)
        )
        fig_loc = px.bar(
            top_localities,
            x="Locality",
            y="Price_in_Lakhs",
            color="Price_in_Lakhs",
            color_continuous_scale="Blues",
            title=f"Top 10 Localities in {city} by Median Price (Lakhs)"
        )
        st.plotly_chart(fig_loc, use_container_width=True)

    with chart_col4:
        bhk_trend = (
            city_df.groupby("BHK", as_index=False)["Price_in_Lakhs"]
            .median()
            .sort_values("BHK")
        )
        fig_bhk = px.area(
            bhk_trend,
            x="BHK",
            y="Price_in_Lakhs",
            markers=True,
            title=f"Median Price Scaling by BHK in {city}"
        )
        st.plotly_chart(fig_bhk, use_container_width=True)


#with open("app.py", "w", encoding="utf-8") as f:
#    f.write(code_content)

#print("app.py written successfully, length:", len(code_content))
