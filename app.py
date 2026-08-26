import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# =========================================================
# CPCB AQI SUB-INDEX CALCULATION
# =========================================================

def calculate_sub_index(concentration, breakpoints):

    for (
        low_concentration,
        high_concentration,
        low_aqi,
        high_aqi
    ) in breakpoints:

        if low_concentration <= concentration <= high_concentration:

            sub_index = (
                (high_aqi - low_aqi)
                / (high_concentration - low_concentration)
            ) * (
                concentration - low_concentration
            ) + low_aqi

            return sub_index

    return None


# =========================================================
# AQI BREAKPOINTS
# =========================================================

PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (31, 60, 51, 100),
    (61, 90, 101, 200),
    (91, 120, 201, 300),
    (121, 250, 301, 400),
    (251, 380, 401, 500)
]

PM10_BREAKPOINTS = [
    (0, 50, 0, 50),
    (51, 100, 51, 100),
    (101, 250, 101, 200),
    (251, 350, 201, 300),
    (351, 430, 301, 400),
    (431, 510, 401, 500)
]

NO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 180, 101, 200),
    (181, 280, 201, 300),
    (281, 400, 301, 400),
    (401, 520, 401, 500)
]

SO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 380, 101, 200),
    (381, 800, 201, 300),
    (801, 1600, 301, 400),
    (1601, 2620, 401, 500)
]

O3_BREAKPOINTS = [
    (0, 50, 0, 50),
    (51, 100, 51, 100),
    (101, 168, 101, 200),
    (169, 208, 201, 300),
    (209, 748, 301, 400),
    (749, 1000, 401, 500)
]

NH3_BREAKPOINTS = [
    (0, 200, 0, 50),
    (201, 400, 51, 100),
    (401, 800, 101, 200),
    (801, 1200, 201, 300),
    (1201, 1800, 301, 400),
    (1801, 2400, 401, 500)
]


# =========================================================
# INDIAN AQI CALCULATION
# =========================================================

def calculate_indian_aqi(row):

    pollutants = {

        "PM2.5": (
            row["pm2_5"],
            PM25_BREAKPOINTS
        ),

        "PM10": (
            row["pm10"],
            PM10_BREAKPOINTS
        ),

        "NO2": (
            row["nitrogen_dioxide"],
            NO2_BREAKPOINTS
        ),

        "SO2": (
            row["sulphur_dioxide"],
            SO2_BREAKPOINTS
        ),

        "O3": (
            row["ozone"],
            O3_BREAKPOINTS
        ),

        "NH3": (
            row["ammonia"],
            NH3_BREAKPOINTS
        )
    }

    sub_indices = {}

    for pollutant, (
        concentration,
        breakpoints
    ) in pollutants.items():

        if pd.notna(concentration):

            sub_index = calculate_sub_index(
                concentration,
                breakpoints
            )

            if sub_index is not None:
                sub_indices[pollutant] = sub_index

    if not sub_indices:
        return None, None

    dominant_pollutant = max(
        sub_indices,
        key=sub_indices.get
    )

    aqi = round(
        sub_indices[dominant_pollutant]
    )

    return aqi, dominant_pollutant


# =========================================================
# CALCULATE AQI FOR COMPLETE DATAFRAME
# =========================================================

def calculate_aqi_for_dataframe(df):

    aqi_values = []
    dominant_pollutants = []

    for _, row in df.iterrows():

        aqi, dominant = calculate_indian_aqi(row)

        aqi_values.append(aqi)

        dominant_pollutants.append(dominant)

    df = df.copy()

    df["AQI"] = aqi_values

    df["Dominant Pollutant"] = dominant_pollutants

    return df


# =========================================================
# AQI CATEGORY
# =========================================================

def get_aqi_category(aqi):

    if pd.isna(aqi):
        return "Unavailable"

    elif aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderately Polluted"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# =========================================================
# AQI HEALTH INTERPRETATION
# =========================================================

def get_health_message(category):

    messages = {

        "Good":
            "Air quality is considered good.",

        "Satisfactory":
            "Air quality is generally acceptable.",

        "Moderately Polluted":
            "Sensitive individuals may experience discomfort.",

        "Poor":
            "People may experience breathing discomfort.",

        "Very Poor":
            "Health effects may occur with prolonged exposure.",

        "Severe":
            "Air quality may pose serious health concerns.",

        "Unavailable":
            "AQI information is currently unavailable."
    }

    return messages.get(
        category,
        "No information available."
    )


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

def train_pm25_models(df):

    model_df = df[
        [
            "time",
            "pm2_5",
            "pm10",
            "nitrogen_dioxide"
        ]
    ].copy()

    # -----------------------------------------------------
    # TIME FEATURES
    # -----------------------------------------------------

    model_df["hour"] = (
        model_df["time"].dt.hour
    )

    model_df["day_of_week"] = (
        model_df["time"].dt.dayofweek
    )

    # -----------------------------------------------------
    # PREVIOUS HOUR FEATURES
    # -----------------------------------------------------

    model_df["previous_pm25"] = (
        model_df["pm2_5"].shift(1)
    )

    model_df["previous_pm10"] = (
        model_df["pm10"].shift(1)
    )

    model_df["previous_no2"] = (
        model_df["nitrogen_dioxide"].shift(1)
    )

    # Remove missing values

    model_df = model_df.dropna()

    if len(model_df) < 10:
        return None

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    features = [

        "hour",

        "day_of_week",

        "previous_pm25",

        "previous_pm10",

        "previous_no2"
    ]

    X = model_df[features]

    y = model_df["pm2_5"]

    # -----------------------------------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------------------------------

    split_index = int(
        len(model_df) * 0.8
    )

    X_train = X.iloc[:split_index]

    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]

    y_test = y.iloc[split_index:]

    # =====================================================
    # LINEAR REGRESSION
    # =====================================================

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    linear_predictions = (
        linear_model.predict(X_test)
    )

    linear_mae = mean_absolute_error(
        y_test,
        linear_predictions
    )

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    random_forest_model = RandomForestRegressor(

        n_estimators=100,

        random_state=42
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    rf_predictions = (
        random_forest_model.predict(X_test)
    )

    rf_mae = mean_absolute_error(
        y_test,
        rf_predictions
    )

    # =====================================================
    # SELECT BEST MODEL
    # =====================================================

    if rf_mae < linear_mae:

        best_model = random_forest_model

        best_model_name = "Random Forest"

        best_mae = rf_mae

        best_predictions = rf_predictions

    else:

        best_model = linear_model

        best_model_name = "Linear Regression"

        best_mae = linear_mae

        best_predictions = linear_predictions

    # =====================================================
    # NEXT-HOUR PREDICTION
    # =====================================================

    last_row = model_df.iloc[-1]

    next_time = (
        last_row["time"]
        + pd.Timedelta(hours=1)
    )

    next_features = pd.DataFrame({

        "hour": [
            next_time.hour
        ],

        "day_of_week": [
            next_time.dayofweek
        ],

        "previous_pm25": [
            last_row["pm2_5"]
        ],

        "previous_pm10": [
            last_row["pm10"]
        ],

        "previous_no2": [
            last_row["nitrogen_dioxide"]
        ]
    })

    next_prediction = (
        best_model.predict(
            next_features
        )[0]
    )

    # PM2.5 cannot be negative

    next_prediction = max(
        0,
        next_prediction
    )

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    comparison_df = pd.DataFrame({

        "Model": [

            "Linear Regression",

            "Random Forest"
        ],

        "MAE": [

            linear_mae,

            rf_mae
        ]
    })

    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================

    prediction_df = pd.DataFrame({

        "Actual PM2.5":
            y_test.values,

        "Predicted PM2.5":
            best_predictions
    })

    # =====================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # =====================================================

    feature_importance_df = None

    if best_model_name == "Random Forest":

        feature_importance_df = pd.DataFrame({

            "Feature": features,

            "Importance":
                random_forest_model.feature_importances_
        })

        feature_importance_df = (
            feature_importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
        )

    return {

        "linear_mae":
            linear_mae,

        "rf_mae":
            rf_mae,

        "comparison_df":
            comparison_df,

        "best_model_name":
            best_model_name,

        "best_mae":
            best_mae,

        "next_prediction":
            next_prediction,

        "prediction_df":
            prediction_df,

        "feature_importance_df":
            feature_importance_df
    }


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(

    page_title="Air Quality Visualizer",

    page_icon="🌍",

    layout="wide"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🌍 Air Quality Visualizer"
)

st.sidebar.write(
    "An air-quality monitoring and "
    "PM2.5 prediction prototype."
)

st.sidebar.markdown(
    """
### Technologies

- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn
- REST API

### ML Models

- Linear Regression
- Random Forest Regression

### Data Source

Open-Meteo Air Quality API
"""
)


# =========================================================
# MAIN TITLE
# =========================================================

st.title(
    "🌍 Air Quality Visualizer"
)

st.write(
    "Real-time air quality monitoring "
    "using live API data with Indian "
    "AQI calculation and machine "
    "learning-based PM2.5 prediction."
)


# =========================================================
# CITY COORDINATES
# =========================================================

cities = {

    "Durgapur": {

        "latitude": 23.5204,

        "longitude": 87.3119
    },

    "Kolkata": {

        "latitude": 22.5726,

        "longitude": 88.3639
    },

    "Delhi": {

        "latitude": 28.6139,

        "longitude": 77.2090
    },

    "Mumbai": {

        "latitude": 19.0760,

        "longitude": 72.8777
    },

    "Bengaluru": {

        "latitude": 12.9716,

        "longitude": 77.5946
    }
}


# =========================================================
# LOCATION SELECTION
# =========================================================

selected_city = st.selectbox(

    "📍 Select a city",

    list(cities.keys())
)

latitude = cities[
    selected_city
]["latitude"]

longitude = cities[
    selected_city
]["longitude"]

st.caption(
    f"Coordinates: {latitude}, {longitude}"
)


# =========================================================
# API REQUEST
# =========================================================

url = (
    "https://air-quality-api.open-meteo.com/"
    "v1/air-quality"
)

params = {

    "latitude": latitude,

    "longitude": longitude,

    "hourly": [

        "pm2_5",

        "pm10",

        "nitrogen_dioxide",

        "sulphur_dioxide",

        "ozone",

        "carbon_monoxide",

        "ammonia"
    ],

    "forecast_days": 2
}


# =========================================================
# GET DATA
# =========================================================

try:

    response = requests.get(

        url,

        params=params,

        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    # -----------------------------------------------------
    # DATAFRAME
    # -----------------------------------------------------

    df = pd.DataFrame(
        data["hourly"]
    )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    # -----------------------------------------------------
    # CALCULATE AQI
    # -----------------------------------------------------

    df = calculate_aqi_for_dataframe(
        df
    )

    st.success(
        "Live air-quality data loaded successfully!"
    )


    # =====================================================
    # CURRENT VALUES
    # =====================================================

    current = df.iloc[0]

    pm25 = current["pm2_5"]

    pm10 = current["pm10"]

    no2 = current["nitrogen_dioxide"]

    so2 = current["sulphur_dioxide"]

    ozone = current["ozone"]

    ammonia = current["ammonia"]

    aqi = current["AQI"]

    dominant_pollutant = (
        current["Dominant Pollutant"]
    )

    category = get_aqi_category(
        aqi
    )


    # =====================================================
    # CURRENT DASHBOARD
    # =====================================================

    st.subheader(
        f"📍 {selected_city} - "
        "Current Air Quality"
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:

        st.metric(

            "Indian AQI",

            round(aqi)
            if pd.notna(aqi)
            else "N/A"
        )

    with col2:

        st.metric(

            "PM2.5",

            f"{pm25:.1f} µg/m³"
        )

    with col3:

        st.metric(

            "PM10",

            f"{pm10:.1f} µg/m³"
        )

    with col4:

        st.metric(

            "NO₂",

            f"{no2:.1f} µg/m³"
        )

    with col5:

        st.metric(

            "O₃",

            f"{ozone:.1f} µg/m³"
        )


    # =====================================================
    # AQI INFORMATION
    # =====================================================

    st.info(
        f"Indian AQI Category: **{category}**"
    )

    st.write(
        get_health_message(category)
    )

    if pd.notna(dominant_pollutant):

        st.write(

            f"**Dominant Pollutant:** "
            f"{dominant_pollutant}"
        )


    # =====================================================
    # POLLUTANT BAR CHART
    # =====================================================

    st.subheader(
        "📊 Current Pollutant Concentrations"
    )

    pollutant_data = pd.DataFrame({

        "Pollutant": [

            "PM2.5",

            "PM10",

            "NO₂",

            "SO₂",

            "O₃",

            "NH₃"
        ],

        "Concentration": [

            pm25,

            pm10,

            no2,

            so2,

            ozone,

            ammonia
        ]
    })

    pollutant_chart = px.bar(

        pollutant_data,

        x="Pollutant",

        y="Concentration",

        title=(
            f"Pollutant Levels - "
            f"{selected_city}"
        ),

        labels={

            "Concentration":
            "Concentration (µg/m³)"
        }
    )

    st.plotly_chart(

        pollutant_chart,

        use_container_width=True
    )


    # =====================================================
    # 24-HOUR POLLUTANT TREND
    # =====================================================

    st.subheader(
        "📈 24-Hour Pollutant Trend"
    )

    trend_df = df.head(24)[

        [
            "time",

            "pm2_5",

            "pm10",

            "nitrogen_dioxide",

            "sulphur_dioxide",

            "ozone",

            "ammonia"
        ]

    ].copy()

    trend_df = trend_df.rename(

        columns={

            "pm2_5": "PM2.5",

            "pm10": "PM10",

            "nitrogen_dioxide": "NO₂",

            "sulphur_dioxide": "SO₂",

            "ozone": "O₃",

            "ammonia": "NH₃"
        }
    )

    trend_df = trend_df.melt(

        id_vars="time",

        var_name="Pollutant",

        value_name="Concentration"
    )

    trend_chart = px.line(

        trend_df,

        x="time",

        y="Concentration",

        color="Pollutant",

        markers=True,

        title=(
            f"24-Hour Pollutant Trend - "
            f"{selected_city}"
        ),

        labels={

            "time": "Time",

            "Concentration":
            "Concentration (µg/m³)"
        }
    )

    st.plotly_chart(

        trend_chart,

        use_container_width=True
    )


    # =====================================================
    # AQI TREND
    # =====================================================

    st.subheader(
        "📈 24-Hour Indian AQI Trend"
    )

    aqi_trend_df = df.head(24).copy()

    aqi_chart = px.line(

        aqi_trend_df,

        x="time",

        y="AQI",

        markers=True,

        title=(
            f"Indian AQI Forecast - "
            f"{selected_city}"
        ),

        labels={

            "time": "Time",

            "AQI": "Indian AQI"
        }
    )

    st.plotly_chart(

        aqi_chart,

        use_container_width=True
    )


    # =====================================================
    # MACHINE LEARNING
    # =====================================================

    st.subheader(
        "🤖 PM2.5 Machine Learning Prediction"
    )

    st.write(
        "Linear Regression and Random Forest "
        "Regression are compared using "
        "Mean Absolute Error (MAE)."
    )

    results = train_pm25_models(df)


    if results is not None:

        # -------------------------------------------------
        # MODEL PERFORMANCE
        # -------------------------------------------------

        st.write(
            "### 📊 Model Performance"
        )

        comparison_df = results[
            "comparison_df"
        ].copy()

        comparison_df["MAE"] = (
            comparison_df["MAE"].round(2)
        )

        st.dataframe(

            comparison_df,

            use_container_width=True,

            hide_index=True
        )


        # -------------------------------------------------
        # BEST MODEL
        # -------------------------------------------------

        best_model_name = results[
            "best_model_name"
        ]

        best_mae = results[
            "best_mae"
        ]

        st.success(

            f"Best Model: **{best_model_name}** "
            f"with MAE = **{best_mae:.2f} µg/m³**"
        )


        # -------------------------------------------------
        # NEXT-HOUR PREDICTION
        # -------------------------------------------------

        next_prediction = results[
            "next_prediction"
        ]

        st.metric(

            "Predicted Next-Hour PM2.5",

            f"{next_prediction:.2f} µg/m³"
        )


        # -------------------------------------------------
        # ACTUAL VS PREDICTED
        # -------------------------------------------------

        st.write(
            "### 📈 Actual vs Predicted PM2.5"
        )

        prediction_df = results[
            "prediction_df"
        ].copy()

        prediction_df["Hour"] = range(

            1,

            len(prediction_df) + 1
        )

        prediction_plot_df = (
            prediction_df.melt(

                id_vars="Hour",

                var_name="Type",

                value_name="PM2.5"
            )
        )

        ml_chart = px.line(

            prediction_plot_df,

            x="Hour",

            y="PM2.5",

            color="Type",

            markers=True,

            title=(
                f"Actual vs Predicted PM2.5 "
                f"({best_model_name})"
            )
        )

        st.plotly_chart(

            ml_chart,

            use_container_width=True
        )


        # -------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------

        feature_importance_df = results[
            "feature_importance_df"
        ]

        if feature_importance_df is not None:

            st.write(
                "### 🔍 Random Forest Feature Importance"
            )

            importance_chart = px.bar(

                feature_importance_df,

                x="Importance",

                y="Feature",

                orientation="h",

                title=(
                    "Features Influencing PM2.5 Prediction"
                )
            )

            st.plotly_chart(

                importance_chart,

                use_container_width=True
            )


        # -------------------------------------------------
        # FEATURES USED
        # -------------------------------------------------

        st.write(
            "### 🔧 Features Used by the Models"
        )

        feature_table = pd.DataFrame({

            "Feature": [

                "Hour of day",

                "Day of week",

                "Previous-hour PM2.5",

                "Previous-hour PM10",

                "Previous-hour NO₂"
            ],

            "Purpose": [

                "Captures hourly patterns",

                "Captures weekly patterns",

                "Previous PM2.5 level",

                "Previous PM10 level",

                "Previous NO₂ level"
            ]
        })

        st.dataframe(

            feature_table,

            use_container_width=True,

            hide_index=True
        )


    else:

        st.warning(
            "Not enough observations "
            "to train the ML models."
        )


    # =====================================================
    # DOWNLOAD DATA
    # =====================================================

    st.subheader(
        "⬇️ Download Air Quality Data"
    )

    download_df = df.copy()

    download_df["time"] = (
        download_df["time"].astype(str)
    )

    csv_data = download_df.to_csv(
        index=False
    )

    st.download_button(

        label="📥 Download CSV",

        data=csv_data,

        file_name=(
            f"{selected_city}_"
            "air_quality.csv"
        ),

        mime="text/csv"
    )


    # =====================================================
    # RAW DATA
    # =====================================================

    with st.expander(
        "View Raw Air Quality Data"
    ):

        st.dataframe(

            df,

            use_container_width=True
        )


# =========================================================
# ERROR HANDLING
# =========================================================

except requests.exceptions.RequestException:

    st.error(
        "Unable to connect to the air-quality API."
    )

except Exception as e:

    st.error(
        f"An error occurred: {e}"
    )