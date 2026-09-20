from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Smart Fitness Recommendation System",
    page_icon="🏃",
    layout="wide",
)

DATA_FILE = Path(__file__).resolve().parent / "Fitness Tracker Dataset.csv"

MET_MAP = {
    "Yoga": 3.0,
    "Strength": 5.0,
    "Cardio": 7.0,
    "HIIT": 8.5,
}


@st.cache_data
def load_and_prepare_data():
    fitness_df = pd.read_csv(DATA_FILE)

    fitness_df = fitness_df.rename(columns={
        "Weight (kg)": "Weight_kg",
        "Height (m)": "Height_m",
        "Session_Duration (hours)": "Session_Duration_hours",
        "Water_Intake (liters)": "Water_Intake_liters",
        "Workout_Frequency (days/week)": "Workout_Frequency_days",
    })

    for df in [fitness_df]:
        df["Gender"] = df["Gender"].astype(str).str.strip()
        df["Workout_Type"] = df["Workout_Type"].astype(str).str.strip()

    numeric_columns = [
        "Age",
        "Weight_kg",
        "Height_m",
        "Max_BPM",
        "Avg_BPM",
        "Resting_BPM",
        "Session_Duration_hours",
        "Calories_Burned",
        "Fat_Percentage",
        "Water_Intake_liters",
        "Workout_Frequency_days",
        "Experience_Level",
        "BMI",
    ]

    for col in numeric_columns:
        if col in fitness_df.columns:
            fitness_df[col] = pd.to_numeric(fitness_df[col], errors="coerce")

    fitness_df = fitness_df.drop_duplicates().copy()

    numeric_columns = fitness_df.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = fitness_df.select_dtypes(include=["object"]).columns

    for col in numeric_columns:
        fitness_df[col] = fitness_df[col].fillna(fitness_df[col].median())

    for col in categorical_columns:
        fitness_df[col] = fitness_df[col].fillna(fitness_df[col].mode()[0])

    fitness_df["BMI_Corrected"] = (
        fitness_df["Weight_kg"] / (fitness_df["Height_m"] ** 2)
    ).round(2)

    fitness_df["BMI_Category"] = fitness_df["BMI_Corrected"].apply(get_bmi_category)

    fitness_df["Session_Minutes"] = (
        fitness_df["Session_Duration_hours"] * 60
    ).round(2)

    fitness_df["Weekly_Workout_Minutes"] = (
        fitness_df["Session_Minutes"] * fitness_df["Workout_Frequency_days"]
    ).round(2)

    fitness_df["Activity_Level"] = fitness_df["Weekly_Workout_Minutes"].apply(
        get_activity_level
    )

    fitness_df["Age_Group"] = fitness_df["Age"].apply(get_age_group)
    fitness_df["BPM_Increase"] = fitness_df["Avg_BPM"] - fitness_df["Resting_BPM"]

    fitness_df["MET"] = fitness_df["Workout_Type"].map(MET_MAP)
    fitness_df["MET"] = fitness_df["MET"].fillna(fitness_df["MET"].median())

    fitness_df["Fitness_Score"] = (
        fitness_df["Weekly_Workout_Minutes"]
        * fitness_df["Workout_Frequency_days"]
        * fitness_df["Avg_BPM"]
        * fitness_df["Calories_Burned"]
    ) / (fitness_df["BMI_Corrected"] * 100000)

    fitness_df["Fitness_Score"] = fitness_df["Fitness_Score"].replace(
        [np.inf, -np.inf], np.nan
    )

    fitness_df["Fitness_Score"] = fitness_df["Fitness_Score"].fillna(
        fitness_df["Fitness_Score"].median()
    )

    min_score = fitness_df["Fitness_Score"].min()
    max_score = fitness_df["Fitness_Score"].max()

    fitness_df["Fitness_Score_100"] = (
        (fitness_df["Fitness_Score"] - min_score) / (max_score - min_score)
    ) * 100

    fitness_df["Fitness_Score_100"] = fitness_df["Fitness_Score_100"].round(2)
    fitness_df["Fitness_Level"] = fitness_df["Fitness_Score_100"].apply(
        get_fitness_level
    )

    return fitness_df


def get_bmi_category(bmi):
    if pd.isna(bmi):
        return np.nan
    elif bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def get_activity_level(weekly_minutes):
    if pd.isna(weekly_minutes):
        return np.nan
    elif weekly_minutes < 150:
        return "Low"
    elif weekly_minutes < 300:
        return "Moderate"
    else:
        return "High"


def get_age_group(age):
    if pd.isna(age):
        return np.nan
    elif age < 25:
        return "Young Adult"
    elif age < 40:
        return "Adult"
    elif age < 55:
        return "Middle Age"
    else:
        return "Older Adult"


def get_fitness_level(score):
    if pd.isna(score):
        return np.nan
    elif score < 34:
        return "Low Fitness"
    elif score < 67:
        return "Moderate Fitness"
    else:
        return "High Fitness"


def generate_fitness_recommendation(row):
    fitness_level = row["Fitness_Level"]
    profile = row["Fitness_Profile_Label"]
    bmi_category = row["BMI_Category"]
    activity_level = row["Activity_Level"]
    age_group = row["Age_Group"]
    workout_type = row["Workout_Type"]

    recommended_intensity = "Moderate"
    recommended_workout = "Cardio and strength training"
    suggested_frequency = "4 days per week"
    suggested_duration = "45–60 minutes per session"
    safety_note = "Maintain proper form and increase intensity gradually."

    if fitness_level == "Low Fitness" or profile == "Beginner / Low Activity Profile":
        recommended_intensity = "Low to Moderate"
        recommended_workout = (
            "Walking, beginner yoga, light cardio, basic bodyweight exercises"
        )
        suggested_frequency = "3 days per week"
        suggested_duration = "30–40 minutes per session"
        safety_note = (
            "Start slowly and focus on consistency before increasing intensity."
        )

    elif fitness_level == "Moderate Fitness" or profile == "Moderate Fitness Profile":
        recommended_intensity = "Moderate"
        recommended_workout = (
            "Cardio, strength training, cycling, swimming, yoga"
        )
        suggested_frequency = "4 days per week"
        suggested_duration = "45–60 minutes per session"
        safety_note = "Balance cardio, strength, and recovery for steady improvement."

    elif fitness_level == "High Fitness" or profile == "High Activity / Advanced Profile":
        recommended_intensity = "Moderate to High"
        recommended_workout = (
            "HIIT, running, advanced strength training, circuit training"
        )
        suggested_frequency = "5 days per week"
        suggested_duration = "60–75 minutes per session"
        safety_note = (
            "Use progressive overload and include recovery days to avoid overtraining."
        )

    if bmi_category in ["Overweight", "Obese"]:
        recommended_workout = (
            "Low-impact cardio, cycling, swimming, walking, beginner strength training"
        )
        safety_note = "Low-impact exercises are recommended to reduce joint pressure."

    elif bmi_category == "Underweight":
        recommended_workout = (
            "Strength training, resistance exercises, yoga, light cardio"
        )
        safety_note = (
            "Focus on strength building and avoid excessive high-intensity cardio."
        )

    if age_group == "Older Adult":
        recommended_intensity = "Low to Moderate"
        suggested_duration = "30–45 minutes per session"
        safety_note = "Prioritise low-impact exercise, mobility, and safe progression."

    return pd.Series({
        "Recommended_Intensity": recommended_intensity,
        "Recommended_Workout": recommended_workout,
        "Suggested_Frequency": suggested_frequency,
        "Suggested_Duration": suggested_duration,
        "Recommendation_Note": safety_note,
    })


@st.cache_resource
def train_models(fitness_df):
    # Model 1: Calories burned prediction
    features_model1 = [
        "Session_Minutes",
        "Avg_BPM",
        "Resting_BPM",
        "BPM_Increase",
        "MET",
        "Weight_kg",
        "Fat_Percentage",
        "Workout_Type",
    ]

    X1 = fitness_df[features_model1]
    y1 = fitness_df["Calories_Burned"]

    numeric_features1 = X1.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features1 = X1.select_dtypes(include=["object"]).columns.tolist()

    preprocessor1 = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features1),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features1),
        ]
    )

    calorie_model = Pipeline(steps=[
        ("preprocessor", preprocessor1),
        ("model", LinearRegression()),
    ])
    calorie_model.fit(X1, y1)

    # Model 2: Fitness level classification
    features_model2 = [
        "Age",
        "Gender",
        "Weight_kg",
        "Height_m",
        "BMI_Corrected",
        "BMI_Category",
        "Session_Minutes",
        "Weekly_Workout_Minutes",
        "Activity_Level",
        "Avg_BPM",
        "Resting_BPM",
        "BPM_Increase",
        "MET",
        "Workout_Type",
        "Fat_Percentage",
        "Water_Intake_liters",
        "Workout_Frequency_days",
        "Experience_Level",
        "Age_Group",
    ]

    X2 = fitness_df[features_model2]
    y2 = fitness_df["Fitness_Level"]

    numeric_features2 = X2.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features2 = X2.select_dtypes(include=["object"]).columns.tolist()

    preprocessor2 = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features2),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features2),
        ]
    )

    fitness_model = Pipeline(steps=[
        ("preprocessor", preprocessor2),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
        ),
    ])
    fitness_model.fit(X2, y2)

    # Model 3: Fitness profile clustering
    cluster_features = [
        "Age",
        "BMI_Corrected",
        "Fat_Percentage",
        "Session_Minutes",
        "Weekly_Workout_Minutes",
        "BPM_Increase",
        "MET",
        "Water_Intake_liters",
        "Workout_Frequency_days",
        "Experience_Level",
    ]

    X_cluster = fitness_df[cluster_features].copy()
    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)

    labelled_df = fitness_df.copy()
    labelled_df["Fitness_Profile_Cluster"] = cluster_labels

    cluster_summary_for_label = labelled_df.groupby("Fitness_Profile_Cluster")[[
        "Weekly_Workout_Minutes",
        "Experience_Level",
        "BPM_Increase",
        "MET",
    ]].mean()

    cluster_activity_rank = (
        cluster_summary_for_label["Weekly_Workout_Minutes"]
        + cluster_summary_for_label["Experience_Level"] * 50
        + cluster_summary_for_label["BPM_Increase"]
    ).sort_values()

    cluster_order = cluster_activity_rank.index.tolist()
    cluster_name_map = {
        cluster_order[0]: "Beginner / Low Activity Profile",
        cluster_order[1]: "Moderate Fitness Profile",
        cluster_order[2]: "High Activity / Advanced Profile",
    }

    return calorie_model, fitness_model, cluster_scaler, kmeans, cluster_name_map


def make_user_row(
    age,
    gender,
    height_m,
    weight_kg,
    workout_type,
    session_minutes,
    workout_frequency,
    avg_bpm,
    resting_bpm,
    fat_percentage,
    water_intake,
    experience_level,
):
    bmi = round(weight_kg / (height_m ** 2), 2)
    bmi_category = get_bmi_category(bmi)
    weekly_workout_minutes = round(session_minutes * workout_frequency, 2)
    activity_level = get_activity_level(weekly_workout_minutes)
    age_group = get_age_group(age)
    bpm_increase = avg_bpm - resting_bpm
    met = MET_MAP[workout_type]

    return {
        "Age": float(age),
        "Gender": gender,
        "Weight_kg": float(weight_kg),
        "Height_m": float(height_m),
        "BMI_Corrected": float(bmi),
        "BMI_Category": bmi_category,
        "Session_Minutes": float(session_minutes),
        "Weekly_Workout_Minutes": float(weekly_workout_minutes),
        "Activity_Level": activity_level,
        "Avg_BPM": float(avg_bpm),
        "Resting_BPM": float(resting_bpm),
        "BPM_Increase": float(bpm_increase),
        "MET": float(met),
        "Workout_Type": workout_type,
        "Fat_Percentage": float(fat_percentage),
        "Water_Intake_liters": float(water_intake),
        "Workout_Frequency_days": float(workout_frequency),
        "Experience_Level": float(experience_level),
        "Age_Group": age_group,
    }


st.title("🏃 Smart Fitness Recommendation System")
st.caption(
    "A student-project prototype using regression, classification, clustering, "
    "feature engineering, and rule-based recommendations."
)

with st.sidebar:
    st.header("About this project")
    st.write(
        "The app uses the same core modelling approach described in the project notebook: "
        "calorie prediction, fitness-level classification, K-Means fitness profiles, "
        "and rule-based workout recommendations."
    )
    st.warning(
        "This app provides general fitness guidance only. It is not medical advice or a clinical diagnosis."
    )

try:
    fitness_df = load_and_prepare_data()
    calorie_model, fitness_model, cluster_scaler, kmeans, cluster_name_map = train_models(
        fitness_df
    )
except Exception as exc:
    st.error(f"The application could not load or train the project models: {exc}")
    st.stop()

st.subheader("Enter your fitness information")

with st.form("fitness_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=16, max_value=90, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height_m = st.number_input(
            "Height (m)", min_value=1.30, max_value=2.20, value=1.75, step=0.01
        )
        weight_kg = st.number_input(
            "Weight (kg)", min_value=35.0, max_value=200.0, value=75.0, step=0.5
        )

    with col2:
        workout_type = st.selectbox("Workout Type", ["Strength", "Cardio", "Yoga", "HIIT"])
        session_minutes = st.number_input(
            "Session Duration (minutes)", min_value=10, max_value=180, value=60, step=5
        )
        workout_frequency = st.slider(
            "Workout Frequency (days/week)", min_value=1, max_value=7, value=4
        )
        experience_level = st.selectbox(
            "Experience Level", [1, 2, 3],
            format_func=lambda x: {1: "1 - Beginner", 2: "2 - Intermediate", 3: "3 - Advanced"}[x],
        )

    with col3:
        avg_bpm = st.number_input(
            "Average BPM", min_value=70, max_value=200, value=140, step=1
        )
        resting_bpm = st.number_input(
            "Resting BPM", min_value=40, max_value=110, value=70, step=1
        )
        fat_percentage = st.number_input(
            "Fat Percentage", min_value=5.0, max_value=50.0, value=22.0, step=0.5
        )
        water_intake = st.number_input(
            "Water Intake (litres/day)", min_value=0.5, max_value=6.0, value=2.5, step=0.1
        )

    fitness_goal = st.selectbox(
        "Fitness Goal",
        ["General Fitness", "Weight Loss", "Muscle Gain"],
    )

    submitted = st.form_submit_button("Generate Recommendation", use_container_width=True)

if submitted:
    if resting_bpm >= avg_bpm:
        st.warning(
            "Average workout BPM should normally be higher than resting BPM. Please check the values you entered."
        )

    user_row = make_user_row(
        age=age,
        gender=gender,
        height_m=height_m,
        weight_kg=weight_kg,
        workout_type=workout_type,
        session_minutes=session_minutes,
        workout_frequency=workout_frequency,
        avg_bpm=avg_bpm,
        resting_bpm=resting_bpm,
        fat_percentage=fat_percentage,
        water_intake=water_intake,
        experience_level=experience_level,
    )

    user_df = pd.DataFrame([user_row])

    # Model 1 prediction, with the notebook-described fallback formula.
    try:
        calorie_features = [
            "Session_Minutes",
            "Avg_BPM",
            "Resting_BPM",
            "BPM_Increase",
            "MET",
            "Weight_kg",
            "Fat_Percentage",
            "Workout_Type",
        ]
        predicted_calories = float(calorie_model.predict(user_df[calorie_features])[0])
        predicted_calories = max(0.0, predicted_calories)
        calorie_method = "Linear Regression model"
    except Exception:
        predicted_calories = (
            user_row["MET"] * 3.5 * user_row["Weight_kg"] / 200 * user_row["Session_Minutes"]
        )
        calorie_method = "MET fallback formula"

    fitness_features = [
        "Age",
        "Gender",
        "Weight_kg",
        "Height_m",
        "BMI_Corrected",
        "BMI_Category",
        "Session_Minutes",
        "Weekly_Workout_Minutes",
        "Activity_Level",
        "Avg_BPM",
        "Resting_BPM",
        "BPM_Increase",
        "MET",
        "Workout_Type",
        "Fat_Percentage",
        "Water_Intake_liters",
        "Workout_Frequency_days",
        "Experience_Level",
        "Age_Group",
    ]
    predicted_fitness_level = fitness_model.predict(user_df[fitness_features])[0]

    cluster_features = [
        "Age",
        "BMI_Corrected",
        "Fat_Percentage",
        "Session_Minutes",
        "Weekly_Workout_Minutes",
        "BPM_Increase",
        "MET",
        "Water_Intake_liters",
        "Workout_Frequency_days",
        "Experience_Level",
    ]
    user_cluster_scaled = cluster_scaler.transform(user_df[cluster_features])
    cluster_id = int(kmeans.predict(user_cluster_scaled)[0])
    profile_label = cluster_name_map[cluster_id]

    recommendation_input = pd.Series({
        **user_row,
        "Fitness_Level": predicted_fitness_level,
        "Fitness_Profile_Label": profile_label,
    })
    recommendation = generate_fitness_recommendation(recommendation_input)

    st.divider()
    st.subheader("Your Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BMI", f"{user_row['BMI_Corrected']:.1f}")
    m2.metric("BMI Category", user_row["BMI_Category"])
    m3.metric("Activity Level", user_row["Activity_Level"])
    m4.metric("MET Value", f"{user_row['MET']:.1f}")

    m5, m6, m7 = st.columns(3)
    m5.metric("Predicted Calories Burned", f"{predicted_calories:.0f} kcal")
    m6.metric("Predicted Fitness Level", predicted_fitness_level)
    m7.metric("Fitness Profile", profile_label)

    st.subheader("Workout Recommendation")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"**Recommended intensity:** {recommendation['Recommended_Intensity']}")
        st.markdown(f"**Workout:** {recommendation['Recommended_Workout']}")
    with r2:
        st.markdown(f"**Frequency:** {recommendation['Suggested_Frequency']}")
        st.markdown(f"**Duration:** {recommendation['Suggested_Duration']}")

    st.info(recommendation["Recommendation_Note"])
    st.caption(f"Selected fitness goal: {fitness_goal}")

    with st.expander("How these results were generated"):
        st.write(f"Calorie estimate method: **{calorie_method}**")
        st.write(
            "Fitness level: Gradient Boosting classification using the engineered project features."
        )
        st.write(
            "Fitness profile: K-Means clustering with three profiles labelled by overall activity level."
        )
        st.write(
            "The final workout recommendation follows the rule-based logic from the project notebook."
        )
        st.caption(
            "The goal input is displayed as context. The notebook's executable recommendation function "
            "does not contain a separate goal-specific rule, so this deployment does not invent one."
        )

st.divider()
st.caption(
    "Educational prototype. Model outputs depend on the supplied project dataset and should not be used as medical advice."
)
