import streamlit as st
import pandas as pd
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Social Media vs Productivity Dashboard",
    layout="wide"
)

st.image("social-media-productivity.png", width=400)

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Sotial_media_vs_productivity_cleaned.csv")
    return df

df = load_data()

# -------------------------
# Page Navigation
# -------------------------
st.sidebar.title("📍Navigation")
page = st.sidebar.radio(
    "",
    [
        "Main Dashboard",
        "Analysis",
        "Predictions"
    ]
)

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.title("🔎 Filters")

if st.sidebar.checkbox("Show Filters", value=True):
    platform_filter = st.sidebar.multiselect(
        "Select Platform",
        options=df["social_platform_preference"].unique(),
        default=df["social_platform_preference"].unique()
    )
    gender_filter = st.sidebar.multiselect(
        "Select Gender",
        options=df["gender"].unique(),
        default=df["gender"].unique()
    )
    job_filter = st.sidebar.multiselect(
        "Select Job Type",
        options=df["job_type"].unique(),
        default=df["job_type"].unique()
    )
else:
    platform_filter = df["social_platform_preference"].unique()
    gender_filter = df["gender"].unique()
    job_filter = df["job_type"].unique()

# -------------------------
# Apply Filters
# -------------------------
filtered_df = df[
    (df["social_platform_preference"].isin(platform_filter)) &
    (df["gender"].isin(gender_filter)) &
    (df["job_type"].isin(job_filter))
]

# =================================================
# MAIN PAGE
# =================================================
if page == "Main Dashboard":
    tab1, tab2 = st.tabs(["Dataset Preview", "About Data"])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(filtered_df.head())

        st.divider()
        st.subheader("Key Performance Indicators")

        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)

        Users = filtered_df["age"].count()
        Avg_Hours = filtered_df["daily_social_media_time"].mean()
        Avg_Productivity = filtered_df["actual_productivity_score"].mean()

        top_platform = (
            filtered_df.groupby("social_platform_preference")["daily_social_media_time"]
            .mean()
            .reset_index()
            .sort_values(by="daily_social_media_time", ascending=False)
            .head(1)
        )

        if not top_platform.empty:
            best_platform = top_platform.iloc[0]["social_platform_preference"]
            best_platform_hours = top_platform.iloc[0]["daily_social_media_time"]
        else:
            best_platform = "No platform found"
            best_platform_hours = 0

        col1.metric("👥 Users", Users)
        col2.metric("⏱️ Avg Hours", round(Avg_Hours, 2))
        col3.metric("💡 Avg Productivity", round(Avg_Productivity, 2))
        col4.metric("🏆 Top Platform", best_platform, f"Avg Hours: {round(best_platform_hours,2)}")

    with tab2:
        st.subheader("Social Media vs Productivity Data")
        st.markdown("This dataset contains user activity on social media platforms and their productivity scores.")
        data_dict = {
            "age": "Age of the individual (18–65 years)",
            "gender": "Gender identity",
            "job_type": "Employment sector or status",
            "daily_social_media_time": "Average daily time spent on social media (hours)",
            "social_platform_preference": "Most-used social platform",
            "number_of_notifications": "Number of notifications per day",
            "work_hours_per_day": "Average hours worked each day",
            "perceived_productivity_score": "Self-rated productivity score (0–10)",
            "actual_productivity_score": "Ground-truth productivity score (0–10)",
            "stress_level": "Stress level (1–10)",
            "sleep_hours": "Average sleep hours per night",
            "screen_time_before_sleep": "Screen time before sleep (hours)",
            "breaks_during_work": "Number of breaks during work",
            "uses_focus_apps": "Uses digital focus apps (True/False)",
            "has_digital_wellbeing_enabled": "Digital Wellbeing enabled (True/False)",
            "coffee_consumption_per_day": "Cups of coffee per day",
            "days_feeling_burnout_per_month": "Days feeling burnout per month",
            "weekly_offline_hours": "Weekly offline hours (excluding sleep)",
            "job_satisfaction_score": "Job satisfaction score (0–10)"
        }
        dict_df = pd.DataFrame(list(data_dict.items()), columns=["Column", "Description"])
        st.table(dict_df)

# =================================================
# ANALYSIS PAGE
# =================================================
elif page == "Analysis":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Univariate Analysis",
        "Bivariate Analysis",
        "Multivariate Analysis",
        "Insights Q&A",
        "Key Insights & Recommendations"
    ])

    with tab1:
        st.subheader("Univariate Analysis")
        fig1 = px.histogram(filtered_df, x="daily_social_media_time", title="Daily Social Media Time Distribution")
        st.plotly_chart(fig1)
        st.markdown("**** Most people spend a few hours daily on social media, but some users spend much longer, showing wide variation in habits.")
        fig2 = px.histogram(filtered_df, x="actual_productivity_score", title="Actual Productivity Score Distribution")
        st.plotly_chart(fig2)
        st.markdown("** Productivity scores are spread across the scale, but many users cluster around mid-level scores, with fewer at very high or very low levels.")

    with tab2:
        st.subheader("Bivariate Analysis")
        # KDE-style plot for Perceived vs Actual Productivity
        x_range = np.linspace(0, 10, 200)
        kde_perceived = gaussian_kde(filtered_df["perceived_productivity_score"])
        kde_actual = gaussian_kde(filtered_df["actual_productivity_score"])

        fig_kde = go.Figure()

        fig_kde.add_trace(go.Scatter(
            x=x_range,
            y=kde_perceived(x_range),
            mode="lines",
            name="Perceived Productivity",
            line=dict(color="blue"),
            fill="tozeroy",
            opacity=0.4
        ))

        fig_kde.add_trace(go.Scatter(
            x=x_range,
            y=kde_actual(x_range),
            mode="lines",
            name="Actual Productivity",
            line=dict(color="red"),
            fill="tozeroy",
            opacity=0.4
        ))

        fig_kde.update_layout(
            title="Perceived vs Actual Productivity Distribution",
            xaxis_title="Productivity Score (1-10)",
            yaxis_title="Density",
            legend_title="Type",
            template="simple_white"
        )

        st.plotly_chart(fig_kde)
        st.markdown("** The perceived scores are consistently higher than the actual scores, showing an overestimation gap (People tend to overestimate their productivity.).")



        avg_prod = filtered_df.groupby("social_platform_preference")["actual_productivity_score"].mean().reset_index()
        fig3 = px.bar(avg_prod, x="social_platform_preference", y="actual_productivity_score", color="social_platform_preference", title="Average Productivity by Platform")
        st.plotly_chart(fig3)
        st.markdown("** Some platforms help people stay more productive, but spending too much time on others lowers productivity.")
        #fig4 = px.scatter(filtered_df, x="daily_social_media_time", y="actual_productivity_score", color="social_platform_preference", title="Daily Social Media Time vs Productivity")
        #st.plotly_chart(fig4)
        #st.markdown("** As daily social media time increases, productivity generally decreases, highlighting a negative correlation.")

        fig5 = px.box(filtered_df, x="gender", y="actual_productivity_score", color="gender", title="Productivity by Gender")
        st.plotly_chart(fig5)
        st.markdown("** Gender differences are observed in productivity distribution, with some variation in average scores across groups.")
    with tab3:
        st.subheader("Multivariate Analysis")
        heatmap_data = filtered_df.groupby(["social_platform_preference","gender"])["actual_productivity_score"].mean().reset_index()
        fig6 = px.density_heatmap(heatmap_data, x="social_platform_preference", y="gender", z="actual_productivity_score", color_continuous_scale="Viridis", title="Productivity Heatmap by Platform & Gender")
        st.plotly_chart(fig6)
        st.markdown("** Productivity varies across platforms and genders, with some combinations showing stronger performance than others.")
        # Grouped Boxplot (Productivity by Job Type & Gender)
        fig_box = px.box(
            filtered_df,
            x="job_type",
            y="actual_productivity_score",
            color="gender",
            title="Productivity by Job Type & Gender"
        )
        st.plotly_chart(fig_box)
        st.markdown("** Job type and gender both influence productivity. Certain job types show wider variability in scores.")
        # calculate average productivity and standard deviation for each job type
        breaks_prod = (
            filtered_df.groupby(["breaks_during_work", "job_type"])["actual_productivity_score"]
            .agg(["mean", "std"])
            .reset_index()
        )

        fig_breaks = px.line(
            breaks_prod,
            x="breaks_during_work",
            y="mean",
            color="job_type",
            error_y="std",
            markers=True,
            title="Relationship Between Breaks and Productivity According to Job Types",
            labels={
                "breaks_during_work": "Breaks During Work",
                "mean": "Average Productivity Score",
                "job_type": "Job Type"
            },
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_breaks.update_layout(
            xaxis=dict(dtick=1),
            yaxis_title="Actual Productivity Score",
            legend_title="Job Type",
            template="simple_white"
        )

        st.plotly_chart(fig_breaks)
        st.markdown("** Moderate breaks during work are linked to higher productivity, while too few or too many breaks reduce efficiency.")
        # 43D Scatter Plot (Social Media Time × Stress × Productivity)
        fig_3d = px.scatter_3d(
            filtered_df,
            x="daily_social_media_time",
            y="stress_level",
            z="actual_productivity_score",
            color="gender",
            title="3D Scatter: Social Media Time, Stress, Productivity"
        )
        st.plotly_chart(fig_3d)
        st.markdown("** Productivity is shaped by both social media time and stress levels. High stress combined with heavy social media use leads to lower scores.")

    with tab4:
        st.subheader("Insights Q&A")
        question = st.selectbox(
            "Select a question:",
            [
                "Which platform has highest average productivity?",
                "Which gender spends more hours?",
                "Average productivity by age?",
                "Correlation between hours and productivity?"
            ]
        )

        if question == "Which platform has highest average productivity?":
            avg_prod = filtered_df.groupby("social_platform_preference")["actual_productivity_score"].mean().reset_index().sort_values(by="actual_productivity_score", ascending=False)
            fig_top_platform = px.bar(avg_prod, x="social_platform_preference", y="actual_productivity_score", color="social_platform_preference", title="Average Productivity by Platform")
            st.plotly_chart(fig_top_platform)
            st.markdown("** Some platforms help people stay more productive, while others are linked to lower productivity.")

        elif question == "Which gender spends more hours?":
            avg_hours_gender = filtered_df.groupby("gender")["daily_social_media_time"].mean().reset_index()
            fig_hours_gender = px.bar(avg_hours_gender, x="gender", y="daily_social_media_time", color="gender", title="Average Hours Spent by Gender")
            st.plotly_chart(fig_hours_gender)
            st.markdown("** On average, one gender spends more time on social media than the other, showing differences in usage patterns.")

        elif question == "Average productivity by age?":
            fig_age_prod = px.scatter(filtered_df, x="age", y="actual_productivity_score", color="gender", title="Productivity by Age")
            st.plotly_chart(fig_age_prod)
            st.markdown("** Productivity changes with age. Younger and older groups show different patterns compared to middle-aged users.")

        elif question == "Relation between social media hours and productivity?":
            fig_corr = px.scatter(filtered_df, x="daily_social_media_time", y="actual_productivity_score", color="gender", title="Correlation: Hours vs Productivity")
            st.plotly_chart(fig_corr)
            st.markdown("** More hours on social media usually mean lower productivity, showing a clear negative relationship.")

    with tab5:
        st.title("🧠 Key Insights")
        st.markdown("""
        ### 📌 Main Findings
        - ⏱️ Productivity drops as daily social media time increases.
        - 🏆 Some platforms are linked to higher productivity, while heavy use of others lowers it.
        - 📊 Productivity changes with age groups.
        - ☕ Moderate breaks during work improve productivity.
        - 😰 High stress combined with heavy social media use reduces performance.
        - 🌐 Platform and gender together influence productivity levels.

        ### 💡 Recommendations
        - Limit time on low-productivity platforms, especially under stress.
        - Encourage balanced use of platforms that support productivity.
        - Provide targeted support for genders and age groups with lower scores.
        - Promote healthy, moderate breaks during work.
        - Support stress management and better sleep habits.
        - Tailor strategies for job types showing wide productivity variation.
        """)



# =================================================
# FINAL INSIGHTS PAGE
# =================================================
elif page == "Predictions":
    st.title("🔮 Productivity Prediction")

    st.sidebar.markdown("### User Specifications")

    # Sidebar inputs
    age = st.sidebar.slider("Age", int(df["age"].min()), int(df["age"].max()), step=1)
    gender = st.sidebar.selectbox("Gender", df["gender"].unique())
    job_type = st.sidebar.selectbox("Job Type", df["job_type"].unique())
    daily_social_media_time = st.sidebar.slider(
        "Daily Social Media Time (hours)",
        int(df["daily_social_media_time"].min()),
        int(df["daily_social_media_time"].max()),
        step=1
    )
    work_hours_per_day = st.sidebar.slider(
        "Work Hours per Day",
        int(df["work_hours_per_day"].min()),
        int(df["work_hours_per_day"].max()),
        step=1
    )

    # Input Data Summary
    input_df = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "job_type": [job_type],
        "daily_social_media_time": [daily_social_media_time],
        "work_hours_per_day": [work_hours_per_day],
        "social_platform_preference": ["Facebook"],
        "number_of_notifications": [0],
        "perceived_productivity_score": [5],
        "stress_level": [5],
        "sleep_hours": [7],
        "screen_time_before_sleep": [2],
        "breaks_during_work": [1],
        "uses_focus_apps": [0],
        "has_digital_wellbeing_enabled": [0],
        "coffee_consumption_per_day": [1],
        "days_feeling_burnout_per_month": [2],
        "weekly_offline_hours": [10],
        "job_satisfaction_score": [6]
    })

    st.markdown("### 📋 Input Summary")
    st.table(input_df)

    # Prediction
    if st.button("Calculate Predicted Productivity"):
        import joblib
        try:
            ml_model = joblib.load("best_model.pkl")

            # تحقق من أن النموذج مدرّب فعلاً
            if hasattr(ml_model, "predict"):
                prediction = ml_model.predict(input_df).round(2)[0]
                st.success(f"💡 Predicted Productivity Score: {prediction}")
                st.markdown("**Note:** Higher scores mean better productivity performance.")
            else:
                st.warning("⚠️ The loaded model is not trained yet. Please retrain and save it again using joblib.dump(best_estimator_, 'best_model.pkl').")

        except FileNotFoundError:
            st.error("❌ Model file not found. Please train and save the model as 'best_model.pkl' before running predictions.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")



# =================================================
# GLOBAL FOOTER
# =================================================
st.markdown(
    """
    <hr style="margin-top:30px; margin-bottom:10px;">
    <div style="text-align:center; color:gray; font-size:14px;">
        © 2026 Social Media Vs Productivity Dashboard| Developed by Eng.Amal Mohamed-Khair | Email: amal4khair@gmail.com
    </div>
    """,
    unsafe_allow_html=True
)
