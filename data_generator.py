# This data is generated with AI 
import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------------------
# CONFIGURATION
# -----------------------------
n_users = 200
days_per_user = 180
total_rows = n_users * days_per_user

# -----------------------------
# USER-LEVEL DATA
# -----------------------------
user_ids = np.repeat(np.arange(1, n_users + 1), days_per_user)

gender = np.random.choice(["Male", "Female", "Other"], n_users, p=[0.48, 0.48, 0.04])
age = np.random.randint(18, 65, n_users)
height_cm = np.random.normal(170, 10, n_users)
weight_kg = np.random.normal(72, 15, n_users)

user_df = pd.DataFrame({
    "user_id": np.arange(1, n_users + 1),
    "gender": gender,
    "age": age,
    "height_cm": height_cm,
    "weight_kg": weight_kg
})

user_df = user_df.loc[user_df.index.repeat(days_per_user)].reset_index(drop=True)

# -----------------------------
# TIME DATA
# -----------------------------
dates = pd.date_range("2024-01-01", periods=days_per_user)
date_col = np.tile(dates, n_users)

# -----------------------------
# ACTIVITY DATA
# -----------------------------
steps = np.random.normal(7500, 3000, total_rows).astype(int)
steps = np.clip(steps, 500, 25000)

active_minutes = (steps / 100) + np.random.normal(0, 10, total_rows)
active_minutes = np.clip(active_minutes, 5, 180)

calories_burned = (
    steps * 0.04
    + active_minutes * 3
    + np.random.normal(0, 50, total_rows)
)

# -----------------------------
# HEALTH METRICS
# -----------------------------
resting_heart_rate = np.random.normal(70, 8, total_rows)
sleep_hours = np.random.normal(7, 1.5, total_rows)
sleep_hours = np.clip(sleep_hours, 3, 10)

stress_level = np.random.choice(
    ["Low", "Medium", "High"],
    total_rows,
    p=[0.4, 0.4, 0.2]
)

# -----------------------------
# LIFESTYLE FACTORS
# -----------------------------
diet_quality = np.random.choice(
    ["Poor", "Average", "Good"],
    total_rows,
    p=[0.25, 0.45, 0.30]
)

water_intake_liters = np.random.normal(2.2, 0.7, total_rows)
water_intake_liters = np.clip(water_intake_liters, 0.5, 5)

# -----------------------------
# PERFORMANCE / TARGET VARIABLES
# -----------------------------
fitness_score = (
    0.3 * (steps / 10000)
    + 0.3 * (active_minutes / 60)
    + 0.2 * (sleep_hours / 8)
    - 0.1 * (resting_heart_rate / 100)
    + np.random.normal(0, 0.2, total_rows)
)

fitness_score = np.clip(fitness_score, 0, 1)

weight_change = np.random.normal(0, 0.15, total_rows) - (calories_burned - 2000) / 10000

# Classification label
goal_achieved = (fitness_score > 0.6).astype(int)

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame({
    "date": date_col,
    "user_id": user_df["user_id"],
    "gender": user_df["gender"],
    "age": user_df["age"],
    "height_cm": user_df["height_cm"],
    "weight_kg": user_df["weight_kg"],
    "steps": steps,
    "active_minutes": active_minutes,
    "calories_burned": calories_burned,
    "resting_heart_rate": resting_heart_rate,
    "sleep_hours": sleep_hours,
    "stress_level": stress_level,
    "diet_quality": diet_quality,
    "water_intake_liters": water_intake_liters,
    "fitness_score": fitness_score,
    "weight_change_kg": weight_change,
    "goal_achieved": goal_achieved
})

# -----------------------------
# INTRODUCE MISSING VALUES
# -----------------------------
for col in ["steps", "sleep_hours", "water_intake_liters", "resting_heart_rate"]:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

# -----------------------------
# INTRODUCE OUTLIERS
# -----------------------------
outlier_idx = df.sample(frac=0.01).index
df.loc[outlier_idx, "steps"] *= 3
df.loc[outlier_idx, "calories_burned"] *= 2

# -----------------------------
# SAVE DATA
# -----------------------------
df.to_csv("personal_fitness_data.csv", index=False)

print("Dataset generated: personal_fitness_data.csv")
print(df.head())
