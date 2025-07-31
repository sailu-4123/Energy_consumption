# 🌟 Energy Consumption Analysis - India Household Dataset 🌟

# === 1. Import Libraries ===
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("energy_data.csv")

# 🔗 MongoDB Connection (Local Server)
     # Collection name


client = MongoClient("mongodb://localhost:27017/")
db = client["energy_analysis_db"]
collection = db["households"]

records = df.to_dict(orient='records')
collection.delete_many({})  # Optional: clear previous data
collection.insert_many(records)
# Set seaborn style for beautiful plots
sns.set(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6), 'axes.titlesize': 16, 'axes.labelsize': 12})

# === 2. Load Dataset ===
print("\n📥 Loading dataset...")
df = pd.read_csv("energy_data.csv")  # Ensure this CSV is in the same folder
print("✅ Dataset loaded successfully!")

# === 3. Data Overview ===
print("\n📊 First 5 rows of the dataset:")
print(df.head())

print("\nℹ️ Dataset Information:")
print(df.info())

print("\n📈 Descriptive Statistics:")
print(df.describe())

# === 4. Data Cleaning & Preprocessing ===
print("\n🧹 Checking for missing values:")
print(df.isnull().sum())

# Convert 'Region' to categorical
df['Region'] = df['Region'].astype('category')
print("\n✅ Data cleaned and types adjusted!")

# === 5. Data Visualization ===

# Energy Consumption by Region
plt.figure()
sns.barplot(x='Region', y='Monthly_Energy_Consumption_kWh', data=df, palette='viridis')
plt.title('⚡ Monthly Energy Consumption by Region')
plt.ylabel('Energy Consumption (kWh)')
plt.xlabel('Region')
plt.tight_layout()
plt.savefig("energy_by_region.png")  # Save image
plt.show()

# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('🔗 Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# === 6. Machine Learning: Predict Energy Usage ===
print("\n🤖 Starting Machine Learning Prediction...")

# Select Features and Target
X = df[['Monthly_Income_INR', 'Appliance_AC', 'Appliance_Fan', 'Appliance_Light',
        'Fridge', 'Washing_Machine', 'EV_Charging']]
y = df['Monthly_Energy_Consumption_kWh']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Show Predictions
print("\n🔮 Predictions on Test Data:")
for i, value in enumerate(y_pred, start=1):
    print(f"Household {i}: {value:.2f} kWh")

# Show MSE
print(f"\n📉 Mean Squared Error: {mse:.2f}")

print("\n🎯 Energy Analysis Complete. Graphs saved as images.")

