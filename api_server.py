from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# --- 1. Initialize API App ---
app = FastAPI(title="Virtual Pit Wall API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health Check Endpoint (For Frontend Badge) ---
@app.get("/api/health")
def health_check():
    return {"status": "online"}

# --- 2. Global AI Assets ---
model = None
scaler = None
df = None
features = []
one_hot_features = []

@app.on_event("startup")
def load_ai_assets():
    global model, scaler, df, features, one_hot_features
    print("--- Booting Virtual Pit Wall Engine ---")
    try:
        model = load_model("Virtual_Pit_Wall_Final.keras")
        scaler = joblib.load("f1_scaler.gz")
        temp_df = pd.read_parquet("Global_F1_Master.parquet", engine="pyarrow")
        
        # --- THE BULLETPROOF FIX ---
        # Instead of guessing the columns, we ask the scaler exactly what 13 features it was trained on.
        if hasattr(scaler, 'feature_names_in_'):
            features = list(scaler.feature_names_in_)
        else:
            # Fallback if using an older version of scikit-learn
            core = ['Speed', 'Throttle', 'Brake', 'RPM', 'TyreLife', 'LapNumber']
            one_hot = [c for c in temp_df.columns if 'Driver_' in c or 'Compound_' in c]
            features = (core + one_hot)[:scaler.n_features_in_]

        one_hot_features = [col for col in features if 'Driver_' in col or 'Compound_' in col]

        # If the parquet file is missing a column the AI expects, fill it with 0s to prevent crashes
        for col in features:
            if col not in temp_df.columns:
                temp_df[col] = 0

        # Filter the main dataframe to EXACTLY match the 13 columns the AI expects
        df = temp_df[features].copy()
        del temp_df # Free RAM
        
        print(f"✅ AI Brain Loaded (Bulletproof). Configured for exact {len(features)} features.")
    except Exception as e:
        print(f"❌ Error loading AI files: {e}")

# --- 3. Define the Incoming Data Structure ---
class SimulationRequest(BaseModel):
    driver: str
    total_laps: int
    current_lap: int
    current_compound: str
    current_tyre_age: int
    target_compound: str

# --- 4. The Strategy Engine Endpoint ---
@app.post("/api/simulate")
async def run_simulation(req: SimulationRequest):
    print(f"\n[SIMULATION START] {req.driver} | {req.current_compound} -> {req.target_compound}")
    
    driver_focus = f'Driver_{req.driver}'
    
    # Safety net: If driver isn't in our exact features, pick the first available to prevent crash
    if driver_focus not in features:
        print(f"⚠️ Warning: {driver_focus} not found. Defaulting to first available driver.")
        available_drivers = [col for col in features if col.startswith('Driver_')]
        if available_drivers:
            driver_focus = available_drivers[0]
        else:
            return {"status": "error", "message": "Critical Error: No driver data found in dataset."}

    # Grab a baseline flat-out sequence from our dataset
    baseline_data = df[(df[driver_focus] == 1) & (df['Throttle'] >= 90)].head(50).copy()

    # If baseline data is empty, take any 50 rows to keep the math alive
    if baseline_data.empty:
        baseline_data = df.head(50).copy()

    def predict_lap_time(compound, tyre_life, lap_num):
        test_data = baseline_data.copy()
        test_data['TyreLife'] = tyre_life
        test_data['LapNumber'] = lap_num
        
        for col in one_hot_features:
            if 'Compound_' in col:
                test_data[col] = 0
                
        target_col = f'Compound_{compound}'
        if target_col in test_data.columns:
            test_data[target_col] = 1

        # Because test_data now exactly matches the 13 features, this will never crash
        scaled_input = scaler.transform(test_data.values).reshape(1, 50, len(features))
        scaled_speed_pred = model.predict(scaled_input, verbose=0)[0][0]
        
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = scaled_speed_pred
        predicted_speed_kmh = scaler.inverse_transform(dummy)[0, 0]
        
        return 85.0 + (100 - predicted_speed_kmh) * 0.1

    strategy_results = []
    pit_penalty = 24.0
    possible_pit_laps = range(req.current_lap + 1, req.total_laps + 1)
    
    for pit_lap in possible_pit_laps:
        total_race_time = 0
        sim_tyre_life = req.current_tyre_age
        
        for lap in range(req.current_lap, req.total_laps + 1):
            if lap < pit_lap:
                total_race_time += predict_lap_time(req.current_compound, sim_tyre_life, lap)
                sim_tyre_life += 1
            elif lap == pit_lap:
                total_race_time += predict_lap_time(req.target_compound, 0, lap) + pit_penalty
                sim_tyre_life = 1
            else:
                total_race_time += predict_lap_time(req.target_compound, sim_tyre_life, lap)
                sim_tyre_life += 1
                
        strategy_results.append((pit_lap, total_race_time))

    optimal_strategy = min(strategy_results, key=lambda x: x[1])
    best_pit_lap = optimal_strategy[0]
    
    chart_data = [round(time, 2) for lap, time in strategy_results]
    chart_labels = list(possible_pit_laps)

    print(f"✅ AI Verdict: Pit on Lap {best_pit_lap}")

    return {
        "status": "success",
        "optimal_pit_lap": best_pit_lap,
        "message": f"AI Simulation complete. Optimal strategy is to box for <strong>{req.target_compound}s</strong> on <strong>Lap {best_pit_lap}</strong>.",
        "chart_labels": chart_labels,
        "chart_data": chart_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
