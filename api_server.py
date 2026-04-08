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
    """The frontend pings this every 5 seconds to confirm the AI is online."""
    return {"status": "online"}

# --- 2. Global AI Assets ---
model = None
scaler = None
df = None
features = []
one_hot_features = []

@app.on_event("startup")
def load_ai_assets():
    """Loads the heavy AI files into RAM only once when the server starts."""
    global model, scaler, df, features, one_hot_features
    print("--- Booting Virtual Pit Wall Engine ---")
    try:
        model = load_model("Virtual_Pit_Wall_Final.keras")
        scaler = joblib.load("f1_scaler.gz")
        df = pd.read_parquet("Global_F1_Master.parquet")
        
        core_features = ['Speed', 'Throttle', 'Brake', 'RPM', 'TyreLife', 'LapNumber']
        one_hot_features = [col for col in df.columns if 'Driver_' in col or 'Compound_' in col]
        features = core_features + one_hot_features
        print("✅ AI Brain Loaded. Waiting for frontend requests...")
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
    
    # --- SAFETY NET ---
    # Check if the requested driver actually exists in your Parquet file
    if driver_focus not in df.columns:
        print(f"⚠️ Warning: {driver_focus} not found. Defaulting to first available driver.")
        available_drivers = [col for col in df.columns if col.startswith('Driver_')]
        if available_drivers:
            driver_focus = available_drivers[0]
        else:
            return {"status": "error", "message": "Critical Error: No driver data found in dataset."}
    # ----------------------

    # Grab a baseline flat-out sequence from our dataset
    baseline_data = df[(df[driver_focus] == 1) & (df['Throttle'] >= 90)].head(50)[features].copy()

    def predict_lap_time(compound, tyre_life, lap_num):
        """Asks the Keras AI to predict speed based on tire age, then converts to lap time."""
        test_data = baseline_data.copy()
        test_data['TyreLife'] = tyre_life
        test_data['LapNumber'] = lap_num
        
        for col in one_hot_features:
            if 'Compound_' in col:
                test_data[col] = 0
        test_data[f'Compound_{compound}'] = 1

        scaled_input = scaler.transform(test_data.values).reshape(1, 50, len(features))
        scaled_speed_pred = model.predict(scaled_input, verbose=0)[0][0]
        
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = scaled_speed_pred
        predicted_speed_kmh = scaler.inverse_transform(dummy)[0, 0]
        
        # Physics conversion: Faster speed = Lower Lap Time
        return 85.0 + (100 - predicted_speed_kmh) * 0.1

    strategy_results = []
    pit_penalty = 24.0

    # The AI simulates every possible pit window from the NEXT lap until the end
    possible_pit_laps = range(req.current_lap + 1, req.total_laps + 1)
    
    for pit_lap in possible_pit_laps:
        total_race_time = 0
        sim_tyre_life = req.current_tyre_age
        
        for lap in range(req.current_lap, req.total_laps + 1):
            if lap < pit_lap:
                # Pre-Pit: Suffer on dying tires
                total_race_time += predict_lap_time(req.current_compound, sim_tyre_life, lap)
                sim_tyre_life += 1
            elif lap == pit_lap:
                # THE PIT STOP: Take the 24s penalty
                total_race_time += predict_lap_time(req.target_compound, 0, lap) + pit_penalty
                sim_tyre_life = 1
            else:
                # Post-Pit: Push on fresh tires
                total_race_time += predict_lap_time(req.target_compound, sim_tyre_life, lap)
                sim_tyre_life += 1
                
        strategy_results.append((pit_lap, total_race_time))

    # --- 5. Find the Mathematical Winner ---
    optimal_strategy = min(strategy_results, key=lambda x: x[1])
    best_pit_lap = optimal_strategy[0]
    
    # Format the data so the frontend Chart.js can draw the graph
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