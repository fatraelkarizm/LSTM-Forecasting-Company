from fastapi import FastAPI
import pandas as pd
from train import train_and_predict

app = FastAPI()

@app.get("/forecast")
def get_forecast():
    df = pd.read_json("outputs/forecast_output.json")
    return df.to_dict(orient="records")

@app.get("/retrain")
def retrain_model():
    df = train_and_predict()
    return {"status": "model retrained", "count": len(df)}
