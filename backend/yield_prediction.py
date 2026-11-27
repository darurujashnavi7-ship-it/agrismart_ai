from pathlib import Path
import joblib, pandas as pd
MODEL_PATH=Path(__file__).resolve().parents[1]/"models"/"yield_model.joblib"
def predict_yield(state,crop,season,year,area_ha,rainfall_mm,fertilizer_kg_ha=100,pesticide_kg_ha=1):
    if not MODEL_PATH.exists(): raise FileNotFoundError("Yield model is not trained. Run the Colab notebook and copy yield_model.joblib into models/.")
    bundle=joblib.load(MODEL_PATH); model=bundle["model"]; features=bundle["features"]
    row=pd.DataFrame([{"Year":int(year),"State":state,"Crop":crop,"Season":season,"Area":float(area_ha),"Annual_Rainfall":float(rainfall_mm),"Fertilizer":float(fertilizer_kg_ha),"Pesticide":float(pesticide_kg_ha)}])
    pred=max(0.0,float(model.predict(row[features])[0]))
    return {"yield_t_ha":pred,"production_tonnes":pred*float(area_ha)}
