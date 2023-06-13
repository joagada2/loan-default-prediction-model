from pydoc import locate
from typing import List
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("model1/loan_default_pred_model_depl.pkl")

def create_type_instance(type_name: str):
    return locate(type_name).__call__()

def get_features_dict(model):
    feature_names = model.get_booster().feature_names
    feature_types = list(map(create_type_instance, model.get_booster().feature_types))
    return dict(zip(feature_names, feature_types))

def create_input_features_class(model):
    return type("InputFeatures", (BaseModel,), get_features_dict(model))

InputFeatures = create_input_features_class(model)
app = FastAPI()

@app.post("/predict", response_model=List)
async def predict_post(datas: List[InputFeatures]):
    return model.predict(np.asarray([list(data.__dict__.values()) for data in datas])).tolist()

if __name__ == "__main__":
    print(get_features_dict(model))
    uvicorn.run(app, host="0.0.0.0", port=8080)

# reference: https://medium.com/predictly-on-tech/serve-any-xgboost-model-with-fastapi-in-less-than-40-lines-85adbb3c43b8