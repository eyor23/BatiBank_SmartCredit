from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression  # Ensure sklearn is imported

app = FastAPI()

# Load the trained model
try:
    with open(r'C:\Users\user\Desktop\BatiBank_SmartCredit\notebooks\logistic_model.pkl', "rb") as f:
        model = pickle.load(f)

    # Check if the model is a valid scikit-learn estimator
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model is not a valid classifier. Please check the saved file.")

except FileNotFoundError:
    raise RuntimeError("Model file not found. Please train and save the model first.")

# Define input data model
class InputData(BaseModel):
    Amount_woe: float
    Total_Transaction_Amount_woe: float

# Define prediction response model
class PredictionResponse(BaseModel):
    prediction: int  # 0 for Good, 1 for Bad
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: InputData):
    """Endpoint for making predictions using the loaded model."""
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Make sure the model is still valid
        if not hasattr(model, "predict_proba"):
            raise HTTPException(status_code=500, detail="Invalid model type. Please reload the correct model.")

        # Make prediction
        probability = model.predict_proba(input_df)[:, 1][0]
        prediction = int(model.predict(input_df)[0])

        return {"prediction": prediction, "probability": probability}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
