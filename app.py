import uvicorn
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)

class DiabetesData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

app = FastAPI()

# Load the pre-trained model from joblib file
try:
    classifier = joblib.load("trained_model.joblib")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed. Please check the model file.")

@app.get('/')
def index():
    return {'message': 'Welcome to the Diabetes Prediction API'}

@app.post('/predict')
def predict_diabetes(data: DiabetesData):
    try:
        input_data = [
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.SkinThickness,
            data.Insulin,
            data.BMI,
            data.DiabetesPedigreeFunction,
            data.Age
        ]

        # Make prediction using the model
        prediction = classifier.predict([input_data])

        # Interpret the prediction result
        result = 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'

        return {
            'input_data': data.dict(),
            'prediction': result
        }

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Please check the input data.")

# Load host and port from environment variables (or default)
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
