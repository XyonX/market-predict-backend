from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import boto3
import os
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import tflite_runtime.interpreter as tflite  # Changed to tflite_runtime

app = FastAPI()

load_dotenv()

# Load R2 credentials from .env
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ENDPOINT = os.getenv("R2_ENDPOINT")

# Boto3 S3 client for Cloudflare R2
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY
)

# Input schema
class StockRequest(BaseModel):
    symbol: str

def download_model(symbol: str):
    model_path = f"model_cache/{symbol}.tflite"
    if not os.path.exists(model_path):
        try:
            s3.download_file(R2_BUCKET, f"{symbol}.tflite", model_path)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model for '{symbol}' not found.")
    return model_path

def DownloadData(ticker : str):

    window_size = 20 # This should match the window_size used during training

    end_date_inference = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date_inference = (pd.Timestamp.now() - pd.Timedelta(days=window_size + 10)).strftime('%Y-%m-%d') # Fetch a few extra days
    try:
        # Fetch the data
        inference_data = yf.download(ticker, start=start_date_inference, end=end_date_inference)

        # Access the Close prices and get the last 'window_size' values
        # Ensure there are enough data points
        if len(inference_data['Close']) < window_size:
            print(f"Not enough data points ({len(inference_data)}) to form a window of size {window_size}.")
            # Exit or handle the error
            exit()

        # Get the last 'window_size' closing prices
        raw_inference_window = inference_data['Close'][ticker].tail(window_size).tolist()
        # raw_inference_window = raw_inference_window[:-1]
        # Get the 20 closing prices ending **yesterday**, not today
        # raw_inference_window = inference_data['Close'][ticker].iloc[-(window_size + 1):-1].tolist()
        # raw_inference_window = inference_data['Close'][ticker].iloc[-window_size-1:-1].tolist()

        # Prepare the inference data in the same format as your training data (normalized)
        # Apply the same normalization logic as in your make_dataset function
        if raw_inference_window: # Check if the list is not empty
            first_price = raw_inference_window[0]
            inference_window_normalized = [p / first_price for p in raw_inference_window]
        else:
            print("Could not get enough data to create an inference window.")
            exit()

        return raw_inference_window
        # Convert the normalized window to a NumPy array and reshape for the model
        # The model expects an input shape like (batch_size, window_size).
        # For a single prediction, batch_size is 1.
        # inference_input = np.array(inference_window_normalized).reshape(1, window_size)
        # return inference_input

    except Exception as e:
        print(f"Error during data fetching or prediction: {e}")

@app.post("/predict")
def predict(request: StockRequest):
    window_size = 20
    model_path = download_model(request.symbol)
    raw_data = DownloadData(request.symbol)
    
    # Normalize data
    first_price = raw_data[0]
    normalized_data = [p / first_price for p in raw_data]
    input_array = np.array(normalized_data, dtype=np.float32).reshape(1, window_size)
    
    # TFLite inference
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # Process results
    current_price = raw_data[-1]
    predicted_change = float(prediction[0][0])
    predicted_price = current_price * (1 + predicted_change)
    
    return {
        "symbol": request.symbol,
        "current_price": current_price,
        "predicted_price": round(predicted_price, 2),
        "prediction_date": pd.Timestamp.now().strftime('%Y-%m-%d'),
        "price_change": round(predicted_price - current_price, 2),
        "price_change_percent": round(predicted_change * 100, 2)
    }