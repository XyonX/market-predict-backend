from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import boto3
import os
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf

app = FastAPI()

load_dotenv()


# Load R2 credentials from .env
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ENDPOINT = os.getenv("R2_ENDPOINT")  # custom R2 endpoint URL


# Boto3 S3 client for Cloudflare R2
s3 = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY
)


# Input schema
class StockRequest(BaseModel):
    symbol: str          # e.g. "AAPL"
    features: list[float]

def download_model(symbol: str):
    model_path = f"model_cache/{symbol}.keras"
    if not os.path.exists(model_path):
        try:
            s3.download_file(R2_BUCKET, f"{symbol}.keras", model_path)
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
def predict(data: StockRequest):
    window_size = 20
    model_path = download_model(data.symbol)
    model = tf.keras.models.load_model(model_path)
    raw_data= DownloadData(data.symbol)
    first_price = raw_data[0]
    inference_window_normalized = [p / first_price for p in raw_data]
    input_array = np.array(inference_window_normalized).reshape(1, window_size)
    # Make a prediction
    prediction = model.predict(input_array)

    print("Prediiction: ",prediction)

    # last_price_in_window = raw_data[-1]
    # predicted_price_change_ratio = prediction[0][0]
    # predicted_next_price = last_price_in_window * (1 + predicted_price_change_ratio)
    last_price_in_window = float(raw_data[-1])
    predicted_price_change_ratio = float(prediction[0][0])
    predicted_next_price = last_price_in_window * (1 + predicted_price_change_ratio)


    return {
        "symbol":data.symbol,
        "current_price": last_price_in_window,
        "predicted_price": predicted_next_price
    }

