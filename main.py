from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import boto3
import os
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import tflite_runtime.interpreter as tflite
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    os.makedirs("model_cache", exist_ok=True)
    
    if not os.path.exists(model_path):
        try:
            logger.info(f"Downloading model for {symbol}")
            s3.download_file(R2_BUCKET, f"{symbol}.tflite", model_path)
            logger.info(f"Model downloaded to {model_path}")
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Model for '{symbol}' not found: {str(e)}"
            )
    return model_path

# def download_data(symbol: str):
#     window_size = 20
#     end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
#     start_date = (pd.Timestamp.now() - pd.Timedelta(days=window_size + 10)).strftime('%Y-%m-%d')
    
#     try:
#         data = yf.download(symbol, start=start_date, end=end_date)
        
#         if len(data) < window_size:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Insufficient data for {symbol}. Only {len(data)} days available."
#             )
            
#         closes = data['Close'].tail(window_size + 1).tolist()
#         return closes[:-1]  # Exclude today's price
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Data download failed: {str(e)}"
#         )

def download_data(symbol: str):
    # Normalize symbol to uppercase
    symbol = symbol.upper()
    window_size = 20
    # Set start date to 100 days ago to ensure enough trading days
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=100)).strftime('%Y-%m-%d')
    
    try:
        # Download data using yfinance
        data = yf.download(symbol, start=start_date, end=end_date)
        
        # Check if enough data is available
        if len(data) < window_size:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {symbol}. Only {len(data)} days available."
            )
        
        # Return the last 20 closing prices
        
        closes = data ['Close'][symbol].tail(window_size).tolist()
        return closes
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data download failed: {str(e)}"
        )

@app.post("/predict")
def predict(request: StockRequest):
    window_size = 20
    model_path = download_model(request.symbol)
    raw_data = download_data(request.symbol)
    
    # Normalize data
    first_price = raw_data[0]
    normalized_data = [p / first_price for p in raw_data]
    input_array = np.array(normalized_data, dtype=np.float32).reshape(1, window_size)
    
    try:
        # Load TFLite model
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Validate input shape
        input_shape = input_details[0]['shape']
        if len(input_shape) != 2 or input_shape[0] != 1 or input_shape[1] != window_size:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected model input shape: {input_shape}"
            )
        
        # Validate output shape (assuming single scalar output)
        output_shape = output_details[0]['shape']
        if len(output_shape) != 2 or output_shape[0] != 1 or output_shape[1] != 1:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected model output shape: {output_shape}"
            )
        
        # Set input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        
        # Get prediction
        prediction = interpreter.get_tensor(output_details[0]['index'])
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"TFLite inference failed: {str(e)}"
        )
    
    # Process results
    current_price = raw_data[-1]
    predicted_change = float(prediction[0][0])
    predicted_price = current_price * (1 + predicted_change)
    
    logger.info(f"Prediction made for {request.symbol}: {predicted_price}")
    
    return {
        "symbol": request.symbol,
        "current_price": current_price,
        "predicted_price": round(predicted_price, 2),
        "prediction_date": pd.Timestamp.now().strftime('%Y-%m-%d'),
        "price_change": round(predicted_price - current_price, 2),
        "price_change_percent": round(predicted_change * 100, 2)
    }