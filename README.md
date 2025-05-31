# Stock Prediction SaaS Backend

This is the backend service for the Stock Prediction SaaS platform, providing AI-powered stock price predictions and trend analysis.

## Features

- Next-day stock price predictions with confidence levels
- 30-day trend predictions with detailed analysis
- Real-time stock data integration
- Machine learning model management
- Historical data analysis
- RESTful API endpoints
- Caching system for optimal performance

## Tech Stack

- **Framework:** FastAPI
- **Language:** Python 3.9+
- **Database:** MongoDB
- **Caching:** Redis
- **ML Libraries:**
  - pandas
  - scikit-learn
  - TensorFlow
  - numpy
  - yfinance
- **Development Tools:**
  - Docker
  - Poetry (dependency management)
  - pytest (testing)
  - black (code formatting)
  - flake8 (linting)

## Prerequisites

- Python 3.9 or higher
- MongoDB 5.0 or higher
- Redis 6.0 or higher
- Docker and Docker Compose (optional)

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── stocks.py
│   │   │   │   ├── predictions.py
│   │   │   │   └── models.py
│   │   │   └── router.py
│   │   └── deps.py
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── db/
│   │   ├── mongodb.py
│   │   └── redis.py
│   ├── models/
│   │   ├── stock.py
│   │   ├── prediction.py
│   │   └── ml_model.py
│   ├── services/
│   │   ├── stock_service.py
│   │   ├── prediction_service.py
│   │   └── ml_service.py
│   └── main.py
├── ml/
│   ├── models/
│   │   ├── next_day_predictor.py
│   │   └── trend_predictor.py
│   ├── training/
│   │   └── trainer.py
│   └── utils/
│       └── feature_engineering.py
├── tests/
│   ├── api/
│   ├── services/
│   └── ml/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Setup and Installation

### Local Development

1. Clone the repository:

```bash
git clone <repository-url>
cd backend
```

2. Install Poetry (if not already installed):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:

```bash
poetry install
```

4. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start MongoDB and Redis:

```bash
# Using Docker
docker-compose up -d mongodb redis
```

6. Run the application:

```bash
poetry run uvicorn app.main:app --reload
```

### Docker Deployment

1. Build and run using Docker Compose:

```bash
docker-compose up --build
```

## Environment Variables

```env
# Application
APP_ENV=development
APP_PORT=8000
DEBUG=True

# MongoDB
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=stock_prediction

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Security
SECRET_KEY=your-secret-key
API_KEY_HEADER=X-API-Key

# External Services
YAHOO_FINANCE_API_KEY=your-api-key
```

## API Documentation

Once the application is running, you can access:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Main Endpoints

#### Stock Data

- `GET /api/v1/stocks/search` - Search stocks by ticker or name
- `GET /api/v1/stocks/{ticker}/info` - Get stock information
- `GET /api/v1/stocks/{ticker}/historical` - Get historical data

#### Predictions

- `POST /api/v1/predictions/next-day` - Get next-day prediction
- `GET /api/v1/predictions/{ticker}/thirty-day` - Get 30-day trend prediction

#### Model Management

- `GET /api/v1/models/{ticker}/status` - Check model status
- `POST /api/v1/models/{ticker}/train` - Train new model
- `GET /api/v1/models/{ticker}/training-status/{job_id}` - Check training status

## Testing

Run tests using pytest:

```bash
poetry run pytest
```

Run tests with coverage:

```bash
poetry run pytest --cov=app tests/
```

## Development Guidelines

1. **Code Style**

   - Use Black for code formatting
   - Follow PEP 8 guidelines
   - Use type hints

2. **Git Workflow**

   - Create feature branches from `develop`
   - Use conventional commits
   - Submit PRs for review

3. **Testing**
   - Write unit tests for new features
   - Maintain minimum 80% coverage
   - Include integration tests for APIs

## Monitoring and Logging

The application uses:

- Prometheus for metrics
- ELK Stack for log aggregation
- Sentry for error tracking

Access monitoring dashboards:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Performance Considerations

1. **Caching Strategy**

   - Stock data: 5-minute TTL
   - Predictions: 1-hour TTL
   - Model metadata: 24-hour TTL

2. **Database Indexing**

   - Indexed fields: ticker, prediction_date
   - Compound indexes for common queries

3. **Rate Limiting**
   - 100 requests per minute per IP
   - 1000 requests per day per API key

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please contact:

- Email: support@stockprediction.com
- Issues: GitHub Issues page
