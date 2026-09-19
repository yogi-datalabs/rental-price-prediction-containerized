# Rental Price Prediction — Containerization Demo

This project evolves the original single-process Streamlit demo into two independently containerized services:

- **Frontend:** Streamlit on port `8501`
- **Backend:** Flask REST API on port `7860`
- **Model:** The existing serialized scikit-learn/XGBoost pipeline

The frontend calls the backend using Docker's internal DNS name: `http://backend:7860`.

## Architecture

```text
Browser -> Streamlit container -> Flask API container -> Joblib model
```

## Run with Docker Compose

```bash
docker compose up --build
```

Open port `8501` for the Streamlit interface. Test the backend separately:

```bash
curl http://localhost:7860/health
```

Stop and remove the containers:

```bash
docker compose down
```

## Manual Docker network demonstration

```bash
docker network create rental-app-network

docker build -t rental-backend:v1 ./backend
docker run -d --name backend --network rental-app-network -p 7860:7860 rental-backend:v1

docker build -t rental-frontend:v1 ./frontend
docker run -d --name frontend --network rental-app-network -p 8501:8501 \
  -e BACKEND_URL=http://backend:7860 rental-frontend:v1
```

Inspect the running system:

```bash
docker images
docker ps
docker network inspect rental-app-network
docker logs backend
docker logs frontend
```

Clean up after the demonstration:

```bash
docker stop frontend backend
docker rm frontend backend
docker network rm rental-app-network
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Verify that the API and model are available |
| `POST` | `/v1/rental` | Predict one rental price from JSON |
| `POST` | `/v1/rentalbatch` | Predict multiple prices from a CSV upload |

Example request:

```bash
curl -X POST http://localhost:7860/v1/rental \
  -H "Content-Type: application/json" \
  -d '{
    "room_type": "Entire home/apt",
    "accommodates": 5,
    "bathrooms": 3,
    "cancellation_policy": "strict",
    "cleaning_fee": true,
    "instant_bookable": "f",
    "review_scores_rating": 90,
    "bedrooms": 3,
    "beds": 3
  }'
```

Batch request:

```bash
curl -X POST http://localhost:7860/v1/rentalbatch \
  -F "file=@sample_batch.csv"
```
