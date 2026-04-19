from fastapi import FastAPI
from api.schema import SalesRequest
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from api.recommender import BookRecommender 
import os

# -------------------------
# Spark + modelo
# -------------------------

spark = SparkSession.builder \
    .appName("penguin-api") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

path_actual = os.path.dirname(os.path.abspath(__file__))
path_model = os.path.join(path_actual, "..", "model", "sales_model")

model = PipelineModel.load(path_model)

# -------------------------
# Recomendador
# -------------------------

path_text = os.path.join(path_actual, "..", "data", "raw", "texts.json")
recommender = BookRecommender(path_text)

# -------------------------
# API
# -------------------------

app = FastAPI()

# -------------------------
# Endpoint predicción
# -------------------------

@app.post("/predict")
def predict(request: SalesRequest):

    # validación manual
    if not (0 <= request.discount_pct <= 1):
        return {"error": "discount_pct must be between 0 and 1"}

    # convertir input en DataFrame
    data = [request.model_dump()]
    df = spark.createDataFrame(data)

    # predicción
    prediction = model.transform(df)

    result = prediction.select("prediction").collect()[0][0]

    return {"prediction": float(result)}

# -------------------------
# Endpoint recomendación
# -------------------------

@app.get("/recommend")
def recommend(isbn: str, top_k: int = 5):

    results = recommender.recommend(isbn, top_k)

    if not results:
        return {"error": "ISBN not found"}

    return {
        "isbn": isbn,
        "recommendations": results
    }