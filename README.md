## Penguin Sales Forecasting & Book Recommendation API

Proyecto de machine learning que combina:

- Forecasting de ventas con PySpark
- Recomendación de libros basada en contenido (NLP)
- Exposición mediante API REST con FastAPI

---

## Objetivo

Construir un pipeline end-to-end que incluya:

* Preparación de datos
* Feature engineering (incluyendo variables temporales y lags)
* Entrenamiento de un modelo de regresión
- Sistema simple de recomendación basado en texto
* Exposición del modelo mediante una API

---

## Requisitos

* Python 3.10
* Java 17
* pip / virtualenv

---

## Instalación

```bash
git clone git@github.com:caroldaza/proyecto_penguin.git
cd proyecto_penguin

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Configuración de Java

PySpark requiere Java 17 para funcionar correctamente.

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

Verificar instalación:

```bash
java -version
```

---

## Pipeline de datos

1. Ejecución notebook de preparación de datos:

```
notebooks/01_data_preparation.ipynb
```

2. Ejecución notebook de modelado:

```
notebooks/02_modeling.ipynb
```

Esto genera el modelo entrenado en:

```
model/sales_model/
```

---

## Modelo de Forecasting

El modelo se implementa como un Pipeline de PySpark, incluyendo:

* Codificación de variables categóricas (country)
* Ensamblado de features (VectorAssembler)
* Escalado (StandardScaler)
* Modelo de regresión lineal

Esto garantiza que las transformaciones de entrenamiento se apliquen de forma consistente en inferencia.

---

## Sistema de Recomendación

Se implementa un recomendador basado en contenido utilizando:

* TF-IDF sobre:
- título
- descripción
- biografía
* Similitud coseno

El sistema también genera explicaciones simples basadas en términos compartidos entre libros.

Ejemplo de explicación:

"Comparten temas como: humor, niños, chistes"

---

## API

Levantar la API desde la raíz del proyecto:

```bash
uvicorn api.app:app --reload
```

---

## Uso

Abrir en el navegador:

```
http://127.0.0.1:8000/docs
```

Desde allí se puede interactuar con los endpoints.

---

## Endpoints

### 1. Predicción de ventas

```
POST /predict
```

---

## Ejemplo de request

```json
{
  "country": "AR",
  "marketing_spend": 5000,
  "discount_pct": 0.1,
  "stock_available": 800,
  "price": 30.5,
  "month": 6,
  "day_of_week": 3,
  "is_weekend": 0,
  "is_holiday": 0,
  "is_non_labour": 0,
  "lag_7": 4000
}
```
## Respuesta

```json
{
  "prediction": 5234.12
}
```

### 2. Recomendación de libros

```
GET /recommend
```

Parámetros:

- isbn (string): identificador del libro
- top_k (int, opcional): cantidad de recomendaciones (default=5)

---

## Ejemplo

```json
/isbn=9789875666627
```

---

## Respuesta

```json
{
  "isbn": "9789875666627",
  "recommendations": [
    {
      "ISBN": "978123...",
      "TITLE": "Otro libro",
      "score": 0.81,
      "reason": "Comparten temas como: humor, niños"
    }
  ]
}
```

## Consideraciones

* El modelo se sirve como un Pipeline completo, incluyendo preprocessing y predicción.
* Las features deben respetar el mismo esquema utilizado durante el entrenamiento.
* El feature lag_7 debe ser provisto como input al momento de inferencia.
* El recomendador es completamente independiente del modelo de ventas.
* No se utilizan modelos LLM; se emplea un enfoque clásico basado en TF-IDF.
---

## Stack tecnológico

* PySpark
* FastAPI
* Uvicorn
* Scikit-learn (soporte)

---

## Estructura del proyecto

```
project/
├── api/
│   ├── app.py
│   └── schema.py
│   └── recommender.py
├── model/
├── data/
├── notebooks/
└── requirements.txt
```

---

## Notas finales

El proyecto implementa un flujo completo de machine learning:

- Entrenamiento reproducible
- Pipeline consistente entre train e inferencia
- Exposición mediante API
- Extensión con un sistema de recomendación explicable

Se prioriza claridad, simplicidad y separación de responsabilidades. 

Esta solución representa un punto de partida que puede extenderse hacia enfoques más complejos, como modelos avanzados de NLP, la incorporación de LLMs para enriquecer el sistema de recomendación (por ejemplo, mediante embeddings o generación de explicaciones), o arquitecturas productivas escalables.

La solución del modelo de predicción y el recomendador se presentan en la misma api por facilidad pero constituyen problemas de negocio distintos.