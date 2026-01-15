# 🥔 Potato Disease Prediction

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Railway-green? style=for-the-badge&logo=railway)](https://flask-frontend-production-cd56.up.railway.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Serving-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/tfx/guide/serving)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)

A deep learning-powered web application that detects diseases in potato leaves.  Upload an image of a potato leaf and get instant predictions for **Early Blight**, **Late Blight**, or **Healthy** status. 

🔗 **Live Demo:** [https://flask-frontend-production-cd56.up.railway.app/](https://flask-frontend-production-cd56.up.railway.app/)

---

## 📸 Screenshots

### Home Page
![Home Page](\images_readme\image_mainpage.png)

### Prediction Result
![Result Page](\images_readme\image_prediction.png)

---

## ✨ Features

- 🔍 **AI-Powered Detection** - Uses deep learning model trained on potato leaf images
- 🎯 **High Accuracy** - Reliable predictions with confidence scores
- 📤 **Drag & Drop Upload** - Easy image upload with preview
- 📱 **Responsive Design** - Works on desktop and mobile devices
- ⚡ **Fast Processing** - Get results in seconds
- 🎨 **Modern UI** - Clean and intuitive user interface

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Flask        │     │    FastAPI      │     │  TensorFlow     │
│   Frontend     │────▶│    Backend      │────▶│   Serving       │
│   (Port 8080)  │     │   (Port 8000)   │     │  (Port 8501)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
       │                       │                       │
       └───────────────────────┴───────────────────────┘
                    Railway Private Network
```

| Service | Technology | Purpose |
|---------|------------|---------|
| **Frontend** | Flask + Jinja2 | Web interface for users |
| **API** | FastAPI | Image processing & API endpoints |
| **ML Model** | TensorFlow Serving | Deep learning inference |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker (for TensorFlow Serving)
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/rahulkate173/potato-disease-prediction.git
   cd potato-disease-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install uv
   uv sync
   ```

3. **Start TensorFlow Serving (Docker)**
   ```bash
   docker run -p 8501:8501 \
     --mount type=bind,source=$(pwd)/models,target=/models \
     -e MODEL_NAME=disease_models \
     tensorflow/serving
   ```

4. **Start FastAPI backend**
   ```bash
   uv run uvicorn api. main:app --host 0.0.0.0 --port 8000
   ```

5. **Start Flask frontend**
   ```bash
   uv run python app.py
   ```

6. **Open your browser**
   ```
   http://localhost:8080
   ```

---

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  flask-frontend:
    build: 
      context: . 
      dockerfile: Dockerfile. flask
    ports:
      - "8080:8080"
    environment:
      - FAST_API_URL=http://fastapi: 8000/predict
    depends_on:
      - fastapi

  fastapi:
    build: 
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - TF_SERVING_URL=http://tfserving:8501/v1/models/disease_models: predict
    depends_on: 
      - tfserving

  tfserving:
    build: 
      context: .
      dockerfile: Dockerfile.tfserving
    ports:
      - "8501:8501"
```

```bash
docker-compose up --build
```

---

## ☁️ Railway Deployment

This project is deployed on [Railway](https://railway.app) with three services:

### Environment Variables

| Service | Variable | Value |
|---------|----------|-------|
| flask-frontend | `FAST_API_URL` | `http://<fastapi-service>. railway.internal: 8000/predict` |
| fastapi | `TF_SERVING_URL` | `http://<tfserving-service>.railway. internal:8501/v1/models/disease_models:predict` |

### Deployment Steps

1. Create a new project on Railway
2. Add three services from the same GitHub repo
3. Configure each service: 
   - **flask-frontend**: Set Dockerfile to `Dockerfile.flask`
   - **fastapi**: Set Dockerfile to `Dockerfile.api`
   - **tfserving**: Set Dockerfile to `Dockerfile.tfserving`
4. Add environment variables (use Railway's private networking)
5. Generate a public domain for flask-frontend

---

## 📁 Project Structure

```
potato-disease-prediction/
├── 📂 api/
│   └── main.py              # FastAPI backend
├── 📂 models/
│   └── 1/                   # TensorFlow SavedModel
│       ├── saved_model.pb
│       └── variables/
├── 📂 static/
│   ├── css/
│   │   └── style.css        # Stylesheet
│   ├── js/
│   │   └── script.js        # JavaScript
│   └── uploads/             # Uploaded images
├── 📂 templates/
│   ├── index.html           # Home page
│   └── result.html          # Result page
├── app.py                   # Flask application
├── Dockerfile.flask         # Flask container
├── Dockerfile.api           # FastAPI container
├── Dockerfile.tfserving     # TF Serving container
├── pyproject.toml           # Python dependencies
├── uv.lock                  # Lock file
└── README.md
```

---

## 🧠 Model Information

| Property | Value |
|----------|-------|
| **Model Type** | Convolutional Neural Network (CNN) |
| **Framework** | TensorFlow/Keras |
| **Input Size** | Variable (RGB images) |
| **Classes** | 3 (Early Blight, Late Blight, Healthy) |
| **Serving** | TensorFlow Serving (REST API) |

### Disease Classes

| Class | Description |
|-------|-------------|
| 🟢 **Healthy** | No disease detected |
| 🟡 **Early Blight** | Caused by *Alternaria solani* fungus |
| 🔴 **Late Blight** | Caused by *Phytophthora infestans* |

---

## 🛠️ API Endpoints

### FastAPI Backend

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/ping` | GET | Health check |
| `/predict` | POST | Predict disease from image |
| `/docs` | GET | Swagger documentation |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@potato_leaf.jpg"
```

### Example Response

```json
{
  "class_name": "Early Blight",
  "confidence": 95.67
}
```

---

## 🤝 Contributing

Contributions are welcome!  Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Rahul Kate**

- GitHub: [@rahulkate173](https://github.com/rahulkate173)

---

## 🙏 Acknowledgments

- [TensorFlow](https://www.tensorflow.org/) for the ML framework
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Railway](https://railway.app/) for hosting
- [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) for training data

---

<p align="center">
  Made with ❤️ by Rahul Kate
</p>

<p align="center">
  <a href="https://flask-frontend-production-cd56.up.railway.app/">
    <img src="https://img.shields.io/badge/🥔_Try_Live_Demo-Click_Here-success?style=for-the-badge" alt="Live Demo">
  </a>
</p>