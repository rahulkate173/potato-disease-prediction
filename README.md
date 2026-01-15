# 🥔 Potato Disease Prediction

A deep learning-based web application that predicts potato plant diseases using image classification.  The system can identify whether a potato plant is **Healthy**, has **Early Blight**, or **Late Blight**. 

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Information](#model-information)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project uses a trained TensorFlow model to classify potato leaf images into three categories:
- **Early Blight** - A fungal disease caused by *Alternaria solani*
- **Late Blight** - A disease caused by *Phytophthora infestans*
- **Healthy** - No disease detected

The application consists of a Flask frontend for user interaction and a FastAPI backend that communicates with TensorFlow Serving for model inference. 

## ✨ Features

- 📷 Upload potato leaf images for disease prediction
- 🔮 Real-time disease classification
- 📊 Confidence score for predictions
- 🎨 Clean and intuitive user interface
- 🚀 Fast inference using TensorFlow Serving

## 🛠 Tech Stack

- **Frontend**: Flask, HTML, CSS
- **Backend API**: FastAPI, Uvicorn
- **Machine Learning**: TensorFlow, TensorFlow Serving
- **Data Processing**: NumPy, Pandas, Pillow
- **Package Manager**: UV
- **Python**:  3.12+

## 📁 Project Structure

```
potato-disease-prediction/
├── api/
│   └── main.py              # FastAPI backend for predictions
├── models/                   # Trained TensorFlow models
├── notebooks/                # Jupyter notebooks for training
├── PlantVillage/            # Dataset directory
├── static/
│   ├── css/                 # Stylesheets
│   ├── images/              # Static images
│   └── uploads/             # Uploaded images storage
├── templates/
│   ├── index.html           # Upload page
│   └── result.html          # Prediction results page
├── app.py                   # Flask web application
├── model. config             # TensorFlow Serving configuration
├── pyproject.toml           # Project dependencies
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- Docker (for TensorFlow Serving)
- UV package manager (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/rahulkate173/potato-disease-prediction.git
   cd potato-disease-prediction
   ```

2. **Install dependencies using UV**
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start TensorFlow Serving with Docker**
   ```bash
   docker pull tensorflow/serving
   docker run -t --rm -p 8051:8051 \
     -v /path/to/potato-disease-prediction:/disease-prediction \
     tensorflow/serving \
     --rest_api_port=8051 \
     --model_config_file=/disease-prediction/model. config
   ```

## 💻 Usage

1. **Start the FastAPI backend** (handles model predictions)
   ```bash
   cd api
   python main.py
   ```
   The API will be available at `http://localhost:8000`

2. **Start the Flask frontend** (in a new terminal)
   ```bash
   python app.py
   ```
   The web application will be available at `http://localhost:8080`

3. **Make predictions**
   - Open your browser and navigate to `http://localhost:8080`
   - Upload an image of a potato leaf
   - Click "Predict" to get the disease classification

## 🔌 API Endpoints

### FastAPI Backend (`http://localhost:8000`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check endpoint |
| `/predict` | POST | Upload image and get prediction |

### Prediction Response

```json
{
  "class_name": "Early Bright",
  "confidence": 95.5
}
```

## 🧠 Model Information

The model is trained on the **PlantVillage** dataset and can classify potato leaves into three categories:

| Class | Description |
|-------|-------------|
| Early Blight | Fungal disease with dark spots and concentric rings |
| Late Blight | Water-soaked lesions that turn brown |
| Healthy | No disease symptoms |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ by <a href="https://github.com/rahulkate173">rahulkate173</a></p>
