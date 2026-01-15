from flask import render_template, redirect, url_for, Flask, request
import requests
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# FastAPI URL - Use environment variable for production
FAST_API_URL = os.environ.get('FAST_API_URL', 'http://localhost:8000/predict')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST': 
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        with open(filepath, "rb") as f:
            files = {"file": (file. filename, f.read(), file.mimetype)}
        
        dict_pred = requests.post(FAST_API_URL, files=files).json()
        confidence = dict_pred['confidence']
        pred_class = dict_pred['class_name']
        
        return render_template(
            "result.html",
            image_url=url_for('static', filename=f"uploads/{file. filename}"),
            pred_class=pred_class,
            confidence=confidence
        )
    
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    file = request.files['file']
    filepath = os.path. join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    with open(filepath, "rb") as f:
        files = {"file": (file.filename, f.read(), file.mimetype)}
    
    dict_pred = requests.post(FAST_API_URL, files=files).json()
    confidence = dict_pred['confidence']
    pred_class = dict_pred['class_name']
    
    return render_template(
        "result.html",
        image_url=url_for('static', filename=f"uploads/{file.filename}"),
        pred_class=pred_class,
        confidence=confidence
    )

if __name__ == '__main__': 
    port = int(os. environ.get('PORT', 8080))
    app.run(debug=False, port=port, host='0.0.0.0')