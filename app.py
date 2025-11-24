from flask import render_template,redirect,url_for,Flask,request
import uvicorn
import requests
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
FAST_API_URL = r"http://localhost:8000/predict"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
    file.save(filepath) 
    ### tensorflow pipeline 
    with open(filepath, "rb") as f:
        files = {"file": (file.filename, f.read(), file.mimetype)}
    dict_pred = requests.post(FAST_API_URL,files=files).json()
    confidence = dict_pred['confidence']
    pred_class = dict_pred['class_name']
    return render_template(
        "result.html",
        image_url=url_for('static',filename=f"uploads/{file.filename}"),
        pred_class=pred_class,
        confidence=confidence
    )  

if __name__ == '__main__':
    app.run(debug=True,port=8080,host='localhost')


### what i had in pipeline 
### app -> localhost:8080
### pipeline -> rest_api_port: 8051
