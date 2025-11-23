from flask import render_template,redirect,url_for,Flask,request
import uvicorn
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
    file.save(filepath) 
    ### tensorflow pipeline 
    confidence = 99.99
    pred_class = 'Healthy'
    return render_template(
        "result.html",
        image_url=url_for('static',filename=f"uploads/{file.filename}"),
        pred_class=pred_class,
        confidence=confidence
    )  

if __name__ == '__main__':
    app.run(debug=True,port=8080,host='localhost')