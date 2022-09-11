import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("./color_model_trunc_unique6.h5")

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
    ii = [float(x) for x in request.form.values()]
    if ii[0] == ii[1] == ii[2]:
        return render_template('result.html',prediction_text='Purity should be : {} %'.format(0))
    elif ii.count(0) in [1,2]:
        return render_template('result.html',prediction_text='Purity should be : {} %'.format(100))
    elif all(ii):
        return render_template('result.html',prediction_text='Purity should be : {} %'.format(model.predict([np.array(sorted([x/255.0 for x in ii])).reshape(1,3)])*100))
    
if __name__ == '__main__':
    app.run(debug=True)
