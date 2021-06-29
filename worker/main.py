import tensorflow as tf
from tensorflow import keras
from flask import Flask, request
import os

app = Flask(__name__)

@app.route("/artifacts/<num>/<id>/<value>")
def hello(num, id, value):
    __dirname = os.getcwd()
    model_path =  __dirname + f'/../exemplo/artifacts/{num}/{id}/artifacts/production-model/tfmodel'
    model = tf.keras.models.load_model(model_path)
    model.summary()
    ypred = float(model.predict(tf.constant([float(value)]))[0][0])
    return {'ypred': ypred}

if __name__ == "__main__":
  app.run(port=8090)