import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import  MeanSquaredError

import pandas as pd
import mlflow

csv_data = pd.read_csv('./data.txt',sep=' ', header=None)

model =keras.Sequential([
    Dense(units=5, input_shape=(1,), activation='relu'),
    Dense(units=10, activation='sigmoid'),
    Dense(units=10, activation='sigmoid'),
    Dense(units=10, activation='relu'),
    Dense(units=1, activation='relu'),
])


model.summary()

# Training

learning_rate = 1e-3
batch_size = 200
epochs = 500

model.compile(
    optimizer='adam', 
    loss='mean_squared_error'
    # metrics=['loss']
)

N = len(csv_data)
num_train = int(0.7*N)

x_train = csv_data.iloc[0:num_train,0]
y_train = csv_data.iloc[0:num_train,1]

history = model.fit(x=x_train,y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True).history


print(model.predict(tf.constant([3]))[0][0])

model.save('./production-model')

print(mlflow.tracking.set_tracking_uri('http://localhost:5000'))
mlflow.set_experiment('teste')

with mlflow.start_run():
    for loss in history.get('loss'):
        mlflow.log_metric('loss', loss)
    mlflow.tensorflow.log_model(tf_saved_model_dir='./production-model', tf_meta_graph_tags='serve', tf_signature_def_key ='serving_default', artifact_path='production-model',registered_model_name='production-model')

