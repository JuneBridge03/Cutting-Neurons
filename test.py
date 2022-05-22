from keras.models import load_model
import json

model = load_model('7_trex_ann_model.h5')

X = []
Y = []

with open('learning_data_10 00 03.json', 'r') as f:
        data = json.load(f)
        X = data[0]
        Y = data[1]
        f.close()

results = model.evaluate([X], [Y], batch_size=128)
print(results)