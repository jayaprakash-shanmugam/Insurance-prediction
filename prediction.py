import numpy as np
import pickle

loaded_model=pickle.load(open('trained_model.sav','rb'))

input_data = (25)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
