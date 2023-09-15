import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def insurance_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    return prediction

  
def main():
    st.title('insurance Prediction Web App')
    Age = st.text_input('Age of the Person')
    diagnosis = ''
    if st.button('Test Result'):
        diagnosis = insurance_prediction([int(Age)])
    st.success(diagnosis)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
  
    
  
