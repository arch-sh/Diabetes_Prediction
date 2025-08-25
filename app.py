# import pickle
# import pandas as pd
# import streamlit as st
# from PIL import Image

# with open('diabetes','rb') as f:
#     model = pickle.load(f)

# def main():
#      img = Image.open('diab.jpg')
#      st.image(img)
#      st.title('DIABETES DIAGNOSIS')
#      col1,col2,col3 = st.columns(3)
#      with col1:
#       Age = st.text_input('AGE :')
#       Pregnancies = st.text_input('NO. OF PREGNANCIES : ')
#       Glucose = st.text_input('GLUCOSE LEVEL : ')
#      with col2:
#       BloodPressure = st.text_input('BLOOD PRESSURE : ')
#       SkinThickness = st.text_input('SKIN THICKNESS : ')
#       Insulin = st.text_input('INSULIN LEVEL : ')
#      with col3:
#       BMI = st.text_input('BMI : ')
#       DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction : ')
       

#      df= pd.DataFrame(
#         {
#             'Pregnancies':[Pregnancies],
#             'Glucose': [Glucose],
#             'BloodPressure':[BloodPressure],
#             'SkinThickness':[SkinThickness],
#             'Insulin':[Insulin],
#             'BMI':[BMI],
#             'DiabetesPedigreeFunction':[DiabetesPedigreeFunction],
#             'Age':[Age]
#         }
#      )
#      if st.button('PREDICT'):
#         result = model.predict(df)
#         if(result==1):
#             st.error('Diagnosis Result: Patient is diabetic.')
#             st.info('FOLLOWING ARE SOME HEALTHCARE ADVISES:')
#             st.info('1. Monitor Blood Sugar Levels: Regularly check your blood sugar levels as advised by your healthcare provider. Monitoring can help you understand how different foods, activities, and medications affect your blood sugar.')
#             st.info('2. Healthy Eating: Follow a balanced and nutritious diet. Focus on whole grains, lean proteins, healthy fats, and a variety of fruits and vegetables. Limit your intake of sugary foods and refined carbohydrates.')
#             st.info('3. Hydration: Stay hydrated by drinking plenty of water throughout the day. Avoid sugary beverages and excessive caffeine.')
#             st.info('4. Sleep: Prioritize getting enough restful sleep each night, as lack of sleep can affect blood sugar control.')
#             st.info('5. Blood Pressure Management: Keep your blood pressure under control, as high blood pressure can contribute to diabetes-related complications.')
#             st.info('6. Cholesterol Control: Manage your cholesterol levels through a combination of a healthy diet, exercise, and, if necessary, medication.')
#         else:
#             st.success('Patient is diabetes free.')

# if __name__ == '__main__':
#     main()



                    # NEW CODEe

# import streamlit as st
# import pickle
# import numpy as np
# import shap
# import matplotlib.pyplot as plt

# # Load model (could be plain model or pipeline)
# with open('diabetes.pkl', 'rb') as f:
#     model = pickle.load(f)

# st.title("Diabetes Risk Predictor with Explainable AI")

# # Inputs (you only take 4 from user)
# glucose = st.number_input("Glucose", 0.0, 300.0, step=1.0)
# insulin = st.number_input("Insulin", 0.0, 900.0, step=1.0)
# bmi = st.number_input("BMI", 0.0, 60.0, step=0.1)
# age = st.number_input("Age", 1, 120, step=1)

# if st.button("Predict Risk Level"):
#     # --- Build full 8-feature input for the model ---
#     # order: ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
#     input_data = np.array([[0, glucose, 0, 0, insulin, bmi, 0, age]])

#     # Predict probability
#     prob = model.predict_proba(input_data)[0][1]

#     # Risk classification
#     if prob < 0.33:
#         risk = "Low Risk"
#     elif prob < 0.66:
#         risk = "Moderate Risk"
#     else:
#         risk = "High Risk"

#     st.write(f"Predicted Risk Level: **{risk}** (Probability: {prob:.2f})")

#     # --- SHAP Explainable AI ---
#     explainer = shap.Explainer(model, input_data)   # works directly on model
#     shap_values = explainer(input_data)

#     st.subheader("🔍 Why this prediction?")
#     fig, ax = plt.subplots()
#     shap.plots.waterfall(shap_values[0], show=False)
#     st.pyplot(fig)




                    # Newwwwwwwwwewweew?

import streamlit as st
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

# Load model
with open('diabetes.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Diabetes Risk Predictor with Explainable AI")

# Inputs
glucose = st.number_input("Glucose", 0.0, 300.0, step=1.0)
insulin = st.number_input("Insulin", 0.0, 900.0, step=1.0)
bmi = st.number_input("BMI", 0.0, 60.0, step=0.1)
age = st.number_input("Age", 1, 120, step=1)

# Feature names in correct order
feature_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

if st.button("Predict Risk Level"):
    # Build full input
    input_data = np.array([[0, glucose, 0, 0, insulin, bmi, 0, age]])

    # Predict probability
    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.33:
        risk = "Low Risk"
    elif prob < 0.66:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    st.write(f"Predicted Risk Level: **{risk}** (Probability: {prob:.2f})")

    # --- SHAP Explainability ---
    st.subheader("🔍 Why this prediction?")

    # Use a small synthetic background for SHAP
    background = np.tile(input_data, (50, 1))  # repeat input 50 times
    explainer = shap.Explainer(model, background, feature_names=feature_names)

    shap_values = explainer(input_data)

    # Plot waterfall
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

