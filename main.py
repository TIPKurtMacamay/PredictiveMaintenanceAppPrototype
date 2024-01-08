import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Check if Firebase app is already initialized
if not firebase_admin._apps:
    # Initialize Firebase credentials
    cred = credentials.Certificate("pdinitialprototype-25e22e0103c7.json")
    firebase_admin.initialize_app(cred, {"databaseURL": "https://pdinitialprototype-default-rtdb.asia-southeast1.firebasedatabase.app/"})

# Function to fetch data from Firebase and convert it to DataFrame
def fetch_data():
    ref = db.reference('/')
    data = ref.get()
    
    # Convert data to DataFrame
    if data:
        df = pd.DataFrame.from_dict(data, orient='index')
        return df
    else:
        return pd.DataFrame()

# Streamlit app
st.title("Firebase Data to DataFrame")

# Create an initial empty DataFrame
merged_data = pd.DataFrame()

# Display placeholder for the data
data_placeholder = st.empty()

# Set the total running time in seconds
total_time_seconds = 10

# Set the interval in milliseconds
interval_ms = 37

# Calculate the total number of iterations
total_iterations = int(total_time_seconds * 1000 / interval_ms)

# Create a placeholder for the progress bar
progress_placeholder = st.empty()

# Get the start time
start_time = time.time()

# Update the progress bar dynamically for 10 seconds
while time.time() - start_time < total_time_seconds:
    elapsed_time = time.time() - start_time
    progress_percentage = int((elapsed_time / total_time_seconds) * 100)
    progress_placeholder.progress(progress_percentage)

    # Fetch data from Firebase
    current_data = fetch_data()

    if not current_data.empty:
        # Append the current data to the merged DataFrame
        merged_data = pd.concat([merged_data, current_data])

        # Display the updated DataFrame
        data_placeholder.write("Merged Data from Firebase:")
        data_placeholder.write(merged_data)

        # Additional DataFrame operations can be performed here if needed
        # For example: merged_data = merged_data.dropna(), merged_data = merged_data.head(10), etc.

        # Wait for the specified interval before fetching the data again
        time.sleep(interval_ms / 1000)
    else:
        data_placeholder.write("No data available in Firebase.")

# Display the final DataFrame after the specified running time
data_placeholder.write("Final Merged Data:")
data_placeholder.write(merged_data)

# Display a success message after the timer completes
st.success("Data Collection Completed, you can now predict RUL!")

# Add the "Predict" button
predict_button = st.button("Predict")

# Placeholder for prediction result
prediction_result_placeholder = st.empty()

# Load the pre-trained model from an HDF5 file
model = load_model("best_model_prototyp.h5")  # Replace "your_model.h5" with your actual model file

# Instantiate MinMaxScaler
scaler = MinMaxScaler()

# Function to preprocess data and make predictions
def predict_rul(data):
    # Fit the scaler with training data
    scaler.fit(data)  # Replace your_training_data with actual training data
    
    # Preprocess the data using MinMaxScaler
    scaled_data = scaler.transform(data)  # Assuming 'scaled_data' is a NumPy array
    
    # Make predictions using the pre-trained model
    predictions = model.predict(scaled_data)
    merged_data['pre'] = predictions
    return predictions
ln = len(merged_data)
def cycleFormat(ln):
    pred=0
    Afil=0
    Pfil=0
    a=merged_data['capacity'].values
    b=merged_data['pre'].values
    j=0
    k=0
    for i in range(len(a)):
        actual=a[i]

        if actual<=1.38:
            j=i
            Afil=j
            break
    for i in range(len(a)):
        pred=b[i]
        if pred< 1.38:
            k=i
            Pfil=k
            break
    
    print("The Actual fail at cycle number: "+ str(Afil+ln))
    print("The prediction fail at cycle number: "+ str(Pfil+ln))
    RULerror=Pfil-Afil
    print("The error of RUL= "+ str(RULerror)+ " Cycle(s)")

    return str(Pfil+ln)

# Execute prediction when the "Predict" button is clicked
if predict_button:
    # Check if there's data available for prediction
    if not merged_data.empty:
        # Perform any necessary preprocessing based on your specific data and model requirements
        # For example, select relevant features, handle missing values, etc.
        # Then, call the predict_rul function to get predictions
        predictions = predict_rul(merged_data)
        
        # Display the predictions
        prediction_result_placeholder.write("Predictions:")
        prediction_result_placeholder.write(predictions)

        wch_colour_box = (49, 150, 38)
        wch_colour_font = (238,238,228)
        fontsize = 80
        valign = "left"
        iconname = "fas fa-asterisk"

        htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                    {wch_colour_box[1]}, 
                                                    {wch_colour_box[2]}, 0.75); 
                                color: rgb({wch_colour_font[0]}, 
                                        {wch_colour_font[1]}, 
                                        {wch_colour_font[2]}, 0.75); 
                                font-size: {fontsize}px; 
                                border-radius: 7px; 
                                padding-left: 50px; 
                                padding-top: 18px; 
                                padding-bottom: 18px; 
                                line-height:65px;'>
                                <i class='{iconname} fa-xs'></i><b> {cycleFormat(len(merged_data))} </b> 
                                </style>
                                <BR>
                                <span style='font-size: 40px;'>{"Cycle Remaining"}</style></span></p>"""

        st.markdown(htmlstr, unsafe_allow_html=True)
    else:
        st.warning("No data available for prediction. Please collect data first.")



