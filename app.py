import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
model=keras.models.load_model("traffic_sign_recognize_model.keras")

# creating a dictionary for sign names/class names
class_names={
    0: "Speed limit (20 km/h)",
    1: "Speed limit (30 km/h)",
    2: "Speed limit (50 km/h)",
    3: "Speed limit (60 km/h)",
    4: "Speed limit (70 km/h)",
    5: "Speed limit (80 km/h)",
    6: "End of speed limit (80 km/h)",
    7: "Speed limit (100 km/h)",
    8: "Speed limit (120 km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}

# creating a dictionary for driver assistance message
driver_assistance = {
    0: "Maintain your speed below 20 km/h.",
    1: "Maintain your speed below 30 km/h.",
    2: "Maintain your speed below 50 km/h.",
    3: "Maintain your speed below 60 km/h.",
    4: "Maintain your speed below 70 km/h.",
    5: "Maintain your speed below 80 km/h.",
    6: "The previous 80 km/h speed limit has ended. Follow new road signs.",
    7: "Maintain your speed below 100 km/h.",
    8: "Maintain your speed below 120 km/h.",
    9: "Do not overtake other vehicles.",
    10: "Vehicles over 3.5 tons must not overtake.",
    11: "Prepare to give priority at the next intersection.",
    12: "You are on a priority road. Continue carefully.",
    13: "Slow down and be ready to give way to other traffic.",
    14: "Bring the vehicle to a complete stop before proceeding.",
    15: "Do not enter this road.",
    16: "Vehicles over 3.5 tons are prohibited.",
    17: "Entry is prohibited. Choose another route.",
    18: "Drive carefully. A hazard lies ahead.",
    19: "Reduce speed. Sharp left curve ahead.",
    20: "Reduce speed. Sharp right curve ahead.",
    21: "Reduce speed. Double curve ahead.",
    22: "Slow down. Uneven road ahead.",
    23: "Drive slowly. Road may be slippery.",
    24: "Be cautious. Road narrows ahead.",
    25: "Slow down. Road work ahead.",
    26: "Prepare to stop if the traffic signal changes.",
    27: "Watch for pedestrians and reduce speed.",
    28: "Slow down. Children may cross the road.",
    29: "Watch for cyclists and give them space.",
    30: "Drive carefully. Road may be icy or snowy.",
    31: "Be alert for animals crossing the road.",
    32: "Previous speed and passing restrictions have ended.",
    33: "Turn right ahead.",
    34: "Turn left ahead.",
    35: "Proceed straight ahead only.",
    36: "Go straight or turn right.",
    37: "Go straight or turn left.",
    38: "Keep to the right.",
    39: "Keep to the left.",
    40: "Follow the roundabout in the indicated direction.",
    41: "The no-passing restriction has ended.",
    42: "The no-passing restriction for heavy vehicles has ended."
}


st.set_page_config(
    page_title="Traffic Sign Recognition with Driver Assistance",
    layout="wide"
)

one,two,three,four=st.columns([0.75,0.25,4,0.75])
with one:
   st.markdown("<br>", unsafe_allow_html=True)
   with st.expander("About this project"):
    st.write("""
    - CNN Model
    - Dataset: GTSRB
    - TensorFlow / Keras
    - Streamlit Interface

    Note:
    Trained on the GTSRB dataset; performance may vary on different real-world images.
    """)

with three:
 st.markdown("""
 <h1 style='font-size:42px; margin-bottom:0px;'>
 Traffic Sign Recognition with Driver Assistance
 </h1>
 """, unsafe_allow_html=True)
 
 uploaded_file=st.file_uploader("Upload a Traffic Sign Image and let the model predict",type=["jpg","jpeg","png"])
 
 if uploaded_file is not None:
    file_bytes=np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8) # first in raw bytes>then an bytearray>numpy array
    img= cv2.imdecode(file_bytes,cv2.IMREAD_COLOR) # gets covert into pixels> becomes an img
    img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
   
    # prepocessing the image uploaded
    img_prepocessed=cv2.resize(img,(32,32))
    img_prepocessed=cv2.GaussianBlur(img_prepocessed,(3,3),0) # blurs or smoothen img, so that model focus more on imp details rather than unwnated noize, 3*3 pixels , 0 is standard deviation
    img_prepocessed=img_prepocessed/255.0
    img_prepocessed = np.expand_dims(img_prepocessed,axis=0)
    
    # predict the image uploaded
    predicted=model.predict(img_prepocessed)
    predicted_class_id=np.argmax(predicted)
    confidence = np.max(predicted)
    
     # to display uploaded image
    col1,col2=st.columns([3,2])
    with col1:
        st.image(img,caption="uploaded image",use_container_width=True)
    with col2:
        # st.write(f"predicted class id:{predicted_class_id}")
        st.subheader("Traffic Sign")
        st.success(class_names[predicted_class_id])
        st.subheader("Confidence Bar")
        st.progress(int(confidence * 100))
        st.write(f"{confidence*100:.2f}%")
        # st.subheader(f"Confidence: {confidence*100:.2f}%")
        st.subheader("Driver Assistance")
        st.info(driver_assistance[predicted_class_id])


# the image file uploaded by user is not an image in pixels, we dont have its path
# so we read it and get raw bytes in form 0100001011
# then we use bytearray by python and coverted into a single list [255,13,45]
# the we converted it into numpy array of datatype unit8 that is unsigned int 8 bits to avoid wastage of memory


