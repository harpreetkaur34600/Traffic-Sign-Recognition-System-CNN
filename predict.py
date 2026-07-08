import tensorflow 
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

model=keras.models.load_model("traffic_sign_recognize_model.keras")
print("model loaded successfully")

# preproccesing the image to be predicted
predict_img=cv2.imread("Test/00025.png")
predict_img=cv2.cvtColor(predict_img,cv2.COLOR_BGR2RGB)
predict_img=cv2.resize(predict_img,(32,32))

# display image
plt.imshow(predict_img)
plt.show()

predict_img = predict_img / 255.0
predict_img=np.expand_dims(predict_img,axis=0) # increse one dimension from start as tf wants batcch size as well batch size, height, width, channels
predictions=model.predict(predict_img)
result_classId=np.argmax(predictions)
print("predicted class id: ",result_classId)


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

print("predicted traffic sign: ",class_names[result_classId])
print("Driver Assistance: ",driver_assistance[result_classId])
