import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# Dataset
temperature = np.array([30, 28, 15, 16, 18, 25])
weather = np.array(['Hot', 'Hot', 'Cool', 'Cool', 'Cool', 'Hot'])
labels = np.array(['No', 'No', 'Yes', 'Yes', 'Yes', 'No'])

# Encoders
weather_encoder = LabelEncoder()
label_encoder = LabelEncoder()

# Encode categorical values
weather_encoded = weather_encoder.fit_transform(weather)
labels_encoded = label_encoder.fit_transform(labels)

# Combine features (temperature + weather)
X = np.column_stack((temperature, weather_encoded))
y = labels_encoded

# Train Naive Bayes model
model = CategoricalNB()
model.fit(X, y)

# Example prediction
test_temp = 20
test_weather = 'Cool'

test_weather_encoded = weather_encoder.transform([test_weather])

test_data = np.array([[test_temp, test_weather_encoded[0]]])

prediction = model.predict(test_data)

predicted_label = label_encoder.inverse_transform(prediction)

print("Prediction to play tennis with 20 degree C:", predicted_label[0])