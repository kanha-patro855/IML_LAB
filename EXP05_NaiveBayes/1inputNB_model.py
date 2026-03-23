import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# Dataset
weather = np.array(['Sunny', 'Sunny', 'Rainy', 'Rainy'])
labels = np.array(['Yes', 'Yes', 'No', 'No'])

# Encoders
weather_encoder = LabelEncoder()
label_encoder = LabelEncoder()

# Encoding
weather_encoded = weather_encoder.fit_transform(weather)
labels_encoded = label_encoder.fit_transform(labels)

X = weather_encoded.reshape(-1,1)
y = labels_encoded

# Train model
model = CategoricalNB()
model.fit(X, y)

# Predict only for Sunny
test_weather = np.array(['Sunny'])
test_encoded = weather_encoder.transform(test_weather).reshape(-1,1)

prediction = model.predict(test_encoded)

# Convert back to label
predicted_label = label_encoder.inverse_transform(prediction)

print("Prediction for Sunny:", predicted_label[0])