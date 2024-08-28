import joblib
from train import extract_features_from_audio_file

model = joblib.load('class.pkl')

test_audio_file = r'\data\doorbell\doorbell-06.wav'
test_features = extract_features_from_audio_file(test_audio_file).reshape(1, -1)

prediction = model.predict(test_features)

result = "Doorbell" if prediction[0] == 1 else "No Doorbell"
print(f'Test Audio File: {test_audio_file}')
print(f'Prediction: {result}')