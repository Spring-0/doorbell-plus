import joblib
from train import extract_features_from_audio

model = joblib.load('class.pkl')

test_audio_file = r'data\bell-test-2\test-doorbell-2-10.wav'
test_features = extract_features_from_audio(test_audio_file).reshape(1, -1)

prediction = model.predict(test_features)

result = "Doorbell" if prediction[0] == 1 else "No Doorbell"
print(f'Test Audio File: {test_audio_file}')
print(f'Prediction: {result}')