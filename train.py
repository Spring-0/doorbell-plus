import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import glob

def extract_features(y, sr):
    features = []

    if not isinstance(y, np.ndarray):
        raise TypeError("y should be a numpy array.")

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.append(np.mean(mfccs, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma, axis=1))

    return np.concatenate(features)

def extract_features_from_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    return extract_features(y, sr)
 
def extract_features_from_memory(y, sr):
    return extract_features(y, sr)

def process_audio_files(file_paths):
    features = []
    labels = []
    for file_path in file_paths:
        label = 1 if 'doorbell' in file_path else 0
        features.append(extract_features_from_audio_file(file_path))
        labels.append(label)
    return np.array(features), np.array(labels)

if __name__ == "__main__":

    doorbell_files = glob.glob("data\\doorbell\\*")
    no_doorbell_files = glob.glob("data\\no-bell\\*")

    doorbell_features, doorbell_labels = process_audio_files(doorbell_files)
    no_doorbell_features, no_doorbell_labels = process_audio_files(no_doorbell_files)

    X = np.vstack((doorbell_features, no_doorbell_features))
    y = np.concatenate((doorbell_labels, no_doorbell_labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')

    print("\nDetailed results:")
    for true_label, pred_label in zip(y_test, y_pred):
        true_label_str = "Doorbell" if true_label == 1 else "No Doorbell"
        pred_label_str = "Doorbell" if pred_label == 1 else "No Doorbell"
        print(f'True Label: {true_label_str}\nPredicted Label: {pred_label_str}\n')

    joblib.dump(model, 'class.pkl')
    print("Saved model...")
