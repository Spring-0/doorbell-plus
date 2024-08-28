import joblib
from recorder import Recorder
from train import extract_features_from_memory
import numpy as np 
import soundfile as sf
import time

def save_audio_to_file(filename, audio_data):
    sf.write(filename, audio_data, 16000)

def main():
    recorder = Recorder()
    model = joblib.load("class.pkl")

    recorder.start()

    while True:
        frame = recorder.get_frame()

        if frame is not None:
            print("[+] Captured frame")

            frame = np.array(frame)

            if frame.dtype != np.float32 and frame.dtype != np.float64:
                print("[+] Converted to float32.")
                frame = frame.astype(np.float32, order="C") / 32768.0

            features = extract_features_from_memory(frame, recorder.sampling_rate * 1000).reshape(1, -1)
            prediction = model.predict(features)

            result = "Doorbell" if prediction[0] == 1 else "No Doorbell"
            print(result)

            if prediction[0] == 1:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_audio_to_file(f"data\\verify\\manual_check-{timestamp}.wav", frame)
                print("Saved file for manual verification.")

        else:
            # print("[!] No audio frame capture")
            pass

    recorder.stop()

if __name__ == "__main__":
    main()