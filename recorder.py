from pvrecorder import PvRecorder
from collections import deque
import threading

class Recorder:
    def __init__(self, window_duration=2000, sampling_rate=16):
        self.windows_q = deque()
        self.window_duration = window_duration
        self.sampling_rate = sampling_rate
        self.frame_length = window_duration * sampling_rate
        self.pv_recorder = PvRecorder(device_index=-1, frame_length=self.frame_length) # -1 to use default system microphone
        self.thread = threading.Thread(target=self.start_recording)
        self.stop_event = threading.Event()
        self.recording = False

    def start_recording(self):
        self.pv_recorder.start()
        self.recording = True

        while not self.stop_event.is_set():
            audio_frame = self.pv_recorder.read()
            self.windows_q.append(audio_frame)

        self.pv_recorder.stop()
        self.recording = False

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def get_frame(self):
        if self.windows_q:
            return self.windows_q.popleft()
        else:
            return None