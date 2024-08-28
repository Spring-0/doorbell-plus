import os
import time
import shutil
import tkinter as tk
from tkinter import messagebox
from pydub import AudioSegment
from pydub.playback import play
import glob

class DoorbellVerifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Doorbell Manual Verifier")
        self.root.geometry("300x100")
        self.root.configure(bg="green")
        
        self.audio_files = glob.glob(r"data\verify\*")
        self.current_file_index = 0
        
        self.play_button = tk.Button(root, text="Play Audio", command=self.play_audio)
        self.play_button.pack(pady=10)
        
        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.yes_button = tk.Button(button_frame, text="Yes", command=lambda: self.verify_audio(True))
        self.yes_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 5))

        self.no_button = tk.Button(button_frame, text="No", command=lambda: self.verify_audio(False))
        self.no_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        
        self.load_next_audio()
        
    def load_next_audio(self):
        if self.current_file_index < len(self.audio_files):
            self.audio_file = self.audio_files[self.current_file_index]
        else:
            messagebox.showinfo("Done", "All files have been processed.")
            self.root.quit()
            
    def play_audio(self):
        try:
            audio = AudioSegment.from_wav(self.audio_file)
            play(audio)
        except Exception as e:
            messagebox.showerror("Error", f"Could not play audio: {str(e)}")
            
    def verify_audio(self, is_doorbell):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        if is_doorbell:
            final_filename = f"manual-bell-{timestamp}.wav"
            final_path = os.path.join("data", "doorbell", final_filename)
        else:
            final_filename = f"manual-no-bell-{timestamp}.wav"
            final_path = os.path.join("data", "no-bell", final_filename)
        
        try:
            shutil.move(self.audio_file, final_path)
            print(f"File saved to: {final_path}")
            self.current_file_index += 1
            self.load_next_audio()
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {str(e)}")
        
if __name__ == "__main__" :
    root = tk.Tk()
    app = DoorbellVerifier(root)
    root.mainloop()
        