import os
import hashlib
import cv2
import librosa
import numpy as np
import faiss
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView
from PIL import Image
from imagehash import phash

# Function to get file hash
def get_file_hash(file_path, algo='md5'):
    hasher = hashlib.md5() if algo == 'md5' else hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Function to extract image hash
def get_image_hash(file_path):
    image = Image.open(file_path)
    return str(phash(image))

# Function to extract audio features
def get_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

# Function to extract video frames and hash
def get_video_frame_hash(file_path):
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return hashlib.md5(frame.tobytes()).hexdigest()
    return None

# Main function to scan directory
def scan_directory(directory):
    file_hashes = {}
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = file.split('.')[-1].lower()
            
            if ext in ['jpg', 'png', 'jpeg']:
                file_hash = get_image_hash(file_path)
            elif ext in ['mp3', 'wav', 'flac']:
                file_hash = str(get_audio_features(file_path))
            elif ext in ['mp4', 'avi', 'mkv']:
                file_hash = get_video_frame_hash(file_path)
            else:
                file_hash = get_file_hash(file_path)
            
            if file_hash in file_hashes:
                file_hashes[file_hash].append(file_path)
            else:
                file_hashes[file_hash] = [file_path]
    
    return {k: v for k, v in file_hashes.items() if len(v) > 1}

# Kivy UI Class
class DuplicateFileApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        self.label = Label(text="Select a folder to scan")
        self.layout.add_widget(self.label)
        
        self.file_chooser = FileChooserListView()
        self.layout.add_widget(self.file_chooser)
        
        self.scan_button = Button(text="Scan for Duplicates")
        self.scan_button.bind(on_press=self.scan_files)
        self.layout.add_widget(self.scan_button)
        
        self.result_list = RecycleView()
        self.layout.add_widget(self.result_list)
        
        return self.layout
    
    def scan_files(self, instance):
        directory = self.file_chooser.path
        duplicates = scan_directory(directory)
        
        self.result_list.adapter.data.clear()
        for hash_value, files in duplicates.items():
            for file in files:
                self.result_list.adapter.data.append(file)
        
        self.result_list._trigger_reset_populate()
        
DuplicateFileApp().run()
