import librosa

y, sr = librosa.load("song.wav")
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beats, sr=sr)

print("Tempo:", tempo)
print("Beat times:", beat_times)