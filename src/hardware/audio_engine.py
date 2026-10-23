import sounddevice as sd
import numpy as np


class AudioEngine:
    """Gestionează exclusiv fluxul de microfon și calculul volumului (RMS)."""

    def __init__(self):
        self.stream = None
        self.is_running = False
        self.current_volume = 0.0

    def _audio_callback(self, indata, frames, time, status):
        rms = np.sqrt(np.mean(indata**2))
        self.current_volume = min(float(rms) * 35.0, 1.0)

    def start(self, device_index=None):
        self.stop()
        self.stream = sd.InputStream(
            device=device_index,
            callback=self._audio_callback,
            channels=1,
            samplerate=44100,
            blocksize=1024,
        )
        self.stream.start()
        self.is_running = True

    def stop(self):
        self.is_running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.current_volume = 0.0
