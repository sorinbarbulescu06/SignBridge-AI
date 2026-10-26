import queue
import sounddevice as sd
import numpy as np


class AudioEngine:
    """Gestionează fluxul de microfon și colectează propoziții complete bazate pe pauze naturale."""

    def __init__(self):
        self.stream = None
        self.is_running = False
        self.current_volume = 0.0
        self.sample_rate = 44100
        self.stt_sample_rate = 16000

        self.phrase_queue = queue.Queue()
        self._speech_buffer = []
        self._pre_roll = []
        self._speaking = False
        self._silence_frames = 0

    def _flush_phrase(self):
        """Trimite întreaga propoziție acumulată la recunoaștere când utilizatorul face o pauză."""
        if len(self._speech_buffer) < 15:
            self._speech_buffer.clear()
            return

        full_audio = np.concatenate(self._speech_buffer)
        self._speech_buffer.clear()

        # Downsampling la 16kHz pentru viteză
        target_len = int(len(full_audio) * self.stt_sample_rate / self.sample_rate)
        if target_len > 0:
            indices = np.linspace(0, len(full_audio) - 1, target_len).astype(np.int32)
            audio_16k = full_audio[indices]
            pcm16 = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16)
            self.phrase_queue.put((pcm16.tobytes(), self.stt_sample_rate))

    def _audio_callback(self, indata, frames, time, status):
        rms = np.sqrt(np.mean(indata**2))
        self.current_volume = min(float(rms) * 35.0, 1.0)

        mono_chunk = indata[:, 0].copy()

        # Prag de volum pentru detectarea vocii
        if self.current_volume > 0.05:
            if not self._speaking:
                self._speaking = True
                self._speech_buffer.extend(self._pre_roll)
                self._pre_roll.clear()

            self._silence_frames = 0
            self._speech_buffer.append(mono_chunk)
        elif self._speaking:
            self._speech_buffer.append(mono_chunk)
            self._silence_frames += 1

            # Așteptăm ~0.8 secunde de tăcere (35 cadre la 1024 blocksize) înainte de a trimite propoziția
            if self._silence_frames >= 35:
                self._flush_phrase()
                self._speaking = False
                self._silence_frames = 0
        else:
            self._pre_roll.append(mono_chunk)
            if len(self._pre_roll) > 10:
                self._pre_roll.pop(0)

    def get_completed_phrase(self):
        try:
            return self.phrase_queue.get_nowait()
        except queue.Empty:
            return None

    def start(self, device_index=None):
        self.stop()
        self.stream = sd.InputStream(
            device=device_index,
            callback=self._audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=1024,
        )
        self.stream.start()
        self.is_running = True

    def stop(self):
        if self._speech_buffer:
            self._flush_phrase()

        self.is_running = False
        self._speech_buffer.clear()
        self._pre_roll.clear()
        self._speaking = False
        self._silence_frames = 0

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.current_volume = 0.0