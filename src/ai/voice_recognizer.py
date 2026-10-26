import queue
import threading
import speech_recognition as sr


class VoiceRecognizer:
    """
    Procesează micro-batch-urile audio de ~1s în ordine strict cronologică
    pentru a construi un paragraf fluent în timp real (Live Captioning).
    """

    def __init__(self, language="ro-RO"):
        self.recognizer = sr.Recognizer()
        self.language = language
        self.text_queue = queue.Queue()
        self._job_queue = queue.Queue()

        # Un singur worker dedicat care procesează batch-urile de 1s strict în ordine
        self._worker_thread = threading.Thread(target=self._process_queue_loop, daemon=True)
        self._worker_thread.start()

    def transcribe_pcm_async(self, pcm_bytes, sample_rate=16000):
        self._job_queue.put(("pcm", pcm_bytes, sample_rate))

    def transcribe_file_async(self, file_path):
        self._job_queue.put(("file", file_path, None))

    def _process_queue_loop(self):
        while True:
            job_type, payload, sample_rate = self._job_queue.get()
            try:
                if job_type == "pcm":
                    audio_data = sr.AudioData(payload, sample_rate, sample_width=2)
                    text = self._recognize_fast(audio_data)
                    if text:
                        self.text_queue.put(text)
                elif job_type == "file":
                    with sr.AudioFile(payload) as source:
                        audio_data = self.recognizer.record(source)
                    text = self._recognize_fast(audio_data)
                    if text:
                        self.text_queue.put(text)
            except Exception as e:
                if job_type == "file":
                    self.text_queue.put(f"[Audio File Error: use .wav format ({e})]")
            finally:
                self._job_queue.task_done()

    def _recognize_fast(self, audio_data):
        try:
            return self.recognizer.recognize_google(audio_data, language=self.language).strip()
        except sr.UnknownValueError:
            try:
                return self.recognizer.recognize_google(audio_data, language="en-US").strip()
            except sr.UnknownValueError:
                return None
        except Exception:
            return None

    def get_recognized_text(self):
        try:
            return self.text_queue.get_nowait()
        except queue.Empty:
            return None