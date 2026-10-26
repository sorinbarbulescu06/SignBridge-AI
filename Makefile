PY = python3.12
LIBS = opencv-python pillow sounddevice numpy customtkinter mediapipe==0.10.14 scikit-learn joblib SpeechRecognition

# Detect operating system
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    PYTHON_CMD := python
    VENV_PY := .\venv\Scripts\python.exe
    RM_DIR := rmdir /s /q
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        DETECTED_OS := Linux
    endif
    ifeq ($(UNAME_S),Darwin)
        DETECTED_OS := macOS
    endif
    PYTHON_CMD := python3.12
    VENV_PY := ./venv/bin/python
    RM_DIR := rm -rf
endif

all: python virtual library

python:
ifeq ($(DETECTED_OS),Windows)
	winget install Python.Python.3.12
else ifeq ($(DETECTED_OS),Linux)
	sudo apt update && sudo apt install software-properties-common -y
	sudo add-apt-repository ppa:deadsnakes/ppa -y
	sudo apt update && sudo apt install python3.12 python3.12-venv -y
else
	@echo "Please install Python 3.12 manually for your operating system."
endif

virtual:
	$(PYTHON_CMD) -m venv venv

library: virtual
ifeq ($(DETECTED_OS),Windows)
	.\venv\Scripts\python -m pip install --upgrade pip
	.\venv\Scripts\pip install $(LIBS)
else
	./venv/bin/python -m pip install --upgrade pip
	./venv/bin/pip install $(LIBS)
endif

run: library
ifeq ($(DETECTED_OS),Windows)
	.\venv\Scripts\python src/main.py
else
	sudo apt update && sudo apt install libportaudio2 -y
	./venv/bin/python src/main.py
endif

train: library
ifeq ($(DETECTED_OS),Windows)
	.\venv\Scripts\python src/dev_trainer_studio.py
else
	./venv/bin/python src/dev_trainer_studio.py
endif

clean:
	$(RM_DIR) venv