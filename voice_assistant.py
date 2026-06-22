import requests
import pyttsx3
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import os
import wave

# ========================
# CONFIGURATION
# ========================
MIC_DEVICE_INDEX = None   # Set to mic index number if default mic is wrong
SAMPLE_RATE      = 16000  # 16kHz optimal for Google Speech Recognition
LISTEN_SECONDS   = 6      # Seconds to record per voice turn
BACKEND_URL      = "http://127.0.0.1:8000"

# ========================
# SUPPORTED LANGUAGES
# ========================
LANGUAGES = {
    "1":  ("en",    "English"),
    "2":  ("es",    "Spanish"),
    "3":  ("hi",    "Hindi"),
    "4":  ("fr",    "French"),
    "5":  ("de",    "German"),
    "6":  ("zh-cn", "Chinese (Simplified)"),
    "7":  ("ar",    "Arabic"),
    "8":  ("pt",    "Portuguese"),
    "9":  ("ru",    "Russian"),
    "10": ("ja",    "Japanese"),
    "11": ("it",    "Italian"),
    "12": ("ko",    "Korean"),
    "13": ("bn",    "Bengali"),
    "14": ("ur",    "Urdu"),
    "15": ("tr",    "Turkish"),
    "16": ("auto",  "Auto-detect from my speech"),
}

# ========================
# TEXT TO SPEECH
# ========================
engine = pyttsx3.init()

def speak(text):
    print(f"\n🔊 Assistant: {text}\n")
    engine.say(text)
    engine.runAndWait()

# ========================
# LANGUAGE SELECTION
# ========================
def select_language():
    print("\n" + "=" * 50)
    print("   🌍  Select Response Language")
    print("=" * 50)
    for num, (code, name) in LANGUAGES.items():
        print(f"  [{num:>2}]  {name}")
    print("=" * 50)
    while True:
        choice = input("Enter number (default = 1 for English): ").strip()
        if choice == "":
            choice = "1"
        if choice in LANGUAGES:
            code, name = LANGUAGES[choice]
            print(f"✅ Language set to: {name}\n")
            return code, name
        print(f"❌ Invalid choice. Enter a number 1–{len(LANGUAGES)}.")

# ========================
# INPUT MODE SELECTION
# ========================
def select_input_mode():
    """Ask the user each turn: voice or text?"""
    print("\n┌─────────────────────────────────┐")
    print("│  How do you want to ask?        │")
    print("│  [1] 🎤  Voice                  │")
    print("│  [2] ⌨️   Text                   │")
    print("│  [L] 🌍  Change language        │")
    print("│  [E] 🚪  Exit                   │")
    print("└─────────────────────────────────┘")
    choice = input("Your choice: ").strip().lower()
    return choice

# ========================
# WINDOWS-SAFE WAV PATH
# ========================
def get_wav_path():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "_recording.wav")

# ========================
# LIST / CHECK MIC
# ========================
def list_microphones():
    print("\n📋 Available audio input devices:")
    print("-" * 50)
    devices = sd.query_devices()
    found_any = False
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            marker = " ← default" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}{marker}")
            found_any = True
    if not found_any:
        print("  ❌ No input devices found!")
    print("-" * 50)
    print("💡 Set MIC_DEVICE_INDEX at top of file if default mic is wrong.\n")

def check_microphone():
    try:
        devs = sd.query_devices()
        if not any(d['max_input_channels'] > 0 for d in devs):
            print("❌ No microphone found.")
            return False
        if MIC_DEVICE_INDEX is not None:
            d = sd.query_devices(MIC_DEVICE_INDEX)
            print(f"✅ Using mic (manual): [{MIC_DEVICE_INDEX}] {d['name']}")
        else:
            d = sd.query_devices(kind='input')
            print(f"✅ Using mic (default): {d['name']}")
        return True
    except Exception as e:
        print(f"❌ Mic check failed: {e}")
        return False

# ========================
# VOICE INPUT
# ========================
def listen():
    """Record from mic and return transcribed text, or None."""
    print("🎤 Listening... (speak now)")
    wav_path = get_wav_path()
    try:
        kwargs = {
            "frames": int(LISTEN_SECONDS * SAMPLE_RATE),
            "samplerate": SAMPLE_RATE,
            "channels": 1,
            "dtype": "int16",
        }
        if MIC_DEVICE_INDEX is not None:
            kwargs["device"] = MIC_DEVICE_INDEX

        recording = sd.rec(**kwargs)
        sd.wait()

        if np.max(np.abs(recording)) < 10:
            print("⚠️  Silent recording — mic may not be capturing.")
            print("   → Set MIC_DEVICE_INDEX to your mic's index at the top of this file.")
            return None

        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(recording.tobytes())

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True

        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        print(f"🧑 You (voice): {text}")
        return text

    except sr.UnknownValueError:
        print("❌ Could not understand. Please speak clearly.")
        return None
    except sr.RequestError as e:
        print(f"❌ Google Speech API error: {e}")
        return None
    except sd.PortAudioError as e:
        print(f"❌ PortAudio error: {e}")
        return None
    except Exception as e:
        print(f"❌ listen() error: {type(e).__name__}: {e}")
        return None
    finally:
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

# ========================
# TEXT INPUT
# ========================
def get_text_input():
    """Get a question from the user via keyboard."""
    try:
        text = input("⌨️  Type your question: ").strip()
        if text:
            print(f"🧑 You (text): {text}")
        return text or None
    except (EOFError, KeyboardInterrupt):
        return None

# ========================
# BACKEND CALL
# ========================
def ask_backend(query, language_code="auto"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/ask",
            json={"query": query, "language": language_code},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return f"Backend error: {data['error']}", None

        answer    = data.get("answer", "No answer returned.")
        lang_name = data.get("language_name", "")
        return answer, lang_name

    except requests.exceptions.ConnectionError:
        return "Cannot connect to backend. Is the server running on port 8000?", None
    except requests.exceptions.Timeout:
        return "Backend timed out. Please try again.", None
    except Exception as e:
        return f"Backend error: {e}", None

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    print("=" * 55)
    print("   🚨  SKY — AI Disaster Assistant  (Voice + Text)")
    print("=" * 55)

    list_microphones()
    mic_available = check_microphone()

    if not mic_available:
        print("⚠️  No mic detected — Voice mode will be unavailable.\n")

    # Language selection at startup
    selected_code, selected_name = select_language()

    speak("Hello! I am Sky, how can i help you today.")
    speak(f"I will respond in {selected_name}. How can I help you?")

    while True:
        # ── Show input mode menu each turn ──
        choice = select_input_mode()

        # ── Exit ──
        if choice in ["e", "exit", "quit", "bye"]:
            speak("Goodbye! Stay safe.")
            break

        # ── Change language ──
        if choice == "l":
            selected_code, selected_name = select_language()
            speak(f"Language changed to {selected_name}.")
            continue

        # ── Voice mode ──
        if choice == "1":
            if not mic_available:
                print("❌ No microphone available. Please choose Text (2) instead.")
                continue
            command = listen()
            if not command:
                print("❌ Could not capture voice. Please try again or switch to Text mode.")
                continue

        # ── Text mode ──
        elif choice == "2":
            command = get_text_input()
            if not command:
                print("❌ Empty input. Please type your question.")
                continue

        # ── Invalid choice ──
        else:
            print(f"❌ Invalid choice '{choice}'. Please enter 1, 2, L, or E.")
            continue

        # ── Check for exit words in the command ──
        if any(w in command.lower() for w in ["exit", "quit", "bye", "goodbye", "stop"]):
            speak("Goodbye! Stay safe.")
            break

        # ── Check for language change in command ──
        if any(p in command.lower() for p in ["change language", "switch language", "change lang"]):
            selected_code, selected_name = select_language()
            speak(f"Language changed to {selected_name}.")
            continue

        # ── Send to backend ──
        print("\n⏳ Thinking...")
        answer, lang_used = ask_backend(command, language_code=selected_code)
        label = lang_used or selected_name

        print(f"\n📝 Answer ({label}):\n{answer}\n")
        speak(answer)