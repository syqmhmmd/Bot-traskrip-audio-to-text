# 🎙️ Realtime Whisper Transcriber

A Streamlit web app for **realtime transcription** using [OpenAI Whisper](https://github.com/openai/whisper).  

### ✨ Features
- Upload audio files (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.aac`)
- Choose Whisper model size (`tiny`, `base`, `small`, `medium`, `large`)
- Realtime transcript with **timestamps**
- Live **runtime timer**
- Fake **download progress bar** (natural like real downloads)
- One-click transcript download (`.txt`)
- **Auto delete uploaded files** after transcription (no disk space wasted)

---

## 🚀 Demo Screenshot
<img width="1888" height="897" alt="image" src="https://github.com/user-attachments/assets/55fdff6d-24a8-4f71-a537-826e29042c61" />



---

## 📦 Requirements

- Python **3.9+**
- [Streamlit](https://streamlit.io)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [FFmpeg](https://ffmpeg.org/) (required for audio decoding)

---

## ⚙️ Installation

  - streamlit
  - openai-whisper
  - torch
  - torchaudio
  - numpy
  - choco install ffmpeg # khusu choco jalanin di powershell admin #
    
  - Note: intstalasi diatas pake pip install semua kecuali chocho
    
    - Kalau sudah install semua dan download zip nya tinggal extact aja
    - Buka foldernya terus klik kanan "Buka di Terminal "
    - Terus masukin perintah di bawah ini

 ****python -m streamlit run transcrip.py****
