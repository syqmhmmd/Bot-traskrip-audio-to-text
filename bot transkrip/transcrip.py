import streamlit as st
from faster_whisper import WhisperModel
import os, time, math, shutil
from datetime import datetime
from pathlib import Path
import ffmpeg

# === DEBUG cek modul ffmpeg yang ke-load ===
try:
    print("DEBUG → ffmpeg module path:", ffmpeg.__file__)
    print("DEBUG → has input():", hasattr(ffmpeg, "input"))
    print("DEBUG → attrs:", dir(ffmpeg)[:20])
except Exception as e:
    print("DEBUG ERROR:", e)

st.set_page_config(page_title="Stable Whisper Transcriber", layout="wide")
st.markdown("""
<style>
.main-title {font-size:2.2rem;font-weight:700;margin-bottom:0.5em;}
.file-info {font-size:1.1rem;color:#555;}
.transcript-box textarea {font-family:monospace; font-size:1.1em;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎙️ Stable Faster-Whisper Transcriber</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    st.markdown("""
    Selamat datang di transcriber!
    
    1. Upload file audio (mp3, wav, dll)
    2. Pilih model & pengaturan
    3. Klik Start Transcription
    4. Download hasilnya
    """)
    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac", "aac"]
    )
    model_size = st.selectbox(
        "Whisper model size", ["tiny", "base", "small", "medium", "large"], index=1
    )
    max_minutes = st.slider("Limit duration (minutes, 0 = full file)", 0, 60, 5)
    language = st.text_input("Language (optional, e.g. 'id' for Indonesian)")
    run_button = st.button("Start Transcription", use_container_width=True)

def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"

if not uploaded_file:
    st.info("""
    **Cara menggunakan aplikasi ini:**
    
    - Upload file audio melalui sidebar
    - Atur model & parameter sesuai kebutuhan
    - Klik Start Transcription untuk mulai
    - Hasil transkrip akan muncul di sini dan bisa diunduh
    """)

if run_button and uploaded_file is not None:
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = Path(uploaded_file.name).suffix
    file_path = UPLOAD_DIR / f"upload_{timestamp}{ext}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_size_mb = os.path.getsize(file_path)/1024/1024
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown(f'<div class="file-info">📂 <b>Saved:</b> `{file_path}`</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="file-info"><b>Size:</b> {file_size_mb:.2f} MB</div>', unsafe_allow_html=True)

    if not shutil.which("ffmpeg"):
        st.error("❌ ffmpeg not found. Please install ffmpeg.")
        st.stop()

    # === Limit durasi kalau slider > 0 ===
    if max_minutes > 0:
        clipped_file = UPLOAD_DIR / f"clip_{timestamp}{ext}"
        (
            ffmpeg
            .input(str(file_path))
            .output(str(clipped_file), t=max_minutes*60)
            .overwrite_output()
            .run(quiet=True)
        )
        file_path = clipped_file
        st.warning(f"⚡ Only transcribing first {max_minutes} minutes")

    # === Auto pilih device ===
    device = "cuda" if shutil.which("nvidia-smi") else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    try:
        with st.spinner(f"Loading Whisper `{model_size}` on {device}..."):
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    transcript_box = st.empty()
    progress_bar = st.progress(0)
    runtime_placeholder = st.empty()

    start_time = time.time()
    transcript_paragraphs = []

    with st.spinner("Transcribing..."):
        segments, info = model.transcribe(str(file_path), beam_size=5, language=language or None)
        segments = list(segments)
        total_segments = len(segments)

        for i, seg in enumerate(segments, start=1):
            transcript_paragraphs.append(
                f"[{format_time(seg.start)} - {format_time(seg.end)}] {seg.text.strip()}"
            )

            transcript_text = "\n\n".join(transcript_paragraphs)
            transcript_box.text_area(
                "Transcript (Realtime)",
                transcript_text,
                height=400,
                key="transcript_box",
                help="Transcription updates in real time. Copy or download after selesai."
            )
            progress_bar.progress(int(i / total_segments * 100))

            elapsed = time.time() - start_time
            runtime_placeholder.info(f"⏱️ Runtime: {format_time(elapsed)}")

    progress_bar.empty()
    st.success("✅ Done!")

    final_text = "\n\n".join(transcript_paragraphs)
    out_filename = f"transcript_{timestamp}.txt"

    st.download_button(
        "💾 Download Transcript",
        data=final_text,
        file_name=out_filename,
        mime="text/plain",
        use_container_width=True
    )

    # Auto cleanup
    if os.path.exists(file_path):
        os.remove(file_path)
