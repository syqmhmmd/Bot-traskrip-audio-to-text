import streamlit as st
import whisper
import os
import time
import math
import random
from datetime import datetime

st.set_page_config(page_title="Realtime Whisper Transcriber", layout="wide")
st.title("🎙️ Realtime Transcription with Whisper + Timestamp + Runtime Timer")

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "m4a", "ogg", "flac", "aac"]
)
model_size = st.selectbox(
    "Whisper model size", ["tiny", "base", "small", "medium", "large"], index=1
)
language = st.text_input("Language (optional, e.g. 'id' for Indonesian)")

run_button = st.button("Start Transcription")


def format_time(seconds: float) -> str:
    """Ubah detik → HH:MM:SS"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"


if run_button and uploaded_file is not None:
    # === Folder custom buat simpan sementara ===
    UPLOAD_DIR = "D:/transcriber_uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(UPLOAD_DIR, f"upload_{timestamp}.mp3")

    # Simpan file upload
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"Saved upload → `{file_path}` — size: {os.path.getsize(file_path):,} bytes")

    # Load Whisper model
    with st.spinner(f"Loading Whisper `{model_size}` model..."):
        model = whisper.load_model(model_size)

    transcript_paragraphs = []
    current_paragraph = ""
    current_start, current_end = None, None
    transcript_box = st.empty()
    progress_bar = st.progress(0)
    runtime_placeholder = st.empty()

    # Timer runtime
    start_time = time.time()

    with st.spinner("Transcribing..."):
        result = model.transcribe(file_path, language=language or None, verbose=False)
        total_segments = len(result["segments"])

        for i, seg in enumerate(result["segments"], start=1):
            line = seg["text"].strip()

            if current_start is None:
                current_start = seg["start"]

            if len(current_paragraph) + len(line) < 120:
                current_paragraph += " " + line
                current_end = seg["end"]
            else:
                if current_paragraph.strip():
                    transcript_paragraphs.append(
                        f"[{format_time(current_start)} - {format_time(current_end)}]\n{current_paragraph.strip()}"
                    )
                current_paragraph = line
                current_start = seg["start"]
                current_end = seg["end"]

            transcript_text = "\n\n".join(
                transcript_paragraphs
                + [
                    f"[{format_time(current_start)} - {format_time(current_end)}]\n{current_paragraph.strip()}"
                ]
            )
            transcript_box.text_area("Transcript (Realtime)", transcript_text, height=400)
            progress_bar.progress(int(i / total_segments * 100))

            elapsed = time.time() - start_time
            runtime_placeholder.info(f"⏱️ Runtime: {format_time(elapsed)}")

    if current_paragraph.strip():
        transcript_paragraphs.append(
            f"[{format_time(current_start)} - {format_time(current_end)}]\n{current_paragraph.strip()}"
        )

    progress_bar.empty()
    st.success("✅ Transcription Completed!")

    elapsed = time.time() - start_time
    runtime_placeholder.success(f"⏱️ Total Runtime: {format_time(elapsed)}")

    final_text = "\n\n".join(transcript_paragraphs)
    file_size = len(final_text.encode("utf-8"))

    # === Progress bar download fake ===
    download_status = st.empty()
    download_bar = st.progress(0)
    download_placeholder = st.empty()

    downloaded = 0
    step = 0

    base_speed = 80
    amplitude = 50
    period = 20

    while downloaded < file_size:
        speed_kb = base_speed + amplitude * math.sin(step / period) + random.uniform(-5, 5)
        speed_kb = max(10, speed_kb)

        chunk_size = speed_kb * 1024 * 0.1
        downloaded += chunk_size
        if downloaded > file_size:
            downloaded = file_size

        pct = int(downloaded / file_size * 100)
        download_bar.progress(pct)
        download_status.text(
            f"📥 Preparing download... {pct}% "
            f"({downloaded/1024:.1f}KB / {file_size/1024:.1f}KB) @ {speed_kb:.1f} KB/s"
        )

        time.sleep(0.1)
        step += 1

    download_bar.empty()
    download_status.success("✅ File ready to download!")

    # === Tombol download ===
    out_filename = f"transcript_{timestamp}.txt"
    download_placeholder.download_button(
        "💾 Download Transcript with Timestamp",
        data=final_text,
        file_name=out_filename,
        mime="text/plain",
    )

    # === Auto delete file upload setelah selesai ===
    if os.path.exists(file_path):
        os.remove(file_path)
        st.info(f"🗑️ File sementara dihapus otomatis: `{file_path}`")
