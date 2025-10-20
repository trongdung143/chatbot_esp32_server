import edge_tts
from faster_whisper import WhisperModel
import io, asyncio
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import uuid
import os
import math


async def stt(wav_path) -> str:
    model_path = "src/model/faster-whisper-base"
    model = WhisperModel(
        model_path, device="cpu", compute_type="int8", local_files_only=True
    )

    segments, _ = model.transcribe(wav_path, language="vi")
    text = "".join([s.text for s in segments])
    return text


async def send_pcm_from_buf(mp3_buf, websocket, sample_rate=16000, chunk_samples=1024):
    mp3_buf.seek(0)
    audio = AudioSegment.from_file(mp3_buf, format="mp3")
    pcm = audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
    pcm_bytes = pcm.raw_data

    bytes_per_chunk = chunk_samples * 2
    for i in range(0, len(pcm_bytes), bytes_per_chunk):
        frame = pcm_bytes[i : i + bytes_per_chunk]
        if len(frame) < bytes_per_chunk:
            continue
        await websocket.send_bytes(frame)
        await asyncio.sleep(0.02)


def estimate_chunks_from_text(text, sample_rate=16000, chunk_samples=1024):
    est_duration = len(text) * 0.06
    chunk_time = chunk_samples / sample_rate
    est_chunks = math.ceil(est_duration / chunk_time)
    return est_chunks, est_duration


async def tts_stream_pcm(
    websocket, text, sample_rate=16000, chunk_samples=1024, save_audio=True
):
    chunks, seconds = estimate_chunks_from_text(text)
    msg = f"start_stream_audio|chunks={chunks}"
    await websocket.send_text(msg)

    communicate = edge_tts.Communicate(text, "vi-VN-HoaiMyNeural")
    mp3_buf = io.BytesIO()
    total_buf = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buf.write(chunk["data"])
            total_buf.write(chunk["data"])

            if mp3_buf.tell() > 30000:
                await send_pcm_from_buf(mp3_buf, websocket, sample_rate, chunk_samples)
                mp3_buf = io.BytesIO()

    if mp3_buf.tell() > 0:
        await send_pcm_from_buf(mp3_buf, websocket, sample_rate, chunk_samples)

    if save_audio and total_buf.tell() > 0:
        total_buf.seek(0)
        audio_id = str(uuid.uuid4())
        save_dir = "src/data"
        os.makedirs(save_dir, exist_ok=True)
        mp3_path = os.path.join(save_dir, f"{audio_id}.mp3")
        wav_path = os.path.join(save_dir, f"{audio_id}.wav")

        with open(mp3_path, "wb") as f:
            f.write(total_buf.getvalue())
        print(f"💾 Saved MP3: {mp3_path}")

        audio = AudioSegment.from_file(total_buf, format="mp3")
        pcm = audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
        pcm.export(wav_path, format="wav")
        print(f"💾 Saved WAV: {wav_path}")

    await websocket.send_text("end_stream_audio")
    print("🔁 Sent: end_stream_audio")


async def save_wav(client_id: str, pcm_buffer: bytearray, sample_rate: int = 16000):
    if not pcm_buffer:
        print("⚠️ Empty buffer, skipping save.")
        return None

    client_id = str(uuid.uuid4())
    wav_path = f"src/data/{client_id}.wav"

    audio = np.frombuffer(pcm_buffer, dtype=np.int16)
    sf.write(wav_path, audio, sample_rate)
    print(
        f"✅ WAV saved: {wav_path} ({len(audio)/sample_rate:.2f}s, {len(pcm_buffer)} bytes)"
    )

    return wav_path
