import asyncio
import wave
import edge_tts
import subprocess
import platform


# ---- Cấu hình ----
# Nếu ffmpeg nằm trong PATH, chỉ cần "ffmpeg"
# Nếu bạn có file ffmpeg.exe riêng, đổi đường dẫn bên dưới
FFMPEG_PATH = (
    r"src/model/mp3-to-pcm/ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
)


async def tts_stream_to_wav(text, wav_path="output.wav", sample_rate=16000):
    """
    Streaming thật sự:
    Edge-TTS -> ffmpeg (pipe) -> PCM -> ghi ra WAV liên tục.
    Không đợi TTS hoàn tất, âm thanh được decode ngay khi sinh ra.
    """

    # Tạo tiến trình ffmpeg
    process = await asyncio.create_subprocess_exec(
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",  # input từ stdin
        "-ac",
        "1",  # mono
        "-ar",
        str(sample_rate),  # 16kHz
        "-f",
        "s16le",  # PCM 16-bit
        "pipe:1",  # output ra stdout
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )

    # Khởi tạo TTS
    communicate = edge_tts.Communicate(text, "vi-VN-HoaiMyNeural")

    # Mở file WAV để ghi dần PCM
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        async def feed_mp3():
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    process.stdin.write(chunk["data"])
                    await process.stdin.drain()
            process.stdin.close()

        async def read_pcm():
            while True:
                pcm = await process.stdout.read(4096)
                if not pcm:
                    break
                wf.writeframes(pcm)

        # --- chạy song song ---
        await asyncio.gather(feed_mp3(), read_pcm())

    print(f"✅ Streamed and saved WAV file: {wav_path}")


if __name__ == "__main__":
    text = "Xin chào! Đây là thử nghiệm phát luồng Edge TTS được decode trực tiếp bằng ffmpeg."
    asyncio.run(tts_stream_to_wav(text))
