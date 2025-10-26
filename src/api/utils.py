import asyncio
import edge_tts
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
import io


async def tts_stream_pcm(websocket, text, sample_rate=16000):

    process = await asyncio.create_subprocess_exec(
        "src/model/mp3-to-pcm/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )

    communicate = edge_tts.Communicate(
        text, "vi-VN-HoaiMyNeural", volume="-40%", rate="+50%"
    )

    pcm_buffer = bytearray()

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
            pcm_buffer.extend(pcm)
            await websocket.send_bytes(pcm)

    await asyncio.gather(feed_mp3(), read_pcm())


async def pcm_to_wav_bytes(pcm_buffer: bytearray, sample_rate: int = 16000) -> bytes:
    if not pcm_buffer:
        print("Empty buffer, skipping.")
        return b""

    audio = np.frombuffer(pcm_buffer, dtype=np.int16)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, sample_rate, format="WAV")
    wav_buf.seek(0)
    return wav_buf.read()


async def stt_from_pcm(pcm_buffer: bytearray, sample_rate: int = 16000) -> str:
    if not pcm_buffer:
        print("Empty audio buffer")
        return ""

    wav_bytes = await pcm_to_wav_bytes(pcm_buffer, sample_rate)
    wav_buf = io.BytesIO(wav_bytes)

    model_path = "src/model/faster-whisper-small"
    model = WhisperModel(
        model_path, device="cpu", compute_type="int8", local_files_only=True
    )

    segments, _ = model.transcribe(wav_buf, language="vi")
    text = "".join([s.text for s in segments])

    print(f"USER: {text.strip()}")
    return text.strip()


async def stream_and_speak(graph, websocket, input_state, config):
    await websocket.send_text(f"start_stream_audio")
    print("Started streaming TTS")
    tts_queue = asyncio.Queue()
    buffer_text = ""
    answer = ""

    async def tts_consumer():
        while True:
            text_part = await tts_queue.get()
            if text_part is None:
                tts_queue.task_done()
                break

            await tts_stream_pcm(websocket, text_part)
            tts_queue.task_done()

    consumer_task = asyncio.create_task(tts_consumer())

    async for event in graph.astream(
        input=input_state,
        config=config,
        stream_mode=["messages", "updates"],
        subgraphs=True,
    ):
        subgraph, data_type, chunk = event

        if data_type == "messages":
            response, meta = chunk
            text = str(response.content).strip() + " "
            if not text:
                continue
            buffer_text += text
            answer += text

            print(text)

            if any(p in buffer_text for p in [".", "!", "?", ","]):
                part = buffer_text.strip()
                buffer_text = ""
                await tts_queue.put(part)

    if buffer_text.strip():
        await tts_queue.put(buffer_text.strip())

    await tts_queue.put(None)
    await tts_queue.join()

    consumer_task.cancel()

    print(f"AI: {answer}")
    await websocket.send_text("end_stream_audio")
    print("Finished streaming TTS")
