from fastapi import APIRouter, WebSocket
from src.api.utils import stt, tts_stream_pcm, save_wav
import os
from langchain_core.messages import HumanMessage
from src.agents.workflow import graph

router = APIRouter()


@router.websocket("/ws_audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    print("✅ ESP32 connected")

    client_id = "unknown"
    pcm_buffer = bytearray()

    try:
        while True:
            data = await websocket.receive()

            if data["type"] == "websocket.disconnect":
                print(f"❌ Disconnected: {client_id}")
                break

            if "text" in data:
                msg = data["text"].strip()

                if msg.startswith("start_chat|"):
                    client_id = msg.split("|")[1]
                    pcm_buffer.clear()
                    print(f"🎙 Start from {client_id}")
                    continue

                elif msg == "end_chat":
                    print(f"🛑 End stream from {client_id}")

                    wav_path = await save_wav(client_id, pcm_buffer)
                    pcm_buffer.clear()

                    if wav_path:

                        text = await stt(wav_path)
                        print(f"🗣 STT result: {text}")

                        input_state = {
                            "client_id": client_id,
                            "messages": HumanMessage(content=text),
                        }

                        config = {
                            "configurable": {
                                "thread_id": client_id,
                            }
                        }

                        response = await graph.ainvoke(input=input_state, config=config)

                        answer = response.get(
                            "answer", "Xin lỗi tôi không hiểu câu hỏi của bạn"
                        )

                        print(f"🤖 Answer: {answer}")

                        try:
                            os.remove(wav_path)
                            print(f"Deleted temporary file: {wav_path}")
                        except Exception as e:
                            print(f"Failed to delete {wav_path}: {e}")

                        await tts_stream_pcm(websocket, answer, 160000, 1024, False)

                    continue

            elif "bytes" in data:
                chunk = data["bytes"]
                pcm_buffer.extend(chunk)

    except Exception as e:
        print(f"⚠️ Error: {e}")

    finally:
        print(f"🔚 Closed connection for {client_id}")
