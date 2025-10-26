from fastapi import APIRouter, WebSocket
from src.api.utils import stream_and_speak, stt_from_pcm
import asyncio
from langchain_core.messages import HumanMessage
from src.agents.workflow import graph

router = APIRouter()


@router.get("/")
async def home():
    return {"message": "hello"}


@router.websocket("/ws_audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    client_id = "unknown"
    pcm_buffer = bytearray()

    try:
        while True:
            data = await websocket.receive()

            if data["type"] == "websocket.disconnect":
                print(f"Disconnected: {client_id}")
                break

            if "text" in data:
                msg = data["text"].strip()

                if msg.startswith("start_chat|"):
                    client_id = msg.split("|")[1]
                    pcm_buffer.clear()
                    print(f"Start from {client_id}")
                    continue

                elif msg == "end_chat":
                    print(f"End from {client_id}")

                    text = await stt_from_pcm(pcm_buffer)
                    pcm_buffer.clear()

                    input_state = {
                        "client_id": client_id,
                        "messages": HumanMessage(content=text),
                    }

                    config = {
                        "configurable": {
                            "thread_id": client_id,
                        }
                    }

                    await stream_and_speak(graph, websocket, input_state, config)

                    continue

            elif "bytes" in data:
                chunk = data["bytes"]
                pcm_buffer.extend(chunk)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        print(f"Closed connection for {client_id}")
