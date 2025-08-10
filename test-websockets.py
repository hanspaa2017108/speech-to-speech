import websocket
import os

# Replace with your key or keep as-is if using dotenv
API_KEY = os.environ.get("OPENAI_API_KEY")
url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
headers = [
    f"Authorization: Bearer {API_KEY}",
    "OpenAI-Beta: realtime=v1"
]

def on_open(ws):
    print("WS open callback")

def on_error(ws, err):
    print("WS error:", err)

def on_close(ws, a, b):
    print("WS closed:", a, b)

def on_message(ws, msg):
    print("WS msg:", msg)

ws = websocket.WebSocketApp(
    url, header=headers,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
)
ws.run_forever()
