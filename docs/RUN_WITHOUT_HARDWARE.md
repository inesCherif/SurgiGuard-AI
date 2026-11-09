# Run SurgiGuard AI without the ESP32 or a phone

Everything runs on one Windows PC.
- The wearable is replaced by a **virtual ESP32**. It publishes the board's exact MQTT message through a local broker, so the real data path is used: MQTT → .NET API → MongoDB → AI backend.
- The phone app runs **in the browser**.

The full explanation is in [`REBUILD_GUIDE.md`](REBUILD_GUIDE.md).

## Requirements
Docker Desktop (running), Python 3.11, Node 22. API keys are optional.

## First time only
```powershell
git clone https://github.com/inesCherif/SurgiGuard-AI.git; cd SurgiGuard-AI
cd backend
python -m venv .venv --system-site-packages
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -e "..\d1[wound,dev]"
copy .env.example .env          # optional: GROQ_API_KEY=..., MISTRAL_API_KEY=...
cd ..\mobile; npm install; cd ..
```

## Every time
```powershell
.\scripts\demo.ps1 -Profile story
```

| Open | Log in |
|---|---|
| http://localhost:8000/story | the whole chain, with live status chips |
| http://localhost:8000/app/ | `sarah@story.test` / `Story2026!` (patient) or `ahmed@story.test` (family); or "continue as demo" |
| http://localhost:8000/agent-demo/ | clinic: alerts, investigation evidence, wound review |
| http://localhost:8000/docs | AI backend API |
| http://localhost:5051/swagger | .NET API (port 8080 if it is free on your PC) |

## Simulated vitals
```powershell
backend\.venv\Scripts\python.exe scripts\simulate_device_mqtt.py --profile normal     # HR ~74, SpO2 ~97
backend\.venv\Scripts\python.exe scripts\simulate_device_mqtt.py --profile desat      # SpO2 ~87 -> urgent alert
backend\.venv\Scripts\python.exe scripts\simulate_device_mqtt.py --profile tachy      # HR ~128 -> investigation
backend\.venv\Scripts\python.exe scripts\simulate_device_mqtt.py --profile nofinger   # sensor not worn -> "unknown", never "normal"
backend\.venv\Scripts\python.exe scripts\simulate_device_mqtt.py --profile story --phase-seconds 60
```
To watch the messages: `docker exec -it mqtt mosquitto_sub -t Hackcure/data -v`

## Stop
Press Ctrl+C in the "AI backend" and "Virtual ESP32" windows, then run `.\scripts\demo.ps1 -Stop`.
