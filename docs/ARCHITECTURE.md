# SurgiGuard AI — architecture

## The problem

After surgery, most recovery happens at home: knee and hip replacements, abdominal surgery, C-sections. Complications show up there, often days after discharge, and patients or their families notice them late:

- wound infection;
- breathing problems and low oxygen;
- fever or a racing heart;
- falls;
- skipped medication.

Hospitals have no view of what happens between discharge and the follow-up visit. Phone calls do not scale.

SurgiGuard closes that gap. Measurements and photos taken at home go to a care team, and an agent turns each concerning change into a **tracked follow-up**. The flow is:

**observation → alert → acknowledged → clinical decision → instructions delivered → follow-up verified**

The system never diagnoses. Code applies auditable rules, the AI explains and drafts, and clinicians decide.

**Demo patient:** Sarah Miller, 64, right total knee replacement. She is fictional.

## Components

```
 ┌──────────────┐   MQTT (JSON, 1 msg/s)   ┌──────────────────────────┐        ┌──────────┐
 │ ESP32 board  │ ───────────────────────► │ SurgiGuard API (.NET 8)  │ ─────► │ MongoDB  │
 │ MAX30102     │   topic Hackcure/data    │ accounts, patients,      │        │ 7        │
 │ MPU6050      │                          │ telemetry, meds, alerts, │        └──────────┘
 │ DHT11        │   (no board? ↓)          │ family, WhatsApp/push    │
 └──────────────┘                          └────────────┬─────────────┘
 ┌──────────────────────────┐                   ▲       │ poll telemetry every 3 s (X-Api-Key)
 │ scripts/                 │  same JSON        │       ▼
 │ simulate_device_mqtt.py  │ ─► Mosquitto ─────┘  ┌─────────────────────────────────────────┐
 └──────────────────────────┘    (Docker)          │ AI backend (Python FastAPI, SQLite)     │
                                                   │  Monitor + urgent rules (code)          │
                                                   │  D1 recovery twin, rehab, WOUND PHOTOS  │
                                                   │  D2 documents → care plan (OCR) + RAG   │
                                                   │  D3 investigation + closed-loop follow-up│
                                                   │  LLM layer: Groq / Mistral / OpenRouter │
                                                   │            or offline "stub" mode       │
                                                   └───────┬─────────────────────────────────┘
                                                           │ HTTP :8000
          ┌───────────────────────┬────────────────────────┼───────────────────────┐
          ▼                       ▼                        ▼                       ▼
   /app/  patient + family   /agent-demo/  clinic     /story  one-page        /prototype
   (React Native, web build) (wound review, alerts)   whole story + chips     D1/D2/D3 console
```

| Component | Folder | Tech | Port |
|---|---|---|---|
| ESP32 firmware | `SurgiGuard/IOT` | C++ / PlatformIO | – |
| SurgiGuard API | `SurgiGuard/Backend/API` | ASP.NET Core 8, MongoDB.Driver, MQTTnet, JWT | 8080 in the container, published as `API_PORT` (8080 default; **5051 on this PC**, because Apache uses 8080) |
| MongoDB | Docker | mongo:7.0 | 27017 |
| Local MQTT broker | Docker | eclipse-mosquitto:2 | 1883 |
| AI backend | `backend/` | FastAPI, SQLModel/SQLite, LangGraph, LangChain, FAISS, OpenCV, PyTorch | 8000 |
| D1 library | `d1/` | Python package `surgiguard_d1` (twin, rehab, movement tests, **wound**) | – |
| Patient / family app | `mobile/` | React Native 0.87 (bare), with a web build through react-native-web + esbuild | served at :8000/app/ |
| Native Android app | `SurgiGuard/MoniteurAndroid` | Kotlin / Compose (needs Android Studio, optional) | – |
| Clinic dashboard (mock) | `dashboard/` | Vite + React (static mock, no API) | 5173 |

## The three AI areas

The AI work was split between three developer roles, D1, D2 and D3 (see `contexte/`).

### D1 — Recovery Twin, rehab and wound photos (`d1/`)
- **Expected recovery trajectory and patient twin:** what a normal recovery looks like for this surgery on day N, compared with what is observed.
- **Rehab:** exercise prescription, plus on-device pose estimation (MediaPipe) that counts repetitions and range of motion.
- **Wound photos at dressing changes:** an OpenCV pipeline, a trained model and a vision-LLM description. The full explanation is in [`docs/wound/HOW_V1_WORKS.md`](wound/HOW_V1_WORKS.md).

### D2 — Documents to care plan (`backend/app/services/ocr_service.py`, `backend/app/agents/protocol.py`)
1. The patient or clinician uploads a discharge letter or prescription.
2. Text is extracted. **OCR:** PyMuPDF reads the text layer; scanned pages are sent to the Mistral Pixtral vision model.
3. An LLM drafts the plan: medications, doses, times, appointments and warning signs.
4. **Code checks that every value appears in the source document.** A human confirms before the plan becomes active.
5. A RAG assistant (FAISS) answers questions from the active plan only.

### D3 — Investigation and follow-up (`backend/app/agents/`, LangGraph)
- Urgent rules are pure code in `backend/app/safety/urgent_rules.py` and are never delayed by the AI. Examples: SpO2 below 90%, temperature of 39.5 °C or more, or a red-flag phrase such as "chest pain". Thresholds come from the care plan. Unusable readings never trigger an urgent alert.
- Non-urgent anomalies open an investigation. The agent may ask the patient a question or request a measurement or photo, then summarise the evidence and its uncertainty.
- Each alert follows the closed loop. Family members are notified through the .NET API (push or WhatsApp, simulated by default).

## Data flow of one heart-rate reading

1. The ESP32 (or the simulator) publishes this on `Hackcure/data`:
   `{"device_id":"ESP32_001","bpm":128,"spo2":97,"temp":23.1,"hum":48,"activity":0.1,"motion":"Immobile"}`
2. `MqttBridgeService.cs` parses it and maps activity to IMMOBILE, LEGER or AGITE. `TelemetryService.IngestAsync` stores it in MongoDB for the patient who owns `ESP32_001`, and flags anomalies.
3. Every 3 s the AI backend pulls new readings from `GET /api/agents/patients/{id}/telemetry/since` (`backend/app/integrations/surgiguard_api.py`). It averages them over 15 s windows and passes them to the Monitor.
4. The Monitor applies its rules. Here it finds tachycardia at rest and opens a D3 investigation. The clinician sees the alert on `/agent-demo/`, and the family app gets an update.

## Data flow of one wound photo

The details are in [`docs/wound/HOW_V1_WORKS.md`](wound/HOW_V1_WORKS.md).

1. The phone camera takes the photo after the patient confirms the dressing change. It goes to `POST /patients/{id}/wound-photos`.
2. **OpenCV analysis** runs immediately: quality check, segmentation, measurements and comparison with the day-1 photo.
3. **Background job:** the trained model reads the photo, then the vision LLM describes it, then a code rule decides whether a clinician should review it.
4. The clinician page shows the overlay, the measurements, the model table and the vision findings. The patient app shows a simplified view.

## What is real and what is simulated in the school demo

| Part | Status |
|---|---|
| .NET API, MongoDB, MQTT bridge, AI backend, rules, agents, wound CV, OCR | **Real code, running** |
| ESP32 readings | **Simulated** by `scripts/simulate_device_mqtt.py`. It sends the same message format through the same MQTT path as the real board. |
| Patient, family, doctor | Fictional accounts (`seed_story.py`) |
| LLM answers | Live with a Groq or Mistral key, otherwise **stub mode**: canned AI text, with all safety rules still real |
| WhatsApp and push to family | Simulated (logged, not sent) |
| Wound photos | Real licensed photos (`backend/seed/photos`, Wikimedia) or your own uploads |

## Safety principles

These are visible in the code:
- The AI never diagnoses. `output_guard` removes diagnosis wording before anything reaches a patient.
- Urgent escalation is pure code and is never waiting on an LLM.
- Every medication, dose or date in an AI-drafted plan must be found in the source document. A human confirms it.
- Every alert, decision and follow-up is audited with an actor and a timestamp.
- Models abstain ("uncertain") instead of guessing on unfamiliar or low-quality inputs.
