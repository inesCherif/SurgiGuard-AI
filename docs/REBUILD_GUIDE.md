# SurgiGuard AI — complete rebuild guide

> **Goal of this document:** someone who has never seen the project can understand it and rebuild it from zero, in the right order, then run it and demo it.
> It follows the order the project was actually built in at the HackCure hackathon (10–11 October 2026, verified in `git log`), then the school-project additions (November 2026).

**Contents**
0. [Read this first](#0-read-this-first)
1. [The idea: problem, users, workflow](#1-the-idea)
2. [Architecture and tech stack](#2-architecture-and-tech-stack)
3. [Every model and algorithm used](#3-every-model-and-algorithm-used)
4. [Repository map](#4-repository-map)
5. [Install the tools (Windows)](#5-install-the-tools-windows)
6. [Rebuild order, step by step](#6-rebuild-order-step-by-step)
   - Step 1: specification
   - Step 2: ESP32 firmware
   - Step 3: SurgiGuard API (.NET + MongoDB)
   - Step 4: AI backend core (safety, alerts, agents)
   - Step 5: D2, documents to care plan (OCR) and RAG assistant
   - Step 6: D1 library (recovery, twin, rehab, movement tests, wound)
   - Step 7: integration (AI backend ↔ .NET API ↔ D1/D2)
   - Step 8: patient/family app (React Native and web)
   - Step 9: web pages (story, clinic, prototype)
   - Step 10: running without hardware (school project)
7. [How to run it](#7-how-to-run-it)
8. [Demo walkthrough (what to click)](#8-demo-walkthrough)
9. [Configuration reference](#9-configuration-reference)
10. [Key API endpoints](#10-key-api-endpoints)
11. [Tests](#11-tests)
12. [Troubleshooting](#12-troubleshooting)

Companion documents:
- [`ARCHITECTURE.md`](ARCHITECTURE.md): short overview.
- [`RUN_WITHOUT_HARDWARE.md`](RUN_WITHOUT_HARDWARE.md): quick start.
- [`wound/HOW_V1_WORKS.md`](wound/HOW_V1_WORKS.md): wound computer vision in depth.
- [`wound/PROBLEMS_AND_SOLUTIONS.md`](wound/PROBLEMS_AND_SOLUTIONS.md): engineering journal.

---

## 0. Read this first

- **Fictional data only.** The demo patient is *Sarah Miller, 64, right total knee replacement*. This is **not a medical device**: medical decisions always stay with clinicians.
- The system is made of **5 programs** that talk to each other:
  1. **ESP32 firmware**: the wearable. In the school demo it is replaced by a simulator.
  2. **SurgiGuard API**: C# .NET 8 with MongoDB. Holds accounts, devices, telemetry, medications and family alerts.
  3. **AI backend**: Python FastAPI with SQLite. Holds the rules, the AI agents, documents, wound photos and the web pages.
  4. **`d1` library**: a Python package used by the AI backend. Recovery prediction, twin, rehab, movement tests, wound CV.
  5. **Patient/family app**: React Native. Runs on Android, or **in the browser** at `/app/`.
- **Three AI areas, named after the three developer roles** (`contexte/`):
  - **D1**, recovery: twin, rehab, wounds.
  - **D2**, documents: care plan and assistant.
  - **D3**, agents: monitoring, investigation, alerts and follow-up.

---

## 1. The idea

### 1.1 Problem
After surgery most of the recovery happens **at home**. Complications appear there, often days after discharge:
- wound infection (most surgical-site infections appear 5–10 days after the operation);
- low oxygen;
- fever or a racing heart;
- falls;
- missed medication.

Between discharge and the follow-up visit, the hospital sees nothing. Phone calls do not scale, and wearables alone produce data nobody acts on.

### 1.2 Solution: "evidence-to-action" recovery
SurgiGuard links **the patient at home**, **a trusted caregiver** (family) and **the clinical team**. It does not stop at collecting data or sending an alert. It **tracks each concern until a clinician has acted and the follow-up is verified**:

```
Monitor → Detect → Investigate → Escalate → Act → Verify → Continue recovery
observation → alert → acknowledged → clinical decision → instructions delivered → follow-up verified
```

### 1.3 Users
| User | Can do | Cannot do |
|---|---|---|
| Patient | upload discharge documents; see the care plan; daily check-in; wound photos at dressing changes; rehab exercises; ask the assistant; see vitals | see risk scores or diagnosis words |
| Family / caregiver | see only what the patient allows; get alerts (push / WhatsApp); help with tasks | make medical decisions |
| Clinician | see the evidence; acknowledge alerts; record decisions; review wound photos; verify follow-up | — (the only one who closes an alert) |

### 1.4 Safety principles (enforced in code, not only in prompts)
1. **AI never diagnoses.** `backend/app/safety/output_guard.py` deletes diagnosis words before anything reaches a patient.
2. **The urgent path is pure code.** `backend/app/safety/urgent_rules.py` never waits for an LLM, and works even if the LLM is down.
3. **Missing data is "unknown", never "normal"** (`safety/quality.py`).
4. **The AI drafts and code verifies.** Every medication, dose and date in an AI-drafted plan must be found in the source document.
5. **Only clinicians acknowledge or close alerts.** No agent can lower an alert (`alerts.py`).
6. **Everything is audited**, with actor and timestamp (`audit.py`).
7. **Models abstain** ("uncertain") on bad or unfamiliar input instead of guessing.

---

## 2. Architecture and tech stack

```
 ┌──────────────┐  MQTT JSON 1/s   ┌──────────────────────────────┐        ┌─────────┐
 │ ESP32 board  │ ───────────────► │ SurgiGuard API  (.NET 8)     │ ─────► │ MongoDB │
 │ MAX30102 HR/ │ topic            │ auth JWT, patients, devices, │        └─────────┘
 │ SpO2, MPU6050│ Hackcure/data    │ telemetry+anomalies, meds,   │
 │ motion, DHT11│                  │ family, alerts→push/WhatsApp │
 └──────────────┘                  └─────────┬────────────▲───────┘
  school demo: scripts/simulate_device_mqtt.py│ poll 3 s   │ alerts, med sync
  → Mosquitto (Docker) → same path            ▼ X-Api-Key  │
                                   ┌──────────────────────────────────────────┐
                                   │ AI backend (FastAPI :8000, SQLite)       │
                                   │ Monitor → urgent rules → investigation   │
                                   │ agents (LangGraph), alerts state machine │
                                   │ D2 OCR + plan compiler + RAG assistant   │
                                   │ D1 twin / rehab / movement / WOUND CV    │
                                   │ llm.py: Groq | Mistral | OpenRouter | stub│
                                   └──────────────────┬───────────────────────┘
                     /app/ (patient+family) · /agent-demo/ (clinic) · /story · /prototype · /docs
```

| Layer | Technology | Why |
|---|---|---|
| Device | ESP32 DevKit, Arduino framework, PlatformIO, PubSubClient (MQTT over TLS) | cheap, Wi-Fi, lots of sensor libraries |
| Messaging | MQTT (EMQX Cloud for the board, Mosquitto locally) | light pub/sub protocol, standard for IoT |
| Account/device API | ASP.NET Core 8, MongoDB.Driver, MQTTnet, JWT, Swagger | team skill; Mongo stores time-series readings easily |
| AI backend | Python 3.11, FastAPI, SQLModel (SQLite), APScheduler | fast to build; Python owns the AI ecosystem |
| Agents | LangGraph, LangChain, Pydantic | explicit state machines that can pause for a human and resume |
| LLMs | Groq (gpt-oss-20b/120b, Qwen 3.8 27B vision), Mistral (Medium/Small, Pixtral, embeddings), OpenRouter; offline **stub** mode | free tiers; one wrapper (`llm.py`) so any agent can switch provider |
| RAG | FAISS + Mistral embeddings (fake embeddings offline) | answer only from the patient's own plan |
| Computer vision | OpenCV, PyTorch, Hugging Face Transformers (CLIP), scikit-learn | classical CV + transfer learning on a small dataset |
| Pose | MediaPipe (on the phone/browser) | landmarks computed on the device; no video leaves the phone |
| Mobile | React Native 0.87 (bare), React Navigation, vision-camera, three.js (3D twin), react-native-web + esbuild | one codebase for Android and the browser |
| Infra | Docker Compose (MongoDB, Mosquitto, API) | runs on any PC |

---

## 3. Every model and algorithm used

| Where | Model / algorithm | Type | Notes |
|---|---|---|---|
| Firmware `vitals.cpp` | SparkFun `checkForBeat` + Maxim `maxim_heart_rate_and_oxygen_saturation` | signal processing | HR = mean of the last 5 valid beats; SpO2 from 100 samples at 25 Hz; automatic LED current |
| Firmware `motion.cpp` | magnitude of acceleration minus gravity | threshold | `activity` < 0.3 = still, < 1.5 = light movement, else agitated |
| API `TelemetryService.DetectAnomalies` | thresholds from the patient's settings | rules | tachycardia, bradycardia, hypoxia, fall |
| AI `safety/urgent_rules.py` | thresholds + red-flag phrase matching | rules | SpO2 < 90 urgent; temperature ≥ 38.5 attention, ≥ 39.5 urgent; phrases like "chest pain" (even when negated, on purpose) |
| AI investigation | LLM tool-calling loop in LangGraph (Groq gpt-oss-120b by default; Mistral / OpenRouter possible) | agent | at most 8 tool calls and 2 patient questions; every claim must cite an observation |
| AI check-in, digest, chat | LLM through `llm.py` | generation | template fallback when the LLM is unavailable |
| D2 OCR | **PyMuPDF** text layer → **Mistral Pixtral-12B** for scanned pages | VLM OCR | Pixtral was chosen over Tesseract because it handles handwriting and layout better |
| D2 extraction | LangGraph `protocol_graph.py` + LLM (Groq gpt-oss-20b) → Pydantic `ExtractedProtocol` | structured extraction | 20+ clinical fields |
| D2 safety | `safety_graph.py`: fixed allergy and interaction rules, then an LLM review | rules + LLM | |
| D2 assistant | FAISS retriever + semantic chunker + LangGraph assistant (memory, self-check) | RAG | answers only from the active plan |
| D1 expected recovery | LLM in 2 steps with a fixed schema + plausibility checks (`trajectory_prediction/predict.py`) | generation + validation | computed once and cached |
| D1 Patient Twin | `PersistenceForecaster`, `TrendForecaster` (optional `ChronosForecaster`) with an 80% band | forecasting | a more complex method is used only if a backtest shows it is better; abstains otherwise |
| D1 rehab | MediaPipe landmarks → joint angles (hip-knee-ankle), smoothing, rep counter, range of motion | geometry | prescription engine by surgery and post-op day |
| D1 movement tests | 6 camera tests (arm drift, face symmetry, finger tapping, eye movement…) comparing left and right | geometry/statistics | engineering cut-offs, not clinically validated |
| D1 wound A | OpenCV: Laplacian sharpness, YCrCb skin mask, CIELAB local contrast, top-hat staple detection, ORB + RANSAC registration | classical CV | see `wound/HOW_V1_WORKS.md` |
| D1 wound B | **CLIP ViT-B/16 (frozen)** + 6 logistic-regression heads trained on **SurgWound** (686 photos) | transfer learning | test AUROC 0.76–0.84; out-of-distribution gate; occlusion map |
| D1 wound C | Vision LLM: Groq Qwen 3.8 27B (was Llama-4-Scout) / Mistral Pixtral / Gemma-3 | VLM | describes, never diagnoses; JSON output; guarded |

---

## 4. Repository map

```
HackCure/
├── PROJECT.md, contexte/          ← STEP 1  specification (vision, users, D1/D2/D3 roles)
├── SurgiGuard/
│   ├── IOT/                       ← STEP 2  ESP32 firmware (PlatformIO)
│   │   ├── platformio.ini
│   │   └── src/ config.example.h → vitals.* → motion.* → ambient.* → network.* → main.cpp
│   ├── Backend/
│   │   ├── API/                   ← STEP 3  ASP.NET Core 8 API
│   │   │   ├── Settings/ Entites/ Data/ Dtos/ Security/ Services/ Controllers/ Program.cs
│   │   │   ├── appsettings.json, Dockerfile, API.csproj
│   │   ├── Infrastructure_hackathon/  docker-compose.yml (+ mosquitto.conf)   ← STEP 3 / 10
│   │   └── scripts/  seed_story.py, simulate_esp32.py
│   ├── MoniteurAndroid/           native Kotlin app (optional, needs Android Studio)
│   └── AppMobile/                 older app versions (replaced by mobile/)
├── backend/                       ← STEPS 4, 5, 7, 9  AI backend (FastAPI)
│   ├── main.py                    uvicorn entry (imports app.main)
│   ├── app/
│   │   ├── config.py db.py models.py clock.py audit.py         core
│   │   ├── safety/  urgent_rules.py quality.py output_guard.py  safety lane (no LLM)
│   │   ├── alerts.py notifications.py monitor.py runner.py scheduler.py sweep.py
│   │   ├── llm.py llm_stubs.py                                   one LLM wrapper + offline stubs
│   │   ├── agents/  investigation*.py evidence*.py checkin.py medication.py digest.py followup.py protocol.py
│   │   ├── tools/investigation_tools.py
│   │   ├── services/ocr_service.py  ai/graphs/*  ai/rag/*  api/v1/*  schemas/*  d2_models/   ← D2
│   │   ├── integrations/ surgiguard_api.py d1_bridge.py d1_motor.py d1_wound.py d2_bridge.py
│   │   ├── wounds.py wound_photos.py wound_image.py requests_service.py medication_guide.py
│   │   └── routers/  one file per HTTP area
│   ├── seed/  demo_seed.py simulator.py photos/ documents/
│   ├── web/   index.html (clinic /agent-demo), story.html, prototype.html
│   └── tests/ 22 pytest files
├── d1/                            ← STEP 6  Python package surgiguard_d1
│   ├── src/surgiguard_d1/ trajectory_prediction/ patient_twin/ physiology/ rehab/ neuro/ wound/
│   ├── scripts/ prepare_surgwound.py train_wound_model.py …
│   ├── models/  wound_heads.joblib  wound_model_metrics.json  WOUND_MODEL_CARD.md
│   └── tests/
├── mobile/                        ← STEP 8  React Native app (+ web build → /app/)
│   ├── src/ context/ services/ navigation/ screens/{onboarding,patient,family}/ components/ i18n/ theme/ web/
│   └── scripts/build-web.mjs
├── dashboard/                     static clinician mock (Vite)
├── scripts/                       ← STEP 10  demo.ps1, simulate_device_mqtt.py, start-local.ps1
└── docs/                          this documentation
```

---

## 5. Install the tools (Windows)

| Tool | Version | Used for | Check |
|---|---|---|---|
| Git | any | code | `git --version` |
| Python | **3.11** (or 3.12) | AI backend, d1, scripts | `python --version` |
| Docker Desktop | recent | MongoDB, Mosquitto, .NET API (so no .NET SDK needed) | `docker info` |
| Node.js | **22+** | building the web app | `node --version` |
| (optional) .NET SDK 8 | 8.x | running the API without Docker | `dotnet --version` |
| (optional) PlatformIO | – | flashing a real ESP32 | `pio --version` |
| (optional) Android Studio + JDK 17 | – | installing the app on an Android phone | – |
| (optional) Groq / Mistral API keys | – | live AI (otherwise stub mode) | – |

```powershell
git clone https://github.com/inesCherif/SurgiGuard-AI.git
cd SurgiGuard-AI

# Python environment (once). --system-site-packages reuses torch etc. if already installed globally.
cd backend
python -m venv .venv --system-site-packages
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -e "..\d1[wound,dev]"
copy .env.example .env        # then put GROQ_API_KEY / MISTRAL_API_KEY in it (optional)
cd ..

# Web app dependencies (once)
cd mobile; npm install; cd ..
```

**About torch:**
- `pip install -r requirements.txt` installs the **CPU** build of torch if none is present. That is enough to run the app.
- For GPU training (Wound CV v2) use `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`. It is a 2.5 GB download.

---

## 6. Rebuild order, step by step

Each step lists **the files in the order to write them**, **what each one does**, and **how to check it** before moving on.

### STEP 1 — Specification
**Files:** `PROJECT.md` (vision, users, workflow, MVP acceptance criteria) and `contexte/` (the three developer roles D1 / D2 / D3).
**Decisions taken:**
- Surgery: total knee replacement.
- Fictional patient: Sarah Miller.
- Sensors: heart rate, SpO2, movement, room temperature.
- Safety first: no AI in the urgent path.

### STEP 2 — ESP32 firmware (`SurgiGuard/IOT`)
| Order | File | What it does |
|---|---|---|
| 1 | `platformio.ini` | board `esp32dev`, Arduino framework, libraries (SparkFun MAX3010x, Adafruit MPU6050, DHT, PubSubClient) |
| 2 | `src/config.example.h` → copy to `config.h` | Wi-Fi, MQTT broker/port/user/password, topic `Hackcure/data`, `DEVICE_ID "ESP32_001"`. `config.h` is git-ignored. |
| 3 | `src/vitals.h/.cpp` | **MAX30102**: red and infrared light through the finger. HR from beat detection (average of the last 5 beats); SpO2 from the red/IR ratio over 100 samples; automatic LED power so the signal does not saturate. Reports 0 when there is no finger. |
| 4 | `src/motion.h/.cpp` | **MPU6050** accelerometer → `activity` (m/s² above gravity) → label Immobile / Leger mouvement / Agite |
| 5 | `src/ambient.h/.cpp` | **DHT11** → room temperature and humidity (NOT body temperature); `null` on a failed read |
| 6 | `src/network.h/.cpp` | Wi-Fi + MQTT over TLS; `publish()` builds the JSON |
| 7 | `src/main.cpp` | setup (I2C at 100 kHz, sensors) and a loop that publishes about once per second |

**Message format:** this is the contract between the device and the backend.
```json
{"device_id":"ESP32_001","temp":23.4,"hum":48.0,"bpm":74,"spo2":97,"activity":0.12,"motion":"Immobile"}
```
**Check:** `pio run -t upload`, then `pio device monitor`. Without a board, skip to Step 10 (simulator).

### STEP 3 — SurgiGuard API: .NET 8 + MongoDB (`SurgiGuard/Backend/API`)
Write it bottom-up: data, then logic, then HTTP.

| Order | Folder / file | What it does |
|---|---|---|
| 1 | `API.csproj` | packages: MongoDB.Driver, MQTTnet, JwtBearer, Swashbuckle, FirebaseAdmin |
| 2 | `Settings/AppSettings.cs` + `appsettings.json` | typed settings: Mongo, Jwt, ServiceApiKeys (Iot, Agents), Mqtt, WhatsApp, Firebase, ChatService |
| 3 | `Entites/` | `User`, `Patient` (device_id, thresholds, link_code), `FamilyMember`, `PatientFamilyRelation`, `TelemetryReading`, `Medication`, `AlertLog`, `Enums` |
| 4 | `Data/MongoDbContext.cs`, `MongoConventions.cs`, `Ids.cs` | collections, snake_case, indexes, telemetry retention (90 days) |
| 5 | `Dtos/` | request/response records (snake_case JSON) and `Mappers.cs` |
| 6 | `Security/` | `TokenService` (JWT), `Roles` (PATIENT, FAMILY_MEMBER, DOCTOR…), `ServiceApiKeyAttribute` (header `X-Api-Key` for the device and the AI) |
| 7 | `Services/TelemetryService.cs` | `IngestAsync`: find the patient by device_id → store the reading → `DetectAnomalies` (tachycardia, bradycardia, hypoxia, fall) |
| 8 | `Services/AlertService.cs` + `Notifications.cs` | store the alert, then notify family by severity: 1 → push; 2 → push + WhatsApp to the main contact; 3 → push + WhatsApp to everyone opted in. Simulated unless enabled. |
| 9 | `Services/MedicationService.cs`, `MedicalFolderService.cs`, `PatientAccessService.cs` | medication schedule and alarms, patient folder, who can see whom |
| 10 | `Services/MqttBridgeService.cs` | background service: subscribes to `Hackcure/data`, parses the device JSON, calls `TelemetryService.IngestAsync` |
| 11 | `Controllers/` | `Auth` (register/login), `Patients` (`/me`, `/me/device`), `Family` (link with a code), `Telemetry` (`/ingest`, history), `Medications`, `Alerts`, `Agents` (routes for the AI backend, key "Agents"), `Chat` and `Wounds` (relays to the AI backend) |
| 12 | `Program.cs` | dependency injection, JWT, Swagger, CORS, `/health` |
| 13 | `Dockerfile` + `Infrastructure_hackathon/docker-compose.yml` | Mongo 7 + API (+ Mosquitto, Step 10) |

**Check:** `docker compose up -d --build`, then open `http://localhost:<API_PORT>/swagger` and `/health`. Then run `seed_story.py` (Step 10) to create accounts.

### STEP 4 — AI backend core (`backend/app`)
This followed the team's "Backend step 1…10" commits. Write it in this order:

| Order | File(s) | What it does |
|---|---|---|
| 1 | `config.py`, `database.py`, `db.py`, `models.py`, `clock.py` | settings from `.env`; SQLite through SQLModel; tables: `Patient`, `SurveillancePlan`, `Observation`, `Investigation`, `PatientRequest`, `Alert`, `PhotoCollectionSlot`, `FollowupTask`/`Verification`, `CaregiverTask`, `Notification`, `Digest`, `Document`, `PlanDraft`, `AuditEvent`; a clock that can be moved forward for demos |
| 2 | `seed/demo_seed.py`, `main.py` (skeleton), `routers/patients.py` | fictional Sarah, her care plan, the `/today` screen |
| 3 | `safety/urgent_rules.py`, `safety/quality.py` | **pure functions, no LLM**: vital thresholds, red-flag phrases, reading quality; missing data is "unknown" |
| 4 | `alerts.py`, `notifications.py`, `audit.py` | alert **state machine**: created → assigned → acknowledged → patient_contacted → action_taken → resolved. Only clinicians move it forward. An alert not acknowledged before the timeout escalates to the backup contact. |
| 5 | `monitor.py`, `seed/simulator.py`, `routers/observations.py` | **Monitor**: quality → urgent rules → deviation → investigation request. Runs synchronously and never calls an LLM. |
| 6 | `agents/medication.py` | rules-only agent: reminders, dose log; 2 or more missed doses in 48 h → caregiver task |
| 7 | `llm.py`, `llm_stubs.py` | **one entry point for every LLM call**. Modes: `live` / `stub` (canned answers) / `off` (simulates an outage). Providers: Mistral, Groq, OpenRouter. Per-feature routing with `LLM_ROUTE_<AGENT>`. Timeout, retry, pacing for free tiers. |
| 8 | `agents/investigation.py`, `tools/investigation_tools.py`, `agents/evidence.py`, `requests_service.py` | **Investigation agent**: tool loop (read vitals, read check-ins, ask the patient, request a measurement or photo). Hard limits are enforced in code. `evidence.py` checks that each claim cites a real observation, and falls back to a rule-based summary otherwise. |
| 9 | `agents/checkin.py`, `routers/checkin.py` | daily check-in (0–10 sliders): urgent rules first, then at most one clarifying question |
| 10 | `agents/digest.py`, `routers/caregiver.py` | evening summary for the family, **filtered by permissions before the LLM sees anything**, then checked |
| 11 | `runner.py`, `scheduler.py`, `sweep.py`, `routers/demo.py` | background jobs (so requests never wait for an LLM); periodic jobs: sweep, medication (60 s), ESP32 sync (3 s), API link (20 s), check-in (8 h/20 h), digest (20:05); demo endpoints (reset, scenario, advance clock, LLM mode) |
| 12 | `agents/investigation_graph.py`, `investigation_workflow.py`, `evidence_review.py`, `followup.py` | investigation moved to **LangGraph**: the graph pauses when the patient or clinician must act and resumes after a server restart (`investigation.recover`) |
| 13 | `account_scope.py`, `safety/output_guard.py`, `routers/*` | ownership checks; diagnosis-word filter; HTTP routes |

**Check:** `uvicorn main:app --port 8000`, then open `http://localhost:8000/docs`. Then `python -m seed.simulator --scenario infection_pattern` and look at the alerts.

### STEP 5 — D2: documents → care plan (OCR) and RAG assistant
| Order | File | What it does |
|---|---|---|
| 1 | `schemas/protocol.py`, `d2_models/` | Pydantic `ExtractedProtocol` (20+ fields: surgery, medications with dose and schedule, allergies, appointments, warning signs, restrictions…) |
| 2 | `services/ocr_service.py` | **OCR in 2 levels.** (1) PyMuPDF native text (about 0.1 s, no API). (2) Scanned page → PNG → base64 → **Mistral Pixtral-12B**. Reports which method was used. |
| 3 | `ai/graphs/protocol_graph.py` | LangGraph: OCR text → LLM → `ExtractedProtocol` |
| 4 | `ai/graphs/safety_graph.py` | fixed allergy and critical-interaction rules, then an LLM review for complex cases |
| 5 | `api/v1/documents.py`, `api/v1/checkin.py` | `/v1` endpoints: upload, complete missing fields, check-in |
| 6 | `ai/rag/vectorstore.py`, `ai/rag/retriever.py`, `ai/graphs/assistant_graph.py`, `ai/guardrails/*` | **RAG assistant**: semantic chunking (split where the topic changes), FAISS index, hybrid keyword ranking, memory, self-check, input sanitising and output validation |
| 7 | `agents/protocol.py` (Recovery Protocol Compiler), `routers/documents.py`, `integrations/d2_bridge.py` | documents → draft plan → **every medication, dose and date checked against the text** → patient confirms → active plan. Safety settings are copied from the current plan, never from the AI. |

**Check:** in the app, onboarding → Upload → sample documents → Protocol review. Or `POST /patients/sarah/documents/sample`.

### STEP 6 — D1 library (`d1/src/surgiguard_d1`)
| Order | Package | What it does |
|---|---|---|
| 1 | `trajectory_prediction/` | D2's JSON (anonymised) → LLM in 2 steps → a normal recovery plan **day by day** (expected HR, SpO2, pain, how the patient should feel), validated against a schema and plausibility checks |
| 2 | `physiology/` | turns readings into series, trends and a comparison with a reference (it never picks thresholds; D3 does) |
| 3 | `patient_twin/` | near-term forecast of the patient's own HR/SpO2 with an 80% band: persistence, then trend (Chronos optional). It **abstains** with too little data. `backtest.py` decides whether a more complex method is worth it. |
| 4 | `rehab/` | `prescription.py` (exercises by surgery and post-op day, clinician limits); `angles.py` (joint angles from MediaPipe landmarks, smoothing); `exercises/` (knee flexion, leg raise, ankle pumps: rep counter, range of motion, hold time); `session.py`, `evaluator.py` |
| 5 | `neuro/` | 6 guided movement tests from video landmarks, comparing left and right; "attention" only, never an emergency |
| 6 | `wound/config.py` → `colour.py` → `quality.py` → `segmentation.py` → `measurement.py` → `registration.py` → `render.py` → `pipeline.py` (Facade) → `model_bundle.py` → `model.py` → `scripts/prepare_surgwound.py` → `train_wound_model.py` → `export_wound_heads.py` → `models/` | wound computer vision: **[wound/V2_ARCHITECTURE.md](wound/V2_ARCHITECTURE.md)** (structure, patterns), **[wound/HOW_V1_WORKS.md](wound/HOW_V1_WORKS.md)** (algorithms) |

**Check:** `cd d1; python -m pytest -q`.

**Retrain the wound model:**
1. Download SurgWound from Hugging Face (`Sonulee/SurgWound`).
2. `python scripts/prepare_surgwound.py <json dir> <prep>`
3. `python scripts/train_wound_model.py <prep>`

### STEP 7 — Integration
| File | What it connects |
|---|---|
| `integrations/surgiguard_api.py` + `scheduler.esp32_sync_job` | AI backend ↔ .NET API. Pulls telemetry every 3 s (key `dev-agents-key`), averages over 15 s, sends it to the Monitor. Pushes alerts to `/api/alerts/trigger` (family notifications) and syncs medications. |
| `integrations/d2_bridge.py` | D2 output → plan draft; the "D2 JSON" handed to D1; RAG indexing |
| `integrations/d1_bridge.py` | expected recovery (cached in `backend/.cache/recovery`) and Patient Twin for the home screen |
| `integrations/d1_motor.py`, `routers/motor.py` | rehab sessions (with ESP32 heart rate) and movement tests (consent is opt-in) |
| `integrations/d1_wound.py`, `wound_views.py` (adapter), `wound_rules.py` (rule strategies), `wounds.py`, `wound_photos.py`, `routers/wounds.py` | wound photos at dressing-change windows → ONE OpenCV analysis → model → vision LLM → review rules |
| `.NET WoundsController`, `ChatController` | let the native Android app reach the AI backend through the .NET API |
| `routers/hub.py`, `routers/integration.py`, `web/story.html` | one page showing the whole chain with status chips (API link, ESP32 online, LLM mode, alerts) |

### STEP 8 — Patient/family app (`mobile/`)
| Order | Folder | What it does |
|---|---|---|
| 1 | `src/theme/`, `src/i18n/` (en, fr) | colours, fonts, translations |
| 2 | `src/context/AppState.tsx` | role, session, the two server addresses (AI backend and .NET API), demo mode |
| 3 | `src/services/backend.ts` (.NET API client), `src/services/api.ts` (AI backend client), `src/hooks/` | data fetching and polling |
| 4 | `src/navigation/RootNavigator.tsx` | onboarding → patient tabs or family tabs |
| 5 | `src/screens/onboarding/` | Welcome, RoleSelection, Auth, Upload (documents), Protocol (plan review) |
| 6 | `src/screens/patient/` | Home, CheckIn, Twin (3D, three.js), WoundScan (camera), Rehab, Medications, AIAssistant, Reports, Requests, Escalation, Consent, FamilyAccess, Profile |
| 7 | `src/screens/family/` | FamilyHome, Updates, FamilyRecovery, JoinPatient (link code), AlertScreens |
| 8 | `src/web/shims/`, `*.web.tsx`, `scripts/build-web.mjs` | **browser build**: esbuild bundles `src/web.tsx`, with react-native mapped to react-native-web and shims for native-only modules. The output `dist-web/` is served by the AI backend at `/app/`. |

**Build for the browser:** `cd mobile; $env:SG_ACCOUNT_URL="http://localhost:5051"; npm run build:web`. `SG_ACCOUNT_URL` is the .NET API address; it defaults to 8080.

**Android (optional):** `npm start`, then `cd android; .\gradlew.bat app:installDebug`. This needs Android Studio, the SDK and JDK 17.

### STEP 9 — Web pages (`backend/web/`)
| URL | File | Audience |
|---|---|---|
| `/story` | `story.html` | jury / professor: the whole chain on one page |
| `/agent-demo/` | `index.html` + `workflow.js` | **clinic**: patients, alerts, investigation evidence, **wound review**, follow-up |
| `/prototype` | `prototype.html` | D1/D2/D3 step-by-step console |
| `/app/` | `mobile/dist-web` | patient and family app |
| `/docs` | FastAPI | AI backend API |

### STEP 10 — Running without hardware (school project, November 2026)
| File | What it does |
|---|---|
| `SurgiGuard/Backend/Infrastructure_hackathon/docker-compose.yml` | adds a **Mosquitto** container (port 1883); the API's MQTT bridge points to it (`MQTT_BROKER=mqtt`); `API_PORT` is configurable (8080 by default); `MONGO_DB` selects the demo database |
| `SurgiGuard/Backend/Infrastructure_hackathon/mosquitto.conf` | listener 1883, anonymous access (local demo only) |
| `scripts/simulate_device_mqtt.py` | **virtual ESP32** that publishes the firmware's exact JSON once per second. Profiles: `normal`, `tachy`, `desat`, `nofinger`, `agitated`, and `story` (normal → tachy → desat → nofinger → normal). Signals drift smoothly, with noise and occasional short walks. |
| `SurgiGuard/Backend/scripts/seed_story.py` (existing) | accounts: Sarah (patient, device ESP32_001), Ahmed (family), Dr Ruiz (doctor); password `Story2026!` |
| `mobile/scripts/build-web.mjs`, `mobile/src/context/AppState.tsx` | `SG_ACCOUNT_URL` at build time, so the web app can reach the API on any port |
| `scripts/demo.ps1` | **one command** that runs everything (see §7.1) |

---

## 7. How to run it

### 7.1 One command
```powershell
cd C:\Users\juba839\OneDrive\3eme\hackathons\hackcure_esen\HackCure
.\scripts\demo.ps1                    # normal vitals
.\scripts\demo.ps1 -Profile story     # scripted scenario with alerts
.\scripts\demo.ps1 -AiMode stub       # force offline AI
.\scripts\demo.ps1 -Stop              # stop the containers
```

What it does:
1. Starts Docker Desktop if needed.
2. Picks the API port: 8080, or **5051 if 8080 is busy** (Apache/XAMPP on this PC).
3. `docker compose up -d --build` (Mongo, Mosquitto, API), then waits for `/health`.
4. Runs `seed_story.py`.
5. Builds the web app if needed.
6. Starts the AI backend in a new window.
7. Starts the virtual ESP32 in a new window.
8. Opens `/story`.

If PowerShell refuses to run scripts, run this once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 7.2 Manually, step by step (to understand each piece)
```powershell
# 1. Database + broker + .NET API (port 5051 here because 8080 is used by Apache)
cd SurgiGuard\Backend\Infrastructure_hackathon
$env:API_PORT="5051"; $env:MONGO_DB="DB_story"
docker compose up -d --build
docker compose ps                     # mongo, mqtt, api should be "running"
curl.exe http://localhost:5051/health
cd ..\..\..

# 2. Demo accounts
backend\.venv\Scripts\python.exe SurgiGuard\Backend\scripts\seed_story.py --api http://localhost:5051

# 3. AI backend (keep this window open)
cd backend
$env:SURGIGUARD_API_URL="http://localhost:5051"
.\.venv\Scripts\python.exe -m uvicorn main:app --port 8000

# 4. In a NEW window: virtual ESP32
backend\.venv\Scripts\python.exe scripts\simulate_device_mqtt.py --profile story --phase-seconds 60

# 5. Browser
start http://localhost:8000/story
```

### 7.3 Each piece alone
| Piece | Command |
|---|---|
| Simulator without a broker | `python scripts/simulate_device_mqtt.py --dry-run --seconds 5` |
| Simulator over HTTP (bypassing MQTT) | `python SurgiGuard/Backend/scripts/simulate_esp32.py --api http://localhost:5051 --profile desat --seconds 60` |
| Vitals straight into the AI backend | `cd backend; .venv\Scripts\python -m seed.simulator --url http://localhost:8000 --scenario infection_pattern --every 5` |
| Wound CV on one photo | see `wound/HOW_V1_WORKS.md` §10 |
| Live AI | put `GROQ_API_KEY=...` (and `MISTRAL_API_KEY=...` for OCR of scans) in `backend/.env`, then restart the backend; `/health` shows `llm_mode: live` |
| Reset the demo | `curl.exe -X POST http://localhost:8000/demo/reset` |
| MQTT traffic | `docker exec -it mqtt mosquitto_sub -t Hackcure/data -v` |
| API logs | `docker logs -f api` |

---

## 8. Demo walkthrough
1. **`/story`**: the chips show the API linked, the ESP32 **online** (the simulator is running) and the LLM mode.
2. **Discharge (D2):** open `/app/` → Patient → Upload → use the sample documents. The AI drafts the plan; on the review screen every medication and dose is checked against the document. Confirm it.
3. **Home:** today's expected recovery card (D1), live vitals from the virtual ESP32, and the Twin forecast.
4. **Concern:** run `-Profile story`, or `simulate_device_mqtt.py --profile desat`.
   - SpO2 drops → urgent rule → urgent alert.
   - With `tachy` → attention → investigation: the agent asks Sarah a question in the app (Requests). Answer it.
5. **Clinic:** `/agent-demo/` → the alert with evidence and uncertainty → acknowledge → record a decision → follow-up → verify.
6. **Family:** log in as `ahmed@story.test` → only the permitted updates, plus the alert.
7. **Wound:**
   - Patient → Wound scan → upload `backend/seed/photos/day1.jpg`, then `day3.jpg`.
   - The analysis shows: overlay, quality check, comparison, model attributes, vision description.
   - The clinic page shows the full table and, if triggered, a "clinician review suggested" alert.
8. **Rehab / movement tests:** the camera runs MediaPipe in the browser (needs `localhost` or https) and counts repetitions and range of motion.

---

## 9. Configuration reference

**`backend/.env`** (see `.env.example`; never commit it):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | auto | groq / mistral / openrouter / auto (the first one with a key) |
| `LLM_MODE` | live if a key is set, else stub | `stub` = canned answers, `off` = simulate an outage |
| `GROQ_API_KEY`, `GROQ_MODEL` | –, openai/gpt-oss-120b | free key at console.groq.com |
| `MISTRAL_API_KEY`, `MISTRAL_MODEL` | –, mistral-medium-latest | needed for Pixtral OCR of scans and for embeddings |
| `OPENROUTER_API_KEY` | – | optional |
| `LLM_ROUTE_<AGENT>` | – | e.g. `LLM_ROUTE_D2=groq:openai/gpt-oss-20b`, `LLM_ROUTE_WOUND_VISION=...` |
| `SURGIGUARD_API_URL` | http://localhost:8080 | address of the .NET API (**5051 on this PC**) |
| `SURGIGUARD_LINK` | on | off = do not talk to the .NET API |
| `ESP32_DEVICE_ID`, `ESP32_POLL_S`, `ESP32_WINDOW_S` | ESP32_001, 3, 15 | telemetry sync |
| `WOUND_MODEL` | on | off = skip the CLIP model |
| `WOUND_EVIDENCE_MAP` | off | on = occlusion heat map (slower) |

**Docker Compose variables:** `API_PORT` (8080), `MONGO_DB` (DB), `MQTT_ENABLED` (true), `MQTT_BROKER` (mqtt), `MQTT_PORT` (1883), `MQTT_TLS` (false), `JWT_KEY`, `IOT_API_KEY` (dev-iot-key), `AGENTS_API_KEY` (dev-agents-key).

To use the **real board** with the cloud broker:
```powershell
$env:MQTT_BROKER="h1211a42.ala.eu-central-1.emqxsl.com"; $env:MQTT_PORT="8883"; $env:MQTT_TLS="true"
docker compose up -d
```

---

## 10. Key API endpoints
**AI backend (:8000)**
- `GET /health`
- `GET /patients/{id}/today`, `/twin`, `/twin-forecast`, `/expected-recovery`
- `POST /observations`
- `POST /checkin`
- `GET /patients/{id}/requests`, `POST /requests/{id}/answer`
- `POST /patients/{id}/documents`, `POST /patients/{id}/plan-drafts`, `POST /plan-drafts/{id}/confirm`
- `POST /patients/{id}/wound-photos`, `GET /patients/{id}/wound-photos/{obs}`, `POST .../review`
- `GET /alerts`, then the lifecycle steps `POST /alerts/{id}/ack`, `/contacted`, `/action`, `/followup`, `/followup-evidence`, `/verify`, `/resolve`
- `/investigations/*`, `/medications/*`, `/caregiver/*`
- `/demo/reset`, `/demo/scenario/{name}`, `/demo/clock/advance`, `/demo/llm_mode/{mode}`
- `/hub/*`
- `/v1/*` (D2)

**.NET API (:5051 here)**
- `POST /api/auth/register|login`
- `GET /api/patients/me`, `PUT /api/patients/me/device`
- `POST /api/family/link-patient`
- `POST /api/telemetry/ingest` (X-Api-Key: dev-iot-key)
- `GET /api/agents/patients/{id}/telemetry/since` (X-Api-Key: dev-agents-key)
- `POST /api/alerts/trigger`
- `/api/medications/*`
- `/swagger`

---

## 11. Tests
**Always run the tests offline.** If `backend/.env` contains an API key, the tests otherwise run in *live* mode: they wait on real LLM calls and free-tier rate limits, and look stuck.
```powershell
$env:LLM_MODE="stub"; $env:SURGIGUARD_LINK="off"; $env:HF_HUB_OFFLINE="1"
cd backend; .\.venv\Scripts\python.exe -m pytest tests -q          # AI backend: about 190 tests, about 1 min
cd ..\d1;   ..\backend\.venv\Scripts\python.exe -m pytest -q       # D1 incl. wound CV: about 185 tests, about 30 s
```
The wound-model tests need the CLIP weights in the Hugging Face cache (one-time download, about 600 MB):
`backend\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as s; s('openai/clip-vit-base-patch16', allow_patterns=['*.json','*.txt','pytorch_model.bin'])"`
- The `backend/test_*.py` files at the top level call live LLMs and are not part of the suite.
- The mobile Jest test is known to fail on its ESM config.
- The .NET API has no tests; `docker compose build` checks that it compiles.

---

## 12. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `port is already allocated` on 8080 | another program (Apache/XAMPP) uses it. Use `API_PORT=5051`; `demo.ps1` does this automatically. |
| `/story` shows "ESP32 silent" | simulator not running, or broker not reachable. Check `docker logs api` for "Pont MQTT connecté", then `docker exec -it mqtt mosquitto_sub -t Hackcure/data`. |
| API log "aucun patient pour le boîtier ESP32_001" | the device is not linked: run `seed_story.py` |
| The app cannot log in | the web build points to another API port: rebuild with `SG_ACCOUNT_URL`, or edit the address in the app's Settings |
| Wound model "unavailable" | CLIP not downloaded yet (needs internet the first time, about 600 MB), or `WOUND_MODEL=off` |
| AI answers look canned | stub mode: add a key in `backend/.env` and restart |
| `docker info` fails | start Docker Desktop and wait until it shows "Engine running" |
| Scripts blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Tests hang with no output | live LLM mode because of a key in `.env`: set `LLM_MODE=stub` (see §11) |
| `Could not import faiss` / `numpy.core.multiarray failed to import` | old faiss built for numpy 1.x: `pip install -U "faiss-cpu>=1.9"` |
| `'NoneType' object has no attribute 'endswith'` when loading CLIP | the CLIP repo has no `model.safetensors`; download `pytorch_model.bin` (command in §11) |
| `docker compose up --build` stays at "0 steps" for a long time | BuildKit stuck resolving base images: stop it, `docker pull mcr.microsoft.com/dotnet/sdk:8.0` and `.../aspnet:8.0`, then build again |
| Slow internet (pip/torch) | create the venv with `--system-site-packages` to reuse packages already installed globally; install CPU torch, not CUDA |
