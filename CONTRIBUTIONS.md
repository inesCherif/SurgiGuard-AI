# Contributions

SurgiGuard AI was built in about 24 hours at the **HackCure** hackathon (October 2026), where it **won 2nd place**. This repository continues it as a school project.

- Original team repository: https://github.com/azizhraghi/HackCure (kept as the `upstream` remote).
- The full git history has been kept unchanged, with every original author.

## Hackathon team (October 2026)

| Member | Main work |
|---|---|
| **Ines Cherif** | **Wound-photo computer vision (D1-F)**: OpenCV pipeline, SurgWound-trained model, vision-LLM review, in-app camera. **D2 documents**: D2 foundation models, dashboard and seed data; **OCR (D2A)** with PyMuPDF text layer + Pixtral vision fallback; extraction schema (20+ clinical fields); safety graph (allergies/interactions); `/v1` API; unification of the D2A compiler with the D2B assistant. |
| Aziz Hraghi | Project brief, scheduled wound photos and alert verification, investigation recovery after restarts, LangGraph investigation, clinic/family integration, theme |
| Karoui Fares | First React Native app + FastAPI scaffold, ESP32/MQTT + .NET API + native Android app, AI backend ↔ SurgiGuard API link, `/story`, README |
| garmayosr | AI backend core (steps 1–10: safety rules, alerts, Monitor, agents, LLM layer), Recovery Protocol Compiler, D1/D2/D3 connection, prototype page, unified app redesign |
| Wassila Benslimene | D2B RAG assistant (Groq, semantic chunking, memory, hybrid ranking) and D1 rehabilitation (pose estimation, prescription, rep counter, range of motion) |
| Elaa Elfahem | D1 normal recovery plan, trajectory prediction / patient twin naming, guided movement tests |

*Roles summarised from `git log`; each member can correct their line.*

### Note on the wound-photo commits

During the hackathon, Ines worked on Karoui Fares's laptop. Because of that, the wound-photo commits of 10–11 October 2026 appear under **Karoui Fares** in `git log`. Those commits are:

- `28a7b4b` D1-F wound photos at dressing changes: OpenCV analysis, overlay, vision LLM, clinician review rule
- `4dd3299` in-app camera on Android and React Native
- `78c2ef5` trained wound-attribute model: CLIP ViT-B/16 + SurgWound
- `2fc5566` unfamiliar-photo gate and faster inference
- `6431361` 48 Mpx capture crash fix

The history has deliberately **not** been rewritten, so it stays exactly as it was pushed during the event.

## School project (from November 2026): Ines Cherif

Every commit after the hackathon is authored by Ines Cherif:

- The system runs end to end **without hardware**: a virtual ESP32 publishes over MQTT, and a local broker runs in Docker. See `docs/RUN_WITHOUT_HARDWARE.md`.
- Documentation of the whole system and of the wound pipeline: `docs/`.
- **Wound CV v2**: calibration, learned segmentation, a better classifier and honest benchmarks. See `docs/wound/`.
