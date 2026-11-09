"""Virtual ESP32: publishes the board's exact MQTT message so SurgiGuard runs without hardware.

The real firmware (SurgiGuard/IOT/src/network.cpp) publishes once per second on topic `Hackcure/data`:

    {"device_id":"ESP32_001","temp":23.4,"hum":48.0,"bpm":74,"spo2":97,"activity":0.12,"motion":"Immobile"}

  temp / hum  room temperature / humidity from the DHT11 (NOT body temperature), null if the read failed
  bpm / spo2  MAX30102 heart rate and oxygen saturation, 0 when no finger is on the sensor
  activity    MPU6050 movement intensity in m/s^2; motion = Immobile (<0.3) | Leger mouvement (<1.5) | Agite

This script sends the same JSON to an MQTT broker, so the whole real chain is exercised:
    simulator -> MQTT broker -> .NET MqttBridgeService -> MongoDB -> AI backend (polls every 3 s) -> apps

Usage (local broker from docker-compose, see docs/RUN_WITHOUT_HARDWARE.md):
    python scripts/simulate_device_mqtt.py                         # normal readings, forever
    python scripts/simulate_device_mqtt.py --profile desat --seconds 60
    python scripts/simulate_device_mqtt.py --profile story          # scripted scenario, phases below
    python scripts/simulate_device_mqtt.py --dry-run --seconds 5    # print messages only, no broker

Profiles:
    normal    resting post-op patient: HR ~74, SpO2 ~97, mostly still, short walks
    desat     oxygen saturation drops to ~87% (respiratory warning)
    tachy     heart rate ~128 bpm at rest (pain, fever, bleeding...)
    nofinger  sensor not worn: bpm = spo2 = 0 (tests the "bad signal" path)
    agitated  restless movement (activity > 1.5)
    story     normal -> rising HR -> desaturation -> sensor off -> recovery, PHASE_S seconds each

Only for demos / throw-away databases: the API cannot tell these readings from the real board's.
Requires: pip install paho-mqtt  (already in backend/requirements.txt)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time

STORY = ["normal", "tachy", "desat", "nofinger", "normal"]


def motion_label(activity: float) -> str:
    """Same thresholds as the firmware (motion.cpp) and the .NET bridge."""
    if activity >= 1.5:
        return "Agite"
    if activity >= 0.3:
        return "Leger mouvement"
    return "Immobile"


class VirtualPatient:
    """Smooth, plausible signals: slow drifts plus small sensor noise, not independent random draws."""

    def __init__(self, seed: int | None = None):
        self.rnd = random.Random(seed)
        self.hr = 74.0
        self.spo2 = 97.0
        self.room_t = 23.0
        self.room_h = 48.0
        self.walk_left = 0  # seconds of a short walk in progress

    def step(self, profile: str, t: int) -> dict:
        r = self.rnd
        target_hr, target_spo2 = 74.0, 97.0
        if profile == "tachy":
            target_hr = 128.0
        elif profile == "desat":
            target_spo2 = 87.0
            target_hr = 92.0  # the heart compensates a little

        # activity: mostly still, occasional short walks (normal) or constant agitation
        if profile == "agitated":
            activity = 1.6 + abs(r.gauss(0, 0.4))
        else:
            if self.walk_left == 0 and r.random() < 0.01:
                self.walk_left = r.randint(10, 30)
            if self.walk_left > 0:
                self.walk_left -= 1
                activity = 0.5 + abs(r.gauss(0, 0.3))
                target_hr += 12  # walking raises HR
            else:
                activity = abs(r.gauss(0.08, 0.05))

        # first-order drift towards the target (about 20 s time constant) + breathing ripple + noise
        self.hr += (target_hr - self.hr) * 0.05 + r.gauss(0, 0.8)
        self.spo2 += (target_spo2 - self.spo2) * 0.08 + r.gauss(0, 0.25)
        self.room_t += (23.0 - self.room_t) * 0.01 + r.gauss(0, 0.02)
        self.room_h += (48.0 - self.room_h) * 0.01 + r.gauss(0, 0.1)
        hr = self.hr + 1.5 * math.sin(t / 4.0)

        bpm, spo2 = round(hr), int(round(min(100.0, self.spo2)))
        if profile == "nofinger":
            bpm, spo2 = 0, 0
        temp = None if r.random() < 0.01 else round(self.room_t, 1)  # the DHT11 sometimes fails a read
        hum = None if temp is None else round(self.room_h, 1)
        return {"temp": temp, "hum": hum, "bpm": bpm, "spo2": spo2,
                "activity": round(activity, 2), "motion": motion_label(activity)}


def connect(args):
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        sys.exit("paho-mqtt is missing: pip install paho-mqtt")
    client_id = f"ESP32_SurgiGuard_sim_{random.randint(1000, 9999)}"
    try:  # paho-mqtt >= 2
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:  # paho-mqtt 1.x
        client = mqtt.Client(client_id=client_id)
    if args.username:
        client.username_pw_set(args.username, args.password)
    if args.tls:
        client.tls_set()
        client.tls_insecure_set(True)  # the firmware also uses setInsecure()
    client.connect(args.broker, args.port, keepalive=30)
    client.loop_start()
    return client


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--tls", action="store_true", help="use TLS (cloud EMQX broker on port 8883)")
    ap.add_argument("--username", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--topic", default="Hackcure/data")
    ap.add_argument("--device", default="ESP32_001")
    ap.add_argument("--profile", default="normal", choices=["normal", "desat", "tachy", "nofinger", "agitated", "story"])
    ap.add_argument("--phase-seconds", type=int, default=60, help="length of each phase in --profile story")
    ap.add_argument("--seconds", type=int, default=0, help="stop after N seconds (0 = run until Ctrl+C)")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between messages (firmware: 1)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducible runs")
    ap.add_argument("--dry-run", action="store_true", help="print the messages, do not connect")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    client = None if args.dry_run else connect(args)
    patient = VirtualPatient(args.seed)
    where = "dry run" if args.dry_run else f"{args.broker}:{args.port} topic {args.topic}"
    print(f"Virtual ESP32 {args.device} -> {where}, profile {args.profile}. Ctrl+C to stop.")
    t = 0
    try:
        while args.seconds == 0 or t < args.seconds:
            profile = args.profile
            if profile == "story":
                profile = STORY[min(t // args.phase_seconds, len(STORY) - 1)]
                if t % args.phase_seconds == 0:
                    print(f"--- story phase: {profile}")
            payload = {"device_id": args.device, **patient.step(profile, t)}
            msg = json.dumps(payload, separators=(",", ":"))
            if client is not None:
                client.publish(args.topic, msg, qos=0)
            if not args.quiet:
                print(t, msg)
            t += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if client is not None:
            client.loop_stop()
            client.disconnect()


if __name__ == "__main__":
    main()
