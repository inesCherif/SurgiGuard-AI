"""Wound photos: ONE analysis (d1 pipeline) for clinician and patient, compared with the first photo only when the
two photos are proven to show the same site, linked to the agent's photo request."""

import io

from PIL import Image

from app import clock, wounds
from app import models as m
from app.integrations import d1_wound
from app.integrations.wound_views import patient_analysis, patient_comparison
from seed import simulator


def jpeg(color=(200, 150, 130), size=(400, 300)) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "JPEG", quality=95)
    return b.getvalue()


def view(data: bytes) -> dict:
    wound = d1_wound._d1()
    return patient_analysis(wound.strip_private(wound.analyze(data)))


def test_real_demo_photos_measured_but_different_patients_are_not_compared():
    """v1 reported day1 -> day3 as "+35 % redness": they are photos of two DIFFERENT patients (seed/photos/CREDITS.md).
    The registered comparison refuses them instead of inventing a change."""
    day1, day3 = view(wounds.demo_bytes("day1")), view(wounds.demo_bytes("day3"))
    assert day1["quality"] == "good" and day3["quality"] == "good" and "redness_area_pct" in day3
    assert set(day3["not_measured"]) >= {"swelling", "drainage"}          # never reported as "none"
    cmp = d1_wound.compare_photos(wounds.demo_bytes("day1"), wounds.demo_bytes("day3"))
    assert cmp["available"] is False and "same area" in cmp["reason"]


def test_same_site_is_compared():
    day1 = wounds.demo_bytes("day1")
    cmp = d1_wound.compare_photos(day1, day1)
    assert cmp["available"] and cmp["trend"] == "stable" and cmp["redness_area_change_points"] == 0
    assert cmp["registration"]["inliers"] >= 12


def test_bad_photos_ask_for_retake_not_numbers():
    dark = view(jpeg((8, 8, 8)))
    assert dark["quality"] == "unusable" and "redness_area_pct" not in dark
    assert any("dark" in p for p in dark["problems"])
    assert view(b"not an image")["quality"] == "unusable"
    assert not d1_wound.compare_photos(jpeg((8, 8, 8)), wounds.demo_bytes("day1"))["available"]


def test_adapter_maps_the_registered_comparison_to_the_patient_view():
    d1_cmp = {"available": True, "trend": "more_red", "redness_area_ref_pct": 5.0, "redness_area_now_pct": 8.0,
              "redness_change_delta_a": 4.5, "area_change_pct": 12, "registration": {"inliers": 40}, "quality": "good"}
    v = patient_comparison(d1_cmp)
    assert v["trend"] == "up" and v["summary"] == "more redness than the reference photo"
    assert v["redness_area_change_points"] == 3.0 and v["redness_area_change_pct"] == 60
    refused = patient_comparison({"available": False, "reason": "the two photos do not seem to show the same area"})
    assert refused == {"available": False, "reason": "the two photos do not seem to show the same area", "registration": None}
    assert patient_comparison(None)["reason"].startswith("first photo")


def test_upload_is_analysed_once_for_both_views(client):
    r = client.post("/patients/sarah/wound-photos", files={"file": ("knee.jpg", wounds.demo_bytes("day3"), "image/jpeg")},
                    data={"dressing_change_confirmed": "true"})
    body = r.json()
    assert r.status_code == 200 and body["url"].startswith("/uploads/")
    assert body["reference_url"] == "/static/photos/day1.jpg"
    # patient and clinician read the same analysis: same quality, same refusal reason
    assert body["analysis"]["quality"] == body["d1"]["quality"]["status"]
    assert body["comparison"]["available"] is False and body["comparison"]["reason"] == body["d1"]["comparison"]["reason"]
    assert body["analysis"]["redness_area_pct"] == body["d1"]["measurements"]["redness_area_pct"]
    listing = client.get("/patients/sarah/wound-photos").json()
    assert listing["reference"]["label"] == "day1" and listing["photos"][0]["observation_id"] == body["observation_id"]


def test_upload_of_the_same_site_shows_a_trend(client):
    r = client.post("/patients/sarah/wound-photos", files={"file": ("knee.jpg", wounds.demo_bytes("day1"), "image/jpeg")},
                    data={"dressing_change_confirmed": "true"}).json()
    assert r["comparison"]["available"] and r["comparison"]["trend"] == "stable"


def test_upload_answers_the_agents_photo_request(repo, client):
    res = simulator.inject(repo, "infection_pattern")
    inv_id = next(r["investigation_id"] for r in res["results"] if r["investigation_id"])
    r = client.post("/patients/sarah/wound-photos", files={"file": ("knee.jpg", wounds.demo_bytes("day3"), "image/jpeg")},
                    data={"dressing_change_confirmed": "true"})
    assert r.json()["answered_request_id"]
    repo.s.expire_all()
    inv = repo.get(m.Investigation, inv_id)
    assert inv.patient_requests == 2  # resumed after the photo, then asked its follow-up question


def test_upload_needs_confirmation_and_a_dressing_window(client):
    photo = {"file": ("knee.jpg", wounds.demo_bytes("day3"), "image/jpeg")}
    assert client.post("/patients/sarah/wound-photos", files=photo).status_code == 409  # not confirmed
    clock.advance(24 * 3600)  # day 4: no dressing change planned
    assert client.post("/patients/sarah/wound-photos", files=photo).status_code == 409  # still needs confirmation
    # the patient changes the dressing on their own: accepted, analysed, flagged as outside the planned window
    r = client.post("/patients/sarah/wound-photos", files=photo, data={"dressing_change_confirmed": "true"})
    body = r.json()
    assert r.status_code == 200 and body["outside_planned_window"] and body["answered_request_id"] is None
    assert body["d1"]["segmentation"]["found"]
