"""Take the screenshots used in docs/INTERFACES.md.

Needs the AI backend on :8000 (live or stub). The .NET API is optional (the app uses demo mode).

    backend\\.venv\\Scripts\\python.exe scripts\\docs_screenshots.py                 # everything
    backend\\.venv\\Scripts\\python.exe scripts\\docs_screenshots.py --only web       # web pages only
    backend\\.venv\\Scripts\\python.exe scripts\\docs_screenshots.py --only patient,family,clinic
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8000"
OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
PHONE = {"width": 390, "height": 844}
DESK = {"width": 1440, "height": 900}
LONG = 180_000  # live LLM calls on free tiers can take a while


def post(path):
    req = urllib.request.Request(BASE + path, method="POST", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


def shot(page, name, full=False, locator=None):
    OUT.mkdir(parents=True, exist_ok=True)
    (locator or page).screenshot(path=str(OUT / f"{name}.png"), **({} if locator else {"full_page": full}))
    print("saved", name, flush=True)


def tap(page, text, timeout=30_000, exact=False):
    el = page.get_by_text(text, exact=exact).first
    el.wait_for(state="visible", timeout=timeout)
    el.click()
    page.wait_for_timeout(1200)


def wait_text(page, text, timeout=LONG):
    page.get_by_text(text, exact=False).first.wait_for(state="visible", timeout=timeout)
    page.wait_for_timeout(800)


def try_step(label, fn):
    try:
        fn()
    except Exception as e:  # keep going: one missing screen must not lose the others
        print(f"!! {label}: {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)


def phone_page(browser):
    ctx = browser.new_context(viewport=PHONE, device_scale_factor=2, is_mobile=True, has_touch=True)
    pg = ctx.new_page()
    pg.goto(BASE + "/app/", wait_until="networkidle")
    pg.wait_for_timeout(1500)
    return pg


def patient_flow(b):
    pg = phone_page(b)
    shot(pg, "app-01-role")
    tap(pg, "Continue")
    shot(pg, "app-02-account")
    tap(pg, "Continue without an account")
    shot(pg, "app-03-welcome")
    tap(pg, "Get started")
    tap(pg, "Use fictional sample")
    shot(pg, "app-04-upload")
    tap(pg, "Analyze with AI")
    wait_text(pg, "Confirm my plan")
    shot(pg, "app-05-plan-review")
    shot(pg, "app-05-plan-review-full", full=True)
    tap(pg, "Confirm my plan")
    wait_text(pg, "YOUR RECOVERY JOURNEY", 60_000)
    pg.wait_for_timeout(2500)
    shot(pg, "app-06-home")
    shot(pg, "app-06-home-full", full=True)

    def home():
        tap(pg, "Home", exact=True)

    def assistant():
        tap(pg, "Ask about your recovery")
        pg.wait_for_timeout(1500)
        shot(pg, "app-07-assistant")
        box = pg.locator("textarea, input[type=text], input:not([type])").last
        box.fill("Can I shower with my dressing?")
        box.press("Enter")
        pg.wait_for_timeout(25_000)
        shot(pg, "app-07-assistant-answer")
        pg.go_back()
        pg.wait_for_timeout(1500)
    try_step("assistant", assistant)

    def wound():
        home()
        tap(pg, "Add a wound photo")
        pg.wait_for_timeout(2000)
        shot(pg, "app-08-wound-request")
        shot(pg, "app-08-wound-request-full", full=True)
    try_step("wound", wound)

    for tab, name in [("Check-in", "app-09-checkin"), ("Twin", "app-10-twin"), ("Reports", "app-11-reports"), ("Profile", "app-12-profile")]:
        def go(tab=tab, name=name):
            tap(pg, tab, exact=True)
            pg.wait_for_timeout(3500)
            shot(pg, name)
            shot(pg, name + "-full", full=True)
        try_step(tab, go)

    def meds():
        home()
        tap(pg, "Medications")
        pg.wait_for_timeout(2500)
        shot(pg, "app-13-medications", full=True)
    try_step("medications", meds)
    pg.context.close()


def family_flow(b):
    pg = phone_page(b)
    tap(pg, "I'm a family member")
    tap(pg, "Continue")
    tap(pg, "Continue without an account")
    pg.wait_for_timeout(2500)
    shot(pg, "app-20-family-start")
    tap(pg, "Open family demo")
    pg.wait_for_timeout(3500)
    shot(pg, "app-21-family-home")
    shot(pg, "app-21-family-home-full", full=True)
    for tab, name in [("Updates", "app-22-family-updates"), ("Recovery", "app-23-family-recovery")]:
        def go(tab=tab, name=name):
            tap(pg, tab, exact=True, timeout=8000)
            pg.wait_for_timeout(3000)
            shot(pg, name, full=True)
        try_step("family " + tab, go)
    pg.context.close()


def web_pages(b):
    page = b.new_context(viewport=DESK).new_page()
    for key, path in [("story", "/story"), ("prototype", "/prototype"), ("apidocs", "/docs")]:
        page.goto(BASE + path, wait_until="networkidle")
        page.wait_for_timeout(2500)
        shot(page, f"web-{key}")
        shot(page, f"web-{key}-full", full=True)
    page.context.close()


def clinic_flow(b):
    """Reset the demo, inject the infection scenario, answer as Sarah, add the day-3 wound photo, then show the clinic."""
    post("/demo/reset")
    page = b.new_context(viewport=DESK).new_page()
    page.goto(BASE + "/agent-demo/", wait_until="networkidle")
    page.wait_for_timeout(2500)
    shot(page, "web-clinic-00-clean", full=True)

    def wound_photo():
        page.get_by_text("I am changing my dressing now", exact=False).first.click()
        tap(page, "Use demo photo (day 3)")
        page.wait_for_timeout(45_000)      # OpenCV now, model + vision LLM in the background job
        page.reload(wait_until="networkidle")
        page.wait_for_timeout(3000)
        sec = page.get_by_text("WOUND PHOTOS", exact=False).first
        sec.scroll_into_view_if_needed()
        shot(page, "web-clinic-01-wound", full=True)
    try_step("clinic wound", wound_photo)

    def scenario():
        tap(page, "Infection pattern")
        page.wait_for_timeout(8000)
        try_step("answer", lambda: tap(page, "Answer all as Sarah", timeout=10_000))
        page.wait_for_timeout(60_000)      # investigation agent (live LLM)
        page.reload(wait_until="networkidle")
        page.wait_for_timeout(3000)
        shot(page, "web-clinic-02-alert-queue", full=True)
        first_case = page.locator("text=/attention|urgent/i").first
        try_step("open case", lambda: (first_case.click(), page.wait_for_timeout(3000)))
        shot(page, "web-clinic-03-case", full=True)
    try_step("clinic scenario", scenario)
    page.context.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of: web, patient, family, clinic")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    with sync_playwright() as p:
        b = p.chromium.launch()
        for key, fn in [("web", web_pages), ("patient", patient_flow), ("family", family_flow), ("clinic", clinic_flow)]:
            if not only or key in only:
                print(f"== {key}", flush=True)
                try_step(key, lambda fn=fn: fn(b))
        b.close()


if __name__ == "__main__":
    sys.exit(main())
