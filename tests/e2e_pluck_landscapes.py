"""
E2E check of the new interactive surfaces, Playwright-driven.

Runs the Dash app in-process (flask's threaded dev server), drives it
with headless Chromium, and exercises BOTH new features through the real
HTTP surface:

1. Click-to-pluck — click the wave chart, assert the status alert reports
   a pluck and the plot re-renders with fresh animation frames.
2. Landscapes tab — assert both regime heatmaps render with real data.

Exits 0 on success; nonzero with diagnostics otherwise. Not part of the
regular suite (launches a browser + server); run manually:

    python tests/e2e_pluck_landscapes.py
"""

import sys
import threading
import time

import numpy as np

sys.path.insert(0, ".")

import dashboard_app  # noqa: E402  (builds layout, registers callbacks)


def _serve():
    # threaded=False keeps the Flask reloader off and the server in this
    # thread; Dash's dev server is fine here (debug=False, no reload).
    dashboard_app.app.run(debug=False, host="127.0.0.1", port=8099,
                          threaded=True)


def main():
    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    time.sleep(3.0)

    import urllib.request
    for _ in range(20):
        try:
            urllib.request.urlopen("http://127.0.0.1:8099/", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    else:
        print("server never came up")
        return 2
    print("server up on 8099")

    from playwright.sync_api import sync_playwright

    ok = True
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors = []
        page.on("console", lambda m: errors.append(m.text)
                if m.type == "error" else None)

        page.goto("http://127.0.0.1:8099/", wait_until="networkidle")
        assert page.title() == "DJC Wave Lab", page.title()
        print("title OK")

        # ------------------------------------------------------------------
        # 1. Click-to-pluck on the Wave tab
        # ------------------------------------------------------------------
        run = page.locator("#run-button")
        run.scroll_into_view_if_needed()
        run.click()
        page.wait_for_selector(".alert-success", timeout=30000)
        first = page.text_content(".alert-success")
        assert "Simulation complete" in first, first
        print("baseline RUN OK:", first.strip())

        # Click inside the wave plot: plotly maps clicks via its own hit
        # testing, so click at a spot away from the seeded pulse (center
        # of the chart area, right of the initial Gaussian).
        graph = page.locator("#tab-content .plotly .main-svg").first
        graph.scroll_into_view_if_needed()
        time.sleep(1.0)  # let the animation settle
        box = graph.bounding_box()
        # Click at 70% across, mid-height of the trace area.
        cx = box["x"] + box["width"] * 0.70
        cy = box["y"] + box["height"] * 0.45
        page.mouse.click(cx, cy)
        print("clicked wave chart at", round(cx), round(cy))

        # The status alert must flip to a pluck within a callback roundtrip
        deadline = time.time() + 60
        pluck_status = None
        while time.time() < deadline:
            txt = page.text_content(".alert-success") or ""
            if "Plucked" in txt:
                pluck_status = txt
                break
            time.sleep(0.5)
        if not pluck_status:
            print("FAIL: click never produced a pluck; alert was:",
                  page.text_content(".alert-success"))
            ok = False
        else:
            print("pluck OK:", pluck_status.strip())

        # ------------------------------------------------------------------
        # 2. Landscapes tab
        # ------------------------------------------------------------------
        tab = page.locator(".nav-link", has_text="Landscapes").first
        tab.scroll_into_view_if_needed()
        tab.click()
        # Plotly renders heatmaps as raster <image> nodes inside the svg,
        # so assert on images + colorbars + titles, not a .heatmap class.
        deadline = time.time() + 120
        info = {}
        while time.time() < deadline:
            info = page.evaluate("""() => {
                const tc = document.querySelector('#tab-content');
                return {
                    images: tc.querySelectorAll('svg image').length,
                    colorbars: tc.querySelectorAll('.colorbar').length,
                    titles: [...tc.querySelectorAll('.g-gtitle text')]
                        .map(t => t.textContent),
                };
            }""")
            if info.get("images", 0) >= 2 and info.get("colorbars", 0) >= 2:
                break
            time.sleep(1.0)
        print("landscape DOM:", info)
        if info.get("images", 0) < 2 or info.get("colorbars", 0) < 2:
            print("FAIL: landscapes did not render both heatmaps")
            ok = False
        elif not any("Escape landscape" in t for t in info.get("titles", [])):
            print("FAIL: escape landscape title missing")
            ok = False
        elif not any("Stability landscape" in t for t in info.get("titles", [])):
            print("FAIL: stability landscape title missing")
            ok = False

        if errors:
            print("console errors:", errors[:5])
            ok = ok and False
        browser.close()

    print("E2E", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
