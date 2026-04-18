"""
Step 1: Record standardized scroll videos from RLDF UI HTML files.

For each HTML file, renders it in a headless browser at a fixed viewport,
then smoothly scrolls from top to bottom at constant speed while recording.

All videos get identical treatment — same viewport, same scroll speed,
same duration, same framerate. The only variable is the pixel content.

Usage:
    python experiments/rldf_ui/01_record_scroll_videos.py
"""

import asyncio
import time
from pathlib import Path

CACHE_DIR = Path("cache/rldf")
VIDEO_DIR = CACHE_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

VIEWPORT_W = 512
VIEWPORT_H = 512
SCROLL_DURATION_S = 10.0
SCROLL_STEP_PX = 2          # pixels per frame
SCROLL_INTERVAL_MS = 16     # ~60fps scroll updates
FPS = 30                    # video recording fps


async def record_scroll_video(html_path: Path, output_path: Path):
    """Record a smooth scroll video of an HTML file."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
            record_video_dir=str(output_path.parent),
            record_video_size={"width": VIEWPORT_W, "height": VIEWPORT_H},
        )
        page = await context.new_page()

        # Load HTML
        file_url = f"file://{html_path.resolve()}"
        await page.goto(file_url, wait_until="networkidle", timeout=15000)

        # Wait for rendering to settle
        await page.wait_for_timeout(500)

        # Get total scrollable height
        scroll_height = await page.evaluate("document.body.scrollHeight")
        max_scroll = max(0, scroll_height - VIEWPORT_H)
        print(f"    Page height: {scroll_height}px, max scroll: {max_scroll}px")

        if max_scroll == 0:
            # Page fits in viewport — just hold for the duration
            print(f"    No scroll needed, holding for {SCROLL_DURATION_S}s")
            await page.wait_for_timeout(int(SCROLL_DURATION_S * 1000))
        else:
            # Calculate scroll speed: cover max_scroll in SCROLL_DURATION_S
            total_steps = int(SCROLL_DURATION_S * 1000 / SCROLL_INTERVAL_MS)
            px_per_step = max_scroll / total_steps

            # Smooth scroll
            for step in range(total_steps):
                scroll_y = min(px_per_step * (step + 1), max_scroll)
                await page.evaluate(f"window.scrollTo(0, {scroll_y})")
                await page.wait_for_timeout(SCROLL_INTERVAL_MS)

        # Brief pause at bottom
        await page.wait_for_timeout(500)

        # Close to finalize video
        video_path = await page.video.path()
        await context.close()
        await browser.close()

        # Rename to desired output path
        if Path(video_path).exists():
            Path(video_path).rename(output_path)
            return True
        return False


async def main():
    html_files = sorted(CACHE_DIR.glob("*.html"))
    print(f"Found {len(html_files)} HTML files to record\n")

    for html_path in html_files:
        name = html_path.stem  # e.g. "chatbot_chosen"
        output_path = VIDEO_DIR / f"{name}.webm"

        if output_path.exists():
            print(f"  [cached] {name}")
            continue

        print(f"  Recording: {name}")
        t0 = time.time()
        success = await record_scroll_video(html_path, output_path)
        elapsed = time.time() - t0

        if success:
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"    Saved: {output_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")
        else:
            print(f"    FAILED: {name}")

    # Convert webm to mp4 for TRIBE compatibility
    print("\nConverting to mp4...")
    import subprocess
    for webm in VIDEO_DIR.glob("*.webm"):
        mp4 = webm.with_suffix(".mp4")
        if mp4.exists():
            print(f"  [cached] {mp4.name}")
            continue
        print(f"  Converting: {webm.name} -> {mp4.name}")
        subprocess.run([
            "ffmpeg", "-i", str(webm),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(FPS),
            "-y", str(mp4),
        ], capture_output=True, check=True)
        print(f"    Done: {mp4.name}")

    # Verify all outputs
    print("\nFinal videos:")
    for mp4 in sorted(VIDEO_DIR.glob("*.mp4")):
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1", str(mp4)],
            capture_output=True, text=True,
        )
        duration = result.stdout.strip()
        size_mb = mp4.stat().st_size / 1024 / 1024
        print(f"  {mp4.name}: {duration}, {size_mb:.1f} MB")


if __name__ == "__main__":
    asyncio.run(main())
