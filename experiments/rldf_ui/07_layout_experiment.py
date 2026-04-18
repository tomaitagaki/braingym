"""
Layout-only experiment: same CSS, colors, fonts — only section order changes.

Tests whether TRIBE responds to spatial information architecture
independent of pixel-level visual features.

Good layout: natural reading flow, important info first
Bad layout: shuffled sections, buries the lede

Usage:
    python experiments/rldf_ui/07_layout_experiment.py
"""

import asyncio
import subprocess
from pathlib import Path

CACHE_DIR = Path("cache/rldf/layout")
VIDEO_DIR = CACHE_DIR / "videos"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Shared CSS — identical for all variants
SHARED_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: white; color: #1a1a2e; }
.container { max-width: 500px; margin: 0 auto; }
.product-image { width: 100%; height: 300px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; color: white; font-size: 48px; }
.content { padding: 20px; }
.section { margin-bottom: 20px; }
.price-row { display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px; }
.price { font-size: 28px; font-weight: 700; }
.price-original { font-size: 16px; color: #9ca3af; text-decoration: line-through; }
.discount { font-size: 13px; color: #059669; font-weight: 600; }
.title { font-size: 20px; font-weight: 600; margin: 8px 0; }
.rating { display: flex; align-items: center; gap: 4px; margin-bottom: 12px; }
.stars { color: #f59e0b; font-size: 14px; }
.rating-text { font-size: 13px; color: #6b7280; }
.description { font-size: 15px; line-height: 1.6; color: #4b5563; }
.specs h3 { font-size: 16px; font-weight: 600; margin-bottom: 8px; }
.spec-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f3f4f6; font-size: 14px; }
.spec-label { color: #6b7280; }
.spec-value { font-weight: 500; }
.btn { width: 100%; padding: 14px; border: none; border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer; margin-bottom: 8px; }
.btn-primary { background: #4f46e5; color: white; }
.btn-secondary { background: #f3f4f6; color: #1a1a2e; }
.reviews h3 { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
.review { padding: 12px 0; border-bottom: 1px solid #f3f4f6; }
.review-header { display: flex; justify-content: space-between; margin-bottom: 4px; }
.review-author { font-size: 14px; font-weight: 600; }
.review-date { font-size: 12px; color: #9ca3af; }
.review-text { font-size: 14px; color: #4b5563; line-height: 1.5; }
"""

# Individual sections — same content, will be reordered
SECTIONS = {
    "hero": '<div class="product-image">🎧</div>',

    "price_title": """<div class="section">
  <div class="price-row"><span class="price">$249</span><span class="price-original">$349</span><span class="discount">Save 29%</span></div>
  <div class="title">ProSound Ultra Wireless Headphones</div>
  <div class="rating"><span class="stars">★★★★★</span><span class="rating-text">4.8 (2,341 reviews)</span></div>
</div>""",

    "description": """<div class="section">
  <p class="description">Premium wireless headphones with active noise cancellation, 40-hour battery life, and studio-quality sound. Designed for all-day comfort with memory foam ear cushions.</p>
</div>""",

    "specs": """<div class="section specs">
  <h3>Specifications</h3>
  <div class="spec-row"><span class="spec-label">Battery Life</span><span class="spec-value">40 hours</span></div>
  <div class="spec-row"><span class="spec-label">Noise Cancellation</span><span class="spec-value">Active (ANC)</span></div>
  <div class="spec-row"><span class="spec-label">Connectivity</span><span class="spec-value">Bluetooth 5.3</span></div>
  <div class="spec-row"><span class="spec-label">Weight</span><span class="spec-value">254g</span></div>
  <div class="spec-row"><span class="spec-label">Driver Size</span><span class="spec-value">40mm</span></div>
</div>""",

    "cta": """<div class="section">
  <button class="btn btn-primary">Add to Cart</button>
  <button class="btn btn-secondary">Add to Wishlist</button>
</div>""",

    "reviews": """<div class="section reviews">
  <h3>Reviews</h3>
  <div class="review"><div class="review-header"><span class="review-author">Alex M.</span><span class="review-date">2 days ago</span></div><span class="stars" style="font-size:12px">★★★★★</span><div class="review-text">Best headphones I've ever owned. The noise cancellation is incredible.</div></div>
  <div class="review"><div class="review-header"><span class="review-author">Sarah K.</span><span class="review-date">1 week ago</span></div><span class="stars" style="font-size:12px">★★★★☆</span><div class="review-text">Great sound quality and comfort. Battery life is amazing. Wish the case was smaller.</div></div>
  <div class="review"><div class="review-header"><span class="review-author">Mike R.</span><span class="review-date">2 weeks ago</span></div><span class="stars" style="font-size:12px">★★★★★</span><div class="review-text">Worth every penny. Use them daily for work calls and music.</div></div>
</div>""",
}


def build_page(section_order: list[str]) -> str:
    """Build an HTML page with sections in the given order."""
    body_parts = []
    for name in section_order:
        if name == "hero":
            body_parts.append(SECTIONS[name])
        else:
            body_parts.append(SECTIONS[name])

    # Hero goes outside .content, everything else inside
    hero = ""
    content_parts = []
    for name in section_order:
        if name == "hero":
            hero = SECTIONS[name]
        else:
            content_parts.append(SECTIONS[name])

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Product Page</title>
<style>{SHARED_CSS}</style></head><body>
<div class="container">
  {hero}
  <div class="content">
    {"".join(content_parts)}
  </div>
</div></body></html>"""


# Layout variants — same sections, different order
LAYOUTS = {
    # Natural e-commerce flow: hero → price → description → specs → CTA → reviews
    "good_hierarchy": ["hero", "price_title", "description", "specs", "cta", "reviews"],

    # Reversed: reviews first, then specs, then description, CTA, price, hero at bottom
    "reversed": ["reviews", "specs", "description", "cta", "price_title", "hero"],

    # CTA-first (aggressive): buttons before you even see the product details
    "cta_first": ["hero", "cta", "price_title", "reviews", "description", "specs"],

    # Specs-heavy: technical details before emotional hook (price/reviews)
    "specs_first": ["hero", "specs", "description", "price_title", "cta", "reviews"],
}


async def record_scroll(html_path: Path, output_mp4: Path):
    from playwright.async_api import async_playwright
    webm = output_mp4.with_suffix('.webm')
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 512, "height": 512},
            record_video_dir=str(output_mp4.parent),
            record_video_size={"width": 512, "height": 512},
        )
        page = await context.new_page()
        await page.goto(f"file://{html_path.resolve()}", wait_until="networkidle", timeout=15000)
        await page.wait_for_timeout(500)
        scroll_height = await page.evaluate("document.body.scrollHeight")
        max_scroll = max(0, scroll_height - 512)
        print(f"    {html_path.stem}: {scroll_height}px, scroll={max_scroll}px")
        total_steps = int(10.0 * 1000 / 16)
        if max_scroll > 0:
            px_per_step = max_scroll / total_steps
            for step in range(total_steps):
                await page.evaluate(f"window.scrollTo(0, {min(px_per_step*(step+1), max_scroll)})")
                await page.wait_for_timeout(16)
        else:
            await page.wait_for_timeout(10000)
        await page.wait_for_timeout(500)
        video_path = await page.video.path()
        await context.close()
        await browser.close()
        Path(video_path).rename(webm)
    subprocess.run(["ffmpeg", "-i", str(webm), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-r", "30", "-y", str(output_mp4)],
                   capture_output=True, check=True)
    webm.unlink(missing_ok=True)


async def main():
    for name, order in LAYOUTS.items():
        html_path = CACHE_DIR / f"{name}.html"
        mp4_path = VIDEO_DIR / f"{name}.mp4"

        html = build_page(order)
        html_path.write_text(html)

        if mp4_path.exists():
            print(f"  [cached] {name}")
            continue

        print(f"  Recording: {name} → {' → '.join(order)}")
        await record_scroll(html_path, mp4_path)
        print(f"    -> {mp4_path.name}")

    # Extract first frames for verification
    for name in LAYOUTS:
        mp4 = VIDEO_DIR / f"{name}.mp4"
        jpg = VIDEO_DIR / f"{name}_frame0.jpg"
        subprocess.run(["ffmpeg", "-y", "-i", str(mp4), "-vf", "select=eq(n\\,0)",
                        "-frames:v", "1", "-q:v", "2", str(jpg)],
                       capture_output=True)

    print("\nDone! Check frames:")
    for name in LAYOUTS:
        print(f"  open {VIDEO_DIR / f'{name}_frame0.jpg'}")


if __name__ == "__main__":
    asyncio.run(main())
