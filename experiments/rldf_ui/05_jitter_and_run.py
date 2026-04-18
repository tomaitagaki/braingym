"""
Jitter experiment: apply controlled design defects to clean UIs,
record scroll videos, and run TRIBE on original vs jittered pairs.

Jitter functions from UIClip paper (CRAP violations):
  - color_swap: randomly swap element foreground/background colors
  - contrast_reduce: lower text-background contrast
  - spacing_noise: add random margin/padding noise
  - font_scramble: randomize font sizes

We craft 3 clean UIs, apply jitter, and compare TRIBE trajectories.

Usage:
    # Step 1: Generate HTML + record videos (local)
    python experiments/rldf_ui/05_jitter_and_run.py generate

    # Step 2: Run TRIBE on Modal
    python experiments/rldf_ui/05_jitter_and_run.py run
"""

import sys
import json
import random
import re
import asyncio
import subprocess
from pathlib import Path

CACHE_DIR = Path("cache/rldf/jitter")
VIDEO_DIR = CACHE_DIR / "videos"


# ============================================================
# CLEAN UI TEMPLATES
# ============================================================

CLEAN_UIS = {
    "dashboard": """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Analytics Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f0f2f5; color: #1a1a2e; }
.container { max-width: 500px; margin: 0 auto; padding: 16px; }
.header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.header h1 { font-size: 22px; font-weight: 700; }
.header .date { font-size: 13px; color: #6b7280; }
.card { background: white; border-radius: 12px; padding: 20px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.card h2 { font-size: 14px; color: #6b7280; font-weight: 500; margin-bottom: 8px; }
.card .value { font-size: 28px; font-weight: 700; color: #1a1a2e; }
.card .change { font-size: 13px; margin-top: 4px; }
.positive { color: #059669; }
.negative { color: #dc2626; }
.metric-row { display: flex; gap: 12px; margin-bottom: 12px; }
.metric-row .card { flex: 1; }
.chart-placeholder { height: 120px; background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); border-radius: 8px; margin-top: 12px; display: flex; align-items: flex-end; padding: 8px; gap: 4px; }
.bar { background: #4f46e5; border-radius: 4px 4px 0 0; flex: 1; }
.table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.table th { text-align: left; font-size: 12px; color: #6b7280; font-weight: 500; padding: 8px 0; border-bottom: 1px solid #e5e7eb; }
.table td { font-size: 14px; padding: 10px 0; border-bottom: 1px solid #f3f4f6; }
.table td:last-child { text-align: right; font-weight: 600; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
.badge-green { background: #d1fae5; color: #065f46; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-red { background: #fee2e2; color: #991b1b; }
</style></head><body>
<div class="container">
  <div class="header"><h1>Analytics</h1><span class="date">April 15, 2026</span></div>
  <div class="metric-row">
    <div class="card"><h2>Revenue</h2><div class="value">$48.2K</div><div class="change positive">+12.5% vs last month</div></div>
    <div class="card"><h2>Users</h2><div class="value">2,847</div><div class="change positive">+8.3% vs last month</div></div>
  </div>
  <div class="card">
    <h2>Weekly Revenue</h2>
    <div class="chart-placeholder">
      <div class="bar" style="height:40%"></div><div class="bar" style="height:65%"></div>
      <div class="bar" style="height:55%"></div><div class="bar" style="height:80%"></div>
      <div class="bar" style="height:70%"></div><div class="bar" style="height:90%"></div>
      <div class="bar" style="height:85%"></div>
    </div>
  </div>
  <div class="card">
    <h2>Top Products</h2>
    <table class="table">
      <tr><th>Product</th><th>Sales</th><th>Status</th></tr>
      <tr><td>Pro Plan</td><td>1,247</td><td><span class="badge badge-green">Active</span></td></tr>
      <tr><td>Basic Plan</td><td>892</td><td><span class="badge badge-green">Active</span></td></tr>
      <tr><td>Enterprise</td><td>156</td><td><span class="badge badge-yellow">Review</span></td></tr>
      <tr><td>Trial</td><td>2,103</td><td><span class="badge badge-green">Active</span></td></tr>
    </table>
  </div>
  <div class="card">
    <h2>Recent Activity</h2>
    <table class="table">
      <tr><td>New signup: john@email.com</td><td>2m ago</td></tr>
      <tr><td>Payment received: $299</td><td>15m ago</td></tr>
      <tr><td>Support ticket #4521 resolved</td><td>1h ago</td></tr>
      <tr><td>New signup: sarah@email.com</td><td>2h ago</td></tr>
      <tr><td>Upgrade: Basic → Pro</td><td>3h ago</td></tr>
    </table>
  </div>
</div></body></html>""",

    "settings": """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Settings</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f7; color: #1d1d1f; }
.container { max-width: 500px; margin: 0 auto; }
.header { padding: 20px; padding-bottom: 8px; }
.header h1 { font-size: 28px; font-weight: 700; }
.section { margin: 16px; }
.section-title { font-size: 13px; color: #86868b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; padding: 0 16px; margin-bottom: 6px; }
.group { background: white; border-radius: 12px; overflow: hidden; margin-bottom: 16px; }
.row { display: flex; align-items: center; padding: 12px 16px; border-bottom: 1px solid #f2f2f7; gap: 12px; }
.row:last-child { border-bottom: none; }
.row-icon { width: 28px; height: 28px; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-size: 14px; }
.row-content { flex: 1; }
.row-label { font-size: 16px; }
.row-sublabel { font-size: 13px; color: #86868b; }
.row-value { font-size: 16px; color: #86868b; }
.toggle { width: 44px; height: 26px; border-radius: 13px; position: relative; }
.toggle-on { background: #34c759; }
.toggle-off { background: #e5e5ea; }
.toggle::after { content: ""; position: absolute; width: 22px; height: 22px; background: white; border-radius: 50%; top: 2px; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.toggle-on::after { right: 2px; }
.toggle-off::after { left: 2px; }
.chevron { color: #c7c7cc; font-size: 14px; }
.icon-blue { background: #007aff; }
.icon-green { background: #34c759; }
.icon-orange { background: #ff9500; }
.icon-purple { background: #af52de; }
.icon-red { background: #ff3b30; }
.icon-gray { background: #8e8e93; }
</style></head><body>
<div class="container">
  <div class="header"><h1>Settings</h1></div>
  <div class="section">
    <div class="section-title">Account</div>
    <div class="group">
      <div class="row"><div class="row-icon icon-blue">👤</div><div class="row-content"><div class="row-label">Profile</div><div class="row-sublabel">John Appleseed</div></div><span class="chevron">›</span></div>
      <div class="row"><div class="row-icon icon-green">🔒</div><div class="row-content"><div class="row-label">Privacy</div></div><span class="chevron">›</span></div>
      <div class="row"><div class="row-icon icon-orange">🔔</div><div class="row-content"><div class="row-label">Notifications</div></div><span class="chevron">›</span></div>
    </div>
  </div>
  <div class="section">
    <div class="section-title">Preferences</div>
    <div class="group">
      <div class="row"><div class="row-icon icon-purple">🌙</div><div class="row-content"><div class="row-label">Dark Mode</div></div><div class="toggle toggle-off"></div></div>
      <div class="row"><div class="row-icon icon-blue">🌐</div><div class="row-content"><div class="row-label">Language</div></div><div class="row-value">English</div><span class="chevron">›</span></div>
      <div class="row"><div class="row-icon icon-green">📍</div><div class="row-content"><div class="row-label">Location</div></div><div class="toggle toggle-on"></div></div>
      <div class="row"><div class="row-icon icon-orange">🔊</div><div class="row-content"><div class="row-label">Sound Effects</div></div><div class="toggle toggle-on"></div></div>
    </div>
  </div>
  <div class="section">
    <div class="section-title">Data & Storage</div>
    <div class="group">
      <div class="row"><div class="row-icon icon-gray">💾</div><div class="row-content"><div class="row-label">Storage Used</div><div class="row-sublabel">4.2 GB of 15 GB</div></div><span class="chevron">›</span></div>
      <div class="row"><div class="row-icon icon-blue">☁️</div><div class="row-content"><div class="row-label">Cloud Backup</div></div><div class="toggle toggle-on"></div></div>
      <div class="row"><div class="row-icon icon-red">🗑️</div><div class="row-content"><div class="row-label">Clear Cache</div><div class="row-sublabel">247 MB</div></div><span class="chevron">›</span></div>
    </div>
  </div>
  <div class="section">
    <div class="section-title">About</div>
    <div class="group">
      <div class="row"><div class="row-content"><div class="row-label">Version</div></div><div class="row-value">3.2.1</div></div>
      <div class="row"><div class="row-content"><div class="row-label">Terms of Service</div></div><span class="chevron">›</span></div>
      <div class="row"><div class="row-content"><div class="row-label">Privacy Policy</div></div><span class="chevron">›</span></div>
    </div>
  </div>
</div></body></html>""",

    "product": """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Product Page</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: white; color: #1a1a2e; }
.container { max-width: 500px; margin: 0 auto; }
.product-image { width: 100%; height: 320px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; color: white; font-size: 48px; }
.content { padding: 20px; }
.price-row { display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px; }
.price { font-size: 28px; font-weight: 700; }
.price-original { font-size: 16px; color: #9ca3af; text-decoration: line-through; }
.discount { font-size: 13px; color: #059669; font-weight: 600; }
.title { font-size: 20px; font-weight: 600; margin: 8px 0; }
.rating { display: flex; align-items: center; gap: 4px; margin-bottom: 12px; }
.stars { color: #f59e0b; font-size: 14px; }
.rating-text { font-size: 13px; color: #6b7280; }
.description { font-size: 15px; line-height: 1.6; color: #4b5563; margin-bottom: 20px; }
.specs { margin-bottom: 20px; }
.specs h3 { font-size: 16px; font-weight: 600; margin-bottom: 8px; }
.spec-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f3f4f6; font-size: 14px; }
.spec-label { color: #6b7280; }
.spec-value { font-weight: 500; }
.btn { width: 100%; padding: 14px; border: none; border-radius: 10px; font-size: 16px; font-weight: 600; cursor: pointer; margin-bottom: 8px; }
.btn-primary { background: #4f46e5; color: white; }
.btn-secondary { background: #f3f4f6; color: #1a1a2e; }
.reviews { margin-top: 20px; }
.reviews h3 { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
.review { padding: 12px 0; border-bottom: 1px solid #f3f4f6; }
.review-header { display: flex; justify-content: space-between; margin-bottom: 4px; }
.review-author { font-size: 14px; font-weight: 600; }
.review-date { font-size: 12px; color: #9ca3af; }
.review-text { font-size: 14px; color: #4b5563; line-height: 1.5; }
</style></head><body>
<div class="container">
  <div class="product-image">🎧</div>
  <div class="content">
    <div class="price-row"><span class="price">$249</span><span class="price-original">$349</span><span class="discount">Save 29%</span></div>
    <div class="title">ProSound Ultra Wireless Headphones</div>
    <div class="rating"><span class="stars">★★★★★</span><span class="rating-text">4.8 (2,341 reviews)</span></div>
    <p class="description">Premium wireless headphones with active noise cancellation, 40-hour battery life, and studio-quality sound. Designed for all-day comfort with memory foam ear cushions.</p>
    <div class="specs">
      <h3>Specifications</h3>
      <div class="spec-row"><span class="spec-label">Battery Life</span><span class="spec-value">40 hours</span></div>
      <div class="spec-row"><span class="spec-label">Noise Cancellation</span><span class="spec-value">Active (ANC)</span></div>
      <div class="spec-row"><span class="spec-label">Connectivity</span><span class="spec-value">Bluetooth 5.3</span></div>
      <div class="spec-row"><span class="spec-label">Weight</span><span class="spec-value">254g</span></div>
      <div class="spec-row"><span class="spec-label">Driver Size</span><span class="spec-value">40mm</span></div>
    </div>
    <button class="btn btn-primary">Add to Cart</button>
    <button class="btn btn-secondary">Add to Wishlist</button>
    <div class="reviews">
      <h3>Reviews</h3>
      <div class="review"><div class="review-header"><span class="review-author">Alex M.</span><span class="review-date">2 days ago</span></div><span class="stars" style="font-size:12px">★★★★★</span><div class="review-text">Best headphones I've ever owned. The noise cancellation is incredible.</div></div>
      <div class="review"><div class="review-header"><span class="review-author">Sarah K.</span><span class="review-date">1 week ago</span></div><span class="stars" style="font-size:12px">★★★★☆</span><div class="review-text">Great sound quality and comfort. Battery life is amazing. Wish the case was smaller.</div></div>
      <div class="review"><div class="review-header"><span class="review-author">Mike R.</span><span class="review-date">2 weeks ago</span></div><span class="stars" style="font-size:12px">★★★★★</span><div class="review-text">Worth every penny. Use them daily for work calls and music.</div></div>
    </div>
  </div>
</div></body></html>""",
}


# ============================================================
# JITTER FUNCTIONS (from UIClip paper)
# ============================================================

def apply_jitter(html: str, seed: int = 42) -> str:
    """Apply multiple CRAP-violating jitter functions to HTML/CSS."""
    rng = random.Random(seed)
    css_block_match = re.search(r'<style>(.*?)</style>', html, re.DOTALL)
    if not css_block_match:
        return html

    css = css_block_match.group(1)
    body_html = html[css_block_match.end():]

    # 1. COLOR SWAP — swap foreground/background colors between elements
    colors_found = re.findall(r'(#[0-9a-fA-F]{6})', css)
    if len(colors_found) >= 4:
        swap_pairs = list(zip(colors_found[::2], colors_found[1::2]))
        rng.shuffle(swap_pairs)
        for c1, c2 in swap_pairs[:3]:
            placeholder = f"__SWAP_{rng.randint(10000,99999)}__"
            css = css.replace(c1, placeholder, 1)
            css = css.replace(c2, c1, 1)
            css = css.replace(placeholder, c2, 1)

    # 2. CONTRAST REDUCE — lighten dark text, darken light backgrounds
    css = re.sub(r'color:\s*#1[a-f0-9]{5}', lambda m: 'color: #9e9e9e', css, count=3)

    # 3. SPACING NOISE — randomize padding and margins
    def jitter_spacing(match):
        val = int(match.group(1))
        jittered = max(0, val + rng.randint(-8, 16))
        return f'{match.group(0).split(":")[0]}: {jittered}px'

    css = re.sub(r'(padding|margin)(?:-(?:top|bottom|left|right))?:\s*(\d+)px',
                 lambda m: f'{m.group(0).split(":")[0]}: {max(0, int(m.group(2)) + rng.randint(-8, 16))}px',
                 css, count=10)

    # 4. FONT SCRAMBLE — randomize font sizes
    def jitter_font(match):
        original = int(match.group(1))
        choices = [10, 11, 13, 15, 18, 22, 28]
        new_size = rng.choice([s for s in choices if s != original] or choices)
        return f'font-size: {new_size}px'

    css = re.sub(r'font-size:\s*(\d+)px', jitter_font, css, count=8)

    # 5. BORDER RADIUS NOISE — break consistent rounding
    css = re.sub(r'border-radius:\s*(\d+)px',
                 lambda m: f'border-radius: {rng.choice([0, 2, 5, 15, 25])}px',
                 css, count=6)

    # 6. ALIGNMENT BREAK — shift some elements
    css = css.replace('align-items: center', f'align-items: flex-start', 1)
    css = css.replace('justify-content: space-between', 'justify-content: flex-start', 1)

    return html[:css_block_match.start(1)] + css + html[css_block_match.end(1):]


# ============================================================
# GENERATE & RECORD
# ============================================================

async def record_scroll(html_path: Path, output_mp4: Path):
    """Record a scroll video of an HTML file."""
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
        total_steps = int(10.0 * 1000 / 16)

        if max_scroll > 0:
            px_per_step = max_scroll / total_steps
            for step in range(total_steps):
                await page.evaluate(f"window.scrollTo(0, {min(px_per_step * (step + 1), max_scroll)})")
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


async def generate():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    for name, html in CLEAN_UIS.items():
        # Save clean version
        clean_path = CACHE_DIR / f"{name}_clean.html"
        clean_path.write_text(html)

        # Apply jitter
        jittered = apply_jitter(html, seed=42)
        jitter_path = CACHE_DIR / f"{name}_jittered.html"
        jitter_path.write_text(jittered)

        print(f"\n{name}:")
        for variant in ["clean", "jittered"]:
            html_path = CACHE_DIR / f"{name}_{variant}.html"
            mp4_path = VIDEO_DIR / f"{name}_{variant}.mp4"
            if mp4_path.exists():
                print(f"  [cached] {variant}")
                continue
            print(f"  Recording: {variant}")
            await record_scroll(html_path, mp4_path)
            print(f"    -> {mp4_path.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "generate":
        asyncio.run(generate())
        print("\nDone! Videos in cache/rldf/jitter/videos/")
        print("Next: upload to Modal and run TRIBE")
    else:
        print("Usage: python 05_jitter_and_run.py generate")
