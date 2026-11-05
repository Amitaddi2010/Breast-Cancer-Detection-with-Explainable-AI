"""
upload_to_paste.py — Uploads the paper section draft to paste.rs
Returns a raw, public, clickable browser URL that can be opened on any device.
"""

import os
import urllib.request

def upload_report_to_web():
    report_path = os.path.join("outputs", "RepSegNet_V2_Paper_Section.md")
    if not os.path.exists(report_path):
        print("[Paste Error] Paper section draft not found.")
        return None
        
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read().encode("utf-8")
            
        req = urllib.request.Request(
            "https://paste.rs",
            data=content,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Content-Type": "text/plain"
            }
        )
        with urllib.request.urlopen(req) as response:
            web_url = response.read().decode("utf-8").strip()
            
        # Save URL locally
        url_path = os.path.join("outputs", "web_results_link.txt")
        with open(url_path, "w", encoding="utf-8") as f:
            f.write(web_url)
            
        print(f"[Paste OK] Successfully uploaded report to web: {web_url}")
        return web_url
    except Exception as e:
        print(f"[Paste Warning] Failed to upload to paste.rs: {e}")
        return None

if __name__ == "__main__":
    upload_report_to_web()
