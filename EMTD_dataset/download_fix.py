import os
import subprocess
import pandas as pd


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')


def download_youtube_video(video_url, out_dir):
    base_cmd = [
        "yt-dlp",
        "-N", "8",  # parallel connections for robustness/speed
        "--concurrent-fragments", "8",
        "--force-ipv4",
        "--no-part",
        "--retries", "10",
        "--fragment-retries", "15",
        "--retry-sleep", "exp=1:30",
        "-f", "bv*+ba/b",  # best video+audio of ANY container, else best single
        "--merge-output-format", "mp4",  # remux to mp4 when codecs allow
        "-o", os.path.join(out_dir, "%(id)s.%(ext)s"),
        video_url,
    ]

    # If some items need login/age/region, uncomment ONE of these:
    # base_cmd += ["--cookies-from-browser", "chrome"]  # or edge/firefox
    # base_cmd += ["--cookies", "cookies.txt"]

    r = subprocess.run(base_cmd, capture_output=True, text=True, encoding="utf-8")
    if r.returncode == 0:
        print(f"OK {video_url}")
        return True

    # Fallback: force re-encode to mp4 if remux failed or formats mismatched
    if "Requested format is not available" in r.stderr or "merge" in r.stderr.lower():
        fallback_cmd = [c for c in base_cmd if c not in ["--merge-output-format", "mp4"]]
        fallback_cmd += ["--recode-video", "mp4"]
        r2 = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding="utf-8")
        if r2.returncode == 0:
            print(f"OK {video_url}")
            return True

    # Only if you see 403 or similar, try cookies as a fallback:
    err = (r.stderr or "") + (r.stdout or "")
    if "403" in err or "fragment 1 not found" in err.lower():
        r2 = subprocess.run(
            base_cmd[:1] + ["--cookies-from-browser", "edge"] + base_cmd[1:],
            capture_output=True, text=True, encoding="utf-8"
        )
        if r2.returncode == 0:
            print(f"OK {video_url}")
            return True

    print(f"Fail to download {video_url}, error info:\n{r.stderr}")
    # Optional: print available formats for this one ID to inspect
    fmts = subprocess.run(["yt-dlp", "-F", video_url], capture_output=True, text=True, encoding="utf-8")
    print(fmts.stdout or fmts.stderr)
    return False


if __name__ == "__main__":
    # Make sure you’re using yt-dlp (not old youtube-dl)
    # pip install -U yt-dlp
    # and have ffmpeg in PATH

    df = pd.read_csv("./echomimicv2_benchmark_url+start_timecode+end_timecode.txt")
    save_dir = "ori_video_dir"
    os.makedirs(save_dir, exist_ok=True)

    urls = list(dict.fromkeys(df["URL"]))  # keep order, drop dups

    ok, bad = 0, []
    for u in urls:
        if download_youtube_video(u, save_dir):
            ok += 1
        else:
            bad.append(u)

    print(f"\nDone. Success: {ok}/{len(urls)}")
    if bad:
        print("Failed URLs:")
        for u in bad:
            print("  ", u)
