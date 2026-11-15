#!/usr/bin/env bash
# Record the MuJoCo mission viewer to an mp4 while you drive it from this terminal (c / v / n / p / q work).
#   scripts/record_viewer.sh <seconds> <out.mp4> <kick_mission.py args...>
# A background watcher waits for the new "MuJoCo : ..." window, reads its position/size, and grabs exactly that
# region of DISPLAY with ffmpeg for <seconds>. The viewer runs in the foreground so it keeps this terminal's keys.
# Keep the window where it opens (moving it mid-run moves it out of frame). Recording only; behaviour unchanged.
set -euo pipefail
secs=$1; out=$2; shift 2
cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$out")"
export DISPLAY=${DISPLAY:-:1}
count() { xwininfo -root -tree | grep -c '"MuJoCo :' || true; }
before=$(count)

record() {
    local win="" info x y w h
    for _ in $(seq 600); do                       # up to 60 s for the window
        if [ "$(count)" -gt "$before" ]; then win=$(xwininfo -root -tree | grep '"MuJoCo :' | tail -1 | awk '{print $1}'); break; fi
        sleep 0.1
    done
    [ -n "$win" ] || { echo "[record] no new MuJoCo window after 60 s"; return 1; }
    sleep 0.3                                     # let the window manager place it
    info=$(xwininfo -id "$win")
    x=$(awk '/Absolute upper-left X/{print $4}' <<<"$info"); y=$(awk '/Absolute upper-left Y/{print $4}' <<<"$info")
    w=$(awk '/^ *Width/{print $2}' <<<"$info");              h=$(awk '/^ *Height/{print $2}' <<<"$info")
    w=$((w / 2 * 2)); h=$((h / 2 * 2))           # x264 wants even sizes
    echo "[record] ${w}x${h} at +${x},${y} for ${secs} s -> $out"
    ffmpeg -loglevel error -y -f x11grab -framerate 30 -video_size "${w}x${h}" -i "${DISPLAY}+${x},${y}" \
           -t "$secs" -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p "$out" </dev/null \
        && echo "[record] saved $out" || echo "[record] ffmpeg failed"
}
record &
rec=$!
env -u PYTHONPATH uv run python scripts/kick_mission.py "$@" || true
wait $rec || true
