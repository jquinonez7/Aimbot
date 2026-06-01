"""
arguments: 
-m <model>   n | s | m | l | path/to/model.pt | path/to/model.onnx
-c <confidence> 0.0-1.0
--imgsz <image size> 640
--no-half Disable FP16 half-precision (FP16 is ON by default for speed)
--region <region> small | medium | large
--max-det <max detections> default 10 (max detections per frame)

py detect_overlay.py -m n   # nano
py detect_overlay.py -m s   # small
py detect_overlay.py -m m   # medium
py detect_overlay.py -m l   # large
py detect_overlay.py -m path/to/model.pt    # .pt model
py detect_overlay.py -m path/to/model.onnx  # ONNX model

default arguments: py detect_overlay.py -m nf -c 0.5 --imgsz 640 --no-half --region small --max-det 10

"""
TRANSPARENT_COLOR = "#010101"
BOX_OUTLINE = "#00FF00"
HUD_COLOR = "#FFFF00"
REGION_OUTLINE = "#00FFFF"
BOX_WIDTH = 2
CONF_STEP = 0.05
CLICK_DELAY = 0.01  # seconds between snap+click cycles
SNAP_MULTIPLIER = 1.9  # initial value; adjust live with [ ]
SNAP_STEP = 0.1  # per [ or ] press
# No-det nudge: only while auto-shoot is on (key 1). After N empty frames, move slightly.
NO_DET_MIN_EMPTY_FRAMES = 3
NO_DET_NUDGE_DX = 5
NO_DET_NUDGE_DY = 0  # relative px; scaled by SNAP_MULTIPLIER

import argparse
import ctypes
import os
import sys
import time

import numpy as np
import mss
import win32api
import win32con
import win32gui
import tkinter as tk
from ultralytics import YOLO

ctypes.windll.user32.SetProcessDPIAware()

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo26_models")
MODELS = {
    "n": os.path.join(MODEL_DIR, "yolo26n_cheat_script_reccomended_150.pt"),
    "s": os.path.join(MODEL_DIR, "yolo26s_cheat_script_reccomended_150.pt"),
    "m": os.path.join(MODEL_DIR, "yolo26m_cheat_script_reccomended_150.pt"),
    "l": os.path.join(MODEL_DIR, "yolo26l_cheat_script_reccomended_150.pt"),
    "nf": os.path.join(MODEL_DIR, "yolo26n_cheat_script_recommended_200_smallflicks_only.pt"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Real-time blue-sphere detector overlay")
    p.add_argument(
        "-m", "--model",
        default="n",
        help="Model: n | s | m | l  OR  path to .pt/.onnx/.engine  [default: n]",
    )
    p.add_argument(
        "-c", "--conf",
        type=float,
        default=0.5,
        help="Initial confidence threshold 0.0-1.0  [default: 0.5]",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference input resolution  [default: 640]",
    )
    p.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 half-precision (FP16 is ON by default for speed)",
    )
    p.add_argument(
        "--region",
        choices=["small", "medium", "large"],
        default="small",
        help="Capture region size: small (35%%), medium (50%%), large (full screen)  [default: small]",
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=10,
        help="Max detections per frame  [default: 10]",
    )
    return p.parse_args()

# ---------------------------------------------------------------------------
# Transparent, click-through overlay using tkinter + Win32
# ---------------------------------------------------------------------------

class Overlay:
    def __init__(self):
        user32 = ctypes.windll.user32
        self.screen_w = user32.GetSystemMetrics(0)
        self.screen_h = user32.GetSystemMetrics(1)

        self.root = tk.Tk()
        self.root.title("_sphere_overlay_")
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", TRANSPARENT_COLOR)
        self.root.config(bg=TRANSPARENT_COLOR)

        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_w,
            height=self.screen_h,
            bg=TRANSPARENT_COLOR,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()

        self.root.update_idletasks()
        self._make_click_through()

        self.show_hud = True

    def _make_click_through(self):
        hwnd = win32gui.FindWindow(None, "_sphere_overlay_")
        if not hwnd:
            return
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        style |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)

    # -- drawing helpers ----------------------------------------------------

    def clear(self):
        self.canvas.delete("all")

    def draw_box(self, x1, y1, x2, y2, conf):
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline=BOX_OUTLINE, width=BOX_WIDTH,
        )
        self.canvas.create_text(
            x1 + 4, y1 - 4,
            text=f"{conf:.0%}",
            fill=BOX_OUTLINE, anchor="sw",
            font=("Consolas", 13, "bold"),
        )

    def draw_region(self, left, top, width, height, crosshair_xy=None, show_aim_dot=True):
        self.canvas.create_rectangle(
            left, top, left + width, top + height,
            outline=REGION_OUTLINE, width=2, dash=(8, 4),
        )
        if not show_aim_dot:
            return
        if crosshair_xy is None:
            cx = left + width // 2
            cy = top + height // 2 + 37
        else:
            cx, cy = crosshair_xy
        r = 2  # small yellow dot
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=HUD_COLOR, outline=HUD_COLOR)

    def draw_hud(self, fps, conf, model_tag, det_count, auto_shoot=False, snap_mult=1.9):
        if not self.show_hud:
            return
        text = (
            f"FPS  {fps:>5.0f}\n"
            f"Conf {conf:>5.0%}\n"
            f"Snap {snap_mult:>5.1f}x\n"
            f"Model    {model_tag}\n"
            f"Dets {det_count:>5}\n"
            f"Auto {'ON ' if auto_shoot else 'OFF'}"
        )
        self.canvas.create_text(
            8, 8,
            text=text,
            fill=HUD_COLOR, anchor="nw",
            font=("Consolas", 14, "bold"),
        )

    def refresh(self):
        self.root.update_idletasks()
        self.root.update()

    def destroy(self):
        self.root.destroy()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    arg = args.model.strip('"')
    if arg.lower().endswith((".onnx", ".pt", ".engine")):
        model_path = os.path.abspath(arg)
    elif arg in MODELS:
        model_path = MODELS[arg]
    else:
        print(f"[ERROR] Model must be n|s|m|l or path to .pt/.onnx/.engine file. Got: {args.model}")
        sys.exit(1)

    if not os.path.isfile(model_path):
        fallback = os.path.join(MODEL_DIR, os.path.basename(model_path))
        if os.path.isfile(fallback):
            model_path = fallback
        else:
            print(f"[ERROR] Model not found: {model_path}")
            sys.exit(1)

    use_half = not args.no_half

    print(f"Loading model  {model_path}")
    model = YOLO(model_path)
    if model_path.lower().endswith((".onnx", ".engine")):
        print(f"Model loaded ({'TensorRT' if model_path.lower().endswith('.engine') else 'ONNX'})")
    else:
        model.to("cuda")
        print(f"Model loaded on CUDA  (FP{'16' if use_half else '32'})")

    overlay = Overlay()
    sct = mss.mss()

    REGION_FRAC = {"small": 0.35, "medium": 0.5, "large": 1.0}
    region_frac = REGION_FRAC[args.region]
    full_mon = sct.monitors[1]
    if region_frac < 1.0:
        side = int(min(full_mon["width"], full_mon["height"]) * region_frac)
        rx = full_mon["left"] + (full_mon["width"] - side) // 2
        ry = full_mon["top"] + (full_mon["height"] - side) // 2
        monitor = {"left": rx, "top": ry, "width": side, "height": side}
        region_offset = (rx - full_mon["left"], ry - full_mon["top"])
    else:
        monitor = full_mon
        region_offset = (0, 0)

    conf = args.conf
    snap_multiplier = SNAP_MULTIPLIER
    auto_shoot = False
    last_key_time = 0.0
    last_click_time = 0.0
    key_cd = 0.2
    fps = 0.0

    # Aim point (fixed_cx, fixed_cy): center of capture + 37px down. Used for nearest-target;
    # dot is hidden during auto-shoot but coordinates are unchanged.
    # If your game's crosshair is elsewhere, adjust the +37 offset.
    fixed_cx = monitor["left"] + monitor["width"] // 2
    fixed_cy = monitor["top"] + monitor["height"] // 2 + 37

    # mouse_event with ABSOLUTE - game receives it (SetCursorPos doesn't). Use virtual
    # screen coords so scaling is correct for multi-monitor / DPI.
    user32 = ctypes.windll.user32
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
    vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_ABSOLUTE = 0x8000

    empty_det_streak = 0

    def move_cursor_to(x, y):
        """Move cursor via mouse_event ABSOLUTE - games receive this."""
        nx = int((x - vx) * 65535 / max(1, vw))
        ny = int((y - vy) * 65535 / max(1, vh))
        win32api.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, nx, ny, 0, 0)

    model_tag = os.path.basename(model_path) if arg.lower().endswith((".onnx", ".pt", ".engine")) else arg.upper()
    print(f"Screen       {overlay.screen_w}x{overlay.screen_h}")
    print(f"Capture      {monitor['width']}x{monitor['height']}  (region {args.region})")
    print(f"Confidence   {conf:.0%}")
    print(f"Img size     {args.imgsz}")
    print(f"Half (FP16)  {use_half}")
    print(f"Max dets     {args.max_det}")
    print()
    print("Hotkeys")
    print("  +/-    adjust confidence")
    print("  [ ]    adjust snap multiplier (±0.1)")
    print("  1      toggle auto-shoot (hides aim dot; uses same aim point in code)")
    print("  F1     toggle HUD")
    print("  F2     quit")
    print()
    print("NOTE: The game must run in Borderless-Windowed mode for the overlay to appear on top.")
    print("Running...")

    try:
        while True:
            t0 = time.perf_counter()

            # ---- hotkeys --------------------------------------------------
            now = time.time()
            if now - last_key_time > key_cd:
                if win32api.GetAsyncKeyState(win32con.VK_F2) & 0x8000:
                    break

                if (win32api.GetAsyncKeyState(0xBB) & 0x8000
                        or win32api.GetAsyncKeyState(0x6B) & 0x8000):
                    conf = min(0.95, round(conf + CONF_STEP, 2))
                    print(f"Confidence -> {conf:.0%}")
                    last_key_time = now

                if (win32api.GetAsyncKeyState(0xBD) & 0x8000
                        or win32api.GetAsyncKeyState(0x6D) & 0x8000):
                    conf = max(0.05, round(conf - CONF_STEP, 2))
                    print(f"Confidence -> {conf:.0%}")
                    last_key_time = now

                if win32api.GetAsyncKeyState(win32con.VK_F1) & 0x8000:
                    overlay.show_hud = not overlay.show_hud
                    last_key_time = now

                if win32api.GetAsyncKeyState(0x31) & 0x8000:  # 1
                    auto_shoot = not auto_shoot
                    print(f"Auto-shoot  {'ON' if auto_shoot else 'OFF'}")
                    last_key_time = now

                if win32api.GetAsyncKeyState(0xDB) & 0x8000:  # [
                    snap_multiplier = max(0.1, round(snap_multiplier - SNAP_STEP, 2))
                    print(f"Snap mult -> {snap_multiplier:.1f}x")
                    last_key_time = now

                if win32api.GetAsyncKeyState(0xDD) & 0x8000:  # ]
                    snap_multiplier = min(10.0, round(snap_multiplier + SNAP_STEP, 2))
                    print(f"Snap mult -> {snap_multiplier:.1f}x")
                    last_key_time = now

            # ---- capture + inference --------------------------------------
            frame = np.array(sct.grab(monitor))[:, :, :3]

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=conf,
                half=use_half,
                max_det=args.max_det,
                verbose=False,
                device="cuda",
            )

            # ---- draw -----------------------------------------------------
            ox, oy = region_offset
            overlay.clear()

            boxes_screen = []
            det_count = 0
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    c = float(box.conf[0])
                    sx1, sy1 = x1 + ox, y1 + oy
                    sx2, sy2 = x2 + ox, y2 + oy
                    overlay.draw_box(sx1, sy1, sx2, sy2, c)
                    boxes_screen.append(((sx1 + sx2) / 2, (sy1 + sy2) / 2))
                    det_count += 1

            if auto_shoot:
                if det_count == 0:
                    empty_det_streak += 1
                else:
                    empty_det_streak = 0

                if empty_det_streak >= NO_DET_MIN_EMPTY_FRAMES:
                    ndx = int(NO_DET_NUDGE_DX * snap_multiplier)
                    ndy = int(NO_DET_NUDGE_DY * snap_multiplier)
                    if ndx != 0 or ndy != 0:
                        win32api.mouse_event(MOUSEEVENTF_MOVE, ndx, ndy, 0, 0)
                    empty_det_streak = 0
            else:
                empty_det_streak = 0

            # Auto-shoot: pick closest -> snap (relative, full distance) -> click -> next target
            # Relative works; absolute did not move the cursor in this game.
            if auto_shoot and boxes_screen and (now - last_click_time) >= CLICK_DELAY:
                best = min(boxes_screen, key=lambda bc: (bc[0] - fixed_cx) ** 2 + (bc[1] - fixed_cy) ** 2)
                tx, ty = best[0], best[1]
                cur = win32api.GetCursorPos()
                dx = int((tx - cur[0]) * snap_multiplier)
                dy = int((ty - cur[1]) * snap_multiplier)
                win32api.mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0)  # instant snap via relative
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                last_click_time = now

            overlay.draw_region(
                monitor["left"], monitor["top"], monitor["width"], monitor["height"],
                show_aim_dot=not auto_shoot,
            )

            elapsed = time.perf_counter() - t0
            fps = 1.0 / max(elapsed, 1e-9)
            overlay.draw_hud(fps, conf, model_tag, det_count, auto_shoot, snap_multiplier)
            overlay.refresh()

    except KeyboardInterrupt:
        pass
    finally:
        overlay.destroy()
        print("Stopped.")


if __name__ == "__main__":
    main()
