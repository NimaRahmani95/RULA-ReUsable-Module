import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROSImage
from hri_msgs.msg import IdsList, LiveSpeech
from body_data.msg import BodyMsg
import customtkinter
import threading
import os
import time
import queue
import warnings
import datetime
import math
import struct
import subprocess
import tempfile
import wave
from PIL import Image, ImageDraw, ImageTk
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory


def _generate_chime(path: str):
    """
    Write a short two-tone success chime (C5 → G5) to *path* as a WAV file.
    Uses only Python stdlib — no extra audio package required.
    """
    rate   = 44100
    dur    = 0.28          # seconds per tone
    volume = 0.88
    tones  = [523.25, 783.99]   # C5, G5

    frames = bytearray()
    for freq in tones:
        n = int(rate * dur)
        for i in range(n):
            t   = i / rate
            env = math.sin(math.pi * t / dur)  # half-sine envelope → no click
            val = int(32767 * volume * env * math.sin(2 * math.pi * freq * t))
            frames += struct.pack('<h', max(-32768, min(32767, val)))

    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(bytes(frames))

package_name = 'rula_gui'
package_share_path = get_package_share_directory(package_name)
resource = os.path.join(package_share_path, 'resource')
frame_path = os.path.join(resource, 'no_frame.png')

warnings.filterwarnings("ignore")

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _MPL = True
except ImportError:
    _MPL = False

# ─── Colour palette ──────────────────────────────────────────────────────────
BG       = "#0d0d0d"
PANEL    = "#141414"
CARD     = "#1c1c1c"
BORDER_C = "#2c2c2c"
TEXT     = "#f0f0f0"
DIM      = "#7a7a7a"

# Score colour ramps (index 0 = score 1 … index 8 = score 9)
_HEX  = ["#00c853","#00c853","#ffd600","#ffd600",
          "#ff6d00","#ff6d00","#d50000","#d50000","#d50000"]
_RGBA = [(0,200,83,255),(0,200,83,255),(255,214,0,255),(255,214,0,255),
         (255,109,0,255),(255,109,0,255),(213,0,0,255),(213,0,0,255),(213,0,0,255)]

PHASE_CFG = {
    "INIT": {
        "bg": "#263238", "dot": "#78909c",
        "label": "INITIALIZING",
        "detail": "Connecting to hardware…",
    },
    "RULA_OPTIMIZING": {
        "bg": "#0d2d6b", "dot": "#64b5f6",
        "label": "AUTO-OPTIMIZING",
        "detail": "RULA gradient descent active — stand in view of all cameras",
    },
    "USER_ADJUSTMENT": {
        "bg": "#1b3a1f", "dot": "#81c784",
        "label": "OPERATOR CONTROL",
        "detail": "Optimum found — Thumbs Up/Down to fine-tune height",
    },
}

MAX_LOG = 8   # number of log entries to keep visible


def _hex(score: int) -> str:
    return _HEX[min(max(score, 1), 9) - 1]

def _rgba(score: int) -> tuple:
    return _RGBA[min(max(score, 1), 9) - 1]

def _label(score: int) -> str:
    if score <= 2: return "Safe"
    if score <= 4: return "Low Risk"
    if score <= 6: return "Moderate"
    return "High Risk"


# ─────────────────────────────────────────────────────────────────────────────
class rula_gui(Node):

    def __init__(self):
        super().__init__('rula_gui')
        self._setup_ros()
        self._setup_state()
        threading.Thread(target=self._build_ui, daemon=True).start()

    # ─── ROS wiring ──────────────────────────────────────────────────────────

    def _setup_ros(self):
        pub = self.create_publisher(IdsList, '/humans/voices/tracked', 10)
        ids_msg = IdsList()
        ids_msg.ids = ['system_controller']
        pub.publish(ids_msg)
        self._speech_pub = self.create_publisher(
            LiveSpeech, '/humans/voices/system_controller/speech', 10)

        self._br = CvBridge()

        self.create_subscription(ROSImage,   '/front_frame_2D',    self._cb_front,   10)
        self.create_subscription(ROSImage,   '/right_frame_2D',    self._cb_right,   10)
        self.create_subscription(ROSImage,   '/left_frame_2D',     self._cb_left,    10)
        self.create_subscription(BodyMsg,    '/full_body_data',     self._cb_rula,    10)
        self.create_subscription(String,     '/gui_notifications',  self._cb_alert,   10)
        self.create_subscription(String,     '/operator_gesture',   self._cb_gesture, 10)
        self.create_subscription(String,     '/speak_text',         self._cb_robot,   10)
        self.create_subscription(
            LiveSpeech, '/humans/voices/user/speech', self._cb_user, 10)

    def _setup_state(self):
        # Queues (bounded so stale frames are dropped)
        self._q_front   = queue.Queue(maxsize=2)
        self._q_right   = queue.Queue(maxsize=2)
        self._q_left    = queue.Queue(maxsize=2)
        self._q_rula    = queue.Queue(maxsize=4)
        self._q_alert   = queue.Queue(maxsize=30)
        self._q_gesture = queue.Queue(maxsize=10)

        self._phase = "INIT"
        self._last_alert_time   = 0.0
        self._last_gesture      = "NONE"
        self._last_gesture_time = 0.0

        # RULA score trend (diagram)
        self._diag_t0      = time.time()
        self._diag_times   = []
        self._diag_ua_l    = []   # upper arm left score
        self._diag_ua_r    = []   # upper arm right score
        self._diag_la_l    = []   # lower arm left score
        self._diag_la_r    = []   # lower arm right score
        self._diag_marks   = []   # [(t, label)] phase-change markers
        self._diag_last    = 0.0  # last redraw timestamp

        # Pre-generate the success chime so playback is instant.
        self._chime_path = os.path.join(tempfile.gettempdir(), 'rula_success_chime.wav')
        try:
            _generate_chime(self._chime_path)
        except Exception as exc:
            self.get_logger().warn(f"Could not generate chime: {exc}")
            self._chime_path = None

        # Avatar scores start at 1 (safe colour)
        self._part_score = {
            "upper_hand_left": 1, "upper_hand_right": 1,
            "lower_hand_left": 1, "lower_hand_right": 1,
            "neck": 1, "back": 1,
            "leg_left": 1, "leg_right": 1,
        }

        ph = Image.open(frame_path)
        self._placeholder = customtkinter.CTkImage(
            light_image=ph, dark_image=ph, size=(420, 560))

    # ─── ROS callbacks ───────────────────────────────────────────────────────

    def _push_frame(self, msg, q: queue.Queue):
        frame = self._br.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        try:
            q.put_nowait(frame)
        except queue.Full:
            try: q.get_nowait()
            except queue.Empty: pass
            q.put_nowait(frame)

    def _cb_front(self, msg): self._push_frame(msg, self._q_front)
    def _cb_right(self, msg): self._push_frame(msg, self._q_right)
    def _cb_left(self,  msg): self._push_frame(msg, self._q_left)

    def _cb_rula(self, msg):
        try:
            self._q_rula.put_nowait(msg)
        except queue.Full:
            try: self._q_rula.get_nowait()
            except queue.Empty: pass
            self._q_rula.put_nowait(msg)

    def _cb_alert(self, msg):
        text = msg.data
        # Phase transitions are tagged as "[OLD → NEW]" by _transition_to().
        if "→ RULA_OPTIMIZING" in text:
            self._phase = "RULA_OPTIMIZING"
        elif "→ USER_ADJUSTMENT" in text:
            self._phase = "USER_ADJUSTMENT"
            self._play_chime()
            t = time.time() - self._diag_t0
            self._diag_marks.append((t, "CTRL"))
        elif "→ INIT" in text or ("Connecting" in text and self._phase == "INIT"):
            self._phase = "INIT"
        self._q_alert.put_nowait((datetime.datetime.now(), text))

    def _play_chime(self):
        """Play the success chime non-blocking. Tries paplay → aplay → play (sox)."""
        if not self._chime_path:
            return
        path = self._chime_path

        def _run():
            for cmd in [
                ['paplay', path],            # PipeWire / PulseAudio (Ubuntu 22.04)
                ['aplay', '-q', path],       # ALSA fallback
                ['play', '-q', path],        # SoX fallback
            ]:
                try:
                    ret = subprocess.call(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if ret == 0:
                        return
                except FileNotFoundError:
                    continue

        threading.Thread(target=_run, daemon=True).start()

    def _cb_gesture(self, msg):
        try:
            self._q_gesture.put_nowait(msg)
        except queue.Full:
            try: self._q_gesture.get_nowait()
            except queue.Empty: pass
            self._q_gesture.put_nowait(msg)

    def _cb_robot(self, msg): self.get_logger().info(f"[COBOT]: {msg.data}")
    def _cb_user(self,  msg): self.get_logger().info(f"[USER]: {msg.final}")

    # ─── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("blue")

        self.app = customtkinter.CTk(fg_color=BG)
        self.app.geometry("2300x1500")
        self.app.title("RULA Ergonomic Assessment System")
        self.app.resizable(False, False)

        self._build_header()
        self._build_body()
        self._poll()
        self.app.mainloop()

    # ── Header bar ───────────────────────────────────────────────
    def _build_header(self):
        hdr = customtkinter.CTkFrame(self.app, fg_color=PANEL, height=62, corner_radius=0)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        customtkinter.CTkLabel(
            hdr,
            text="RULA  Ergonomic Assessment System",
            font=customtkinter.CTkFont(family="Helvetica", size=21, weight="bold"),
            text_color=TEXT,
        ).pack(side="left", padx=22, pady=16)

        # Phase pill (right side)
        pill = customtkinter.CTkFrame(hdr, fg_color="#263238", corner_radius=8, height=38)
        pill.pack(side="right", padx=20, pady=12)
        pill.pack_propagate(False)

        self._phase_dot = customtkinter.CTkLabel(
            pill, text="●", text_color="#78909c",
            font=customtkinter.CTkFont(size=13))
        self._phase_dot.pack(side="left", padx=(14, 4), pady=8)

        self._phase_label = customtkinter.CTkLabel(
            pill, text="INITIALIZING",
            font=customtkinter.CTkFont(size=13, weight="bold"), text_color=TEXT)
        self._phase_label.pack(side="left", padx=(0, 6))

        self._phase_detail = customtkinter.CTkLabel(
            pill, text="Connecting to hardware…",
            font=customtkinter.CTkFont(size=11), text_color=DIM)
        self._phase_detail.pack(side="left", padx=(0, 18))

        self._phase_pill = pill

        # Separator
        customtkinter.CTkFrame(self.app, height=1, fg_color=BORDER_C, corner_radius=0).pack(fill="x")

    # ── Two-column body ──────────────────────────────────────────
    def _build_body(self):
        body = customtkinter.CTkFrame(self.app, fg_color=BG, corner_radius=0)
        body.pack(fill="both", expand=True)

        # Left column: cameras + angles + log
        left = customtkinter.CTkFrame(body, fg_color=BG, width=1450, corner_radius=0)
        left.pack(side="left", fill="y", padx=(4, 0), pady=8)
        left.pack_propagate(False)

        customtkinter.CTkFrame(body, width=1, fg_color=BORDER_C, corner_radius=0).pack(
            side="left", fill="y", pady=16)

        # Right column: avatar
        right = customtkinter.CTkFrame(body, fg_color=BG, corner_radius=0)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self._build_cameras(left)
        self._build_angles(left)
        self._build_log(left)
        self._build_diagram(left)
        self._build_avatar(right)

    # ── Camera row ───────────────────────────────────────────────
    def _build_cameras(self, parent):
        row = customtkinter.CTkFrame(parent, fg_color=BG)
        row.pack(fill="x", pady=(0, 6))

        cam_cfg = [
            ("LEFT CAMERA",  self._q_left),
            ("FRONT CAMERA", self._q_front),
            ("RIGHT CAMERA", self._q_right),
        ]
        self._cam_img    = []
        self._cam_status = []

        for col, (title, _) in enumerate(cam_cfg):
            card = customtkinter.CTkFrame(row, fg_color=CARD, corner_radius=10)
            card.grid(row=0, column=col, padx=7, pady=2, sticky="n")

            # Card header
            hdr = customtkinter.CTkFrame(card, fg_color=CARD)
            hdr.pack(fill="x", padx=10, pady=(8, 3))
            customtkinter.CTkLabel(
                hdr, text=title, text_color=DIM,
                font=customtkinter.CTkFont(size=11, weight="bold")).pack(side="left")
            status_lbl = customtkinter.CTkLabel(
                hdr, text="● WAITING",
                font=customtkinter.CTkFont(size=10), text_color="#546e7a")
            status_lbl.pack(side="right")
            self._cam_status.append(status_lbl)

            # Image feed
            img_lbl = customtkinter.CTkLabel(
                card, text="", image=self._placeholder,
                width=420, height=560, fg_color="#0a0a0a", corner_radius=8)
            img_lbl.pack(padx=8, pady=(0, 8))
            self._cam_img.append(img_lbl)

    # ── Joint angle readout ──────────────────────────────────────
    def _build_angles(self, parent):
        card = customtkinter.CTkFrame(parent, fg_color=CARD, corner_radius=10)
        card.pack(fill="x", padx=4, pady=(0, 6))

        arow = customtkinter.CTkFrame(card, fg_color=CARD)
        arow.pack(fill="x", padx=16, pady=(10, 10))

        customtkinter.CTkLabel(
            arow, text="JOINT ANGLES",
            font=customtkinter.CTkFont(size=12, weight="bold"),
            text_color=DIM).grid(row=0, column=0, columnspan=12, sticky="w", pady=(0, 6))

        self._al = {}
        angle_defs = [
            ("upper_arm_l", "Upper Arm L", 1, 0),
            ("upper_arm_r", "Upper Arm R", 1, 2),
            ("neck",        "Neck",        1, 4),
            ("lower_arm_l", "Lower Arm L", 1, 6),
            ("lower_arm_r", "Lower Arm R", 1, 8),
            ("trunk",       "Trunk",       1, 10),
        ]
        for key, lbl, r, c in angle_defs:
            customtkinter.CTkLabel(
                arow, text=lbl, text_color=DIM,
                font=customtkinter.CTkFont(size=11), width=110,
                anchor="w").grid(row=r, column=c, padx=(10, 2), pady=4, sticky="w")
            val = customtkinter.CTkLabel(
                arow, text="—°",
                font=customtkinter.CTkFont(size=15, weight="bold"),
                text_color=TEXT, width=80, anchor="w")
            val.grid(row=r, column=c + 1, padx=(0, 12), pady=4, sticky="w")
            self._al[key] = val

    # ── Notification log ─────────────────────────────────────────
    def _build_log(self, parent):
        log_card = customtkinter.CTkFrame(parent, fg_color=CARD, corner_radius=10)
        log_card.pack(fill="x", padx=4, pady=(0, 4))

        lhdr = customtkinter.CTkFrame(log_card, fg_color=CARD)
        lhdr.pack(fill="x", padx=16, pady=(10, 4))
        customtkinter.CTkLabel(
            lhdr, text="SYSTEM LOG",
            font=customtkinter.CTkFont(size=12, weight="bold"), text_color=DIM).pack(side="left")

        self._log_scroll = customtkinter.CTkScrollableFrame(
            log_card, fg_color=CARD, height=200, corner_radius=0)
        self._log_scroll.pack(fill="x", padx=6, pady=(0, 6))
        self._log_entries: list = []

    def _add_log(self, ts: datetime.datetime, text: str):
        # Colour-code by message intent
        if any(k in text for k in ["Moving", "Adjusting", "Optimizer", "Auto-Opt"]):
            col = "#64b5f6"
        elif any(k in text for k in ["Achieved", "PAUSED", "restarted", "OPERATOR"]):
            col = "#81c784"
        elif any(k in text for k in ["Limit", "Warning", "failed", "error"]):
            col = "#ffb74d"
        else:
            col = DIM

        entry = customtkinter.CTkFrame(self._log_scroll, fg_color="#191919", corner_radius=6)
        entry.pack(fill="x", padx=4, pady=2)

        customtkinter.CTkLabel(
            entry, text=ts.strftime("%H:%M:%S"),
            font=customtkinter.CTkFont(size=10, family="Courier"),
            text_color="#546e7a", width=65).pack(side="left", padx=(8, 4), pady=4)

        customtkinter.CTkLabel(
            entry,
            text=text.replace("\n", "  "),
            font=customtkinter.CTkFont(size=12),
            text_color=col, anchor="w",
            wraplength=1060).pack(side="left", fill="x", expand=True, padx=(0, 8), pady=4)

        self._log_entries.append(entry)
        if len(self._log_entries) > MAX_LOG:
            self._log_entries.pop(0).destroy()

        self._log_scroll._parent_canvas.yview_moveto(1.0)

    # ── Avatar panel ─────────────────────────────────────────────
    def _build_avatar(self, parent):
        # ── Dynamic phase state card (replaces static gesture reference) ──────
        self._pc = customtkinter.CTkFrame(
            parent, fg_color=PHASE_CFG["INIT"]["bg"], corner_radius=10, height=78)
        self._pc.pack(fill="x", padx=4, pady=(0, 4))
        self._pc.pack_propagate(False)

        # Left: dot + phase name + description
        info = customtkinter.CTkFrame(self._pc, fg_color="transparent")
        info.pack(side="left", padx=14, pady=8, fill="y")

        self._pc_dot = customtkinter.CTkLabel(
            info, text="●", text_color=PHASE_CFG["INIT"]["dot"],
            font=customtkinter.CTkFont(size=15))
        self._pc_dot.pack(side="left", padx=(0, 8))

        txt_col = customtkinter.CTkFrame(info, fg_color="transparent")
        txt_col.pack(side="left")
        self._pc_label = customtkinter.CTkLabel(
            txt_col, text=PHASE_CFG["INIT"]["label"],
            font=customtkinter.CTkFont(size=13, weight="bold"),
            text_color=TEXT, anchor="w")
        self._pc_label.pack(anchor="w")
        self._pc_detail = customtkinter.CTkLabel(
            txt_col, text=PHASE_CFG["INIT"]["detail"],
            font=customtkinter.CTkFont(size=10), text_color=DIM, anchor="w")
        self._pc_detail.pack(anchor="w")

        # Right: gesture icons — only visible in USER_ADJUSTMENT
        self._pc_gestures = customtkinter.CTkFrame(self._pc, fg_color="transparent")
        for icon, lbl in [("👍", "+15 mm"), ("👎", "−15 mm")]:
            fr = customtkinter.CTkFrame(self._pc_gestures, fg_color="#1f2a1f", corner_radius=6)
            fr.pack(side="left", padx=5, pady=6)
            customtkinter.CTkLabel(
                fr, text=icon, font=customtkinter.CTkFont(size=18)).pack(
                    side="left", padx=(8, 3), pady=4)
            customtkinter.CTkLabel(
                fr, text=lbl, text_color=TEXT,
                font=customtkinter.CTkFont(size=11)).pack(
                    side="left", padx=(0, 8), pady=4)
        # (not packed yet — shown only on USER_ADJUSTMENT in _refresh_phase_card)

        # ── Gesture detected indicator ────────────────────────────────────────
        self._gesture_card = customtkinter.CTkFrame(
            parent, fg_color=CARD, corner_radius=10, height=82)
        self._gesture_card.pack(fill="x", padx=4, pady=(0, 4))
        self._gesture_card.pack_propagate(False)

        customtkinter.CTkLabel(
            self._gesture_card, text="DETECTED",
            font=customtkinter.CTkFont(size=10, weight="bold"),
            text_color=DIM).pack(side="left", padx=(16, 8), pady=10)

        self._gesture_icon_lbl = customtkinter.CTkLabel(
            self._gesture_card, text="—",
            font=customtkinter.CTkFont(size=40))
        self._gesture_icon_lbl.pack(side="left", padx=(0, 6), pady=6)

        self._gesture_text_lbl = customtkinter.CTkLabel(
            self._gesture_card, text="No gesture",
            font=customtkinter.CTkFont(size=14, weight="bold"),
            text_color=DIM)
        self._gesture_text_lbl.pack(side="left", padx=(0, 16), pady=6)

        # ── Avatar ────────────────────────────────────────────────────────────
        self._avatar_label = customtkinter.CTkLabel(
            parent, text="", fg_color="#050505", corner_radius=10)
        self._avatar_label.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        self._draw_avatar()

    def _build_diagram(self, parent):
        """Embedded matplotlib chart — shoulder + lower-arm RULA scores over time."""
        if not _MPL:
            return

        card = customtkinter.CTkFrame(parent, fg_color=CARD, corner_radius=10)
        card.pack(fill="x", padx=4, pady=(0, 4))

        # ── Header row ────────────────────────────────────────────────────────
        hdr = customtkinter.CTkFrame(card, fg_color=CARD)
        hdr.pack(fill="x", padx=14, pady=(10, 0))
        customtkinter.CTkLabel(
            hdr, text="RULA SCORE TREND",
            font=customtkinter.CTkFont(size=12, weight="bold"),
            text_color=TEXT).pack(side="left")

        # ── Colour legend row (below header, above chart) ─────────────────────
        _LINE_DEFS = [
            ("ua_l", "#ff9800", "Shoulder L"),
            ("ua_r", "#ff5722", "Shoulder R"),
            ("la_l", "#2196f3", "Lower Arm L"),
            ("la_r", "#29b6f6", "Lower Arm R"),
        ]
        leg = customtkinter.CTkFrame(card, fg_color=CARD)
        leg.pack(fill="x", padx=14, pady=(4, 2))
        for _, col, label in _LINE_DEFS:
            entry = customtkinter.CTkFrame(leg, fg_color="transparent")
            entry.pack(side="left", padx=(0, 18))
            customtkinter.CTkLabel(
                entry, text="───",
                font=customtkinter.CTkFont(size=13, weight="bold"),
                text_color=col).pack(side="left", padx=(0, 4))
            customtkinter.CTkLabel(
                entry, text=label,
                font=customtkinter.CTkFont(size=11), text_color="#aaa").pack(side="left")

        # ── Matplotlib figure ─────────────────────────────────────────────────
        fig = Figure(figsize=(8.5, 3.2), dpi=90)
        fig.patch.set_facecolor("#1c1c1c")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#111111")
        ax.set_ylim(0.5, 7.5)
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
        ax.tick_params(colors="#888", labelsize=10)
        ax.set_ylabel("RULA Score", color="#999", fontsize=10)
        ax.set_xlabel("Time (s)", color="#999", fontsize=10)
        ax.grid(axis="y", color="#2a2a2a", linewidth=0.8, linestyle="--")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        # Risk-zone background bands (subtle, informative)
        ax.axhspan(0.5, 2.5, alpha=0.10, color="#00c853", linewidth=0)  # safe
        ax.axhspan(2.5, 4.5, alpha=0.10, color="#ffd600", linewidth=0)  # low risk
        ax.axhspan(4.5, 6.5, alpha=0.10, color="#ff6d00", linewidth=0)  # moderate
        ax.axhspan(6.5, 7.5, alpha=0.12, color="#d50000", linewidth=0)  # high risk

        # Right-side risk labels (secondary y-axis, no ticks)
        ax2 = ax.twinx()
        ax2.set_ylim(0.5, 7.5)
        ax2.set_yticks([1.5, 3.5, 5.5, 7.1])
        ax2.set_yticklabels(["Safe", "Low Risk", "Moderate", "High Risk"],
                            fontsize=8, color="#555")
        ax2.tick_params(right=False, labelright=True)
        for sp in ax2.spines.values():
            sp.set_visible(False)

        # Score lines — thicker, clearly distinct colours
        self._diag_lines = {}
        for key, col, _ in _LINE_DEFS:
            self._diag_lines[key] = ax.plot(
                [], [], color=col, linewidth=2.2, alpha=0.95,
                solid_capstyle="round")[0]

        fig.tight_layout(pad=0.8)
        canvas = FigureCanvasTkAgg(fig, master=card)
        canvas.get_tk_widget().pack(fill="x", padx=8, pady=(4, 10))

        self._diag_fig    = fig
        self._diag_ax     = ax
        self._diag_canvas = canvas
        self._diag_vlines = []

    # ─── Poll loop (≈30 fps) ─────────────────────────────────────────────────

    def _poll(self):
        # Camera feeds
        for q, img_wgt, status_wgt in zip(
            [self._q_left, self._q_front, self._q_right],
            self._cam_img,
            self._cam_status,
        ):
            try:
                frame = q.get_nowait()
                pil   = Image.fromarray(frame).resize((420, 560))
                photo = ImageTk.PhotoImage(pil)
                img_wgt.configure(image=photo)
                img_wgt.image = photo
                status_wgt.configure(text="● LIVE", text_color="#00c853")
            except queue.Empty:
                pass

        # RULA scores
        try:
            msg = self._q_rula.get_nowait()
            self._update_dashboard(msg)
        except queue.Empty:
            pass

        # Notifications
        try:
            while True:
                ts, text = self._q_alert.get_nowait()
                self._add_log(ts, text)
                self._refresh_phase_pill()
        except queue.Empty:
            pass

        # Gesture indicator — drain queue, keep most recent
        try:
            while True:
                g = self._q_gesture.get_nowait()
                self._last_gesture      = g.data
                self._last_gesture_time = time.time()
        except queue.Empty:
            pass

        # Show icon for 3 s then fade back to idle.
        # Guard with hasattr in case _poll fires before _build_avatar completes.
        if hasattr(self, '_gesture_icon_lbl'):
            elapsed = time.time() - self._last_gesture_time
            if elapsed < 3.0 and self._last_gesture in ("THUMBS_UP", "THUMBS_DOWN"):
                self._update_gesture_display(self._last_gesture)
            else:
                self._update_gesture_display("NONE")

        # Redraw diagram every second (matplotlib is not free)
        now = time.time()
        if hasattr(self, '_diag_canvas') and now - self._diag_last > 1.0:
            self._diag_last = now
            self._update_diagram()

        self.app.after(33, self._poll)

    def _update_diagram(self):
        if not _MPL or len(self._diag_times) < 2:
            return
        ax = self._diag_ax
        ts = self._diag_times
        window = 90.0          # show last 90 seconds
        t_end  = ts[-1]
        t_start = max(0.0, t_end - window)
        ax.set_xlim(t_start, t_end + 1)

        self._diag_lines["ua_l"].set_data(ts, self._diag_ua_l)
        self._diag_lines["ua_r"].set_data(ts, self._diag_ua_r)
        self._diag_lines["la_l"].set_data(ts, self._diag_la_l)
        self._diag_lines["la_r"].set_data(ts, self._diag_la_r)

        # Redraw phase-change vertical markers
        for vl in self._diag_vlines:
            try: vl.remove()
            except Exception: pass
        self._diag_vlines = []
        for t_mark, label in self._diag_marks:
            if t_mark >= t_start:
                vl = ax.axvline(t_mark, color="#81c784", linewidth=1.2,
                                linestyle="--", alpha=0.8)
                ax.text(t_mark + 0.5, 7.2, label, color="#81c784",
                        fontsize=7, va="top")
                self._diag_vlines.append(vl)

        self._diag_canvas.draw_idle()

    # ─── Dashboard update ─────────────────────────────────────────────────────

    def _update_dashboard(self, msg: BodyMsg):
        # ── Joint angles ────────────────────────────────────────
        self._al["neck"].configure(text=f"{msg.neck_angle:.1f}°")
        self._al["trunk"].configure(text=f"{msg.trunk_angle:.1f}°")
        if msg.right:
            self._al["upper_arm_r"].configure(text=f"{msg.right_arm_up:.1f}°")
            self._al["lower_arm_r"].configure(text=f"{msg.right_low_angle:.1f}°")
        if msg.left:
            self._al["upper_arm_l"].configure(text=f"{msg.left_arm_up:.1f}°")
            self._al["lower_arm_l"].configure(text=f"{msg.left_low_angle:.1f}°")

        # ── Avatar part scores (drive colour via RULA sub-scores) ─
        # neck_score  (1–6): neck flexion + twist + bending modifiers
        # trunk_score (1–5): trunk flexion + side-bending modifier
        # up_arm_score (1–7): upper-arm flexion + raised/abduction modifiers
        # lower_arm_score (1–3): elbow flexion + abduction modifier
        self._part_score["neck"] = msg.neck_score
        self._part_score["back"] = msg.trunk_score

        if msg.left:
            self._part_score["upper_hand_left"] = msg.up_arm_score_left
            self._part_score["lower_hand_left"] = msg.lower_arm_score_left
            if msg.left_rula_score >= 7:
                self._voice_alert()

        if msg.right:
            self._part_score["upper_hand_right"] = msg.up_arm_score_right
            self._part_score["lower_hand_right"] = msg.lower_arm_score_right
            if msg.right_rula_score >= 7:
                self._voice_alert()

        # ── Record score history for diagram ────────────────────────
        t = time.time() - self._diag_t0
        self._diag_times.append(t)
        self._diag_ua_l.append(self._part_score["upper_hand_left"])
        self._diag_ua_r.append(self._part_score["upper_hand_right"])
        self._diag_la_l.append(self._part_score["lower_hand_left"])
        self._diag_la_r.append(self._part_score["lower_hand_right"])
        # Keep last 600 samples (~60 s at 10 Hz)
        if len(self._diag_times) > 600:
            self._diag_times = self._diag_times[-600:]
            self._diag_ua_l  = self._diag_ua_l[-600:]
            self._diag_ua_r  = self._diag_ua_r[-600:]
            self._diag_la_l  = self._diag_la_l[-600:]
            self._diag_la_r  = self._diag_la_r[-600:]

        self._draw_avatar()

    def _voice_alert(self):
        if time.time() - self._last_alert_time < 30:
            return
        self._last_alert_time = time.time()
        m = LiveSpeech()
        m.final = ("Paraphrase and tell the user: "
                   "Your posture is at high risk. Immediate adjustment is recommended.")
        self._speech_pub.publish(m)

    # ─── Gesture indicator ───────────────────────────────────────────────────

    def _update_gesture_display(self, gesture: str):
        if gesture == "THUMBS_UP":
            self._gesture_card.configure(fg_color="#1b3a1f")
            self._gesture_icon_lbl.configure(text="👍", text_color="#81c784")
            self._gesture_text_lbl.configure(text="Raise PCB  +15 mm", text_color="#81c784")
        elif gesture == "THUMBS_DOWN":
            self._gesture_card.configure(fg_color="#3a1a1a")
            self._gesture_icon_lbl.configure(text="👎", text_color="#ef9a9a")
            self._gesture_text_lbl.configure(text="Lower PCB  −15 mm", text_color="#ef9a9a")
        else:
            self._gesture_card.configure(fg_color=CARD)
            self._gesture_icon_lbl.configure(text="—", text_color=DIM)
            self._gesture_text_lbl.configure(text="No gesture", text_color=DIM)

    # ─── Phase pill + right-panel phase card refresh ─────────────────────────

    def _refresh_phase_pill(self):
        cfg = PHASE_CFG.get(self._phase, PHASE_CFG["INIT"])
        self._phase_pill.configure(fg_color=cfg["bg"])
        self._phase_dot.configure(text_color=cfg["dot"])
        self._phase_label.configure(text=cfg["label"])
        self._phase_detail.configure(text=cfg["detail"])
        if hasattr(self, "_pc"):
            self._refresh_phase_card(cfg)

    def _refresh_phase_card(self, cfg: dict):
        self._pc.configure(fg_color=cfg["bg"])
        self._pc_dot.configure(text_color=cfg["dot"])
        self._pc_label.configure(text=cfg["label"])
        self._pc_detail.configure(text=cfg["detail"])
        # Gesture icons only visible during operator control
        if self._phase == "USER_ADJUSTMENT":
            self._pc_gestures.pack(side="right", padx=10, pady=8)
        else:
            self._pc_gestures.pack_forget()

    # ─── Avatar rendering ─────────────────────────────────────────────────────

    def _draw_avatar(self):
        W, H = 660, 1080
        img  = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx, cy = W // 2, H // 2

        def rr(xy, r, part_key):
            c = _rgba(self._part_score[part_key])
            draw.rounded_rectangle(xy, r, fill=c, outline="#000000", width=3)

        # Head
        draw.ellipse(
            (cx - 65, cy - 450, cx + 65, cy - 290),
            fill=_rgba(self._part_score["neck"]),
            outline="#000000", width=3)

        # Neck
        rr((cx - 20, cy - 296, cx + 20, cy - 236), 10, "neck")

        # Torso
        rr((cx - 115, cy - 245, cx + 115, cy + 120), 35, "back")

        # Upper arms
        ua_w, ua_h = 58, 180
        rr((cx - 115 - ua_w, cy - 235, cx - 115, cy - 235 + ua_h), ua_w // 2, "upper_hand_left")
        rr((cx + 115,        cy - 235, cx + 115 + ua_w, cy - 235 + ua_h), ua_w // 2, "upper_hand_right")

        # Lower arms
        la_w, la_h, off = 44, 160, 7
        ua_bot = cy - 235 + ua_h
        rr((cx - 115 - ua_w + off, ua_bot + 6, cx - 115 - off, ua_bot + 6 + la_h), la_w // 2, "lower_hand_left")
        rr((cx + 115 + off,        ua_bot + 6, cx + 115 + ua_w - off, ua_bot + 6 + la_h), la_w // 2, "lower_hand_right")

        # Legs — RULA does not produce a leg risk score; rendered in neutral gray
        leg_gray = (70, 75, 85, 255)
        lg_w, lg_h, gap = 72, 300, 12
        draw.rounded_rectangle(
            (cx - gap - lg_w, cy + 100, cx - gap,        cy + 100 + lg_h),
            lg_w // 2, fill=leg_gray, outline="#000000", width=3)
        draw.rounded_rectangle(
            (cx + gap,        cy + 100, cx + gap + lg_w, cy + 100 + lg_h),
            lg_w // 2, fill=leg_gray, outline="#000000", width=3)

        ctk_img = customtkinter.CTkImage(light_image=img, dark_image=img, size=(W, H))
        self._avatar_label.configure(image=ctk_img)
        self._avatar_img_ref = ctk_img


# ─── Entry point ─────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = rula_gui()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
