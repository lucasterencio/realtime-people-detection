import tkinter as tk
from PIL import Image, ImageTk
import cv2

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import config
from detector import detect_persons
from face_recognizer import load_known_faces, recognize_faces
from dwell_time import DwellTimeTracker
from visualization import draw_person
from report import load_report, add_session, get_totals


BG = '#1a1a2e'
FG = '#e0e0e0'
ACCENT = '#e94560'
BTN_BG = '#16213e'


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('FC — Face Control')
        self.root.configure(bg=BG)
        self.root.minsize(520, 440)
        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        load_known_faces()

        self._cap = None
        self._capture_running = False
        self._report = load_report()
        self._tracker = DwellTimeTracker()
        self._last_persons = []
        self._last_faces = []
        self._frame_count = 0
        self._after_id = None

        self._main_frame = None
        self._capture_frame = None
        self._video_label = None

        self._show_main()

    def _show_main(self):
        if self._capture_frame:
            self._capture_frame.destroy()
            self._capture_frame = None

        self._main_frame = tk.Frame(self.root, bg=BG)
        self._main_frame.pack(expand=True, fill='both', padx=48, pady=48)

        tk.Label(
            self._main_frame, text='FC — Face Control',
            font=('Helvetica', 28, 'bold'), fg=ACCENT, bg=BG,
        ).pack(pady=(0, 4))

        tk.Label(
            self._main_frame, text='Reconhecimento Facial com Registro de Permanencia',
            font=('Helvetica', 11), fg='#a0a0b0', bg=BG,
        ).pack(pady=(0, 24))

        self._make_btn(self._main_frame, 'Iniciar Captura', self._start_capture).pack(pady=6)
        self._make_btn(self._main_frame, 'Ver Relatorio', self._on_report).pack(pady=6)

        self._status = tk.Label(
            self._main_frame, text='Pronto',
            font=('Helvetica', 10), fg='#a0a0b0', bg=BG,
        )
        self._status.pack(pady=(20, 0))

    def _show_capture(self):
        if self._main_frame:
            self._main_frame.destroy()
            self._main_frame = None

        self._capture_frame = tk.Frame(self.root, bg=BG)
        self._capture_frame.pack(expand=True, fill='both', padx=12, pady=12)

        self._video_label = tk.Label(self._capture_frame, bg='#000000')
        self._video_label.pack(padx=4, pady=4)

        btn_bar = tk.Frame(self._capture_frame, bg=BG)
        btn_bar.pack(pady=(8, 0))

        self._make_btn(btn_bar, 'Parar Captura', self._stop_capture).pack()

        self._status_label = tk.Label(
            self._capture_frame, text='',
            font=('Helvetica', 10), fg='#a0a0b0', bg=BG,
        )
        self._status_label.pack(pady=(4, 0))

    def _make_btn(self, parent, text, command):
        btn = tk.Button(
            parent, text=text, command=command,
            font=('Helvetica', 15, 'bold'),
            fg=FG, bg=BTN_BG,
            activeforeground='#ffffff', activebackground=ACCENT,
            relief='solid', bd=2,
            highlightthickness=0,
            padx=28, pady=10,
            cursor='hand2',
            highlightbackground=ACCENT, highlightcolor=ACCENT,
        )
        btn.bind('<Enter>', lambda e: e.widget.configure(bg=ACCENT, fg='#ffffff'))
        btn.bind('<Leave>', lambda e: e.widget.configure(bg=BTN_BG, fg=FG))
        return btn

    def _start_capture(self):
        self._show_capture()

        self._cap = cv2.VideoCapture(config.CAMERA_ID)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._tracker = DwellTimeTracker()
        self._last_persons = []
        self._last_faces = []
        self._frame_count = 0
        self._capture_running = True

        self._process_frame()

    def _stop_capture(self):
        self._capture_running = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

        remaining = self._tracker.flush()
        for name, duration, start_ts, end_ts in remaining:
            add_session(self._report, name, duration, start_ts, end_ts)

        if self._cap:
            self._cap.release()
            self._cap = None

        self._show_main()

    def _process_frame(self):
        if not self._capture_running:
            return

        ret, frame = self._cap.read()
        if not ret:
            self._after_id = self.root.after(10, self._process_frame)
            return

        display = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        if self._frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
            small = cv2.resize(frame, (320, 240))
            raw_persons = detect_persons(small)
            raw_faces = recognize_faces(small)

            sx = config.FRAME_WIDTH / 320
            sy = config.FRAME_HEIGHT / 240

            self._last_persons = []
            for p in raw_persons:
                x1, y1, x2, y2 = p['bbox']
                self._last_persons.append({
                    'bbox': (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
                    'confidence': p['confidence'],
                })

            self._last_faces = []
            for f in raw_faces:
                ft, fr, fb, fl = f['bbox']
                self._last_faces.append({
                    'bbox': (int(ft * sy), int(fr * sx), int(fb * sy), int(fl * sx)),
                    'name': f['name'],
                })

            names_in_frame = []
            for person in self._last_persons:
                best = 'Unknown'
                for face in self._last_faces:
                    x1, y1, x2, y2 = person['bbox']
                    ft, fr, fb, fl = face['bbox']
                    fx, fy = (fl + fr) // 2, (ft + fb) // 2
                    if x1 <= fx <= x2 and y1 <= fy <= y2:
                        best = face['name']
                        break
                if best != 'Unknown':
                    names_in_frame.append(best)
                dwell = self._tracker.get_dwell_time(best)
                draw_person(display, person['bbox'], best, person['confidence'], dwell)

            departed = self._tracker.update(names_in_frame)
            for name, duration, start_ts, end_ts in departed:
                session = add_session(self._report, name, duration, start_ts, end_ts)
                self._status_label.configure(
                    text=f'{session["name"]} saiu: {session["duration_seconds"]:.1f}s',
                )
        else:
            for person in self._last_persons:
                best = 'Unknown'
                for face in self._last_faces:
                    x1, y1, x2, y2 = person['bbox']
                    ft, fr, fb, fl = face['bbox']
                    fx, fy = (fl + fr) // 2, (ft + fb) // 2
                    if x1 <= fx <= x2 and y1 <= fy <= y2:
                        best = face['name']
                        break
                dwell = self._tracker.get_dwell_time(best)
                draw_person(display, person['bbox'], best, person['confidence'], dwell)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self._video_label.configure(image=imgtk)
        self._video_label.image = imgtk

        self._frame_count += 1
        self._after_id = self.root.after(10, self._process_frame)

    def _on_report(self):
        report = load_report()
        totals = get_totals(report)

        dlg = tk.Toplevel(self.root)
        dlg.title('Relatorio de Permanencia')
        dlg.configure(bg=BG)
        dlg.minsize(580, 480)
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        f = tk.Frame(dlg, bg=BG)
        f.pack(expand=True, fill='both', padx=24, pady=20)

        tk.Label(
            f, text='Relatorio de Permanencia',
            font=('Helvetica', 20, 'bold'), fg=ACCENT, bg=BG,
        ).pack(pady=(0, 16))

        if not totals:
            tk.Label(
                f, text='Nenhum registro de permanencia.',
                font=('Helvetica', 12), fg='#c0c0d0', bg=BG,
            ).pack(pady=40)
        else:
            chart_frame = tk.Frame(f, bg=BG)
            chart_frame.pack(fill='both', expand=True, pady=(0, 12))
            self._embed_chart(chart_frame, totals)

            for name, total in sorted(totals.items()):
                text = f'{name}: {total / 60:.1f} min' if total >= 60 else f'{name}: {total:.1f}s'
                tk.Label(
                    f, text=text, font=('Helvetica', 12), fg='#c0c0d0', bg=BG,
                ).pack()

        tk.Button(
            f, text='Fechar', command=dlg.destroy,
            font=('Helvetica', 14, 'bold'),
            fg=FG, bg=BTN_BG,
            activeforeground='#ffffff', activebackground=ACCENT,
            relief='solid', bd=2,
            highlightthickness=0,
            padx=24, pady=8,
            cursor='hand2',
            highlightbackground=ACCENT, highlightcolor=ACCENT,
        ).pack(pady=(16, 0))

    def _embed_chart(self, parent, totals):
        names = list(totals.keys())
        values = list(totals.values())

        colors = ['#e94560', '#0f3460', '#16213e', '#533483'] * 3

        fig = Figure(figsize=(5.5, 2.2), facecolor=BG, dpi=100)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)

        ax.barh(
            names, values,
            color=[colors[i] for i in range(len(names))],
            height=0.55, edgecolor=ACCENT, linewidth=0.8,
        )

        mx = max(values) if values else 1
        for bar, val in zip(ax.containers[0], values):
            label = f'{val / 60:.1f} min' if val >= 60 else f'{val:.1f}s'
            ax.text(
                bar.get_width() + mx * 0.02,
                bar.get_y() + bar.get_height() / 2,
                label, va='center', color=FG, fontsize=9, fontweight='bold',
            )

        ax.set_xlim(0, mx * 1.18)
        ax.invert_yaxis()
        ax.tick_params(colors='#a0a0b0', labelsize=9)
        ax.xaxis.set_visible(False)
        for spine in ('top', 'right', 'bottom'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#444466')
        fig.tight_layout(pad=0.8)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _on_close(self):
        self._capture_running = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
        if self._cap:
            self._cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
