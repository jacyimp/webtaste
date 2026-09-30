"""Windows screen scoring with a transparent, click-through top-right overlay.

Place beside taste_ml.py. Uses the model from WebTaste's train.py.

    .\\.venv\\Scripts\\pythonw.exe screen_taste.py

Ctrl+Alt+Q quits. No main window, taskbar button, screenshot files, or frame cache.
Options: --model models/taste_model.joblib --interval 1 --monitor 1
         --crop-top 90 --crop-bottom 48 --device cuda --font-size 22
Troubleshooting: screen_taste.log beside this script.
"""
import argparse
import ctypes
from ctypes import wintypes
import logging
import math
import os
from pathlib import Path
import queue
import threading
import time


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Must be finite and greater than zero')
    return number


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Must be greater than zero')
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('Must be nonnegative')
    return number


def capture_box(monitor, crop_top=0, crop_bottom=0):
    left, top, right, bottom = monitor['box']
    if crop_top < 0 or crop_bottom < 0 or crop_top + crop_bottom >= bottom - top:
        raise ValueError('Cropping must leave a positive screen height')
    return (left, top + crop_top, right, bottom - crop_bottom)


class WindowsAPI:
    def __init__(self):
        self.user = ctypes.WinDLL('user32', use_last_error=True)
        self.user.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        self.user.GetAncestor.restype = wintypes.HWND
        self.get_style = getattr(self.user, 'GetWindowLongPtrW', self.user.GetWindowLongW)
        self.set_style = getattr(self.user, 'SetWindowLongPtrW', self.user.SetWindowLongW)
        self.get_style.argtypes = [wintypes.HWND, ctypes.c_int]
        self.get_style.restype = ctypes.c_ssize_t
        self.set_style.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        self.set_style.restype = ctypes.c_ssize_t
        self.user.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
        self.user.SetWindowPos.restype = wintypes.BOOL
        self.user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.user.ShowWindow.restype = wintypes.BOOL
        self.user.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user.GetAsyncKeyState.restype = ctypes.c_short
        self.user.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        self.user.SetWindowDisplayAffinity.restype = wintypes.BOOL
        try:
            self.dwm = ctypes.WinDLL('dwmapi', use_last_error=True)
            self.dwm.DwmFlush.argtypes = []
            self.dwm.DwmFlush.restype = ctypes.c_long
        except OSError:
            self.dwm = None

    def dpi_awareness(self):
        # Must run before creating Tk windows or asking for monitor coordinates.
        try:
            function = self.user.SetProcessDpiAwarenessContext
            function.argtypes = [ctypes.c_void_p]
            function.restype = wintypes.BOOL
            if function(ctypes.c_void_p(-4)):  # Per-monitor-aware v2
                return
        except AttributeError:
            pass
        try:
            shcore = ctypes.WinDLL('shcore')
            function = shcore.SetProcessDpiAwareness
            function.argtypes = [ctypes.c_int]
            function.restype = ctypes.c_long
            if function(2) == 0:
                return
        except (AttributeError, OSError):
            pass
        self.user.SetProcessDPIAware()

    def monitors(self):
        class MonitorInfo(ctypes.Structure):
            _fields_ = [('cbSize', wintypes.DWORD), ('rcMonitor', wintypes.RECT),
                        ('rcWork', wintypes.RECT), ('dwFlags', wintypes.DWORD)]

        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HANDLE,
            wintypes.HDC, ctypes.POINTER(wintypes.RECT), ctypes.c_ssize_t)
        self.user.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
        self.user.GetMonitorInfoW.restype = wintypes.BOOL
        self.user.EnumDisplayMonitors.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT),
                                                 callback_type, ctypes.c_ssize_t]
        self.user.EnumDisplayMonitors.restype = wintypes.BOOL
        found = []

        def callback(handle, dc, rect, data):
            info = MonitorInfo()
            info.cbSize = ctypes.sizeof(info)
            if self.user.GetMonitorInfoW(handle, ctypes.byref(info)):
                r = info.rcMonitor
                found.append({'box': (r.left, r.top, r.right, r.bottom),
                              'primary': bool(info.dwFlags & 1)})
            return True

        callback_reference = callback_type(callback)
        if not self.user.EnumDisplayMonitors(None, None, callback_reference, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        found.sort(key=lambda item: (not item['primary'], item['box'][0], item['box'][1]))
        if not found:
            raise RuntimeError('No monitors found')
        return found

    def configure_overlay(self, window):
        hwnd = self.user.GetAncestor(window.winfo_id(), 2)  # GA_ROOT; use the Tk wrapper HWND
        if not hwnd:
            raise RuntimeError('Cannot find overlay window handle')
        style = self.get_style(hwnd, -20)  # GWL_EXSTYLE
        style |= 0x00080000 | 0x00000020 | 0x00000080 | 0x08000000
        style &= ~0x00040000  # Remove WS_EX_APPWINDOW
        ctypes.set_last_error(0)
        self.set_style(hwnd, -20, style)  # Layered, transparent, tool window, no activate
        error = ctypes.get_last_error()
        if error:
            raise ctypes.WinError(error)
        # Additional exclusion on supported Windows builds. We always hide before
        # capture too, because affinity support differs between capture APIs.
        self.user.SetWindowDisplayAffinity(hwnd, 0x11)
        return hwnd

    def position(self, hwnd, left, top, width, height):
        if not self.user.SetWindowPos(hwnd, wintypes.HWND(-1), left, top, width, height,
                                       0x0010 | 0x0020):  # NOACTIVATE | FRAMECHANGED
            raise ctypes.WinError(ctypes.get_last_error())

    def show(self, hwnd):
        self.user.ShowWindow(hwnd, 4)  # SW_SHOWNOACTIVATE
        self.user.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0,
                               0x0001 | 0x0002 | 0x0010)  # NOSIZE | NOMOVE | NOACTIVATE

    def hide(self, hwnd):
        self.user.ShowWindow(hwnd, 0)
        if self.dwm:
            self.dwm.DwmFlush()

    def quit_pressed(self):
        return all(self.user.GetAsyncKeyState(key) & 0x8000 for key in (0x11, 0x12, 0x51))


class ScreenScorer:
    def __init__(self, bundle, device='auto', candidate='selected'):
        from taste_ml import ClipEncoder, DEFAULT_ENCODER
        if bundle.get('format_version') != 1:
            raise ValueError('Unsupported model format. Use the output of WebTaste train.py.')
        self.name = bundle['selected'] if candidate == 'selected' else candidate
        self.model = bundle['models'][self.name]
        self.dimension = bundle['feature_dimension']
        self.encoder = None
        self.encoder_config = bundle['encoder']
        if self.encoder_config['image_policy'] != DEFAULT_ENCODER['image_policy']:
            raise ValueError('Unsupported screenshot preprocessing configuration')
        if self.name != 'baseline':
            # Loaded once; each frame stays in RAM and does not create cache files.
            self.encoder = ClipEncoder(self.encoder_config, device)

    def score(self, image):
        import numpy as np
        from taste_ml import square_image
        if self.encoder is None:
            features = np.zeros((1, self.dimension), dtype=np.float32)
        else:
            encoder = self.encoder
            tensor = encoder.preprocess(square_image(image)).unsqueeze(0).to(encoder.device)
            with encoder.torch.inference_mode():
                features = encoder.model.encode_image(tensor).float()
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            features = features.cpu().numpy()
            if features.shape != (1, self.dimension):
                raise ValueError('Screen embedding dimension does not match the trained model')
        score = float(self.model.predict(features)[0])
        if not math.isfinite(score):
            raise ValueError('The model produced a nonfinite score')
        return max(0.0, min(10.0, score))


def scoring_worker(args, jobs, results, stop):
    try:
        import joblib
        from PIL import ImageGrab
        # Load only the model produced by your local training run.
        scorer = ScreenScorer(joblib.load(args.model), args.device, args.candidate)
        results.put(('ready', scorer.name))
    except Exception:
        logging.exception('Model initialization failed')
        results.put(('fatal', None))
        return
    while not stop.is_set():
        try:
            rectangle = jobs.get(timeout=0.1)
        except queue.Empty:
            continue
        if rectangle is None or stop.is_set():
            return
        try:
            # The UI has hidden its overlay and waited for compositor refresh.
            image = ImageGrab.grab(bbox=rectangle, all_screens=True, include_layered_windows=False)
        except Exception:
            logging.exception('Screen capture failed')
            results.put(('capture_error', None))
            continue
        # Restore the last score while inference runs in the worker thread.
        results.put(('captured', None))
        try:
            results.put(('score', scorer.score(image)))
        except Exception:
            logging.exception('Screen prediction failed')
            results.put(('prediction_error', None))
        finally:
            image.close()


class ScoreOverlay:
    def __init__(self, api, monitor, args):
        import tkinter as tk
        from tkinter import font
        self.api = api
        self.args = args
        self.rectangle = capture_box(monitor, args.crop_top, args.crop_bottom)
        self.root = tk.Tk()
        self.root.withdraw()
        self.window = tk.Toplevel(self.root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes('-topmost', True)
        self.window.attributes('-transparentcolor', '#ff00ff')
        self.window.configure(bg='#ff00ff')
        self.font = font.Font(root=self.root, family='Consolas', size=args.font_size, weight='bold')
        width = max(self.font.measure('10.0 / 10'), self.font.measure('Loading...')) + 24
        height = self.font.metrics('linespace') + 16
        self.window.geometry(f'{width}x{height}')
        self.canvas = tk.Canvas(self.window, width=width, height=height, bg='#ff00ff',
                                highlightthickness=0, bd=0)
        self.canvas.pack()
        self.root.update_idletasks()
        self.hwnd = api.configure_overlay(self.window)
        # Let Tk mark its widgets as mapped, with NOACTIVATE already applied.
        # Reapply native styles after mapping in case Tk updates its wrapper.
        self.window.deiconify()
        self.root.update_idletasks()
        self.hwnd = api.configure_overlay(self.window)
        left, top, right, bottom = monitor['box']
        api.position(self.hwnd, max(left, right - width - args.margin), top + args.margin, width, height)
        self.width, self.height = width, height
        self.jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue()
        self.stop = threading.Event()
        self.ready = False
        self.busy = False
        self.closed = False
        self.next_capture = 0.0
        self.draw('Loading...')
        api.show(self.hwnd)
        self.thread = threading.Thread(target=scoring_worker,
            args=(args, self.jobs, self.results, self.stop), daemon=True, name='screen-predictor')
        self.thread.start()
        self.root.after(30, self.tick)

    def draw(self, text):
        self.canvas.delete('all')
        x, y = self.width // 2, self.height // 2
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)):
            self.canvas.create_text(x + dx, y + dy, text=text, font=self.font, fill='#000000')
        self.canvas.create_text(x, y, text=text, font=self.font, fill='#ffffff')

    def submit_capture(self):
        if not self.closed:
            self.jobs.put_nowait(self.rectangle)

    def tick(self):
        if self.closed:
            return
        if self.api.quit_pressed():
            self.close()
            return
        while True:
            try:
                kind, value = self.results.get_nowait()
            except queue.Empty:
                break
            if kind == 'ready':
                self.ready = True
                self.draw('-- / 10')
                logging.info('Ready: %s model; capture rectangle %s', value, self.rectangle)
                if value == 'baseline':
                    logging.warning('Baseline selected: its score is constant, not a visual preference prediction.')
            elif kind == 'captured':
                self.api.show(self.hwnd)
            elif kind == 'score':
                self.draw(f'{value:.1f} / 10')
                self.api.show(self.hwnd)
                self.busy = False
                self.next_capture = time.monotonic() + self.args.interval
            elif kind == 'fatal':
                self.draw('ERROR')
                self.ready = False
                self.busy = False
                self.api.show(self.hwnd)
            elif kind in ('capture_error', 'prediction_error'):
                self.draw('ERROR')
                self.api.show(self.hwnd)
                self.busy = False
                self.next_capture = time.monotonic() + self.args.interval
        if self.ready and not self.busy and time.monotonic() >= self.next_capture:
            self.busy = True
            self.api.hide(self.hwnd)
            # Keep Tk responsive; do not block the UI during capture or inference.
            self.root.after(60, self.submit_capture)
        self.root.after(30, self.tick)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.root.destroy()
        logging.info('Overlay stopped')

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.close()


def main(argv=None):
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=script_dir / 'models' / 'taste_model.joblib')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--candidate', choices=('selected', 'ridge', 'mlp', 'baseline'), default='selected')
    parser.add_argument('--interval', type=positive_float, default=1.0, help='Seconds between completed predictions')
    parser.add_argument('--monitor', type=positive_int, default=1, help='1 = primary; others ordered left to right')
    parser.add_argument('--list-monitors', action='store_true')
    parser.add_argument('--crop-top', type=nonnegative_int, default=0, help='Pixels to omit from screen top')
    parser.add_argument('--crop-bottom', type=nonnegative_int, default=0, help='Pixels to omit from screen bottom')
    parser.add_argument('--font-size', type=positive_int, default=22)
    parser.add_argument('--margin', type=nonnegative_int, default=12)
    args = parser.parse_args(argv)
    if os.name != 'nt':
        parser.error('This overlay targets Windows. Run it on your Windows WebTaste machine.')
    logging.basicConfig(filename=script_dir / 'screen_taste.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')
    overlay = None
    try:
        api = WindowsAPI()
        api.dpi_awareness()
        monitors = api.monitors()
        if args.list_monitors:
            for i, monitor in enumerate(monitors, 1):
                print(f'{i}: {monitor["box"]}' + (' (primary)' if monitor['primary'] else ''))
            return 0
        if args.monitor > len(monitors):
            raise ValueError(f'Monitor {args.monitor} is unavailable; found {len(monitors)} monitors')
        if not args.model.is_file():
            raise ValueError(f'Model file not found: {args.model}')
        capture_box(monitors[args.monitor - 1], args.crop_top, args.crop_bottom)
        overlay = ScoreOverlay(api, monitors[args.monitor - 1], args)
        overlay.run()
        return 0
    except Exception:
        logging.exception('Overlay failed')
        if overlay is not None:
            overlay.close()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
