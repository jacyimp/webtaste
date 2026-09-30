"""Keyboard-first desktop screenshot rating app."""
import random
import time
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk


class RatingApp:
    def __init__(self, dataset, state='unrated', shuffle=True):
        self.dataset = dataset
        self.state = state
        self.shuffle = shuffle
        self.queue = []
        self.current = None
        self.original = None
        self.photo = None
        self.last_action = 0.0
        self.pressed_keys = set()
        self.resize_job = None
        self.root = tk.Tk()
        self.root.title('WebTaste — rate website designs')
        self.root.geometry('1200x850')
        self.root.minsize(760, 500)
        self.root.configure(bg='#f5f2ef')
        self.show_details = tk.BooleanVar(value=False)
        self.progress = tk.StringVar()
        self.details = tk.StringVar()
        tk.Label(self.root, text='How much do I like this first screen?',
                 font=('Segoe UI', 18, 'bold'), bg='#f5f2ef', fg='#722f37').pack(pady=(12, 3))
        tk.Label(self.root, textvariable=self.progress, bg='#f5f2ef').pack()
        self.canvas = tk.Canvas(self.root, bg='#dedad6', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=16, pady=10)
        self.canvas.bind('<Configure>', self.resize)
        tk.Label(self.root, textvariable=self.details, bg='#f5f2ef').pack()
        tk.Label(self.root, text='0 = strongly dislike     5 = neutral     10 = strongly like',
                 bg='#f5f2ef').pack(pady=4)
        scores = tk.Frame(self.root, bg='#f5f2ef')
        scores.pack()
        for score in range(11):
            tk.Button(scores, text=str(score), width=4, font=('Segoe UI', 12, 'bold'),
                      bg='#722f37', fg='white', command=lambda s=score: self.assign(s)).pack(side='left', padx=3)
        actions = tk.Frame(self.root, bg='#f5f2ef')
        actions.pack(pady=10)
        for text, command in [('Skip (S)', self.skip), ('Undo (U)', self.undo),
                              ('Refresh queue', self.refresh), ('Export CSV', self.export)]:
            tk.Button(actions, text=text, command=command).pack(side='left', padx=5)
        tk.Checkbutton(actions, text='Show website details', variable=self.show_details,
                       command=self.update_details, bg='#f5f2ef').pack(side='left', padx=8)
        tk.Label(self.root, text='Keys: 0–9 • X = 10 • S = skip • U / Ctrl+Z = undo • Esc = save and quit',
                 bg='#f5f2ef').pack(pady=(0, 10))
        self.root.bind('<Key>', self.on_key)
        self.root.bind('<KeyRelease>', self.on_release)
        self.root.bind('<FocusOut>', lambda event: self.pressed_keys.clear())
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.refresh()

    def refresh(self):
        self.queue = self.dataset.rating_queue(self.state)
        if self.shuffle:
            random.shuffle(self.queue)
        self.next_image()

    def next_image(self):
        self.current = None
        self.original = None
        self.photo = None
        self.canvas.delete('all')
        while self.queue:
            candidate = self.queue.pop(0)
            try:
                with Image.open(self.dataset.image(candidate)) as image:
                    self.original = ImageOps.exif_transpose(image).convert('RGB')
                self.current = candidate
                break
            except (OSError, ValueError) as exc:
                messagebox.showwarning('Cannot load image', f'Screenshot {candidate}: {exc}')
        stats = self.dataset.stats()
        self.progress.set(f"Rated: {stats['rated']}   •   Skipped: {stats['skipped']}   •   Remaining in this queue: {len(self.queue) + bool(self.current)}")
        self.update_details()
        self.render()

    def update_details(self):
        if self.current is None:
            self.details.set('Queue complete. Use Refresh queue after capturing more sites.')
        elif self.show_details.get():
            row = self.dataset.row(self.current)
            text = f"{row['domain']}  |  {row['title'] or ''}"
            if row['score'] is not None:
                text += f"  |  Current score: {row['score']}"
            self.details.set(text[:160])
        else:
            self.details.set('Website details hidden')

    def resize(self, event=None):
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(80, self.render)

    def render(self):
        self.resize_job = None
        self.canvas.delete('all')
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if self.original is None:
            self.canvas.create_text(width // 2, height // 2, text='No screenshots in this queue',
                                    font=('Segoe UI', 16), fill='#555555')
            return
        image = ImageOps.contain(self.original, (width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(width // 2, height // 2, image=self.photo)

    def assign(self, score=None, *, skip=False):
        if self.current is None or time.monotonic() - self.last_action < 0.25:
            return
        self.last_action = time.monotonic()
        try:
            self.dataset.rate(self.current, score, skip=skip)
        except Exception as exc:
            messagebox.showerror('Rating could not be completed', str(exc))
            return
        self.next_image()

    def skip(self):
        self.assign(skip=True)

    def undo(self):
        try:
            site_id = self.dataset.undo()
        except Exception as exc:
            messagebox.showerror('Undo could not be completed', str(exc))
            return
        if site_id is None:
            return
        if self.current is not None and self.current != site_id:
            self.queue.insert(0, self.current)
        self.queue = [x for x in self.queue if x != site_id]
        self.queue.insert(0, site_id)
        self.last_action = time.monotonic()
        self.next_image()

    def on_key(self, event):
        if event.state & 4 and event.keysym.lower() != 'z':
            return
        key = event.keysym.lower()
        if key in self.pressed_keys:
            return 'break'
        self.pressed_keys.add(key)
        if key in '0123456789' and len(key) == 1:
            self.assign(int(key))
        elif key.startswith('kp_') and key[3:].isdigit() and len(key[3:]) == 1:
            self.assign(int(key[3:]))
        elif key == 'x':
            self.assign(10)
        elif key == 's':
            self.skip()
        elif key == 'u' or (key == 'z' and event.state & 4):
            self.undo()
        elif key == 'escape':
            self.close()
        return 'break'

    def on_release(self, event):
        self.pressed_keys.discard(event.keysym.lower())

    def export(self):
        try:
            path = self.dataset.export()
            messagebox.showinfo('Exported', f'Saved {path}')
        except Exception as exc:
            messagebox.showerror('Export failed', str(exc))

    def close(self):
        try:
            self.dataset.export()
        except Exception as exc:
            messagebox.showerror('CSV export failed', f'Ratings remain in SQLite. {exc}')
        self.root.destroy()

    def run(self):
        self.root.mainloop()
