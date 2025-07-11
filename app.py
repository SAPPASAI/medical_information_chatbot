
import tkinter as tk
from tkinter import font as tkFont, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import sqlite3
import hashlib
import os
import chatbot # Your existing chatbot logic module
import threading
import pyttsx3
import time

# Add ffmpeg path manually for soundfile/whisper to find it
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# Speech-to-Text specific imports
try:
    import whisper
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    import tempfile
    STT_ENABLED = True
except ImportError as e:
    print(f"Speech-to-Text dependencies not found: {e}. STT will be disabled.")
    STT_ENABLED = False

# ===================================================================================
# 2. CONFIGURATION AND PATH HELPER
# ===================================================================================
RECORD_SECONDS = 7  # Duration of the audio recording in seconds

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATH = os.path.join(BASE_DIR, 'assets')

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller. """
    return os.path.join(ASSETS_PATH, relative_path)

# ===================================================================================
# 3. DATABASE HELPER (Unchanged)
# ===================================================================================
class Database:
    def __init__(self, db_name="medical_chatbot.db"):
        self.db_name = os.path.join(BASE_DIR, db_name)
        self._check_and_recreate_db_if_needed()
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def _check_and_recreate_db_if_needed(self):
        if not os.path.exists(self.db_name): return
        try:
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM User LIMIT 1")
                cursor.execute("SELECT 1 FROM ChatHistory LIMIT 1")
        except sqlite3.OperationalError as e:
            print(f"Database schema error: {e}. Backing up and recreating database.")
            backup_name = self.db_name + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(self.db_name, backup_name)
            print(f"Backed up corrupt database to {backup_name}")

    def create_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS User (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL, password TEXT NOT NULL)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS ChatHistory (id INTEGER PRIMARY KEY, user_id INTEGER, message_text TEXT, sender_type TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES User(id))''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS Prediction (id INTEGER PRIMARY KEY, user_id INTEGER, symptoms TEXT, predicted_disease TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES User(id))''')
        self.conn.commit()

    def hash_password(self, password): return hashlib.sha256(password.encode()).hexdigest()
    def add_user(self, username, email, password):
        try:
            self.cursor.execute("INSERT INTO User (username, email, password) VALUES (?, ?, ?)", (username, email, self.hash_password(password)))
            self.conn.commit(); return True
        except sqlite3.IntegrityError: return False
    def check_user(self, username, password):
        self.cursor.execute("SELECT id, username, email FROM User WHERE username = ? AND password = ?", (username, self.hash_password(password)))
        return self.cursor.fetchone()
    def update_password(self, email, new_password):
        self.cursor.execute("UPDATE User SET password = ? WHERE email = ?", (self.hash_password(new_password), email))
        self.conn.commit(); return self.cursor.rowcount > 0
    def add_chat_message(self, user_id, message, sender):
        self.cursor.execute("INSERT INTO ChatHistory (user_id, message_text, sender_type) VALUES (?, ?, ?)", (user_id, message, sender))
        self.conn.commit()
    def get_chat_history(self, user_id):
        self.cursor.execute("SELECT message_text, sender_type FROM ChatHistory WHERE user_id = ? ORDER BY timestamp ASC", (user_id,))
        return self.cursor.fetchall()
    def clear_user_history(self, user_id):
        self.cursor.execute("DELETE FROM ChatHistory WHERE user_id = ?", (user_id,)); self.conn.commit()

# ===================================================================================
# 4. SPEECH AND AUDIO HANDLERS
# ===================================================================================
class TTSHandler:
    def __init__(self):
        try:
            self.engine = pyttsx3.init(driverName='sapi5')
            self.engine.setProperty('rate', 150); self.engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"Error initializing TTS engine: {e}. TTS will be disabled."); self.engine = None
    def speak_text(self, text, on_finish_callback=None):
        if not self.engine:
            if on_finish_callback: on_finish_callback(); return
        try: self.engine.say(text); self.engine.runAndWait()
        except Exception as e: print(f"Text-to-speech error: {e}")
        finally:
            if on_finish_callback: on_finish_callback()
    def stop_speaking(self):
        if self.engine: self.engine.stop()

class SpeechRecognitionHandler:
    """Handles STT using a fixed-duration recording for maximum reliability."""
    def __init__(self, controller, model_size="base.en"):
        self.controller = controller
        self.model = None
        self.model_size = model_size
        self.is_loading_model = False
        self.is_recording = False # Use a new flag to prevent multiple clicks

    def _load_model(self, on_load_callback):
        self.is_loading_model = True
        print(f"Loading Whisper model '{self.model_size}' for the first time...")
        try:
            self.model = whisper.load_model(self.model_size)
            print("Whisper model loaded successfully.")
        except Exception as e:
            self.model = None
            self.controller.after(0, lambda: messagebox.showerror("Whisper Error", f"Could not load speech model: {e}\n\nPlease ensure FFmpeg is installed."))
        finally:
            self.is_loading_model = False
            self.controller.after(0, on_load_callback)

    def start_recording_session(self, on_transcription_result, on_state_change):
        if not STT_ENABLED or self.is_loading_model or self.is_recording:
            return

        if self.model is None:
            on_state_change("loading_model")
            threading.Thread(target=self._load_model, args=(lambda: self.start_recording_session(on_transcription_result, on_state_change),), daemon=True).start()
            return

        self.is_recording = True
        on_state_change("listening")
        threading.Thread(target=self._record_and_transcribe, args=(on_transcription_result, on_state_change), daemon=True).start()

    def _record_and_transcribe(self, on_transcription_result, on_state_change):
        """Records for a fixed duration and then transcribes."""
        samplerate = 16000
        try:
            print(f"🎙️ Recording for {RECORD_SECONDS} seconds...")
            # This is the reliable, blocking recording method
            audio_data = sd.rec(int(RECORD_SECONDS * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait() # Wait for the recording to complete
            
            print("Recording finished. Processing...")
            self.controller.after(0, on_state_change, "processing")

        except Exception as e:
            print(f"Audio recording error: {e}")
            self.controller.after(0, lambda: messagebox.showerror("Audio Error", f"Could not start recording: {e}\nPlease check your microphone."))
            self.controller.after(0, on_state_change, "idle")
            self.is_recording = False
            return

        # Create, close, and then write to a temporary file
        temp_file_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio_file = temp_file_handle.name
        temp_file_handle.close()

        try:
            sf.write(temp_audio_file, audio_data, samplerate)
            
            result = self.model.transcribe(temp_audio_file)
            transcribed_text = result['text']
            print(f"Transcription: {transcribed_text}")
            self.controller.after(0, on_transcription_result, transcribed_text)
        except Exception as e:
            print(f"Transcription failed: {e}")
            self.controller.after(0, on_transcription_result, "")
        finally:
            self.controller.after(0, on_state_change, "idle")
            self.is_recording = False # Allow new recordings
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)

# ===================================================================================
# 5. UI HELPER CLASSES (Unchanged)
# ===================================================================================
class ResponsiveBgFrame(tk.Frame):
    def __init__(self, parent, controller, image_path):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        try:
            self.original_image = Image.open(resource_path(image_path))
        except Exception:
            print(f"Image not found: {image_path}. Using fallback background.")
            self.original_image = Image.new("RGB", (1, 1), color="#EAEAF2")
        self.bg_label = tk.Label(self); self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.bind("<Configure>", self._resize_image)

    def _resize_image(self, event):
        resized = self.original_image.resize((event.width, event.height), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(resized)
        self.bg_label.configure(image=self.bg_photo)

class ModernMenuButton(tk.Frame):
    def __init__(self, parent, text, icon, command):
        super().__init__(parent, bg="#ffffff", bd=1, relief="solid")
        self.command = command
        self.icon_label = tk.Label(self, text=icon, font=('Segoe UI Emoji', 36), bg="white"); self.icon_label.pack(pady=(20, 10))
        self.text_label = tk.Label(self, text=text, font=('Segoe UI', 14, 'bold'), bg="white"); self.text_label.pack(pady=(0, 20))
        self._bind_all_children("<Button-1>", self.on_click); self._bind_all_children("<Enter>", self.on_hover); self._bind_all_children("<Leave>", self.on_leave)
    def _bind_all_children(self, seq, func): self.bind(seq, func); [c.bind(seq, func) for c in self.winfo_children()]
    def on_hover(self, e): self.config(bg="#e9ecef"); [c.config(bg="#e9ecef") for c in self.winfo_children()]
    def on_leave(self, e): self.config(bg="#ffffff"); [c.config(bg="#ffffff") for c in self.winfo_children()]
    def on_click(self, e): self.command()

# ===================================================================================
# 6. MAIN APPLICATION AND PAGES (Unchanged)
# ===================================================================================
class MainApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.db = Database(); self.speech_handler = TTSHandler()
        self.speech_recognizer = SpeechRecognitionHandler(self) if STT_ENABLED else None
        self.current_user_id = None; self.current_username = None
        self.title("Medicine Information and Advice System"); self.geometry("1200x750"); self.minsize(1000, 700)
        container = tk.Frame(self); container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1); container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        pages = (HomePage, MenuPage, LoginPage, RegisterPage, ForgotPasswordPage, ChatbotApp, AbstractPage, AlgorithmPage, ExamplePage, DatasetPage, HelpPage)
        for F in pages:
            frame = F(parent=container, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("HomePage")
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        if page_name == 'ChatbotApp' and not self.current_user_id:
            messagebox.showwarning("Login Required", "You must be logged in."); self.show_frame("LoginPage"); return
        if hasattr(frame, 'on_show'): frame.on_show()
        frame.tkraise()
    def logout(self):
        self.current_user_id = None; self.current_username = None
        self.frames["ChatbotApp"].clear_chat(show_confirmation=False); self.show_frame("LoginPage")

class ImageContentPage(tk.Frame):
    def __init__(self, parent, controller, image_file):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.responsive_bg = ResponsiveBgFrame(self, controller, image_file); self.responsive_bg.place(x=0, y=0, relwidth=1, relheight=1)
        tk.Button(self, text="🔙 Back to Menu", font=("Segoe UI", 12, "bold"), cursor="hand2", relief="raised", command=lambda: controller.show_frame("MenuPage")).place(x=20, y=20)
class HomePage(ResponsiveBgFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "AI_in_Medicine.png")

        # Transparent button container (no bg/frame color)
        btn_frame = tk.Frame(self, bg="", bd=0, highlightthickness=0)
        
        # Styled Buttons
        proceed_btn = tk.Button(
            btn_frame, text="PROCEED", font=('Helvetica', 14, 'bold'),
            relief="flat", cursor="hand2", bg="#007bff", fg="white",
            activebackground="#0056b3", activeforeground="white",
            command=lambda: self.controller.show_frame("MenuPage")
        )
        exit_btn = tk.Button(
            btn_frame, text="EXIT", font=('Helvetica', 14, 'bold'),
            relief="flat", cursor="hand2", bg="#dc3545", fg="white",
            activebackground="#a71d2a", activeforeground="white",
            command=self.controller.destroy
        )

        proceed_btn.pack(side="left", padx=10, pady=10, ipadx=20, ipady=10)
        exit_btn.pack(side="left", padx=10, pady=10, ipadx=20, ipady=10)

        btn_frame.place(relx=0.5, rely=0.8, anchor="center")

class MenuPage(ResponsiveBgFrame):
    def __init__(self, parent, controller):
        # Using a solid color for the background for this runnable example
        super().__init__(parent, controller, "medicineshutterstock_1421041688.jpg")
        
        tk.Button(self, text="🔙 Back to Home", font=("Segoe UI", 12, "bold"), cursor="hand2", relief="raised", command=lambda: self.controller.show_frame("HomePage")).place(x=20, y=20)
        
        center_frame = tk.Frame(self, bg=self.cget('bg'))
        center_frame.place(relx=0.5, rely=0.5, anchor="center")

        # --- CHANGE 1: Create the heading Label ---
        # We create a label with larger, bold font to serve as the heading.
        # It's placed inside the center_frame to be part of the "card".
        heading_label = tk.Label(center_frame, text="Main Menu", 
                                 font=("Segoe UI", 28, "bold"), 
                                 bg=center_frame.cget('bg'), 
                                 fg="#333333")

        # --- CHANGE 2: Place the heading at the top of the grid ---
        # We use grid() to place it in the first row (row=0).
        # columnspan=3 makes it span across all three columns the buttons will use.
        # pady adds some vertical spacing between the heading and the buttons below it.
        heading_label.grid(row=0, column=0, columnspan=3, pady=(0, 25), sticky="ew")

        buttons = {
            "Login / Chat": ("👤", lambda: self.controller.show_frame("LoginPage")),
            "Abstract": ("📄", lambda: self.controller.show_frame("AbstractPage")),
            "Algorithm": ("⚙️", lambda: self.controller.show_frame("AlgorithmPage")),
            "About": ("💡", lambda: self.controller.show_frame("ExamplePage")),
            "Dataset": ("🗂️", lambda: self.controller.show_frame("DatasetPage")),
            "Help": ("❓", lambda: self.controller.show_frame("HelpPage"))
        }

        for i, (text, (icon, command)) in enumerate(buttons.items()):
            # --- CHANGE 3: Adjust the button loop to start from the next row ---
            # Since the heading is in row=0, the buttons must start from row=1.
            # We change `row=i // 3` to `row=(i // 3) + 1`.
            ModernMenuButton(center_frame, text, icon, command).grid(
                row=(i // 3) + 1,  # Start buttons from row 1
                column=i % 3, 
                padx=20, 
                pady=20, 
                sticky="nsew"
            )


class AuthPage(ResponsiveBgFrame):
    def __init__(self, parent, controller, image_file):
        super().__init__(parent, controller, image_file)
        self.main_frame = tk.Frame(self, bg="white", bd=0, highlightbackground="#ccc", highlightthickness=1)
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center", width=460, height=560)

        # Card shadow background effect
        self.shadow = tk.Frame(self, bg="#d9d9d9")
        self.shadow.place(relx=0.5, rely=0.5, anchor="center", width=470, height=570)

        self.main_frame.lift()

        tk.Button(self.main_frame, text="← Back to Menu", font=('Segoe UI', 10, 'bold'), bg="white", fg="#007bff",
                  activeforeground="#0056b3", relief="flat", cursor="hand2",
                  command=lambda: self.controller.show_frame("MenuPage")
                  ).place(x=10, y=10)

    def _create_entry(self, label_text, is_password):
        container = tk.Frame(self.main_frame, bg="white")
        container.pack(pady=(10, 5), padx=40, fill="x")

        tk.Label(container, text=label_text, font=('Segoe UI', 11), bg="white").pack(anchor="w")
        entry = tk.Entry(container, font=('Segoe UI', 12), bd=1, relief="solid",
                         highlightthickness=1, highlightbackground="#ccc",
                         show="*" if is_password else "")
        entry.pack(ipady=6, fill="x", pady=(5, 0))
        return entry


class LoginPage(AuthPage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "travel-with-medicine.jpg")
        tk.Label(self.main_frame, text="User Login", font=('Segoe UI', 22, 'bold'), bg="white").pack(pady=(60, 20))

        self.username_entry = self._create_entry("Username", False)
        self.password_entry = self._create_entry("Password", True)

        tk.Button(self.main_frame, text="Login", font=('Segoe UI', 13, 'bold'),
                  bg="#007bff", fg="white", activebackground="#0056b3",
                  relief="flat", cursor="hand2", command=self._login_handler
                  ).pack(pady=20, ipadx=8, ipady=6, fill="x", padx=40)

        tk.Button(self.main_frame, text="Forgot Password?", font=('Segoe UI', 10), bg="white", fg="#007bff",
                  activeforeground="#0056b3", relief="flat",
                  command=lambda: self.controller.show_frame("ForgotPasswordPage")
                  ).pack()

        tk.Button(self.main_frame, text="Don't have an account? Sign Up", font=('Segoe UI', 10), bg="white", fg="#007bff",
                  activeforeground="#0056b3", relief="flat",
                  command=lambda: self.controller.show_frame("RegisterPage")
                  ).pack(pady=5)

    def _login_handler(self):
        username, password = self.username_entry.get(), self.password_entry.get()
        if not all((username, password)):
            return messagebox.showerror("Input Error", "All fields are required.")
        user_data = self.controller.db.check_user(username, password)
        if user_data:
            self.controller.current_user_id, self.controller.current_username = user_data[0], user_data[1]
            self.controller.show_frame("ChatbotApp")
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")


class RegisterPage(AuthPage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "travel-with-medicine.jpg")
        tk.Label(self.main_frame, text="Create Account", font=('Segoe UI', 22, 'bold'), bg="white").pack(pady=(60, 10))

        self.username_entry = self._create_entry("Username", False)
        self.email_entry = self._create_entry("Email", False)
        self.password_entry = self._create_entry("Password", True)

        tk.Button(self.main_frame, text="Register", font=('Segoe UI', 13, 'bold'),
                  bg="#28a745", fg="white", activebackground="#218838",
                  cursor="hand2", relief="flat", command=self._register_handler
                  ).pack(pady=20, ipadx=8, ipady=6, fill="x", padx=40)

        tk.Button(self.main_frame, text="Already have an account? Login", font=('Segoe UI', 10), bg="white", fg="#007bff",
                  activeforeground="#0056b3", relief="flat",
                  command=lambda: self.controller.show_frame("LoginPage")
                  ).pack()

    def _register_handler(self):
        username, email, password = self.username_entry.get(), self.email_entry.get(), self.password_entry.get()
        if not all((username, email, password)):
            return messagebox.showerror("Error", "All fields are required.")
        if self.controller.db.add_user(username, email, password):
            messagebox.showinfo("Success", "Account created successfully!")
            self.controller.show_frame("LoginPage")
        else:
            messagebox.showerror("Error", "Username or Email already exists.")


class ForgotPasswordPage(AuthPage):
    def __init__(self, parent, controller):
        super().__init__(parent, controller, "travel-with-medicine.jpg")
        tk.Label(self.main_frame, text="Reset Password", font=('Segoe UI', 22, 'bold'), bg="white").pack(pady=(60, 20))

        self.email_entry = self._create_entry("Enter your Email", False)
        self.new_pass_entry = self._create_entry("New Password", True)
        self.confirm_pass_entry = self._create_entry("Confirm New Password", True)

        tk.Button(self.main_frame, text="Reset Password", font=('Segoe UI', 13, 'bold'),
                  bg="#dc3545", fg="white", activebackground="#c82333",
                  cursor="hand2", command=self._reset_password_handler,
                  relief="flat"
                  ).pack(pady=25, ipadx=8, ipady=6, fill="x", padx=40)

        tk.Button(self.main_frame, text="Back to Login", font=('Segoe UI', 10), bg="white", fg="#007bff",
                  activeforeground="#0056b3", relief="flat",
                  command=lambda: self.controller.show_frame("LoginPage")
                  ).pack()

    def _reset_password_handler(self):
        email, new_pass, confirm_pass = self.email_entry.get(), self.new_pass_entry.get(), self.confirm_pass_entry.get()
        if not all((email, new_pass, confirm_pass)):
            return messagebox.showerror("Error", "All fields are required.")
        if new_pass != confirm_pass:
            return messagebox.showerror("Error", "Passwords do not match.")
        if self.controller.db.update_password(email, new_pass):
            messagebox.showinfo("Success", "Password updated successfully.")
            self.controller.show_frame("LoginPage")
        else:
            messagebox.showerror("Error", "No account found with that email address.")

class AbstractPage(ImageContentPage):
    def __init__(self, parent, controller): super().__init__(parent, controller, "abstract.png")
class AlgorithmPage(ImageContentPage):
    def __init__(self, parent, controller): super().__init__(parent, controller, "11.jpg")
class ExamplePage(ImageContentPage):
    def __init__(self, parent, controller): super().__init__(parent, controller, "5.jpg")
class DatasetPage(ImageContentPage):
    def __init__(self, parent, controller): super().__init__(parent, controller, "12.jpg")
class HelpPage(ImageContentPage):
    def __init__(self, parent, controller): super().__init__(parent, controller, "6.jpg")

class ChatbotApp(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent); self.controller = controller
        self.theme = "dark"; self.typing_indicator = None; self.is_speaking = False; self.style = {}
        self.dark_theme = {"bg": "#121212", "sidebar_bg": "#1e1e1e", "input_bg": "#1e1e1e", "user_bg": "#2c2c2c", "bot_bubble_bg": "#0052d6", "send_btn_bg": "#0052d6", "text": "#e0e0e0", "send_btn_text": "white"}
        self.light_theme = {"bg": "#ffffff", "sidebar_bg": "#f5f5f5", "input_bg": "#f0f0f0", "user_bg": "#d9fdd3", "bot_bubble_bg": "#e9ecef", "send_btn_bg": "#007bff", "text": "#1c2c1c", "send_btn_text": "white"}
        self._build_ui(); self.update_theme()
    def on_show(self):
        self.username_label.config(text=f"👤 {self.controller.current_username}"); self._load_user_data()
    def _build_ui(self):
        main_area = tk.Frame(self); main_area.pack(fill=tk.BOTH, expand=True)
        self.sidebar = tk.Frame(main_area, width=180); self.sidebar.pack(side="left", fill="y", padx=10)
        self.username_label = tk.Label(self.sidebar, text="", font=("Segoe UI", 12, "bold"), anchor="w"); self.username_label.pack(pady=10, padx=10, fill="x")
        tk.Button(self.sidebar, text="🌙/☀️ Toggle Theme", command=self.toggle_theme, bd=0, cursor="hand2", anchor="w").pack(fill="x", padx=10, pady=5)
        self.sidebar_speak_button = tk.Button(self.sidebar, text="🔊 Speak Response", command=self.toggle_speaking, bd=0, cursor="hand2", anchor="w"); self.sidebar_speak_button.pack(fill="x", padx=10, pady=5)
        tk.Button(self.sidebar, text="➕ New Chat", command=self.start_new_chat, bd=0, cursor="hand2", anchor="w").pack(fill="x", padx=10, pady=5)
        tk.Button(self.sidebar, text="🗑️ Clear History", command=self.clear_chat, bd=0, cursor="hand2", anchor="w").pack(fill="x", padx=10, pady=5)
        tk.Button(self.sidebar, text="🚪 Logout", command=self.controller.logout, bd=0, cursor="hand2", anchor="w").pack(side="bottom", fill="x", padx=10, pady=10)
        chat_container = tk.Frame(main_area); chat_container.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(chat_container, highlightthickness=0); self.scrollbar = tk.Scrollbar(chat_container, orient="vertical", command=self.canvas.yview); self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.configure(yscrollcommand=self.scrollbar.set); self.canvas_frame_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))); self.canvas.bind("<Configure>", self._on_chat_canvas_resize)
        self.canvas.pack(side="left", fill="both", expand=True); self.scrollbar.pack(side="right", fill="y")
        self.input_frame = tk.Frame(self); self.input_frame.pack(side="bottom", fill="x", padx=20, pady=20)
        self.mic_button = tk.Button(self.input_frame, text="🎤", font=("Segoe UI", 14), command=self._start_listening_session, relief="flat", bd=0)
        if STT_ENABLED: self.mic_button.pack(side="left", padx=(0, 10))
        self.input_canvas = tk.Canvas(self.input_frame, height=55, highlightthickness=0); self.input_canvas.pack(fill="x", expand=True)
        self.entry = tk.Entry(self.input_canvas, font=("Segoe UI", 12), relief="flat", bd=0); self.entry.bind("<Return>", self._send_message)
        self.send_button = tk.Button(self.input_canvas, text="➤", font=("Segoe UI", 16), relief="flat", bd=0, cursor="hand2", command=self._send_message); self.input_canvas.bind("<Configure>", self._on_input_canvas_resize)
    def _start_listening_session(self):
        if self.controller.speech_recognizer: self.controller.speech_recognizer.start_recording_session(self._on_transcription_result, self._on_listening_state_change)
    def _on_listening_state_change(self, state):
        entry_text = {"loading_model": "Loading speech model...", "listening": f"Listening for {RECORD_SECONDS} sec...", "processing": "Processing audio..."}.get(state)
        mic_text = {"loading_model": "⏳", "listening": "...", "processing": "🤔", "idle": "🎤"}.get(state)
        is_disabled = state in ["loading_model", "listening", "processing"]
        self.mic_button.config(text=mic_text, state=tk.DISABLED if is_disabled else tk.NORMAL)
        if entry_text: self.entry.delete(0, tk.END); self.entry.insert(0, entry_text)
        elif state == "idle" and "..." in self.entry.get(): self.entry.delete(0, tk.END)
    def _on_transcription_result(self, text): self.entry.delete(0, tk.END); self.entry.insert(0, text)
    def _load_user_data(self):
        self._clear_chat_display()
        for msg, sender in self.controller.db.get_chat_history(self.controller.current_user_id): self.add_message(msg, sender, save_to_db=False)
        self.canvas.yview_moveto(1.0)
        if not self.scrollable_frame.winfo_children(): self.add_message("👋 Hello! I'm your medical assistant.", "bot", save_to_db=False)
    def _clear_chat_display(self): [w.destroy() for w in self.scrollable_frame.winfo_children()]
    def _on_input_canvas_resize(self, event=None):
        self.update_idletasks(); w, h = self.input_canvas.winfo_width(), self.input_canvas.winfo_height()
        if w < 10 or h < 10: return
        r = h // 2; self.input_canvas.delete("all")
        self.input_canvas.create_oval(0, 0, h, h, fill=self.style['input_bg'], outline=""); self.input_canvas.create_oval(w - h, 0, w, h, fill=self.style['send_btn_bg'], outline=""); self.input_canvas.create_rectangle(r, 0, w - r, h, fill=self.style['input_bg'], outline="")
        self.entry.place(x=r, y=5, width=w - h - 15, height=h - 10); self.send_button.place(x=w - h, y=0, width=h, height=h)
    def toggle_theme(self): self.theme = "light" if self.theme == "dark" else "dark"; self.update_theme()
    def update_theme(self):
        self.style = self.light_theme if self.theme == "light" else self.dark_theme
        self.configure(bg=self.style['bg']); self.input_frame.configure(bg=self.style['bg']); self.sidebar.configure(bg=self.style['sidebar_bg']); self.canvas.configure(bg=self.style['bg']); self.scrollable_frame.configure(bg=self.style['bg']); self.input_canvas.configure(bg=self.style['bg'])
        self.entry.configure(bg=self.style['input_bg'], fg=self.style['text'], insertbackground=self.style['text'])
        self.send_button.configure(bg=self.style['send_btn_bg'], fg=self.style['send_btn_text'], activebackground=self.style['send_btn_bg'], activeforeground=self.style['send_btn_text'])
        if STT_ENABLED: self.mic_button.configure(bg=self.style['bg'], fg=self.style['text'])
        self._on_input_canvas_resize(); self._update_speaker_buttons_state()
        for child in self.sidebar.winfo_children():
            if isinstance(child, (tk.Button, tk.Label)): child.configure(bg=self.style['sidebar_bg'], fg=self.style['text'])
        for frame in self.scrollable_frame.winfo_children():
            if hasattr(frame, 'sender_type'):
                bubble_color = self.style['bot_bubble_bg'] if frame.sender_type == 'bot' else self.style['user_bg']
                text_color = self.style['send_btn_text'] if frame.sender_type == 'bot' and self.theme == 'dark' else self.style['text']
                frame.configure(bg=self.style['bg']); frame.winfo_children()[0].configure(bg=self.style['bg'], fg=self.style['text'])
                frame.winfo_children()[1].configure(bg=bubble_color); frame.winfo_children()[1].winfo_children()[0].configure(bg=bubble_color, fg=text_color)
    def add_message(self, message, sender, is_typing=False, save_to_db=True):
        frame = tk.Frame(self.scrollable_frame, bg=self.style['bg']); frame.sender_type = sender
        tk.Label(frame, text="🤖" if sender == "bot" else "🧑", bg=self.style['bg'], fg=self.style['text'], font=("Segoe UI", 16)).pack(side=tk.LEFT, padx=(10, 5), anchor='n')
        bubble_color = self.style['bot_bubble_bg'] if sender == 'bot' else self.style['user_bg']
        text_color = self.style['send_btn_text'] if sender == 'bot' and self.theme == 'dark' else self.style['text']
        bubble_frame = tk.Frame(frame, bg=bubble_color)
        wrap_len = self.canvas.winfo_width() - 150 if self.canvas.winfo_width() > 150 else 300
        text_label = tk.Label(bubble_frame, text=message, fg=text_color, bg=bubble_color, font=("Segoe UI", 11, "italic" if is_typing else "normal"), wraplength=wrap_len, justify='left'); text_label.pack(padx=12, pady=8)
        bubble_frame.pack(side=tk.LEFT, pady=(0, 2)); frame.pack(anchor="w" if sender == "bot" else "e", fill=tk.X, pady=(10, 0), padx=10, expand=True)
        self.after(100, lambda: self.canvas.yview_moveto(1.0))
        if save_to_db and not is_typing: self.controller.db.add_chat_message(self.controller.current_user_id, message, sender)
        return frame
    def _on_chat_canvas_resize(self, event):
        wrap_len = event.width - 150 if event.width > 150 else 300
        for frame in self.scrollable_frame.winfo_children():
            if len(frame.winfo_children()) > 1 and len(frame.winfo_children()[1].winfo_children()) > 0:
                frame.winfo_children()[1].winfo_children()[0].configure(wraplength=wrap_len)
    def _send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input or "..." in user_input: return
        self.add_message(user_input, "user"); self.entry.delete(0, tk.END)
        self.typing_indicator = self.add_message("HealthBot is typing...", "bot", is_typing=True, save_to_db=False)
        threading.Thread(target=self._get_and_display_bot_response, args=(user_input,), daemon=True).start()
    def _get_and_display_bot_response(self, message):
        response = chatbot.get_bot_response(message, self.controller.current_user_id)
        self.after(0, self._update_ui_with_response, response)
    def _update_ui_with_response(self, response):
        if self.typing_indicator:
            self.typing_indicator.destroy(); self.typing_indicator = None
        self.add_message(response, "bot")
        if self.is_speaking: self.toggle_speaking(speak_now=True)
    def start_new_chat(self): self._clear_chat_display(); self.add_message("👋 Hello! I'm your medical assistant.", "bot", save_to_db=False)
    def clear_chat(self, show_confirmation=True):
        if show_confirmation and not messagebox.askyesno("Confirm", "Delete chat history? This cannot be undone."): return
        self._clear_chat_display()
        if self.controller.current_user_id: self.controller.db.clear_user_history(self.controller.current_user_id)
        if show_confirmation: messagebox.showinfo("Success", "Chat history cleared.")
        self.add_message("Chat history cleared. How can I help you now?", "bot", save_to_db=False)
    def _update_speaker_buttons_state(self): self.sidebar_speak_button.config(text="⏹️ Stop Speaking" if self.is_speaking else "🔊 Speak Response")
    def _reset_speaker_ui(self): self.is_speaking = False; self._update_speaker_buttons_state()
    def toggle_speaking(self, speak_now=False):
        if self.is_speaking and not speak_now:
            self.controller.speech_handler.stop_speaking(); self._reset_speaker_ui(); return
        last_bot_message = None
        for widget in reversed(self.scrollable_frame.winfo_children()):
            if hasattr(widget, 'sender_type') and widget.sender_type == 'bot':
                last_bot_message = widget.winfo_children()[1].winfo_children()[0].cget("text"); break
        if last_bot_message and "typing" not in last_bot_message:
            self.is_speaking = True; self._update_speaker_buttons_state()
            threading.Thread(target=self.controller.speech_handler.speak_text, args=(last_bot_message, self._reset_speaker_ui), daemon=True).start()
        elif not speak_now: self._reset_speaker_ui()

# ===================================================================================
# 7. APPLICATION EXECUTION
# ===================================================================================
if __name__ == "__main__":
    if not os.path.exists(ASSETS_PATH):
        os.makedirs(ASSETS_PATH)
        print("="*60 + "\nIMPORTANT: 'assets' directory created.\n" + f"Please place your image files inside: {ASSETS_PATH}\n" + "e.g., 'AI_in_Medicine.png', 'abstract.png', etc.\n" + "="*60)
    try: app = MainApp(); app.mainloop()
    except Exception as e:
        messagebox.showerror("Application Error", f"A fatal error occurred: {e}")
        print(f"Application error: {e}")