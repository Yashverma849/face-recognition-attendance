import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import ttk
from PIL import Image, ImageTk
import pyttsx3
import threading
import customtkinter as ctk

# Set appearance mode and default theme
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Helper function for safely managing dialog focus and grabbing
def safe_grab_set(dialog):
    """Safely set grab on a dialog, with fallbacks if it fails."""
    try:
        # First try to lift the window
        try:
            dialog.lift()
        except Exception as e:
            print(f"Warning: Could not lift window: {e}")
            
        # Then try to force focus
        try:
            dialog.focus_force()
        except Exception as e:
            print(f"Warning: Could not force focus: {e}")
            
        # Finally try to grab
        try:
            dialog.grab_set()
            return True
        except Exception as e:
            print(f"Warning: Could not grab focus: {e}")
            return False
    except Exception as e:
        print(f"Critical error in safe_grab_set: {e}")
        return False

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Face Recognition Attendance System")
        self.geometry("1200x750")
        self.minsize(900, 600)
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        # Create necessary directories
        os.makedirs("src/images/train", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        # Load the face cascade
        self.face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Try to load recognizer if model exists
        try:
            self.face_recognizer.read("data/trainer.yml")
            print("Model loaded successfully")
        except Exception as e:
            print(f"No existing model found: {e}")
        
        # Initialize variables
        self.cap = None
        self.is_capturing = False
        self.capture_thread = None
        self.current_student_id = None
        self.current_student_name = None
        self.student_records = self.load_student_data()
        
        # Track current appearance mode
        self.appearance_mode = ctk.StringVar(value=ctk.get_appearance_mode())
        
        # Active dialogs tracking (to prevent multiple dialogs)
        self.active_dialogs = set()
        
        # Create UI
        self.create_ui()
        
    def load_student_data(self):
        # Load student data from CSV or create new one if it doesn't exist
        try:
            # Check if file exists and is not empty
            if os.path.exists('data/students.csv') and os.path.getsize('data/students.csv') > 0:
                df = pd.read_csv('data/students.csv')
                return df.to_dict('records')
            else:
                # Create a new file with headers if it doesn't exist or is empty
                columns = ['id', 'name', 'enrollment', 'semester', 'branch']
                df = pd.DataFrame(columns=columns)
                df.to_csv('data/students.csv', index=False)
                return []
        except FileNotFoundError:
            # Create a new file with headers
            columns = ['id', 'name', 'enrollment', 'semester', 'branch']
            df = pd.DataFrame(columns=columns)
            df.to_csv('data/students.csv', index=False)
            return []
        except Exception as e:
            print(f"Error loading student data: {e}")
            # Create a new file with headers in case of error
            columns = ['id', 'name', 'enrollment', 'semester', 'branch']
            df = pd.DataFrame(columns=columns)
            df.to_csv('data/students.csv', index=False)
            return []

    def save_student_data(self):
        # Save student data to CSV
        df = pd.DataFrame(self.student_records)
        df.to_csv('data/students.csv', index=False)
    
    def update_status(self, message):
        self.status_label.configure(text=f"Status: {message}")
        print(message)
    
    def speak_text(self, text):
        # Function to speak text using pyttsx3
        def speak():
            self.engine.say(text)
            self.engine.runAndWait()
        
        # Start in a separate thread to avoid blocking
        t = threading.Thread(target=speak)
        t.daemon = True
        t.start()
    
    def toggle_appearance_mode(self):
        # Toggle between light and dark mode
        new_mode = "Light" if ctk.get_appearance_mode() == "Dark" else "Dark"
        ctk.set_appearance_mode(new_mode)
        self.appearance_slider.configure(text="Dark Mode" if new_mode == "Dark" else "Light Mode")
    
    def create_ui(self):
        # Create a grid layout
        self.grid_columnconfigure(0, weight=1)  # Sidebar
        self.grid_columnconfigure(1, weight=5)  # Main content
        self.grid_rowconfigure(0, weight=1)
        
        # Create sidebar frame
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        # Configure sidebar grid
        self.sidebar.grid_rowconfigure(0, minsize=80)  # Logo
        self.sidebar.grid_rowconfigure(7, weight=1)  # Spacer (increased by 1)
        
        # App logo/title
        self.logo_label = ctk.CTkLabel(
            self.sidebar, text="Face Recognition\nAttendance System", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Navigation buttons
        self.register_button = ctk.CTkButton(
            self.sidebar, text="Register Student",
            font=ctk.CTkFont(size=14), height=40,
            command=self.register_student
        )
        self.register_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.train_button = ctk.CTkButton(
            self.sidebar, text="Train Model",
            font=ctk.CTkFont(size=14), height=40,
            command=self.train_model
        )
        self.train_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.attendance_button = ctk.CTkButton(
            self.sidebar, text="Take Attendance",
            font=ctk.CTkFont(size=14), height=40,
            command=self.start_attendance
        )
        self.attendance_button.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        self.view_button = ctk.CTkButton(
            self.sidebar, text="View Attendance",
            font=ctk.CTkFont(size=14), height=40,
            command=self.view_attendance
        )
        self.view_button.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        # Add reset data button
        self.reset_button = ctk.CTkButton(
            self.sidebar, text="Reset Data",
            font=ctk.CTkFont(size=14), height=40,
            fg_color="#FF9800", hover_color="#F57C00",
            command=self.reset_data
        )
        self.reset_button.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        
        self.exit_button = ctk.CTkButton(
            self.sidebar, text="Exit",
            font=ctk.CTkFont(size=14), height=40,
            fg_color="#E53935", hover_color="#C62828",
            command=self.exit_app
        )
        self.exit_button.grid(row=6, column=0, padx=20, pady=10, sticky="ew")
        
        # Appearance mode frame
        appearance_frame = ctk.CTkFrame(self.sidebar)
        appearance_frame.grid(row=8, column=0, padx=20, pady=20, sticky="ew")
        
        appearance_label = ctk.CTkLabel(appearance_frame, text="Appearance Mode:")
        appearance_label.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        
        # Use a switch to toggle dark/light mode
        self.appearance_slider = ctk.CTkSwitch(
            appearance_frame, text="Dark Mode" if ctk.get_appearance_mode() == "Dark" else "Light Mode",
            command=self.toggle_appearance_mode
        )
        self.appearance_slider.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="w")
        
        # Set initial switch state based on current appearance mode
        if ctk.get_appearance_mode() == "Dark":
            self.appearance_slider.select()
        else:
            self.appearance_slider.deselect()
        
        # Create main frame for content
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        # Configure main frame grid
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)  # Video/content area
        self.main_frame.grid_rowconfigure(1, minsize=100)  # Status bar
        
        # Video display frame
        self.video_frame = ctk.CTkFrame(self.main_frame, corner_radius=5)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Video label
        self.video_label = ctk.CTkLabel(self.video_frame, text="Camera feed will appear here")
        self.video_label.pack(expand=True, fill="both")
        
        # Status frame
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.status_frame, text="Status: Ready",
            font=ctk.CTkFont(size=14)
        )
        self.status_label.pack(pady=10)

    def register_student(self):
        # Check if we already have an active dialog
        if 'register_dialog' in self.active_dialogs:
            messagebox.showinfo("Already Running", "Registration dialog is already open.")
            return
        
        # Add to active dialogs
        self.active_dialogs.add('register_dialog')
        
        # Process directly instead of delegating to original app
        self.update_status("Starting student registration")
        
        # Stop any existing camera capture
        self.stop_capture()
        
        # Ask for student ID using customtkinter dialog
        try:
            student_id_dialog = ctk.CTkInputDialog(text="Enter Student ID:", title="Student ID")
            student_id = student_id_dialog.get_input()
            if not student_id:
                self.active_dialogs.remove('register_dialog')
                return
            
            # Ask for student name
            student_name_dialog = ctk.CTkInputDialog(text="Enter Student Name:", title="Student Name")
            student_name = student_name_dialog.get_input()
            if not student_name:
                self.active_dialogs.remove('register_dialog')
                return
                
            # Ask for enrollment number
            enrollment_dialog = ctk.CTkInputDialog(text="Enter Enrollment/Roll Number:", title="Enrollment")
            student_enrollment = enrollment_dialog.get_input()
            if not student_enrollment:
                self.active_dialogs.remove('register_dialog')
                return
        except Exception as e:
            print(f"Dialog error: {e}")
            messagebox.showerror("Error", "There was an issue with the dialog. Please try again.")
            self.active_dialogs.remove('register_dialog')
            return
            
        # Create a dialog for semester and branch selection
        try:
            semester_branch_dialog = ctk.CTkToplevel(self)
            semester_branch_dialog.title("Select Semester & Branch")
            semester_branch_dialog.geometry("350x250")
            semester_branch_dialog.transient(self)
            
            # Use our safe grab function instead of direct calls
            safe_grab_set(semester_branch_dialog)
            
            # Semester selection
            semester_frame = ctk.CTkFrame(semester_branch_dialog)
            semester_frame.pack(fill=tk.X, padx=20, pady=20)
            
            ctk.CTkLabel(semester_frame, text="Semester:").pack(side=tk.LEFT, padx=10)
            
            semester_var = ctk.StringVar()
            semesters = ["1", "2", "3", "4", "5", "6", "7", "8"]
            semester_menu = ctk.CTkOptionMenu(semester_frame, values=semesters, variable=semester_var)
            semester_menu.pack(side=tk.LEFT, padx=10)
            semester_var.set("1")  # Default value
            
            # Branch selection
            branch_frame = ctk.CTkFrame(semester_branch_dialog)
            branch_frame.pack(fill=tk.X, padx=20, pady=20)
            
            ctk.CTkLabel(branch_frame, text="Branch:").pack(side=tk.LEFT, padx=10)
            
            # Add branch dropdown
            branch_var = ctk.StringVar()
            all_branches = ["CSE", "IT", "ECE", "EEE"]
            branch_menu = ctk.CTkOptionMenu(branch_frame, values=all_branches, variable=branch_var)
            branch_menu.pack(side=tk.LEFT, padx=10)
            branch_var.set("CSE")  # Default value
            
            # Result variables
            details_result = {"semester": None, "branch": None}
            
            def on_ok():
                details_result["semester"] = semester_var.get()
                details_result["branch"] = branch_var.get()
                semester_branch_dialog.destroy()
            
            # OK button
            ok_btn = ctk.CTkButton(
                semester_branch_dialog, text="Continue",
                command=on_ok,
                fg_color="#2196F3", hover_color="#1976D2"
            )
            ok_btn.pack(pady=20)
            
            # Ensure dialog close removes active state
            def on_dialog_close():
                self.active_dialogs.remove('register_dialog')
                semester_branch_dialog.destroy()
                
            semester_branch_dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
            
            # Wait for dialog to close
            self.wait_window(semester_branch_dialog)
            
            # Check if user canceled
            if details_result["semester"] is None:
                return
                
            semester = details_result["semester"]
            branch = details_result["branch"]
        except Exception as e:
            print(f"Dialog creation error: {e}")
            messagebox.showerror("Error", "There was an issue with the dialog. Please try again.")
            if 'register_dialog' in self.active_dialogs:
                self.active_dialogs.remove('register_dialog')
            return
        
        # Check if ID already exists
        for record in self.student_records:
            if record['id'] == student_id:
                messagebox.showerror("Error", "Student ID already exists!")
                if 'register_dialog' in self.active_dialogs:
                    self.active_dialogs.remove('register_dialog')
                return
        
        # Save student info
        self.current_student_id = student_id
        self.current_student_name = student_name
        
        new_student = {
            'id': student_id,
            'name': student_name,
            'enrollment': student_enrollment,
            'semester': semester,
            'branch': branch
        }
        
        self.student_records.append(new_student)
        self.save_student_data()
        
        # Start face capture
        self.update_status(f"Registering {student_name} (Semester: {semester}, Branch: {branch}). Please look at the camera.")
        self.speak_text(f"Registering {student_name}. Please look at the camera.")
        
        # Create directory for this student
        os.makedirs(f"src/images/train/{student_id}", exist_ok=True)
        
        # Clear from active dialogs
        if 'register_dialog' in self.active_dialogs:
            self.active_dialogs.remove('register_dialog')
        
        # Start capture thread
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self.capture_faces)
        self.capture_thread.daemon = True
        self.capture_thread.start()
    
    def capture_faces(self):
        # Open camera
        self.cap = cv2.VideoCapture(0)
        
        count = 0
        max_images = 50  # Number of images to capture per person
        
        while self.is_capturing and count < max_images:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save the face image
                if count < max_images:
                    face_img = gray[y:y+h, x:x+w]
                    img_path = f"src/images/train/{self.current_student_id}/face_{count}.jpg"
                    cv2.imwrite(img_path, face_img)
                    count += 1
                    
                    self.update_status(f"Capturing image {count}/{max_images}")
                    time.sleep(0.1)  # Small delay between captures
            
            # Display the frame
            self.display_frame(frame)
            
            # Break if enough images captured
            if count >= max_images:
                self.update_status(f"Registration complete for {self.current_student_name}")
                self.speak_text(f"Registration complete for {self.current_student_name}")
                break
                
            # Process GUI events
            self.update()
        
        # Close camera
        self.stop_capture()
        
        # Ask to train the model
        if count > 0:
            if messagebox.askyesno("Training", "Registration complete! Do you want to train the model now?"):
                self.train_model()
    
    def train_model(self):
        # Process directly
        self.update_status("Training model... Please wait")
        
        # Get all face samples
        face_samples = []
        face_ids = []
        
        # Iterate through all student directories
        student_dirs = os.listdir("src/images/train")
        
        if not student_dirs:
            messagebox.showwarning("Warning", "No student data found for training")
            self.update_status("No student data found for training")
            return
            
        for student_id in student_dirs:
            student_path = os.path.join("src/images/train", student_id)
            
            if os.path.isdir(student_path):
                # Get all face images for this student
                face_images = [os.path.join(student_path, f) for f in os.listdir(student_path)]
                
                if not face_images:
                    continue
                    
                # Add each face image to training data
                for face_img_path in face_images:
                    img = cv2.imread(face_img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        face_samples.append(img)
                        face_ids.append(int(student_id))
        
        if not face_samples:
            messagebox.showwarning("Warning", "No face images found for training")
            self.update_status("No face images found for training")
            return
            
        # Train the model
        self.update_status(f"Training with {len(face_samples)} images")
        
        try:
            self.face_recognizer.train(face_samples, np.array(face_ids))
            
            # Save the model
            self.face_recognizer.save("data/trainer.yml")
            
            self.update_status("Model trained successfully")
            self.speak_text("Model trained successfully")
            messagebox.showinfo("Success", "Model trained successfully!")
        except Exception as e:
            self.update_status(f"Error training model: {e}")
            messagebox.showerror("Error", f"Error training model: {e}")
        
    def start_attendance(self):
        # Check if we already have an active dialog
        if 'attendance_dialog' in self.active_dialogs:
            messagebox.showinfo("Already Running", "Attendance dialog is already open.")
            return
            
        # Add to active dialogs
        self.active_dialogs.add('attendance_dialog')
        
        # Process directly instead of delegating to original app
        self.update_status("Starting attendance process")
        
        # Stop any existing camera capture
        self.stop_capture()
        
        # Create a dialog for selecting semester, branch and subject
        try:
            attendance_dialog = ctk.CTkToplevel(self)
            attendance_dialog.title("Attendance Details")
            attendance_dialog.geometry("350x300")
            attendance_dialog.transient(self)
            
            # Use our safe grab function instead of direct calls
            safe_grab_set(attendance_dialog)
            
            # Semester selection
            semester_frame = ctk.CTkFrame(attendance_dialog)
            semester_frame.pack(fill=tk.X, padx=20, pady=10)
            
            ctk.CTkLabel(semester_frame, text="Semester:").pack(side=tk.LEFT, padx=10)
            
            semester_var = ctk.StringVar()
            semesters = ["1", "2", "3", "4", "5", "6", "7", "8"]
            semester_menu = ctk.CTkOptionMenu(semester_frame, values=semesters, variable=semester_var)
            semester_menu.pack(side=tk.LEFT, padx=10)
            semester_var.set("1")  # Default value
            
            # Branch selection
            branch_frame = ctk.CTkFrame(attendance_dialog)
            branch_frame.pack(fill=tk.X, padx=20, pady=10)
            
            ctk.CTkLabel(branch_frame, text="Branch:").pack(side=tk.LEFT, padx=10)
            
            # Only include the four specified branches
            all_branches = ["CSE", "IT", "ECE", "EEE"]
            
            # Set default to CSE
            branch_var = ctk.StringVar()
            branch_var.set("CSE")
            
            branch_menu = ctk.CTkOptionMenu(branch_frame, values=all_branches, variable=branch_var, width=150)
            branch_menu.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            # Subject input
            subject_frame = ctk.CTkFrame(attendance_dialog)
            subject_frame.pack(fill=tk.X, padx=20, pady=10)
            
            ctk.CTkLabel(subject_frame, text="Subject:").pack(side=tk.LEFT, padx=10)
            
            # Get common subjects for default value
            common_subjects = []
            attendance_files = [f for f in os.listdir("data") if f.startswith("attendance_") and f.endswith(".csv")]
            for f in attendance_files:
                parts = f.split('_')
                if len(parts) > 4:
                    subject = parts[4].replace('.csv', '')
                    if subject not in common_subjects:
                        common_subjects.append(subject)
            
            # Sort subjects alphabetically
            common_subjects = sorted(common_subjects)
            
            # Only manual entry for subject
            subject_entry = ctk.CTkEntry(subject_frame, width=200)
            subject_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            # Set default subject if one exists
            if common_subjects:
                subject_entry.insert(0, common_subjects[0])
            
            # Result variables
            details_result = {"semester": None, "branch": None, "subject": None}
            
            def on_ok():
                # Validate subject is not empty
                subject = subject_entry.get().strip()
                if not subject:
                    messagebox.showwarning("Warning", "Please enter a subject.", parent=attendance_dialog)
                    return
                    
                details_result["semester"] = semester_var.get()
                details_result["branch"] = branch_var.get()
                details_result["subject"] = subject
                attendance_dialog.destroy()
            
            # OK button
            ok_btn = ctk.CTkButton(
                attendance_dialog, text="Start Attendance",
                command=on_ok,
                fg_color="#4CAF50", hover_color="#388E3C"
            )
            ok_btn.pack(pady=20)
            
            # Ensure dialog close removes active state
            def on_dialog_close():
                self.active_dialogs.remove('attendance_dialog')
                attendance_dialog.destroy()
                
            attendance_dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
            
            # Wait for dialog to close
            self.wait_window(attendance_dialog)
            
            # Check if user canceled
            if details_result["semester"] is None:
                return
                
            semester = details_result["semester"]
            branch = details_result["branch"]
            subject = details_result["subject"]
        except Exception as e:
            print(f"Dialog creation error: {e}")
            messagebox.showerror("Error", "There was an issue with the dialog. Please try again.")
            if 'attendance_dialog' in self.active_dialogs:
                self.active_dialogs.remove('attendance_dialog')
            return
        
        # Set up for attendance
        self.update_status(f"Taking attendance for {branch} Semester {semester} - {subject}")
        self.speak_text(f"Taking attendance for {subject}")
        
        # Create today's date string (YYYY-MM-DD)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Create CSV file for attendance if it doesn't exist
        csv_path = f"data/attendance_{today}_{semester}_{branch}_{subject}.csv"
        
        # Check if file exists, create if not
        if not os.path.exists(csv_path):
            # Create new attendance file
            columns = ['id', 'name', 'enrollment', 'semester', 'branch', 'subject', 'time']
            df = pd.DataFrame(columns=columns)
            df.to_csv(csv_path, index=False)
            print(f"Created new attendance file: {csv_path}")
        
        # Load existing attendance
        try:
            attendance_df = pd.read_csv(csv_path)
        except Exception:
            # Create a fresh dataframe if loading failed
            attendance_df = pd.DataFrame(columns=['id', 'name', 'enrollment', 'semester', 'branch', 'subject', 'time'])
        
        # Start attendance thread
        self.is_capturing = True
        self.capture_thread = threading.Thread(
            target=self.recognize_faces,
            args=(attendance_df, csv_path, semester, branch, subject)
        )
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Clear from active dialogs (will be handled by the recognition thread)
        if 'attendance_dialog' in self.active_dialogs:
            self.active_dialogs.remove('attendance_dialog')
    
    def recognize_faces(self, attendance, attendance_file, semester, branch, subject):
        self.cap = cv2.VideoCapture(0)
        
        # Dictionary to avoid multiple detections in short time
        detected_students = {}
        
        while self.is_capturing:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Try to recognize the face
                try:
                    face_img = gray[y:y+h, x:x+w]
                    id_predicted, confidence = self.face_recognizer.predict(face_img)
                    
                    # Lower confidence is better in LBPH
                    if confidence < 70:  # Threshold for recognition
                        # Find student details
                        student = None
                        for record in self.student_records:
                            if int(record['id']) == id_predicted:
                                student = record
                                break
                                
                        if student:
                            student_id = student['id']
                            student_name = student['name']
                            enrollment = student['enrollment']
                            
                            # Check if this student was recently detected
                            current_time = time.time()
                            if student_id in detected_students:
                                # Only register attendance if more than 5 seconds passed
                                if current_time - detected_students[student_id] < 5:
                                    # Add text to frame but don't record again
                                    cv2.putText(frame, f"{student_name} - {enrollment}", (x, y-10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                    continue
                            
                            # Update detection time
                            detected_students[student_id] = current_time
                            
                            # Add text to frame
                            cv2.putText(frame, f"{student_name} - {enrollment}", (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            # Check if already marked attendance today
                            student_attendance = attendance[attendance['id'] == student_id]
                            
                            if student_attendance.empty:
                                # Mark attendance
                                time_now = datetime.datetime.now().strftime("%H:%M:%S")
                                new_attendance = pd.DataFrame([[student_id, student_name, enrollment, semester, branch, subject, time_now]], 
                                                            columns=['id', 'name', 'enrollment', 'semester', 'branch', 'subject', 'time'])
                                attendance = pd.concat([attendance, new_attendance], ignore_index=True)
                                attendance.to_csv(attendance_file, index=False)
                                
                                self.update_status(f"Attendance marked for {student_name}")
                                self.speak_text(f"Attendance marked for {student_name}")
                    else:
                        # Unknown face
                        cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Error in recognition: {e}")
            
            # Display the frame
            self.display_frame(frame)
            
            # Process GUI events to keep UI responsive
            self.update_idletasks()
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
        
        # Release camera but don't call stop_capture (which would try to join the current thread)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Reset the video label
        self.video_label.configure(image=None, text="Camera feed will appear here")
        self.update_status("Attendance session ended")
    
    def view_attendance(self):
        # Check if we already have an active dialog
        if 'view_attendance_dialog' in self.active_dialogs:
            messagebox.showinfo("Already Running", "View attendance window is already open.")
            return
            
        # Add to active dialogs
        self.active_dialogs.add('view_attendance_dialog')
        
        # Process directly instead of delegating to original app
        
        # Open a new window to view attendance
        try:
            attendance_window = ctk.CTkToplevel(self)
            attendance_window.title("Attendance Records")
            attendance_window.geometry("1000x600")
            attendance_window.transient(self)
            
            # Use safe grab set instead of separate try/except blocks
            safe_grab_set(attendance_window)
            
            # Ensure dialog close removes active state
            def on_window_close():
                # Check if dialog is still in active_dialogs before trying to remove
                if 'view_attendance_dialog' in self.active_dialogs:
                    self.active_dialogs.remove('view_attendance_dialog')
                attendance_window.destroy()
                
            attendance_window.protocol("WM_DELETE_WINDOW", on_window_close)
            
            # Get list of attendance files
            attendance_files = [f for f in os.listdir("data") if f.startswith("attendance_") and f.endswith(".csv")]
            
            if not attendance_files:
                ctk.CTkLabel(attendance_window, text="No attendance records found", 
                          font=ctk.CTkFont(size=14)).pack(pady=20)
                return
                
            # Sort files by date (newest first)
            attendance_files.sort(reverse=True)
            
            # Title label
            title_label = ctk.CTkLabel(
                attendance_window, 
                text="Attendance Records", 
                font=ctk.CTkFont(size=20, weight="bold")
            )
            title_label.pack(pady=(20, 5))
            
            # Subtitle
            subtitle_label = ctk.CTkLabel(
                attendance_window, 
                text="Select filters to view specific attendance records", 
                font=ctk.CTkFont(size=14)
            )
            subtitle_label.pack(pady=(0, 10))
            
            # Frame for filters with rounded corners and padding
            filter_frame = ctk.CTkFrame(attendance_window, corner_radius=10)
            filter_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Filter title
            filter_title = ctk.CTkLabel(
                filter_frame, 
                text="Filter Options", 
                font=ctk.CTkFont(size=16, weight="bold")
            )
            filter_title.pack(pady=(10, 5))
            
            # Two-column layout for filters
            filter_grid = ctk.CTkFrame(filter_frame, fg_color="transparent")
            filter_grid.pack(fill=tk.X, padx=20, pady=10)
            
            # Left column
            left_column = ctk.CTkFrame(filter_grid, fg_color="transparent")
            left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
            
            # Right column
            right_column = ctk.CTkFrame(filter_grid, fg_color="transparent")
            right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
            
            # Date selection (left column)
            date_frame = ctk.CTkFrame(left_column)
            date_frame.pack(fill=tk.X, pady=5)
            
            ctk.CTkLabel(date_frame, text="Date:", font=ctk.CTkFont(weight="bold")).pack(side=tk.LEFT, padx=10)
            
            # Extract unique dates from filenames - fix to ensure all dates are captured
            dates = []
            for f in attendance_files:
                parts = f.split('_')
                if len(parts) > 1:
                    # Get the date part
                    date = parts[1]
                    if date not in dates:
                        dates.append(date)
            
            # Sort dates in reverse (newest first) and set default
            dates = sorted(dates, reverse=True)
            date_var = ctk.StringVar(value=dates[0] if dates else "")
            
            date_menu = ctk.CTkOptionMenu(date_frame, values=dates, variable=date_var, width=150)
            date_menu.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            # Semester selection (left column)
            semester_frame = ctk.CTkFrame(left_column)
            semester_frame.pack(fill=tk.X, pady=5)
            
            ctk.CTkLabel(semester_frame, text="Semester:", font=ctk.CTkFont(weight="bold")).pack(side=tk.LEFT, padx=10)
            
            # Always include all possible semesters
            all_semesters = ["1", "2", "3", "4", "5", "6", "7", "8"]
            
            # Add any additional semesters found in attendance files
            for f in attendance_files:
                parts = f.split('_')
                if len(parts) > 2:
                    semester = parts[2]
                    if semester not in all_semesters:
                        all_semesters.append(semester)
            
            # Sort and set default
            all_semesters = sorted(all_semesters)
            semester_var = ctk.StringVar(value=all_semesters[0] if all_semesters else "")
            
            semester_menu = ctk.CTkOptionMenu(semester_frame, values=all_semesters, variable=semester_var, width=150)
            semester_menu.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            # Branch selection (right column)
            branch_frame = ctk.CTkFrame(right_column)
            branch_frame.pack(fill=tk.X, pady=5)
            
            ctk.CTkLabel(branch_frame, text="Branch:", font=ctk.CTkFont(weight="bold")).pack(side=tk.LEFT, padx=10)
            
            # Only include the four specified branches
            all_branches = ["CSE", "IT", "ECE", "EEE"]
            
            # Set default to CSE
            branch_var = ctk.StringVar()
            branch_var.set("CSE")
            
            branch_menu = ctk.CTkOptionMenu(branch_frame, values=all_branches, variable=branch_var, width=150)
            branch_menu.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            # Subject input
            subject_frame = ctk.CTkFrame(right_column)
            subject_frame.pack(fill=tk.X, pady=5)
            
            ctk.CTkLabel(subject_frame, text="Subject:").pack(side=tk.LEFT, padx=10)
            
            # Get common subjects for default value
            common_subjects = []
            for f in attendance_files:
                parts = f.split('_')
                if len(parts) > 4:
                    # Get the subject part and remove .csv extension
                    subject = parts[4].replace('.csv', '')
                    if subject not in common_subjects:
                        common_subjects.append(subject)
            
            # Sort subjects alphabetically
            common_subjects = sorted(common_subjects)
            
            # Only manual entry for subject
            subject_entry = ctk.CTkEntry(subject_frame, width=200)
            subject_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            # Set default subject if one exists
            if common_subjects:
                subject_entry.insert(0, common_subjects[0])
            
            # Function to load attendance data
            def load_attendance(event=None):
                # Clear existing data
                for item in tree.get_children():
                    tree.delete(item)
                    
                # Get selected filters
                selected_date = date_var.get()
                selected_semester = semester_var.get()
                selected_branch = branch_var.get()
                selected_subject = subject_entry.get().strip()  # Get and trim the subject text
                
                # Ensure we have a subject entered
                if not selected_subject:
                    messagebox.showinfo("Missing Subject", "Please enter a subject name", parent=attendance_window)
                    return
                    
                # Update data label
                data_label.configure(text=f"Attendance Data - {selected_date}, {selected_branch} Sem {selected_semester}, {selected_subject}")
                
                file_path = f"data/attendance_{selected_date}_{selected_semester}_{selected_branch}_{selected_subject}.csv"
                
                try:
                    # Check if file exists first
                    if not os.path.exists(file_path):
                        # Show a simple "No data available" message
                        tree.insert("", tk.END, values=["No data available", "", "", "", "", "", ""], tags=('odd_row',))
                        status_label.configure(text="No attendance records found for the selected filters.")
                        return
                        
                    # Load attendance data
                    df = pd.read_csv(file_path)
                    
                    if len(df) == 0:
                        # Handle empty data file
                        tree.insert("", tk.END, values=["No data available", "", "", "", "", "", ""], tags=('odd_row',))
                        status_label.configure(text="No attendance records found for the selected filters.")
                        return
                    
                    # Add data to treeview with alternating row colors
                    for i, row in df.iterrows():
                        values = [row['id'], row['name'], row['enrollment']]
                        
                        # Add optional fields if they exist
                        if 'semester' in df.columns:
                            values.append(row['semester'])
                        else:
                            values.append("")
                            
                        if 'branch' in df.columns:
                            values.append(row['branch'])
                        else:
                            values.append("")
                            
                        if 'subject' in df.columns:
                            values.append(row['subject'])
                        else:
                            values.append("")
                        
                        values.append(row['time'])
                        
                        # Add row with alternating background
                        row_tag = 'even_row' if i % 2 == 0 else 'odd_row'
                        tree.insert("", tk.END, values=values, tags=(row_tag,))
                    
                    # Show the count of records
                    status_text = f"Loaded {len(df)} records for {selected_date}, {selected_branch} Semester {selected_semester} - {selected_subject}"
                    status_label.configure(text=status_text)
                    self.update_status(status_text)
                except Exception as e:
                    print(f"Error loading attendance: {e}")
                    tree.insert("", tk.END, values=["Error loading data", "", "", "", "", "", ""], tags=('odd_row',))
                    status_label.configure(text=f"Error loading data: {e}")
                    # Don't show a messagebox to avoid disturbing the user experience
            
            # Refresh button - define after the load_attendance function
            refresh_btn = ctk.CTkButton(
                filter_frame, 
                text="Refresh Data",
                command=load_attendance,  # Use the function directly now that it's defined
                fg_color="#2196F3", hover_color="#1976D2",
                width=150,
                height=32
            )
            refresh_btn.pack(pady=(5, 15))
            
            # Create a frame for the treeview with border and rounded corners
            tree_frame = ctk.CTkFrame(attendance_window, corner_radius=10)
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            # Data view label
            data_label = ctk.CTkLabel(
                tree_frame, 
                text="Attendance Data", 
                font=ctk.CTkFont(size=16, weight="bold")
            )
            data_label.pack(pady=(10, 5))
            
            # Create a sub-frame for the actual treeview to control its appearance
            treeview_container = ctk.CTkFrame(tree_frame, fg_color="transparent")
            treeview_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            
            # Create outer frame for border effect in dark mode
            if ctk.get_appearance_mode() == "Dark":
                outer_frame = ctk.CTkFrame(treeview_container, fg_color="#444444", corner_radius=0)
                outer_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
                parent_frame = outer_frame
            else:
                parent_frame = treeview_container
            
            # Create treeview with dark headers
            columns = ("ID", "Name", "Enrollment", "Semester", "Branch", "Subject", "Time")
            tree = ttk.Treeview(parent_frame, columns=columns, show="headings", style="Custom.Treeview")
            
            # Style for the treeview - try to match the appearance mode
            style = ttk.Style()
            if ctk.get_appearance_mode() == "Dark":
                # Configure a custom dark theme for the treeview
                theme_name = "dark_theme"
                
                # Check if theme already exists to avoid errors
                existing_themes = style.theme_names()
                if theme_name not in existing_themes:
                    style.theme_create(theme_name, parent="alt", settings={
                        "Treeview": {
                            "configure": {
                                "background": "#1a1a1a",
                                "foreground": "white",
                                "fieldbackground": "#1a1a1a",
                                "borderwidth": 1,
                                "relief": "solid",
                                "rowheight": 30
                            },
                            "map": {
                                "background": [("selected", "#344e70")],
                                "foreground": [("selected", "white")]
                            }
                        },
                        "Treeview.Heading": {
                            "configure": {
                                "background": "black",
                                "foreground": "white",
                                "relief": "raised",
                                "borderwidth": 2,
                                "font": ('Arial', 10, 'bold')
                            },
                            "map": {
                                "background": [("active", "black"), ("pressed", "black")],
                                "foreground": [("active", "white")]
                            }
                        },
                        "Vertical.TScrollbar": {
                            "configure": {
                                "background": "#0a0a0a",
                                "troughcolor": "#1a1a1a",
                                "arrowcolor": "white",
                                "bordercolor": "#444444",
                                "relief": "flat"
                            }
                        }
                    })
                
                # Use the custom theme
                try:
                    style.theme_use(theme_name)
                except Exception as e:
                    print(f"Error using theme: {e}")
                    
                # Configure custom Treeview style
                style.configure("Custom.Treeview", 
                              background="#1a1a1a", 
                              foreground="white", 
                              fieldbackground="#1a1a1a",
                              rowheight=30)
                
                style.configure("Custom.Treeview.Heading",
                              background="black",
                              foreground="white", 
                              relief="raised",
                              borderwidth=2)
                
                # Add cell borders by configuring tags with contrasting colors
                tree.tag_configure('even_row', background='#262626')
                tree.tag_configure('odd_row', background='#1a1a1a')
            else:
                # Configure for light mode
                style.configure("Custom.Treeview", 
                              background="white", 
                              foreground="black", 
                              fieldbackground="white",
                              borderwidth=1,
                              relief="solid",
                              rowheight=30)
                style.configure("Custom.Treeview.Heading", 
                              background="#e1e1e1", 
                              foreground="black",
                              borderwidth=1,
                              relief="raised",
                              font=('Arial', 10, 'bold'))
                style.map('Custom.Treeview', 
                      background=[('selected', '#4a6ea9')],
                      foreground=[('selected', 'white')])
                
                # Improve cell borders
                tree.tag_configure('even_row', background='#f0f0f0')
                tree.tag_configure('odd_row', background='white')
            
            # Set column headings with optimized widths
            column_widths = {
                "ID": 80,
                "Name": 160,
                "Enrollment": 120,
                "Semester": 100,
                "Branch": 100,
                "Subject": 140,
                "Time": 100
            }
            
            for col in columns:
                tree.heading(col, text=col)
                width = column_widths.get(col, 120)
                tree.column(col, width=width, anchor=tk.CENTER if col != "Name" else tk.W, 
                        stretch=True, minwidth=50)
            
            # Add scrollbar with styling to match theme
            scrollbar = ttk.Scrollbar(parent_frame, orient=tk.VERTICAL, command=tree.yview, style="Custom.Vertical.TScrollbar")
            tree.configure(yscroll=scrollbar.set)
            
            # Customize scrollbar appearance
            if ctk.get_appearance_mode() == "Dark":
                # Dark mode scrollbar styling comes from the theme
                style.configure("Custom.Vertical.TScrollbar",
                           background="black",
                           troughcolor="#1a1a1a",
                           arrowcolor="white",
                           bordercolor="#444444",
                           relief="flat")
            else:
                # Light mode scrollbar styling
                style.configure("Custom.Vertical.TScrollbar",
                           background="#e1e1e1",
                           troughcolor="white",
                           arrowcolor="black",
                           bordercolor="#cccccc",
                           relief="flat")
            
            # Pack treeview and scrollbar
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Bind selection change events
            date_menu.configure(command=lambda choice: load_attendance())
            semester_menu.configure(command=lambda choice: load_attendance())
            branch_menu.configure(command=lambda choice: load_attendance())
            
            # Bind Return/Enter key to the subject entry for searching
            subject_entry.bind("<Return>", load_attendance)
            
            # Add a search button next to the subject entry for better usability
            search_btn = ctk.CTkButton(
                filter_frame, 
                text="Search",
                command=load_attendance,
                fg_color="#FF9800", hover_color="#F57C00",
                width=100,
                height=32
            )
            search_btn.pack(pady=(5, 15))
            
            # Status and export frame 
            bottom_frame = ctk.CTkFrame(attendance_window)
            bottom_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
            
            # Status label
            status_label = ctk.CTkLabel(
                bottom_frame, 
                text="Ready to load attendance data", 
                font=ctk.CTkFont(size=12)
            )
            status_label.pack(side=tk.LEFT, padx=20, pady=10)
            
            # Export button
            def export_to_excel():
                # Get selected filters
                selected_date = date_var.get()
                selected_semester = semester_var.get()
                selected_branch = branch_var.get()
                selected_subject = subject_entry.get().strip()
                
                file_path = f"data/attendance_{selected_date}_{selected_semester}_{selected_branch}_{selected_subject}.csv"
                
                try:
                    df = pd.read_csv(file_path)
                    excel_path = f"data/attendance_{selected_date}_{selected_semester}_{selected_branch}_{selected_subject}.xlsx"
                    df.to_excel(excel_path, index=False)
                    messagebox.showinfo("Export Successful", f"Attendance exported to {excel_path}", parent=attendance_window)
                except Exception as e:
                    messagebox.showerror("Export Failed", f"Error: {e}", parent=attendance_window)
            
            export_btn = ctk.CTkButton(
                bottom_frame, 
                text="Export to Excel",
                command=export_to_excel,
                fg_color="#4CAF50", hover_color="#388E3C",
                width=150,
                height=32
            )
            export_btn.pack(side=tk.RIGHT, padx=20, pady=10)
            
            # Load initial data if possible
            try:
                load_attendance()
            except Exception as e:
                print(f"Could not load initial data: {e}")
        except Exception as e:
            print(f"Error creating attendance window: {e}")
            messagebox.showerror("Error", "There was an issue opening the attendance records. Please try again.")
            if 'view_attendance_dialog' in self.active_dialogs:
                self.active_dialogs.remove('view_attendance_dialog')
            return
    
    def display_frame(self, frame):
        if frame is not None:
            try:
                # Convert the frame from BGR to RGB for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Adapt size to the frame
                frame_h, frame_w = frame.shape[:2]
                
                # Calculate scaling factor to fit within the video frame
                video_w = self.video_frame.winfo_width() or 600  # Default if not yet rendered
                video_h = self.video_frame.winfo_height() or 500
                
                # Calculate scaling factor to fit the frame
                scale_w = video_w / frame_w
                scale_h = video_h / frame_h
                scale = min(scale_w, scale_h)
                
                # Calculate new dimensions
                new_w = int(frame_w * scale)
                new_h = int(frame_h * scale)
                
                # Resize the frame
                frame = cv2.resize(frame, (new_w, new_h))
                
                # Convert to PhotoImage
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update the label
                self.video_label.configure(image=img_tk, text="")
                self.video_label.image = img_tk  # Keep a reference to prevent garbage collection
                
                # Update the UI to show the frame immediately
                self.video_label.update()
            except Exception as e:
                print(f"Error displaying frame: {e}")
                # If there's an error, reset the label
                self.video_label.configure(image=None, text="Error displaying camera feed")
        else:
            # Reset the video label if no frame
            self.video_label.configure(image=None, text="Camera feed will appear here")
    
    def stop_capture(self):
        self.is_capturing = False
        if self.capture_thread is not None:
            if self.capture_thread.is_alive():
                # Only join if not the current thread
                if threading.current_thread() != self.capture_thread:
                    self.capture_thread.join(1.0)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Reset the video label
        self.video_label.configure(image=None, text="Camera feed will appear here")

    def exit_app(self):
        # Stop any capture threads
        self.stop_capture()
        
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.destroy()

    def reset_data(self):
        """Delete all stored data, including attendance records, training images, and model file."""
        # Confirm with the user before deleting data
        if not messagebox.askyesno("Confirm Reset", 
                                "This will delete ALL data, including:\n\n"
                                "• All student records\n"
                                "• All attendance records\n"
                                "• All face images\n"
                                "• The trained model\n\n"
                                "This action cannot be undone. Continue?"):
            return
            
        self.update_status("Resetting data...")
        
        # Stop any running capture
        self.stop_capture()
        
        try:
            import shutil
            import glob
            
            # 1. Delete all student records (reset student CSV)
            self.student_records = []
            self.save_student_data()
            
            # 2. Delete all attendance records (CSV files)
            attendance_files = glob.glob("data/attendance_*.csv")
            for file in attendance_files:
                try:
                    os.remove(file)
                    print(f"Deleted: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
            
            # Also delete any Excel exports
            excel_files = glob.glob("data/attendance_*.xlsx")
            for file in excel_files:
                try:
                    os.remove(file)
                    print(f"Deleted: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
            
            # 3. Delete all training images
            train_dir = "src/images/train"
            if os.path.exists(train_dir):
                for student_dir in os.listdir(train_dir):
                    student_path = os.path.join(train_dir, student_dir)
                    if os.path.isdir(student_path):
                        try:
                            shutil.rmtree(student_path)
                            print(f"Deleted student folder: {student_path}")
                        except Exception as e:
                            print(f"Error deleting {student_path}: {e}")
            
            # 4. Delete the model file
            model_file = "data/trainer.yml"
            if os.path.exists(model_file):
                try:
                    os.remove(model_file)
                    print(f"Deleted model: {model_file}")
                except Exception as e:
                    print(f"Error deleting model {model_file}: {e}")
            
            # Reset any current values
            self.current_student_id = None
            self.current_student_name = None
            
            # Show success message
            self.update_status("Application reset successfully")
            messagebox.showinfo("Reset Complete", "All data has been deleted. The application is now fresh.")
            
        except Exception as e:
            self.update_status(f"Error during reset: {e}")
            messagebox.showerror("Reset Error", f"An error occurred during reset: {e}")

# If this module is run directly, create and start the app
if __name__ == "__main__":
    try:
        # Enable exception catching for better error handling
        tk.Tk.report_callback_exception = lambda self, exc, val, tb: print(f"Exception: {val}")
        
        app = FaceRecognitionApp()
        
        # Set up a custom error handler for Tkinter
        def handle_tk_error(exc, val, tb):
            import traceback
            print(f"Tkinter Error: {val}")
            traceback.print_exception(exc, val, tb)
            
            # If it's a grab error, recover by releasing all grabs
            if "grab" in str(val).lower():
                try:
                    app.grab_release()
                    print("Released grab on main window")
                except Exception as e:
                    print(f"Could not release grab: {e}")
            
            # Don't terminate the application on non-fatal errors
            
        # Set the custom error handler
        app.report_callback_exception = handle_tk_error
        
        # Start the main event loop
        app.mainloop()
    except Exception as e:
        print(f"Fatal error in main application: {e}")
        import traceback
        traceback.print_exc() 