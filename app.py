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
from enum import Enum
import base64
import io

class FaceRecognitionAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")

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
        
        # Create GUI
        self.create_gui()

    def load_student_data(self):
        # Load student data from CSV or create new one if it doesn't exist
        try:
            df = pd.read_csv('data/students.csv')
            return df.to_dict('records')
        except FileNotFoundError:
            columns = ['id', 'name', 'enrollment', 'semester', 'branch']
            df = pd.DataFrame(columns=columns)
            df.to_csv('data/students.csv', index=False)
            return []

    def save_student_data(self):
        # Save student data to CSV
        df = pd.DataFrame(self.student_records)
        df.to_csv('data/students.csv', index=False)

    def create_gui(self):
        # Top title
        title_frame = tk.Frame(self.root, bg="#009688", bd=5, relief=tk.GROOVE)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(title_frame, text="Face Recognition Attendance Management System", 
                             font=("times new roman", 30, "bold"), bg="#009688", fg="white")
        title_label.pack(pady=10)

        # Main frame
        main_frame = tk.Frame(self.root, bd=5, relief=tk.GROOVE, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left frame - Video feed
        self.video_frame = tk.Frame(main_frame, bd=2, relief=tk.RIDGE, bg="#f0f0f0")
        self.video_frame.place(x=20, y=20, width=600, height=500)
        
        # Video display label
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right frame - Controls
        control_frame = tk.Frame(main_frame, bd=2, relief=tk.RIDGE, bg="#f0f0f0")
        control_frame.place(x=640, y=20, width=520, height=500)
        
        # Controls title
        control_label = tk.Label(control_frame, text="Control Panel", 
                                font=("times new roman", 20, "bold"), bg="#f0f0f0")
        control_label.pack(pady=10)
        
        # Buttons
        btn_frame = tk.Frame(control_frame, bg="#f0f0f0")
        btn_frame.pack(pady=20)
        
        # Register new student button
        register_btn = tk.Button(btn_frame, text="Register New Student", 
                                width=20, font=("times new roman", 13, "bold"),
                                command=self.register_student, bg="#2196F3", fg="white")
        register_btn.grid(row=0, column=0, padx=10, pady=10)
        
        # Train model button
        train_btn = tk.Button(btn_frame, text="Train Model", 
                            width=20, font=("times new roman", 13, "bold"),
                            command=self.train_model, bg="#4CAF50", fg="white")
        train_btn.grid(row=1, column=0, padx=10, pady=10)
        
        # Take attendance button
        attendance_btn = tk.Button(btn_frame, text="Take Attendance", 
                                width=20, font=("times new roman", 13, "bold"),
                                command=self.start_attendance, bg="#FF9800", fg="white")
        attendance_btn.grid(row=2, column=0, padx=10, pady=10)
        
        # View attendance button
        view_btn = tk.Button(btn_frame, text="View Attendance", 
                            width=20, font=("times new roman", 13, "bold"),
                            command=self.view_attendance, bg="#9C27B0", fg="white")
        view_btn.grid(row=3, column=0, padx=10, pady=10)
        
        # Exit button
        exit_btn = tk.Button(btn_frame, text="Exit", 
                            width=20, font=("times new roman", 13, "bold"),
                            command=self.exit_app, bg="#f44336", fg="white")
        exit_btn.grid(row=4, column=0, padx=10, pady=10)
        
        # Status frame
        status_frame = tk.Frame(main_frame, bd=2, relief=tk.RIDGE, bg="#f0f0f0")
        status_frame.place(x=20, y=540, width=1140, height=120)
        
        # Status label
        self.status_label = tk.Label(status_frame, text="Status: Ready", 
                                  font=("times new roman", 14), bg="#f0f0f0")
        self.status_label.pack(pady=10)

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        print(message)

    def register_student(self):
        # Stop any existing camera capture
        self.stop_capture()
        
        # Ask for student ID and name
        student_id = simpledialog.askstring("Student ID", "Enter Student ID:", parent=self.root)
        if not student_id:
            return
            
        student_name = simpledialog.askstring("Student Name", "Enter Student Name:", parent=self.root)
        if not student_name:
            return
            
        student_enrollment = simpledialog.askstring("Enrollment", "Enter Enrollment/Roll Number:", parent=self.root)
        if not student_enrollment:
            return
            
        # Create a dialog for semester and branch selection
        semester_branch_dialog = tk.Toplevel(self.root)
        semester_branch_dialog.title("Select Semester & Branch")
        semester_branch_dialog.geometry("300x200")
        semester_branch_dialog.transient(self.root)
        semester_branch_dialog.grab_set()
        
        # Semester selection
        semester_frame = tk.Frame(semester_branch_dialog)
        semester_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(semester_frame, text="Semester:").pack(side=tk.LEFT, padx=5)
        
        semester_var = tk.StringVar()
        semesters = ["1", "2", "3", "4", "5", "6", "7", "8"]
        semester_dropdown = ttk.Combobox(semester_frame, textvariable=semester_var, values=semesters, state="readonly", width=5)
        semester_dropdown.pack(side=tk.LEFT, padx=5)
        semester_dropdown.current(0)
        
        # Branch selection
        branch_frame = tk.Frame(semester_branch_dialog)
        branch_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(branch_frame, text="Branch:").pack(side=tk.LEFT, padx=5)
        
        branch_var = tk.StringVar()
        branches = ["CSE", "IT", "ECE", "EEE"]
        branch_dropdown = ttk.Combobox(branch_frame, textvariable=branch_var, values=branches, state="readonly", width=5)
        branch_dropdown.pack(side=tk.LEFT, padx=5)
        branch_dropdown.current(0)
        
        # Result variables
        semester_result = [None]
        branch_result = [None]
        
        def on_ok():
            semester_result[0] = semester_var.get()
            branch_result[0] = branch_var.get()
            semester_branch_dialog.destroy()
        
        # OK button
        ok_btn = tk.Button(semester_branch_dialog, text="OK", command=on_ok)
        ok_btn.pack(pady=20)
        
        # Wait for dialog to close
        self.root.wait_window(semester_branch_dialog)
        
        # Check if user canceled
        if semester_result[0] is None or branch_result[0] is None:
            return
            
        semester = semester_result[0]
        branch = branch_result[0]
        
        # Check if ID already exists
        for record in self.student_records:
            if record['id'] == student_id:
                messagebox.showerror("Error", "Student ID already exists!")
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
            self.root.update()
        
        # Close camera
        self.stop_capture()
        
        # Ask to train the model
        if count > 0:
            if messagebox.askyesno("Training", "Registration complete! Do you want to train the model now?"):
                self.train_model()

    def display_frame(self, frame):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (600, 500))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

    def stop_capture(self):
        self.is_capturing = False
        if self.capture_thread is not None:
            if self.capture_thread.is_alive():
                self.capture_thread.join(1.0)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Reset the video label
        self.video_label.config(image='')

    def train_model(self):
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
        # Stop any existing camera capture
        self.stop_capture()
        
        # Create a dialog for selecting semester, branch and subject
        attendance_dialog = tk.Toplevel(self.root)
        attendance_dialog.title("Attendance Details")
        attendance_dialog.geometry("350x250")
        attendance_dialog.transient(self.root)
        attendance_dialog.grab_set()
        
        # Semester selection
        semester_frame = tk.Frame(attendance_dialog)
        semester_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(semester_frame, text="Semester:").pack(side=tk.LEFT, padx=5)
        
        semester_var = tk.StringVar()
        semesters = ["1", "2", "3", "4", "5", "6", "7", "8"]
        semester_dropdown = ttk.Combobox(semester_frame, textvariable=semester_var, values=semesters, state="readonly", width=5)
        semester_dropdown.pack(side=tk.LEFT, padx=5)
        semester_dropdown.current(0)
        
        # Branch selection
        branch_frame = tk.Frame(attendance_dialog)
        branch_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(branch_frame, text="Branch:").pack(side=tk.LEFT, padx=5)
        
        branch_var = tk.StringVar()
        branches = ["CSE", "IT", "ECE", "EEE"]
        branch_dropdown = ttk.Combobox(branch_frame, textvariable=branch_var, values=branches, state="readonly", width=5)
        branch_dropdown.pack(side=tk.LEFT, padx=5)
        branch_dropdown.current(0)
        
        # Subject input
        subject_frame = tk.Frame(attendance_dialog)
        subject_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(subject_frame, text="Subject:").pack(side=tk.LEFT, padx=5)
        
        subject_entry = tk.Entry(subject_frame, width=20)
        subject_entry.pack(side=tk.LEFT, padx=5)
        
        # Result variables
        details_result = [None, None, None]
        
        def on_ok():
            if not subject_entry.get().strip():
                messagebox.showerror("Error", "Please enter a subject", parent=attendance_dialog)
                return
                
            details_result[0] = semester_var.get()
            details_result[1] = branch_var.get()
            details_result[2] = subject_entry.get().strip()
            attendance_dialog.destroy()
        
        # OK button
        ok_btn = tk.Button(attendance_dialog, text="Start Attendance", command=on_ok, bg="#FF9800", fg="white")
        ok_btn.pack(pady=20)
        
        # Wait for dialog to close
        self.root.wait_window(attendance_dialog)
        
        # Check if user canceled
        if details_result[0] is None:
            return
            
        semester = details_result[0]
        branch = details_result[1]
        subject = details_result[2]
        
        # Create today's attendance file if it doesn't exist
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        attendance_file = f"data/attendance_{today}_{semester}_{branch}_{subject}.csv"
        
        if not os.path.exists(attendance_file):
            df = pd.DataFrame(columns=['id', 'name', 'enrollment', 'semester', 'branch', 'subject', 'time'])
            df.to_csv(attendance_file, index=False)
        
        # Load existing attendance
        try:
            attendance = pd.read_csv(attendance_file)
        except Exception:
            attendance = pd.DataFrame(columns=['id', 'name', 'enrollment', 'semester', 'branch', 'subject', 'time'])
            
        # Start capture thread for attendance
        self.update_status(f"Starting attendance for {branch} Semester {semester} - {subject}. Press ESC to stop.")
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=lambda: self.recognize_faces(attendance, attendance_file, semester, branch, subject))
        self.capture_thread.daemon = True
        self.capture_thread.start()

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
            
            # Process GUI events
            self.root.update()
            
            # Check for ESC key press
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        
        # Close camera
        self.stop_capture()
        self.update_status("Attendance session ended")

    def view_attendance(self):
        # Open a new window to view attendance
        attendance_window = tk.Toplevel(self.root)
        attendance_window.title("Attendance Records")
        attendance_window.geometry("1000x600")
        
        # Get list of attendance files
        attendance_files = [f for f in os.listdir("data") if f.startswith("attendance_") and f.endswith(".csv")]
        
        if not attendance_files:
            tk.Label(attendance_window, text="No attendance records found", font=("times new roman", 14)).pack(pady=20)
            return
            
        # Sort files by date (newest first)
        attendance_files.sort(reverse=True)
        
        # Frame for filters
        filter_frame = tk.Frame(attendance_window)
        filter_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Date selection
        date_frame = tk.Frame(filter_frame)
        date_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(date_frame, text="Date:", font=("times new roman", 12)).pack(side=tk.LEFT, padx=5)
        
        # Extract unique dates from filenames
        dates = sorted(list(set([f.split('_')[1] for f in attendance_files])), reverse=True)
        date_var = tk.StringVar(value=dates[0] if dates else "")
        
        date_dropdown = ttk.Combobox(date_frame, textvariable=date_var, values=dates, state="readonly", width=12)
        date_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Semester selection
        semester_frame = tk.Frame(filter_frame)
        semester_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(semester_frame, text="Semester:", font=("times new roman", 12)).pack(side=tk.LEFT, padx=5)
        
        # Extract unique semesters from filenames
        semesters = sorted(list(set([f.split('_')[2] for f in attendance_files if len(f.split('_')) > 2])))
        semester_var = tk.StringVar(value=semesters[0] if semesters else "")
        
        semester_dropdown = ttk.Combobox(semester_frame, textvariable=semester_var, values=semesters, state="readonly", width=5)
        semester_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Branch selection
        branch_frame = tk.Frame(filter_frame)
        branch_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(branch_frame, text="Branch:", font=("times new roman", 12)).pack(side=tk.LEFT, padx=5)
        
        # Extract unique branches from filenames
        branches = sorted(list(set([f.split('_')[3] for f in attendance_files if len(f.split('_')) > 3])))
        branch_var = tk.StringVar(value=branches[0] if branches else "")
        
        branch_dropdown = ttk.Combobox(branch_frame, textvariable=branch_var, values=branches, state="readonly", width=5)
        branch_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Subject selection
        subject_frame = tk.Frame(filter_frame)
        subject_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(subject_frame, text="Subject:", font=("times new roman", 12)).pack(side=tk.LEFT, padx=5)
        
        # Extract unique subjects from filenames
        subjects = sorted(list(set([f.split('_')[4].replace('.csv', '') for f in attendance_files if len(f.split('_')) > 4])))
        subject_var = tk.StringVar(value=subjects[0] if subjects else "")
        
        subject_dropdown = ttk.Combobox(subject_frame, textvariable=subject_var, values=subjects, state="readonly", width=15)
        subject_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Create treeview to display attendance
        columns = ("ID", "Name", "Enrollment", "Semester", "Branch", "Subject", "Time")
        tree = ttk.Treeview(attendance_window, columns=columns, show="headings")
        
        # Set column headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor=tk.CENTER)
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Function to load attendance data
        def load_attendance(event=None):
            # Clear existing data
            for item in tree.get_children():
                tree.delete(item)
                
            # Get selected filters
            selected_date = date_var.get()
            selected_semester = semester_var.get()
            selected_branch = branch_var.get()
            selected_subject = subject_var.get()
            
            file_path = f"data/attendance_{selected_date}_{selected_semester}_{selected_branch}_{selected_subject}.csv"
            
            try:
                # Load attendance data
                df = pd.read_csv(file_path)
                
                # Add data to treeview
                for index, row in df.iterrows():
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
                    tree.insert("", tk.END, values=values)
                    
                self.update_status(f"Loaded attendance for {selected_date}, {selected_branch} Semester {selected_semester} - {selected_subject}")
            except Exception as e:
                print(f"Error loading attendance: {e}")
                messagebox.showinfo("Not Found", "No attendance records found for the selected filters.", parent=attendance_window)
        
        # Bind selection change events
        date_dropdown.bind("<<ComboboxSelected>>", load_attendance)
        semester_dropdown.bind("<<ComboboxSelected>>", load_attendance)
        branch_dropdown.bind("<<ComboboxSelected>>", load_attendance)
        subject_dropdown.bind("<<ComboboxSelected>>", load_attendance)
        
        # Export button
        def export_to_excel():
            # Get selected filters
            selected_date = date_var.get()
            selected_semester = semester_var.get()
            selected_branch = branch_var.get()
            selected_subject = subject_var.get()
            
            file_path = f"data/attendance_{selected_date}_{selected_semester}_{selected_branch}_{selected_subject}.csv"
            
            try:
                df = pd.read_csv(file_path)
                excel_path = f"data/attendance_{selected_date}_{selected_semester}_{selected_branch}_{selected_subject}.xlsx"
                df.to_excel(excel_path, index=False)
                messagebox.showinfo("Export Successful", f"Attendance exported to {excel_path}", parent=attendance_window)
            except Exception as e:
                messagebox.showerror("Export Failed", f"Error: {e}", parent=attendance_window)
        
        # Export button
        export_btn = tk.Button(attendance_window, text="Export to Excel", 
                             font=("times new roman", 12), bg="#4CAF50", fg="white",
                             command=export_to_excel)
        export_btn.pack(pady=10)
        
        # Load initial data if possible
        try:
            load_attendance()
        except Exception as e:
            print(f"Could not load initial data: {e}")

    def speak_text(self, text):
        # Function to speak text using pyttsx3
        def speak():
            self.engine.say(text)
            self.engine.runAndWait()
        
        # Start in a separate thread to avoid blocking
        t = threading.Thread(target=speak)
        t.daemon = True
        t.start()

    def exit_app(self):
        # Stop any capture threads
        self.stop_capture()
        
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionAttendanceSystem(root)
    root.mainloop() 