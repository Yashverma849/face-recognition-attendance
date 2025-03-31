# Face Recognition Attendance System

A modern attendance management system using facial recognition technology. This Python application provides an efficient way to register students, track attendance, and manage attendance records with an easy-to-use graphical interface.

## Features

- **Student Registration**: Register new students with their details and face data
- **Face Recognition**: Automatically recognize registered students using webcam
- **Attendance Tracking**: Mark and store attendance with date, time, and subject information
- **Records Management**: View and filter attendance records by date, semester, branch, and subject
- **Modern UI**: Clean and responsive interface using CustomTkinter
- **Voice Feedback**: Audio confirmation when attendance is marked

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam

### Setup

1. Clone this repository:
```
git clone https://github.com/yourusername/face-recognition-attendance.git
cd face-recognition-attendance
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Create required directories (if they don't exist):
```
mkdir -p data src/images/train src/images/temp
```

## Usage

### Running the Application

On Windows:
```
run.bat
```

Or directly with Python:
```
python app_modern.py
```

### Getting Started

1. **Register Students**:
   - Click on "Register Student" in the sidebar
   - Enter student details (ID, name, enrollment number)
   - Select semester and branch
   - The system will capture face images for training

2. **Take Attendance**:
   - Click on "Take Attendance" in the sidebar
   - Select semester, branch, and enter subject
   - The system will recognize students and mark attendance automatically

3. **View Attendance Records**:
   - Click on "View Attendance" in the sidebar
   - Filter records by date, semester, branch, and subject
   - Attendance data will be displayed in a table format

4. **Reset System** (if needed):
   - Click on "Reset Data" in the sidebar
   - Confirm to delete all attendance records and registered student data

## Project Structure

- `app_modern.py`: Main application file with the modern UI
- `requirements.txt`: List of Python dependencies
- `run.bat`: Batch file to run the application on Windows
- `data/`: Directory for storing attendance records and the face recognition model
- `src/images/`: Directory structure for storing student face images

## Technologies Used

- **Python**: Core programming language
- **OpenCV**: Computer vision and face recognition
- **CustomTkinter**: Modern UI components
- **Pandas**: Data management
- **Pillow (PIL)**: Image processing
- **pyttsx3**: Text-to-speech functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CustomTkinter for the modern UI components
- OpenCV team for the computer vision library
- All contributors and testers who helped improve this project 