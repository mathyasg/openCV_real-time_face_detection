# Real-Time Face & Eye Detection

A simple yet powerful real-time face and eye detection app built with OpenCV and Haar Cascades.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)

## ✨ Features

- Real-time face detection with blue boxes
- Improved eye detection (no more nose false positives!)
- Eye detection can be toggled on/off
- Capture button (`c`) saves the full screen **and** individual face crops
- Resizable window + true fullscreen mode (`f`)
- Clean on-screen instructions

## 🚀 Installation

1. Clone the repository:
   git clone https://github.com/mathyasg/openCV_real-time_face_detection
   cd opencv_real-time_face_detection

2. Install dependencies:
		pip install -r requirements.txt
	Make sure the two .xml files are in the folder (they are already included in this repo).

🎮 How to Use

Run the script:
	python face_detector.py
	
	Keyboard Controls
		q → Quit
		c → Capture detected faces (saves images)
		f → Toggle fullscreen
		e → Toggle eye detection


📁 Project Structure
	textopencv-face-eye-detection/
	├── face_detector.py
	├── haarcascade_frontalface_default.xml
	├── haarcascade_eye.xml
	├── requirements.txt
	├── README.md
	└── .gitignore

🤝 Contributing
	Feel free to open issues or pull requests!
	Ideas for improvements: add smile detection, age/gender estimation, or switch to DNN model.

📄 License
	This project is open-source under the MIT License.