# Face Recognition System

A Python-based Face Recognition System that detects and recognizes human faces using computer vision and machine learning techniques. This project uses OpenCV and the `face_recognition` library to identify faces from webcam input in real time.

---

## Features

- Real-time face detection
- Face recognition using known images
- Webcam integration
- Multiple face detection support
- Fast and accurate recognition
- Simple Python implementation

---

## Tech Stack

- Python
- OpenCV
- face_recognition
- NumPy

---

## Project Structure

```bash
Face_Recognition/
│── images/
│   ├── person1.jpg
│   ├── person2.jpg
│── main.py
│── requirements.txt
│── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Puspanjali05/Face_Recognition.git
cd Face_Recognition
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Create a `requirements.txt` file and add:

```txt
opencv-python
face-recognition
numpy
```

---

## Usage

Run the project using:

```bash
python main.py
```

The webcam will open and start detecting and recognizing faces.

Press **Q** to exit the application.

---

## main.py Code

```python
import cv2
import face_recognition
import numpy as np
import os

# Load known images
path = 'images'
images = []
classNames = []

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to encode faces
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # Resize image for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc

            # Scale back up
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0,255,0), cv2.FILLED)

            # Display name
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1, (255,255,255), 2)

    cv2.imshow('Webcam', img)

    # Exit on pressing Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Applications

- Smart attendance systems
- Security systems
- Face-based authentication
- AI surveillance
- Visitor management

---

## Future Improvements

- GUI support
- Database integration
- Attendance logging
- Better accuracy
- Cloud deployment

---

## Author

**Puspanjali Behera**

GitHub: https://github.com/Puspanjali05
