---

# Fruit Detection and Assistance for the Visually Impaired

[Download all project files from Google Drive]([https://your-google-drive-link.com](https://drive.google.com/drive/folders/1A-d76iUM-7fFsvNVw66uvY0s1mMz7atz?usp=sharing)) *(Due to large file sizes, the complete dataset, model, and other assets can be downloaded from this link.)*

This project leverages object detection to count and identify fruits using camera input, aimed at assisting visually impaired individuals in grocery stores. The application integrates a TensorFlow-Keras model for fruit detection, MediaPipe's hand-tracking for gesture recognition, and generative AI for providing additional assistance and instructions. The user interacts with the system via voice commands, and the app responds with text-to-speech (TTS) feedback.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
  - [Backend](#backend)
  - [Frontend](#frontend)
  - [MyEnv](#myenv)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)

---

## Overview

This system is designed to help visually impaired customers in grocery stores by detecting and counting fruits within a camera's field of view. The backend runs a fruit detection model that uses object detection, while the frontend renders the application on a ReactJS-based interface. The app also uses speech recognition to understand voice commands and text-to-speech (TTS) to provide audible feedback to the user.

---

## Features

- **Fruit Detection**: Identifies and counts multiple fruit types using a trained neural network.
- **Hand Gesture Tracking**: Uses a hand-tracking module to detect gestures for interaction without touch.
- **Generative AI Support**: Provides contextual assistance using generative AI models for deeper interaction.
- **Voice Interaction**: Captures user voice input for triggering actions, and uses TTS for verbal feedback.
- **User Interface**: Clean and intuitive UI built with ReactJS, designed for easy interaction.

---

## Tech Stack

- **Generative AI**: The application leverages generative AI to provide advanced, contextual responses. For instance, it explains the fruit types or counts based on the user's command and offers additional assistance in real-time.
  
- **Cameras and Microphones**: A camera captures the fruits in real time and sends the video feed for analysis. A microphone allows users to provide voice commands for fruit detection and system interaction.

- **TTS (Text-to-Speech)**: The system uses TTS to provide audio feedback, ensuring that the visually impaired user gets verbal responses and information about detected fruits.

- **ReactJS**: The frontend is built using ReactJS, offering a modern, responsive, and interactive UI for the web app. It manages the visual representation of detected fruits, as well as video player interactions.

---

## Folder Structure

### Backend
The backend folder contains the core logic for fruit detection and hand tracking. 

- **`__pycache__`**: This directory stores the compiled Python files used for optimizing module loading time. Specifically, it contains the hand-tracking module.

- **`dataset3/`**: 
  - *train/*: Training dataset for the fruit detection model.
  - *test/*: Testing dataset to validate the fruit detection model.

- **`app.py`**: This is the main backend script, which connects the TensorFlow-Keras model with the generative AI. It processes the camera feed, runs the fruit detection model, and interfaces with ReactJS on the frontend.

- **`fruitRecogV3.h5`**: The pre-trained TensorFlow-Keras model file used to detect fruits from the camera feed. This model was trained using the datasets provided in the `dataset3` folder.

- **`requirements.txt`**: Contains the list of dependencies needed to run the Python backend. Libraries include TensorFlow, OpenCV, MediaPipe, pyttsx3, and more.

- **`handtrackingmodule.py`**: Implements hand gesture recognition using MediaPipe and OpenCV to track user gestures and interact with the system.

### Frontend
The frontend folder contains the UI components and static assets for rendering the web app in the browser.

- **`public/`**: This folder holds static assets like images and logos that are used in the frontend interface.

- **`src/`**:
  - *index.css*: Stylesheet for basic UI design and layout.
  - *index.jsx*: The entry point of the React app. It initializes the app and renders the various components.
  - **`components/`**: Contains reusable components such as:
    - *Footer*: Displays at the bottom of the app.
    - *Header*: The top navigation and title of the app.
    - *MainContent*: The core content area where the video feed and fruit detection output are displayed.
    - *VideoPlayer*: Component for displaying the camera feed and interacting with the backend for fruit detection.

### MyEnv
This folder contains miscellaneous environment dependencies required to run the app smoothly. You might find virtual environment configuration files here, ensuring the correct versions of Python and dependencies are used.

---

## Setup Instructions

To run this project locally, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fruit-detection-assistance.git
cd fruit-detection-assistance
```

### 2. Backend Setup
- Navigate to the `backend` directory:
```bash
cd backend
```

- Install the required Python libraries using `requirements.txt`:
```bash
pip install -r requirements.txt
```

- Ensure that the TensorFlow model `fruitRecogV3.h5` and the datasets in `dataset3/` are properly placed in the backend folder.

### 3. Frontend Setup
- Navigate to the `frontend` directory:
```bash
cd ../frontend
```

- Install the dependencies using npm:
```bash
npm install
```

### 4. Run the Application
- Start the backend server:
```bash
python app.py
```

- In another terminal, start the frontend ReactJS application:
```bash
npm start
```

The app will be available at `http://localhost:3000`.

---

## Usage

1. **Fruit Detection**: 
   - Open the app and allow camera access. The system will start detecting fruits in real-time.
   - The identified fruits will be counted and announced verbally using TTS.

2. **Hand Gestures**: 
   - Place your hand in the camera frame to trigger specific commands (e.g., start/stop detection).

3. **Voice Commands**: 
   - Speak commands like "Detect" or "Count fruits" for hands-free interaction.

4. **Generative AI Responses**:
   - The AI can assist by explaining detected fruits or giving contextual information based on user queries.

---

## Future Enhancements

- **Multi-language Support**: Implementing multi-language TTS for broader accessibility.
- **Expanded Dataset**: Training the model with a wider variety of fruits to improve accuracy.
- **Mobile App**: Developing a mobile version of the app for more convenient use in grocery stores.
- **Gesture Customization**: Allow users to customize hand gestures for specific actions like switching between different detection modes.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Contact
If you have any questions or issues with the project, please reach out at [your.email@example.com].

---

