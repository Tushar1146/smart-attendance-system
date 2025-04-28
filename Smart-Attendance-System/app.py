import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import csv
import os
from datetime import datetime

# Streamlit page settings
st.set_page_config(page_title="Smart Attendance System", layout="centered")

# Custom CSS for gradient background + animated header
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #e0f7fa, #ffffff);
        padding: 1rem;
        font-family: 'Segoe UI', sans-serif;
    }
    .animated-title {
        font-size: 45px;
        font-weight: bold;
        text-align: center;
        animation: glow 1.5s ease-in-out infinite alternate;
        color: #003366;
    }
    @keyframes glow {
        from {
            text-shadow: 0 0 10px #66ccff, 0 0 20px #66ccff, 0 0 30px #66ccff;
        }
        to {
            text-shadow: 0 0 20px #0099cc, 0 0 30px #0099cc, 0 0 40px #0099cc;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Display animated title
st.markdown('<h1 class="animated-title">🧠 Smart Attendance System</h1>', unsafe_allow_html=True)

# Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

def register_face(name):
    cap = cv2.VideoCapture(0)
    success = False
    message = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([[p.x * w, p.y * h] for p in result.multi_face_landmarks[0].landmark], dtype=np.float32)
            for x, y in landmarks:
                cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 255), -1)

        cv2.imshow("Webcam Feed - Register", frame)

        key = cv2.waitKey(1)
        if key == ord('s') and result.multi_face_landmarks:
            np.save(f"face_data_{name}.npy", landmarks)
            success = True
            message = f"✅ Face registered for {name}!"
            break
        if key == ord('q'):
            message = "⚠️ Registration cancelled."
            break

    cap.release()
    cv2.destroyAllWindows()
    return success, message

def mark_attendance(name):
    if not os.path.exists(f"face_data_{name}.npy"):
        return False, "❌ No registration data found for this user."

    reference = np.load(f"face_data_{name}.npy")
    cap = cv2.VideoCapture(0)
    logged = False
    matched = False
    message = "Face not matched."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([[p.x * w, p.y * h] for p in result.multi_face_landmarks[0].landmark], dtype=np.float32)

            if reference.shape == landmarks.shape:
                distance = np.linalg.norm(reference - landmarks)
                if distance < 1000 and not logged:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("attendance.csv", "a", newline="") as f:
                        csv.writer(f).writerow([name, now])
                    message = f"✅ Attendance marked for {name} at {now}"
                    logged = True
                    matched = True

        cv2.imshow("Webcam Feed - Attendance", frame)

        if matched or cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return matched, message

# Streamlit Layout
tab1, tab2, tab3 = st.tabs(["📸 Register Face", "✅ Mark Attendance", "📋 View Attendance"])

with tab1:
    st.header("📸 Register Your Face")
    name_input = st.text_input("Enter your Name to Register:")
    if st.button("Start Face Registration"):
        if name_input.strip():
            with st.spinner("Opening Camera... Press 'S' to save face. Press 'Q' to cancel."):
                success, msg = register_face(name_input.strip())
                if success:
                    st.success(msg)
                else:
                    st.warning(msg)
        else:
            st.warning("Please enter your name.")

with tab2:
    st.header("✅ Mark Your Attendance")
    name_input = st.text_input("Enter your Name to Mark Attendance:")
    if st.button("Start Attendance"):
        if name_input.strip():
            with st.spinner("Opening Camera... Looking for your face."):
                matched, msg = mark_attendance(name_input.strip())
                if matched:
                    st.success(msg)
                else:
                    st.warning(msg)
        else:
            st.warning("Please enter your name.")

with tab3:
    st.header("📋 Attendance Records")
    if os.path.exists("attendance.csv"):
        import pandas as pd
        data = pd.read_csv("attendance.csv", names=["Name", "Date & Time"])
        st.dataframe(data)
        st.download_button("Download Attendance CSV", data.to_csv(index=False), "attendance.csv", "text/csv")
    else:
        st.info("No attendance data found yet.")

