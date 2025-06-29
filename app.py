import streamlit as st
import pandas as pd
from textblob import TextBlob
import speech_recognition as sr
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import random
from deepface import DeepFace
import base64
import io
import numpy as np
import datetime
import cv2
import random
import torch
import tempfile
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import mediapipe as mp
from fer import FER

# Patch to prevent Streamlit from trying to access __path__ of torch.classes
if hasattr(torch, 'classes') and not hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = []

# Page config
st.set_page_config(page_title="InQuest Pro", layout="centered")

# App title and intro
st.title('InQuest Pro')
st.header('Welcome!')
st.write("This is a smart interview assistant designed to help you improve your interview skills. Select a job type, answer questions, and get instant feedback!")

# Load dataset
DATASET_PATH = os.path.join("datasets", "refined_realistic_interview_questions.csv")
df = pd.read_csv(DATASET_PATH)

# Load the Sentence Transformer model

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
# Load model once
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Filter roles dynamically
job_type = st.selectbox('👔 Select the Job Type:', sorted(df['Role'].unique()))

# Initialize session state
if 'prev_job_type' not in st.session_state:
    st.session_state.prev_job_type = job_type
if 'selected_questions' not in st.session_state:
    st.session_state.selected_questions = None

# If job type changed, reset
if st.session_state.prev_job_type != job_type:
    st.session_state.selected_questions = None
    st.session_state.prev_job_type = job_type

# Load questions
if st.session_state.selected_questions is None:
    # Filter questions
    job_specific = df[(df['Role'] == job_type) & ((df['Type'] == 'Technical') | (df['Type'] == 'Behavioral') | (df['Type'] == 'Situational'))]
    general_questions = df[df['Type'] == 'General']

    if len(job_specific) > 0:
        num_job_questions = min(len(job_specific), 5)
        selected_job_questions = job_specific.sample(n=num_job_questions)
        selected_general_questions = general_questions.sample(n=10 - num_job_questions)
        
        selected_questions = pd.concat([selected_job_questions, selected_general_questions]).sample(frac=1).reset_index(drop=True)
    else:
        # No technical questions found
        st.warning("⚠️ No technical questions available for this job type. Showing only general questions.")
        selected_questions = general_questions.sample(n=10).reset_index(drop=True)

    st.session_state.selected_questions = selected_questions
else:
    selected_questions = st.session_state.selected_questions

# Display questions
st.subheader("📋 Your Interview Questions")
answers = []
for i, row in selected_questions.iterrows():
    question = row['Question']
    q_type = row['Type']
    st.markdown(f"**{i+1}. ({q_type}) {question}**")
    answer = st.text_area(f"Your answer to Question {i+1}:", key=f"answer_{i}")
    answers.append(answer)

# Function to analyze answer using semantic similarity
def analyze_text_answer(user_answer, question_text):
    user_embed = sbert_model.encode(user_answer, convert_to_tensor=True)
    best_similarity = 0.0
    best_relevance_score = 0.0
    best_feedback = "No relevant data found."
    relevance_score = 0.0

    # Check USER ANSWER LOGS first
    if os.path.exists("updated_user_answer_log.csv"):
        logs_df = pd.read_csv("updated_user_answer_log.csv")
        similar_qs = logs_df[logs_df['Question'].str.contains(question_text[:10], case=False, na=False)]

        if not similar_qs.empty:
            past_answers = similar_qs['Answer'].dropna().tolist()
            past_embeddings = sbert_model.encode(past_answers, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(user_embed, past_embeddings)[0]
            max_score = float(similarities.max())

            if max_score > best_similarity:
                best_similarity = max_score
                relevance_score = max_score
                if max_score > 0.8:
                    best_feedback = f"Highly relevant (similarity: {max_score:.2f})"
                elif max_score > 0.5:
                    best_feedback = f"Moderately relevant (similarity: {max_score:.2f})"
                else:
                    best_feedback = f"Low relevance (similarity: {max_score:.2f})"

    # Check STRUCTURED MODEL ANSWERS
    model_path = os.path.join("datasets", "updated_model_answers.csv")

    if os.path.exists(model_path):
        model_df = pd.read_csv(model_path).dropna(subset=["Model_Answers"])
        model_qs = model_df[model_df['Question'].str.contains(question_text[:10], case=False, na=False)]

        if not model_qs.empty:
            model_answers = model_qs['Model_Answers'].tolist()
            model_scores = model_qs['Relevance Score'].tolist()
            model_embeddings = sbert_model.encode(model_answers, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(user_embed, model_embeddings)[0]
            model_max_idx = int(similarities.argmax())
            model_score = float(similarities[model_max_idx])
            model_relevance = float(model_scores[model_max_idx])

            # Use model if it has a higher score
            if model_score > best_similarity:
                best_similarity = model_score
                relevance_score = model_relevance
                label = "Highly relevant" if model_relevance >= 0.85 else (
                    "Moderately relevant" if model_relevance >= 0.5 else "Low relevance"
                )
                best_feedback = f"{label} (model match, similarity: {model_score:.2f}, relevance: {model_relevance:.2f})"

    return best_feedback, round(relevance_score, 2)

    
# Feedback function using TextBlob (Tone analysis)
def analyze_tone(answer):
    if not answer.strip():
        return "No answer provided.", 0.0
    
    blob = TextBlob(answer)
    polarity = blob.sentiment.polarity
    
    # Passive-aggressive and defensive cues
    passive_cues = ["I guess", "sure", "whatever", "if you say so", "I don't know", "I mean", "I guess"]
    defensive_cues = ["I tried", "no one helped", "I had to", "nobody listened", "not my fault"]

    passive_score = sum(1 for word in passive_cues if word in answer.lower()) / len(passive_cues)
    defensive_score = sum(1 for word in defensive_cues if word in answer.lower()) / len(defensive_cues)

    tone_score = polarity - (0.5 * passive_score) - (0.5 * defensive_score)

    # Determine tone label
    if tone_score > 0.2:
        return "✅ Positive tone – sounds confident!", tone_score
    elif tone_score < -0.2:
        if passive_score > 0.3:
            return "⚠️ Passive-aggressive tone detected. Try to be more direct and positive.", tone_score
        elif defensive_score > 0.3:
            return "⚠️ Defensive tone detected. Focus on your strengths instead.", tone_score
        else:
            return "⚠️ Negative tone – try to frame things more optimistically.", tone_score
    else:
        return "😐 Neutral – consider showing more enthusiasm or details.", tone_score


def calculate_acceptance(relevance_score, tone_score):
    # Weighted average of relevance and tone
    acceptance_score = (0.7 * relevance_score) + (0.3 * tone_score)
    if acceptance_score >= 0.7:
        return "✅ Highly likely to be well-received by the interviewer.", acceptance_score
    elif acceptance_score >= 0.4:
        return "😐 Could go either way. Consider refining your response.", acceptance_score
    else:
        return "⚠️ Unlikely to make a positive impression. Consider revising the content and tone.", acceptance_score

# Show feedback for each answer
# After asking questions and collecting all answers into 'answers' list
submit = st.button("🚀 Submit All Answers")

# Sidebar Setup
st.sidebar.title("Performance Overview")

# Sidebar Dropdowns
text_dropdown = st.sidebar.expander("📄 Text Interview Progress", expanded=False)
video_dropdown = st.sidebar.expander("📹 Video Interview Progress", expanded=False)

# Initialize placeholders
text_progress_placeholder = text_dropdown.empty()
text_score_placeholder = text_dropdown.empty()
video_progress_placeholder = video_dropdown.empty()
video_score_placeholder = video_dropdown.empty()

# Initialize mode flags
is_text_mode = False
is_video_mode = False

# Check for text mode
answered_count = len([answer for answer in answers if answer.strip()])
if answered_count > 0:
    is_text_mode = True

# Update the sidebar based on mode
def update_sidebar():
    # Reset placeholders
    text_progress_placeholder.empty()
    text_score_placeholder.empty()
    video_progress_placeholder.empty()
    video_score_placeholder.empty()

    # Text Interview Progress
    if is_text_mode:
        total_questions = len(answers)
        completion_percentage = answered_count / total_questions if total_questions > 0 else 0

        with text_dropdown:
            text_progress_placeholder.progress(completion_percentage, text="Questions Answered")
            text_score_placeholder.text("Overall Score will be displayed upon submission.")

    # Video Interview Progress
    if is_video_mode:
        with video_dropdown:
            video_progress_placeholder.progress(0.0, text="Analyzing Video...")
            video_score_placeholder.text("Overall Score will be displayed after analysis.")

update_sidebar()
    
def plot_radar_chart(scores):
    categories = ["General", "Behavioral", "Technical", "Situational"]
    category_scores = [0, 0, 0, 0]
    category_counts = [0, 0, 0, 0]

    # Group scores based on categories
    for i, score in enumerate(scores):
        question_type = selected_questions.iloc[i]['Type']
        
        if question_type == "General":
            category_scores[0] += score
            category_counts[0] += 1
        elif question_type == "Behavioral":
            category_scores[1] += score
            category_counts[1] += 1
        elif question_type == "Technical":
            category_scores[2] += score
            category_counts[2] += 1
        elif question_type == "Situational":
            category_scores[3] += score
            category_counts[3] += 1

    # Calculate average score for each category
    average_scores = [cs / cc if cc > 0 else 0 for cs, cc in zip(category_scores, category_counts)]

    # Radar chart setup
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Make the radar chart a closed loop
    average_scores += average_scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.fill(angles, average_scores, color='royalblue', alpha=0.25)
    ax.plot(angles, average_scores, color='royalblue', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)

    ax.set_title("Performance Across Categories", size=14, color='black', pad=20)

    st.sidebar.pyplot(fig)

if is_text_mode:
    if submit:
        unanswered_indices = [i for i, answer in enumerate(answers) if not answer.strip()]
        
        if unanswered_indices:
            # Inform the user of unanswered questions
            st.warning(f"Please provide answers to all questions before submitting. You have {len(unanswered_indices)} unanswered question(s).")
        else:
            st.subheader("🧠 Feedback on Your Answers")

            acceptance_scores = []  # List to store acceptance scores for the current interview session

            for i, answer in enumerate(answers, 1):
                question = selected_questions.iloc[i-1]['Question']
                
                # Analyze relevance and tone
                relevance_feedback, relevance_score = analyze_text_answer(answer, question)
                tone_feedback, tone_score = analyze_tone(answer)
                interviewer_feedback, acceptance_score = calculate_acceptance(relevance_score, tone_score)

                # Store the acceptance score for the current answer
                acceptance_scores.append(acceptance_score)
                
                st.write(f"**Answer {i}:**")
                st.write(f"**Relevance Feedback:** {relevance_feedback}")
                st.write(f"**Tone Feedback:** {tone_feedback}")
                st.write(f"**Interviewer acceptance:** {interviewer_feedback}")

            # --- New Code for Overall Score Calculation ---
            if acceptance_scores:
                # Calculate overall score
                total_score = sum(acceptance_scores)
                average_score = total_score / len(acceptance_scores)

                # Determine verdict
                if average_score >= 0.7:
                    verdict = "Nail It"
                elif average_score >= 0.6:
                    verdict = "Likely to Nail It"
                elif average_score >= 0.5:
                    verdict = "Could Go Either Way"
                else:
                    verdict = "Likely to Fail"

                # Display overall score and verdict
                st.markdown(f"### Overall Score: {average_score * 100:.2f}% - {verdict}")

                # Update the sidebar with the same overall score and progress bar
                overall_score = sum(acceptance_scores) / len(acceptance_scores) if acceptance_scores else 0
                with text_dropdown:
                    text_score_placeholder.progress(overall_score, text=f"Overall Score: {overall_score * 100:.2f}%")

                st.sidebar.markdown("### 🛠️ Category Performance")
                plot_radar_chart(acceptance_scores)

            # Path to store answer logs
            LOG_FILE = "updated_user_answer_log.csv"

            # Create log entry list
            log_entries = []

            # Generate log for each answer
            for i, answer in enumerate(answers, 1):
                question = selected_questions.iloc[i-1]['Question']
                q_type = selected_questions.iloc[i-1]['Type']
                relevance_feedback, relevance_score = analyze_text_answer(answer, question)
                tone_feedback, tone_score = analyze_tone(answer)
                interviewer_feedback, acceptance_score = calculate_acceptance(relevance_score, tone_score)

                # Append log entry
                log_entries.append({
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Job Type": job_type,
                    "Question Type": q_type,
                    "Question": question,
                    "Answer": answer,
                    "Relevance Feedback": relevance_feedback,
                    "Tone Feedback": tone_feedback,
                    "Relevance Score": round(relevance_score, 2),
                    "Tone Score": round(tone_score, 2),
                    "Acceptance Score": round(acceptance_score, 2)
                })

            # Save to CSV
            if log_entries:
                df_logs = pd.DataFrame(log_entries)
                if os.path.exists(LOG_FILE):
                    df_logs.to_csv(LOG_FILE, mode='a', header=False, index=False)
                else:
                    df_logs.to_csv(LOG_FILE, index=False)
                st.success("✅ Your answers and feedback have been saved to the learning log.")

selected_q = st.selectbox("🔎 Choose a question to see top answers:", selected_questions['Question'].tolist())

if st.button("Show Top Answers for this Question"):
    top_answers = None

    # Normalize selected question
    normalized_q = selected_q.strip().lower()

    # First check user log
    if os.path.exists("updated_user_answer_log.csv"):
        logs_df = pd.read_csv("updated_user_answer_log.csv")
        logs_df['Question_norm'] = logs_df['Question'].str.strip().str.lower()

        # Filter matching question and high relevance answers
        matched_logs = logs_df[
            (logs_df['Question_norm'] == normalized_q) & 
            (logs_df['Relevance Score'] >= 0.80)
        ]

        if not matched_logs.empty:
            top_answers = matched_logs.sort_values(by="Relevance Score", ascending=False).head(3)

    # Fallback to model answers only if no good user answers found
    if top_answers is None or top_answers.empty:
        model_path = os.path.join("datasets", "updated_model_answers.csv")

        if os.path.exists(model_path):
            model_df = pd.read_csv(model_path)
            model_df['Question_norm'] = model_df['Question'].str.strip().str.lower()

            matched_model = model_df[model_df['Question_norm'] == normalized_q]
            if not matched_model.empty:
                top_answers = matched_model.head(3)
            else:
                st.warning("No model answers available for this question.")
        else:
            st.warning("Model answer file not found.")

    # Display results
    if top_answers is not None and not top_answers.empty:
        # Decide which column to show (only one will be valid per file)
        if 'Model_Answers' in top_answers.columns:
            answer_column = 'Model_Answers'
        elif 'Answer' in top_answers.columns:
            answer_column = 'Answer'
        else:
            st.warning("❌ Could not find a valid answer column.")
            answer_column = None

        if answer_column:
            for idx, row in top_answers.iterrows():
                answer = row.get(answer_column, None)
                if pd.isna(answer):
                    st.markdown("❌ No answer available in this row.")
                else:
                    st.markdown(f"**Answer:** {answer}")

                # Show relevance if present
                if 'Relevance Score' in row and not pd.isna(row['Relevance Score']):
                    st.markdown(f"_Relevance Score: {row['Relevance Score']:.2f}_")
                st.markdown("---")
    else:
        st.warning("No suitable answers found for this question.")

# Speech recording (optional)
# st.subheader("🎙️ Try Speaking Your Answer (optional)")

# def record_audio():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Recording... Please speak now.")
#         audio = recognizer.listen(source, timeout=5)
#     try:
#         result = recognizer.recognize_google(audio)
#         return result
#     except sr.UnknownValueError:
#         return "Could not understand audio."
#     except sr.RequestError:
#         return "Speech recognition service failed."

# if st.button("🎤 Record Audio for Practice"):
#     try:
#         speech_text = record_audio()
#         st.write("🗣️ You said:", speech_text)

#         # Analyze the speech response
#         st.write("🧠 Feedback:", analyze_tone(speech_text))

#         # Speech Confidence & Pace Analysis
#         st.write("🎯 Speech Analysis:")
#         st.write("💬 Confidence: Based on clarity and fluency")
#         st.write("⏱️ Pace: Analyze speed and pauses")

#         # For simplicity, let's assume no pauses or hesitations for now
#         if "uh" in speech_text.lower() or "um" in speech_text.lower():
#             st.write("⚠️ Speech might lack confidence due to frequent pauses.")
#         else:
#             st.write("✅ Confident tone – great!")

#     except:
#         st.error("Microphone not found or error during recording.")

# Placeholder for video input

st.title("InQuest Pro - Emotion & Body Language Analysis")


# Upload video
video_file = st.file_uploader("📤 Upload your interview video (MP4/WebM/MOV):", type=["mp4", "webm", "mov"])

if video_file is not None:
    is_video_mode = True
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name

    st.success("✅ Video uploaded successfully!")

    # Convert video to base64 to display
    video_base64 = base64.b64encode(open(video_path, "rb").read()).decode("utf-8")
    video_html = f"""
    <div style="max-width: 100%; height: auto;">
        <video style="width:100%; height:auto; max-height:300px;" controls>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    """
    st.markdown(video_html, unsafe_allow_html=True)

    st.subheader("🔍 Analyzing Facial Emotions & Body Language")

    # Initialize detectors
    emotion_detector = FER(mtcnn=True)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    emotions_data = []
    posture_flags = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:
            continue  # Skip frames for faster processing

        # Emotion Detection
        result = emotion_detector.detect_emotions(frame)
        if result:
            emotions_data.append(result[0]['emotions'])

        # Body Pose Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_pose = pose.process(rgb_frame)
        if result_pose.pose_landmarks:
            posture_flags.append("✅ Detected")
        else:
            posture_flags.append("❌ Not Detected")

    cap.release()

    # Emotion Summary
    if emotions_data:
        st.subheader("🙂 Detected Emotions Summary")
        avg_emotions = {k: np.mean([e[k] for e in emotions_data]) for k in emotions_data[0].keys()}
        for emo, val in avg_emotions.items():
            st.write(f"*{emo.capitalize()}*: {val:.2f}")
    else:
        st.warning("❌ Could not detect facial emotions.")

    # Posture Summary
    st.subheader("💼 Body Posture Analysis")
    detected_count = posture_flags.count("✅ Detected")
    st.write(f"Detected posture in {detected_count} of {len(posture_flags)} frames.")
    if detected_count / len(posture_flags) > 0.6:
        st.success("👍 Consistent posture detected — indicates confidence.")
    else:
        st.warning("⚠️ Inconsistent posture — try to sit upright and stay steady.")

    st.success("✅ Emotion & Body Language Analysis Complete.")
    # Video analysis progress (simulated full progress after completion)
    video_progress_placeholder.progress(1.0, text="Video analysis complete")

    # --- Calculate Scores ---

    # Emotion Score
    positive_emotions = ['happy', 'neutral']
    emotion_score = 0
    if emotions_data:
        positive_avg = np.mean([avg_emotions[e] for e in avg_emotions if e in positive_emotions])
        emotion_score = min(positive_avg, 1.0) * 0.5  # max 0.5 for emotions

    # Posture Score
    posture_score = 0
    if posture_flags:
        consistency_ratio = detected_count / len(posture_flags)
        if consistency_ratio >= 0.6:
            posture_score = 0.5  # Good posture
        else:
            posture_score = 0.25  # Partial credit

    # Total Score
    video_overall_score = emotion_score + posture_score

    # Verdict
    if video_overall_score >= 0.8:
        video_verdict = "✅ Excellent confidence and expression!"
    elif video_overall_score >= 0.5:
        video_verdict = "⚖️ Fair posture and emotion — room for improvement."
    else:
        video_verdict = "❌ Poor engagement — work on expression and body language."

    # --- Show in Sidebar ---
    with video_dropdown:
        video_score_placeholder.progress(video_overall_score, text=f"Overall Score: {video_overall_score * 100:.2f}%")
        st.sidebar.markdown(f"### 🎯 Video Verdict: {video_verdict}")
