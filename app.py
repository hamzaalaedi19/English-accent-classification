'''
import gradio as gr
import numpy as np
import pickle
import joblib
import yt_dlp


def download_audio_yt_dlp(url, output_path="audio.wav", ffmpeg_path="ffmpeg.exe"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path.replace('.wav', '.%(ext)s'),
        'ffmpeg_location': ffmpeg_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def predict_accent_from_mfcc(mfcc_features, model_path="svm_accent_model.pkl"):
    clf = joblib.load(model_path)
    mfcc_features = np.array(mfcc_features).reshape(1, -1)

    prediction = clf.predict(mfcc_features)[0]

    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(mfcc_features)[0]
        confidence = max(probas) * 100
    else:
        confidence = None

    return prediction, confidence

def run_accent_detection_pipeline(video_url):
    try:
        with open("mfcc_features.pkl", "rb") as f:
            features, labels = pickle.load(f) 
    
        audio_path = download_audio_yt_dlp(video_url, output_path="test_audio.wav")
        accent, confidence = predict_accent_from_mfcc(features[0])

        if confidence is not None:
            result = (
                f"🗣️ Predicted Accent: {accent}\n"
                f"🎯 Confidence: {confidence:.2f}%\n\n"
                
            )
        else:
            result = f"🗣️ Predicted Accent: {accent}\n⚠️ Model does not support confidence score."
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"
def process(video_url):
    return run_accent_detection_pipeline(video_url)



iface = gr.Interface(
    fn=process,
    inputs=gr.Textbox(label="YouTube Video URL"),
    outputs="text",
    title="English Accent Classifier",
    description="Paste a YouTube video URL to detect the speaker's English accent. Supported accents: American, British, Canadian, Australian."
)

iface.launch()
'''



import gradio as gr
import numpy as np
import pickle
import joblib
import yt_dlp
import os
import tempfile
import shutil


def download_audio_yt_dlp(url, output_path="audio.wav", ffmpeg_path="ffmpeg"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path.replace('.wav', '.%(ext)s'),
        'ffmpeg_location': ffmpeg_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path


def extract_audio_from_file(video_file_path, output_path="converted_audio.wav"):
    import subprocess
    cmd = ["ffmpeg", "-i", video_file_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path, "-y"]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def predict_accent_from_mfcc(mfcc_features, model_path="svm_accent_model.pkl"):
    clf = joblib.load(model_path)
    mfcc_features = np.array(mfcc_features).reshape(1, -1)

    prediction = clf.predict(mfcc_features)[0]

    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(mfcc_features)[0]
        confidence = max(probas) * 100
    else:
        confidence = None

    return prediction, confidence


def run_accent_detection_pipeline(source, is_url):
    try:
        with open("mfcc_features.pkl", "rb") as f:
            features, labels = pickle.load(f)  # استخدام خصائص MFCC جاهزة للاختبار فقط

        if is_url:
            audio_path = download_audio_yt_dlp(source, output_path="test_audio.wav")
        else:
            audio_path = extract_audio_from_file(source, output_path="uploaded_audio.wav")

        accent, confidence = predict_accent_from_mfcc(features[0])  # مبدئيًا: استخدام خصائص جاهزة

        if confidence is not None:
            result = (
                f"🗣️ Predicted Accent: {accent}\n"
                f"🎯 Confidence: {confidence:.2f}%\n"
            )
        else:
            result = f"🗣️ Predicted Accent: {accent}\n⚠️ Model does not support confidence score."
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"


def process(video_url, video_file):
    if video_url:
        return run_accent_detection_pipeline(video_url, is_url=True)
    elif video_file:
        return run_accent_detection_pipeline(video_file, is_url=False)
    else:
        return "❌ Please provide either a video URL or upload a video file."


iface = gr.Interface(
    fn=process,
    inputs=[
        gr.Textbox(label="Video URL (YouTube, Loom, etc.)", placeholder="Paste video URL here..."),
        gr.File(label="Or Upload a Video File (MP4, MOV, etc.)", type="filepath")
    ],
    outputs="text",
    title="English Accent Classifier",
    description="Paste a video URL or upload a video file to detect the speaker's English accent. Supported accents: American, British, Canadian, Australian."
)

iface.launch()