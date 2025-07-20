import gradio as gr
import openai
import tempfile
import soundfile as sf
import traceback
import os
import json

openai.api_key = "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"

if not openai.api_key:
    print("Please set it or replace this line with your key.")


def transcribe_audio(audio):
    if audio is None:
        return "Error: No audio input received."
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        # Ensure the temporary file is closed before trying to open it again
        tmp.close()
        try:
            sf.write(tmp.name, audio[1], audio[0])  # audio[0]-sampling rate, audio[1]-data
            with open(tmp.name, "rb") as f:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1", # Using whisper-1 as it's the standard transcription model
                    file=f
                )
                return transcript.text
        finally:
            # Clean up the temporary file
            os.remove(tmp.name)


def generate_medical_summary(transcript_text):
    if not transcript_text:
        return "Error: No transcription available to summarize."
    prompt = f"""You are a professional medical assistant. Based on the following transcript of a medical consultation, generate a clear and concise report with the following two sections:

    1. Summary: Provide a professional summary of the consultation, including main symptoms, patient complaints, and condition descriptions. This also includes any possible diagnosis or observations made during the consultation.
    2. Treatment: List treatment suggestions or follow-up recommendations based on the transcript.

    Ensure that each section has a bolded title (Summary, Treatment). Be accurate, structured, and use a professional tone.

    Transcript:
    {transcript_text}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred during summary generation: {e}")
        traceback.print_exc()
        return f"Error: {e}"

def generate_dental_treatment_json(transcript_text):
    if not transcript_text:
        return "Error: No transcription available to generate JSON."

    prompt = f"""
    You are a professional dental assistant. Extract dental treatment information from the transcript below.

    For each tooth mentioned, output a JSON object where the key is the tooth number (e.g., "3") and the value is its condition (e.g., "cavity", "filled", "missing", etc.). If a tooth is not mentioned, do not include it.

    Use "cavity" instead of "caries".

    Example:
    Transcript: My third tooth has a cavity.
    Expected JSON: {{"3": "cavity"}}

    Transcript:
    {transcript_text}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        # Try to extract the JSON from the response
        import re
        import json
        match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if match:
            return json.dumps(json.loads(match.group()), indent=2)
        else:
            return "Error: No JSON found in response."
    except Exception as e:
        print(f"An error occurred during JSON generation: {e}")
        traceback.print_exc()
        return f"Error: {e}"


transcription_text = ""

def do_transcription(audio):
    global transcription_text
    try:
        transcription_text = transcribe_audio(audio)
        return transcription_text
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        traceback.print_exc()
        return f"Error: {e}"


def do_medical_summary():
    try:
        return generate_medical_summary(transcription_text)
    except Exception as e:
        print(f"An error occurred during medical summary generation: {e}")
        traceback.print_exc()
        return f"Error: {e}"

def do_dental_json():
    try:
        return generate_dental_treatment_json(transcription_text)
    except Exception as e:
        print(f"An error occurred during dental JSON generation: {e}")
        traceback.print_exc()
        return f"Error: {e}"


with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>Doctor Assistant</h1>")
    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="numpy", label="🎙️ Click to Record")
        output_textbox = gr.Textbox(label="📝 Transcription", lines=10)

    with gr.Row():
        transcribe_btn = gr.Button("🎧 Transcribe")
        summary_btn = gr.Button("🩺 Generate Medical Summary")
        dental_json_btn = gr.Button("🦷 Generate Dental Treatment JSON") # New button

    summary_output = gr.Textbox(label="📄 Medical Summary", lines=10)
    dental_json_output = gr.Textbox(label="🦷 Dental Treatment JSON", lines=10) # New output textbox

    transcribe_btn.click(fn=do_transcription, inputs=mic, outputs=output_textbox)
    summary_btn.click(fn=do_medical_summary, outputs=summary_output)
    dental_json_btn.click(fn=do_dental_json, outputs=dental_json_output) # Link new button to new function

# Set share=True to get a public link when running in environments like Colab or Hugging Face Spaces
demo.launch(share=True)