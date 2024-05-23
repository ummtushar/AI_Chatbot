import PyPDF2
import requests
import json
import gradio as gr
import markdown

url = "http://localhost:11434/api/generate"
headers = {
    'Content-Type': 'application/json',
}

note_taking_instructions = """
Do not start responses with opening words like 'Certainly!' etc. Keep your responses strictly to the content of the pdf. Your responses should only include 
headers for topics and subtopics discussed in the pdf and explanation in bullet points. The goal is to prepare for exams by reading these slides so keep your responses
as similar to the wordings used in slides, that is, avoid paraphrasing or rewording unless absolutely necessary.
Please take notes of the following lecture notes in a detailed manner:
- Identify the main topics and subtopics.
- Provide detailed explanations for each topic and detailed explanation also for subtopic.
- If any topic lacks detail, use your own knowledge to add more information.
- The structure of your response should be exactly as the structure they follow on slide. This is very important.
- Ensure the notes are very detailed and include every single detail that is mentioned on the slides as everything is crucial to understand for the exam.
"""

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def generate_notes(text):
    full_prompt = note_taking_instructions + "\n\n" + text
    data = {
        "model": "llama3",
        "stream": False,
        "prompt": full_prompt,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        return actual_response
    else:
        print("Error:", response.status_code, response.text)
        return "An error occurred while generating the notes."

def enhance_notes(text):
    initial_notes = generate_notes(text)
    additional_info = generate_notes(f"Add more details to: {initial_notes}")
    enhanced_notes = initial_notes + "\n\n" + additional_info
    return enhanced_notes

def process_and_take_notes(pdf_file):
    text = extract_text_from_pdf(pdf_file.name)
    notes = enhance_notes(text)
    # Convert the notes to markdown format
    notes_markdown = markdown.markdown(notes)
    return notes_markdown

iface = gr.Interface(
    fn=process_and_take_notes,
    inputs=gr.File(label="Upload Lecture PDF"),
    outputs=gr.Markdown(),
    title="Lecture Note Taker",
    description="Upload your lecture slides or notes to get detailed notes."
)

iface.launch(share=True)