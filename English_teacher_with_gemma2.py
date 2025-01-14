import gradio as gr
import torch
import torchaudio
import scipy.io.wavfile
from transformers import AutoProcessor, SeamlessM4Tv2Model
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain.memory import ConversationBufferWindowMemory
import os

# Set up Google API key
#os.environ['GOOGLE_API_KEY'] = 'YOUR_GOOGLE_API_KEY_HERE'

# Initialize LLM
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
llm = OllamaLLM(model="gemma2")
# Set up prompt and conversation
prompt_system = '''Act as an English teacher. Your job is to teach and provide practical exercises.
Always respond in English and remember that you're teaching someone who speaks Spanish. Be concise and precise in your response.
dont use markdown or emojis, just plain text.
'''

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(prompt_system),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=3)
conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Set up Seamless model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to(device)

AUDIO_SAMPLE_RATE = model.config.sampling_rate
MAX_INPUT_AUDIO_LENGTH = 60

def preprocess_audio(input_audio: str):
    arr, org_sr = torchaudio.load(input_audio)
    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
        gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
    audio_inputs = processor(audios=new_arr, return_tensors="pt").to(device)
    return audio_inputs

def speech_to_text(audio_inputs):
    output_tokens = model.generate(**audio_inputs, tgt_lang='eng', generate_speech=False)
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return translated_text_from_audio

def generate_llm_response(transcribed_text, messages):
    response = conversation(transcribed_text)
    messages.extend(['User: ' + transcribed_text, 'AI: ' + response['text']])
    chat_transcription = "\n".join(messages)
    return chat_transcription, response['text']

def text_to_speech(text_input):
    text_inputs = processor(text=text_input, src_lang=["eng"], return_tensors="pt").to(device)
    spanish_audio = model.generate(**text_inputs, tgt_lang="spa")[0].cpu().numpy().squeeze()
    english_audio = model.generate(**text_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()

    scipy.io.wavfile.write("spanish_audio.wav", rate=AUDIO_SAMPLE_RATE, data=spanish_audio)
    scipy.io.wavfile.write("english_audio.wav", rate=AUDIO_SAMPLE_RATE, data=english_audio)

    return './spanish_audio.wav', './english_audio.wav'

def text_to_text(text_input):
    text_inputs = processor(text=text_input, src_lang=["eng"], return_tensors="pt").to(device)
    output_tokens = model.generate(**text_inputs, tgt_lang="spa", generate_speech=False)
    translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return translated_text_from_text

messages = []

def process_interaction(audio):
    global messages
    arr_audio = preprocess_audio(audio)
    query_input = speech_to_text(arr_audio)
    llm_response, last_response = generate_llm_response(query_input, messages)
    spanish_output, english_output = text_to_speech(last_response)
    spanish_text = text_to_text(last_response)
    return llm_response, spanish_output, english_output, spanish_text

# Gradio interface
iface = gr.Interface(
    fn=process_interaction,
    inputs=gr.Audio(type="filepath", label="Record or upload audio"),
    outputs=[
        gr.Textbox(label="Conversation History"),
        gr.Audio(label="Spanish Audio Response", autoplay=False),
        gr.Audio(label="English Audio Response", autoplay=False),
        gr.Textbox(label="Spanish Text Translation"),
    ],
    title="AI-Powered English Tutor",
    description="Speak in Spanish or English to interact with your AI English tutor.",
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch(share=True)