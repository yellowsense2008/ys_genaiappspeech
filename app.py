from flask import Flask, render_template, request, send_file, jsonify
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from googletrans import Translator
from gtts import gTTS
from io import BytesIO
from playsound import playsound
import os
import re

app = Flask(__name__)

a = "sk-wLQRwTr8RS"
b = "Z1ph7bIlovT3B"
c = "lbkFJe49s56n"
d = "KfeWgts8MpMQC"
oak = a + b + c + d

directory = './datatset'

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)

def split_docs(documents, chunk_size=800, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embeddings)

os.environ["OPENAI_API_KEY"] = oak

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)

chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

translator = Translator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form.get('query')
    selected_language = request.form.get('selected_language')

    print(f"Selected language: {selected_language}")  # Add this line to check the selected language

    translated_query = translate_to_english(user_query)

    ranking_prompt = ".rank by descending order of rating and dont show their phone number. Make sure you dont mention that you have omitted their phone number. "
    user_query_with_ranking = translated_query + ranking_prompt

    matching_docs = db.similarity_search(user_query_with_ranking)

    answer = chain.run(input_documents=matching_docs, question=user_query_with_ranking)

    processed_answer = process_answer(answer)

    # Translate the processed answer to the selected language
    translated_processed_answer = translate_to_selected_language(processed_answer, selected_language)

    return render_template('result.html', processed_answer=processed_answer, translated_processed_answer=translated_processed_answer, selected_language=selected_language)


@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        selected_language = data.get('selected_language', '')  # Update key name here

        if not text or not selected_language:
            return jsonify({"error": "Invalid request"}), 400

        # Translate the text to the specified language
        translation = translator.translate(text, dest=selected_language)
        translated_text = translation.text

        return jsonify({"translatedText": translated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def translate_to_english(text):
    try:
        translation = translator.translate(text, src='auto', dest='en')
        return translation.text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def translate_to_selected_language(text, selected_language):
    try:
        if selected_language:
            print(f"Translating to language code: {selected_language}")
            chunk_size = 500
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            translated_chunks = [translator.translate(chunk, dest=selected_language).text for chunk in chunks]
            translated_text = ''.join(translated_chunks)
            return translated_text
    except Exception as e:
        print(f"Translation error: {str(e)}")
    return text



def process_answer(answer):
    lines = answer.split('\n')
    processed_lines = [line for line in lines if "Phone Number:" not in line]
    processed_answer = '<br>'.join(processed_lines)
    return processed_answer

def strip_html_tags(text):
    # Use regular expression to remove HTML tags
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text
    
@app.route('/play_speech')
def play_speech():
    # Retrieve parameters from the URL
    translated_processed_answer = request.args.get('translated_processed_answer', '')
    selected_language = request.args.get('selected_language', '')

    # Strip HTML tags from the translated answer
    cleaned_text = strip_html_tags(translated_processed_answer)

    # Generate speech audio for the cleaned text
    speech_text = cleaned_text
    language = selected_language if selected_language else 'en'

    if speech_text:
        tts = gTTS(text=speech_text, lang=language, slow=False)
        audio_stream = BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)

        # Save the audio stream to a file
        audio_path = 'static/audio/speech.mp3'
        with open(audio_path, 'wb') as audio_file:
            audio_file.write(audio_stream.read())

        # Render the template with the audio URL
        return render_template('result.html', audio_url=audio_path)
    else:
        # Handle the case where there is no text to speak
        return "No text to speak"


if __name__ == '__main__':
    app.run(debug=True)
