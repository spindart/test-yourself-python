from flask import Flask, request, jsonify
import yt_dlp
import io
import ffmpeg
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import vosk
import os
from dotenv import load_dotenv
import time
import json
from nltk.corpus import stopwords
from nltk import FreqDist
from pydub import AudioSegment

load_dotenv()

app = Flask(__name__)
port = int(os.getenv('PORT', 3000))

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
tokenizer = nltk.tokenize.word_tokenize
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'vosk', 'models', 'vosk-model-small-en-us-0.15')

if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError(f"O modelo Vosk não foi encontrado em {VOSK_MODEL_PATH}")

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/transcribe', methods=['POST'])
def transcribe():
    video_url = request.json['videoUrl']
    start_time = time.time()

    if not yt_dlp.YoutubeDL().extract_info(video_url, download=False):
        return jsonify({'error': 'URL de vídeo inválida'}), 400

    try:
        # Download do áudio do vídeo
        print('Iniciando download...')
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
            'outtmpl': 'audio.%(ext)s'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Processamento de áudio
        audio = AudioSegment.from_wav("audio.wav")
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Transcrição com Vosk
        model = vosk.Model(VOSK_MODEL_PATH)
        rec = vosk.KaldiRecognizer(model, 16000)

        full_transcription = []
        for i in range(0, len(audio), 4000):  # Processa em chunks de 250ms
            chunk = audio[i:i+4000]
            raw_data = chunk.raw_data
            if rec.AcceptWaveform(raw_data):
                result = json.loads(rec.Result())
                if result.get('text'):
                    full_transcription.append(result['text'])

        final_result = json.loads(rec.FinalResult())
        if final_result.get('text'):
            full_transcription.append(final_result['text'])

        transcription = ' '.join(full_transcription)
        print(f'Transcrição completa: {transcription}')
        
        if not transcription.strip():
            raise ValueError("A transcrição está vazia")

        # Geração de perguntas
        questions = generate_questions(transcription)
        os.remove('audio.wav')
        return jsonify({'transcription': transcription, 'questions': questions})

    except Exception as e:
        return jsonify({'error': 'Erro ao processar o vídeo', 'details': str(e)}), 500

def generate_questions(transcription):
    try:
        print("Iniciando geração de questões...")
        chunks = split_transcription(transcription, max_length=300)  # Alterado de chunk_size para max_length
        model_name = "valhalla/t5-small-qg-hl"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

        questions = []
        for chunk in chunks:
            input_text = f"generate question: {chunk}"
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
            outputs = model.generate(
                input_ids,
                max_length=2048,
                num_return_sequences=2,
                num_beams=4,
                do_sample=True,
                temperature=0.7,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            for output in outputs:
                question = tokenizer.decode(output, skip_special_tokens=True)
                if not question.endswith('?'):
                    question += '?'
                questions.append({"question": question, "context": chunk})

        # Filtragem de questões
        unique_questions = []
        for q in questions:
            if not any(similar_questions(q['question'], uq['question']) for uq in unique_questions) and is_relevant_question(q['question'], q['context']):
                unique_questions.append(q)

        final_questions = post_process_questions(unique_questions)
        return questions

    except Exception as e:
        print(f'Erro ao gerar questões: {str(e)}')
        return []

def split_transcription(transcription, max_length=512):
    words = transcription.split()
    segments = []
    while words:
        segment = " ".join(words[:max_length])
        segments.append(segment)
        words = words[max_length:]
    return segments

def preprocess_question(question):
    tokens = word_tokenize(question.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return set(filtered_tokens)

def similar_questions(q1, q2):
    q1_tokens = preprocess_question(q1)
    q2_tokens = preprocess_question(q2)
    return jaccard_similarity(q1_tokens, q2_tokens) > 0.6

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def is_relevant_question(question, context):
    question_words = set(word_tokenize(question.lower()))
    context_words = set(word_tokenize(context.lower()))
    overlap = len(question_words.intersection(context_words))
    return overlap > max(2, len(context_words) * 0.1)

def post_process_questions(questions):
    processed_questions = []
    for q in questions:
        question = q['question']
        context = q['context']
        if len(question.split()) > 5 and not question.lower().startswith(('what is', 'who is', 'when is')):
            processed_questions.append({"question": question, "context": context})
    return processed_questions

if __name__ == '__main__':
    app.run(port=port)
