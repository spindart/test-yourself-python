from flask import Flask, request, jsonify
import yt_dlp
import ffmpeg
import io
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import vosk
import os
from dotenv import load_dotenv
import time

load_dotenv()

app = Flask(__name__)
port = int(os.getenv('PORT', 3000))

nltk.download('punkt')
tokenizer = nltk.tokenize.word_tokenize
stemmer = PorterStemmer()

with open('config/stopwords.txt', 'r') as f:
    stopwords = set(f.read().splitlines())

FEEDBACK_THRESHOLDS = {
    'HIGH': 0.7,
    'MEDIUM': 0.4
}

VOSK_MODEL_PATH = os.getenv('VOSK_MODEL_PATH', 'path/to/vosk/model')

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
        print('iniciando download...')
        download_start = time.time()
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': 'audio.%(ext)s'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print(f'download: {time.time() - download_start:.3f}s')

        print('iniciando processamento de áudio...')
        audio_processing_start = time.time()
        stream = ffmpeg.input('audio.wav')
        stream = ffmpeg.output(stream, 'pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16k')
        out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        print(f'audio_processing: {time.time() - audio_processing_start:.3f}s')

        print('iniciando transcrição...')
        transcription_start = time.time()
        model = vosk.Model(VOSK_MODEL_PATH)
        rec = vosk.KaldiRecognizer(model, 16000)
        rec.AcceptWaveform(out)
        result = rec.FinalResult()
        transcription = result['text']
        print(f'transcription: {time.time() - transcription_start:.3f}s')

        print('gerando perguntas...')
        question_generation_start = time.time()
        questions = generate_questions(transcription)
        print(f'question_generation: {time.time() - question_generation_start:.3f}s')

        os.remove('audio.wav')
        print(f'total: {time.time() - start_time:.3f}s')
        return jsonify({'transcription': transcription, 'questions': questions})

    except Exception as e:
        print(f'Erro ao processar o vídeo: {str(e)}')
        return jsonify({'error': 'Erro ao processar o vídeo', 'details': str(e)}), 500

@app.route('/answer', methods=['POST'])
def answer():
    data = request.json
    question = data['question']
    user_answer = data['userAnswer']
    context = data['context']

    try:
        feedback = evaluate_answer(question, user_answer, context)
        return jsonify({'feedback': feedback})
    except Exception as e:
        print(f'Erro ao avaliar resposta: {str(e)}')
        return jsonify({'error': 'Erro ao processar a resposta'}), 500

def generate_questions(text):
    try:
        tokens = tokenizer(text.lower())
        filtered_tokens = [token for token in tokens if token not in stopwords]

        frequency = {}
        for token in filtered_tokens:
            frequency[token] = frequency.get(token, 0) + 1

        keywords = sorted(frequency, key=frequency.get, reverse=True)[:5]

        questions = [generate_question_for_keyword(keyword, text) for keyword in keywords]
        return [q for q in questions if q is not None]
    except Exception as e:
        print(f'Erro ao gerar perguntas: {str(e)}')
        return []

def generate_question_for_keyword(keyword, context):
    try:
        generator = pipeline('text2text-generation', model='unicamp-dl/ptt5-base-portuguese-vocab')
        prompt = f'Gere uma pergunta em português usando a palavra-chave "{keyword}" no seguinte contexto: "{context}"'
        result = generator(prompt, max_length=50)
        return result[0]['generated_text'].strip()
    except Exception as e:
        print(f'Erro ao gerar pergunta: {str(e)}')
        return None

def evaluate_answer(question, user_answer, context):
    try:
        context_tokens = tokenize_and_stem(context)
        user_answer_tokens = tokenize_and_stem(user_answer)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([' '.join(context_tokens), ' '.join(user_answer_tokens)])
        relevance_score = tfidf_matrix[0].dot(tfidf_matrix[1].T).toarray()[0][0]

        similarity_score = jaccard_similarity(set(context_tokens), set(user_answer_tokens))

        feedback = generate_feedback(relevance_score, similarity_score, question, user_answer, context)

        return feedback
    except Exception as e:
        print(f'Erro ao avaliar resposta: {str(e)}')
        return "Desculpe, não foi possível avaliar a resposta no momento."

def tokenize_and_stem(text):
    return [stemmer.stem(token) for token in tokenizer(text.lower())]

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def generate_feedback(relevance_score, similarity_score, question, user_answer, context):
    overall_score = (relevance_score + similarity_score) / 2

    if overall_score > FEEDBACK_THRESHOLDS['HIGH']:
        feedback_text = 'Sua resposta está muito boa e relevante. Parabéns!'
    elif overall_score > FEEDBACK_THRESHOLDS['MEDIUM']:
        feedback_text = 'Sua resposta está parcialmente correta. Há espaço para melhorias.'
    else:
        feedback_text = 'Sua resposta precisa de mais trabalho. Tente revisar o conteúdo novamente.'

    detailed_feedback = generate_detailed_feedback(question, user_answer, context, overall_score)

    return f'{feedback_text}\n\nFeedback detalhado: {detailed_feedback}'

def generate_detailed_feedback(question, user_answer, context, overall_score):
    try:
        generator = pipeline('text2text-generation', model='unicamp-dl/ptt5-base-portuguese-vocab')
        prompt = f'''
        Pergunta: "{question}"
        Resposta do usuário: "{user_answer}"
        Contexto: "{context}"
        Pontuação geral: {overall_score:.2f}
        
        Forneça um feedback detalhado e construtivo em português. Explique por que a resposta está correta, parcialmente correta ou incorreta. Sugira melhorias específicas e indique quais informações do contexto poderiam ter sido incluídas para uma resposta mais completa.
        '''
        result = generator(prompt, max_length=200)
        return result[0]['generated_text'].strip()
    except Exception as e:
        print(f'Erro ao gerar feedback detalhado: {str(e)}')
        return "Não foi possível gerar um feedback detalhado no momento."

if __name__ == '__main__':
    app.run(port=port)