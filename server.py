from flask import Flask, request, jsonify
import yt_dlp
import ffmpeg
import io
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForQuestionAnswering
import vosk
import os
from dotenv import load_dotenv
import time
import json
import random
from nltk.corpus import stopwords
from nltk import FreqDist
import torch
from pydub import AudioSegment

load_dotenv()

app = Flask(__name__)
port = int(os.getenv('PORT', 3000))

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')  # Adicione esta linha
nltk.download('stopwords')
tokenizer = nltk.tokenize.word_tokenize
stemmer = PorterStemmer()

FEEDBACK_THRESHOLDS = {
    'HIGH': 0.7,
    'MEDIUM': 0.4
}

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
        
        # Carregar o áudio usando pydub
        audio = AudioSegment.from_wav("audio.wav")
        
        # Converter para o formato correto para o Vosk (16kHz, 16-bit, mono)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        print(f'audio_processing: {time.time() - audio_processing_start:.3f}s')

        print('iniciando transcrição...')
        transcription_start = time.time()
        model = vosk.Model(VOSK_MODEL_PATH)
        rec = vosk.KaldiRecognizer(model, 16000)

        # Processar o áudio em chunks
        chunk_size = 4000  # 250ms
        full_transcription = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
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
        print(f'Tamanho da transcrição: {len(transcription)} caracteres')

        if not transcription.strip():
            print("Aviso: A transcrição está vazia. Resultado completo:", final_result)
            raise ValueError("A transcrição está vazia")

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

def analyze_content(transcription):
    # Tokenização e remoção de stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(transcription.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    # Análise de frequência de palavras
    freq_dist = FreqDist(filtered_words)
    
    # Extração de frases-chave usando as palavras mais frequentes
    most_common = freq_dist.most_common(20)  # Aumentado de 10 para 20
    key_phrases = []
    for word, _ in most_common:
        for sentence in sent_tokenize(transcription):
            if word in sentence.lower():
                key_phrases.append(sentence)
                break
    
    return key_phrases

def generate_questions(transcription):
    try:
        print("Iniciando geração de questões...")
        
        # Identificar tópicos importantes usando palavras-chave e frases frequentes
        important_topics = identify_important_topics(transcription)
        if not important_topics:
            print("Nenhum tópico importante identificado.")
            return []

        model_name = "valhalla/t5-small-qg-hl"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

        questions = []

        # Gerar perguntas sobre tópicos importantes
        for topic in important_topics:
            input_text = f"generate question about {topic} in the context of the entire transcription: {transcription[:1000]}"
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

            outputs = model.generate(
                input_ids,
                max_length=64,
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
                questions.append({"question": question.strip(), "context": transcription[:1000]})

        # Gerar perguntas gerais sobre a transcrição
        input_text = f"generate general questions about the main ideas in: {transcription[:1000]}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

        outputs = model.generate(
            input_ids,
            max_length=64,
            num_return_sequences=4,
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
            questions.append({"question": question.strip(), "context": transcription[:1000]})

        print(f"Geradas {len(questions)} questões iniciais")

        # Remover questões similares e irrelevantes
        unique_questions = []
        for q in questions:
            if not any(similar_questions(q['question'], uq['question']) for uq in unique_questions) and is_relevant_question(q['question'], q['context']):
                unique_questions.append(q)

        print(f"Ao remover similares e irrelevantes: {len(unique_questions)} questões")

        # Processar as perguntas finais
        final_questions = post_process_questions(unique_questions)[:10]
        
        print(f"Questões finais: {len(final_questions)}")
        
        return final_questions

    except Exception as e:
        print(f'Erro ao gerar questões: {str(e)}')
        return []

def similar_questions(q1, q2):
    return jaccard_similarity(set(q1.lower().split()), set(q2.lower().split())) > 0.6

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def identify_important_topics(transcription):
    if not transcription.strip():
        print("Transcrição vazia. Nenhum tópico pode ser identificado.")
        return []

    words = word_tokenize(transcription.lower())
    stop_words = set(stopwords.words('english'))
    important_words = [word for word in words if word not in stop_words and len(word) > 3]
    word_freq = FreqDist(important_words)
    
    # Retornar as 5 palavras mais comuns
    return [word for word, freq in word_freq.most_common(5)]

def is_relevant_question(question, context):
    question_words = set(word_tokenize(question.lower()))
    context_words = set(word_tokenize(context.lower()))
    overlap = len(question_words.intersection(context_words))
    
    # Ajuste dinâmico baseado no tamanho do contexto
    return overlap > max(2, len(context_words) * 0.1)  # Exemplo: 10% do contexto

def evaluate_answer(question, user_answer, context):
    try:
        context_tokens = tokenize_and_stem(context)
        user_answer_tokens = tokenize_and_stem(user_answer)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([' '.join(context_tokens), ' '.join(user_answer_tokens)])
        relevance_score = tfidf_matrix[0].dot(tfidf_matrix[1].T).toarray()[0][0]

        similarity_score = jaccard_similarity(set(context_tokens), set(user_answer_tokens))

        overall_score = (relevance_score + similarity_score) / 2

        feedback = generate_detailed_feedback(question, user_answer, context, overall_score)

        return feedback
    except Exception as e:
        print(f'Erro ao avaliar resposta: {str(e)}')
        return "Desculpe, não foi possível avaliar a resposta no momento."

def tokenize_and_stem(text):
    return [stemmer.stem(token) for token in tokenizer(text.lower())]


def generate_detailed_feedback(question, user_answer, context, overall_score):
    try:
        model_name = "valhalla/t5-small-qg-hl"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

        input_text = f"generate answer: {question} context: {context}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            input_ids, 
            max_length=128, 
            num_return_sequences=1, 
            num_beams=4,
            do_sample=True,
            temperature=0.7,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        feedback = f"Question: {question}\n\n"
        feedback += f"Your answer: {user_answer}\n\n"
        feedback += f"Model answer: {model_answer}\n\n"

        # Compare user answer with model answer
        user_keywords = set(tokenize_and_stem(user_answer))
        model_keywords = set(tokenize_and_stem(model_answer))
        common_keywords = user_keywords.intersection(model_keywords)
        missing_keywords = model_keywords - user_keywords

        if overall_score > 0.8:
            feedback += "Excellent! Your answer is very close to the model answer. "
            if missing_keywords:
                feedback += f"To improve, consider including: {', '.join(missing_keywords)}."
        elif overall_score > 0.5:
            feedback += "Good job! Your answer contains correct elements, but there's room for improvement. "
            feedback += f"You correctly mentioned: {', '.join(common_keywords)}. "
            if missing_keywords:
                feedback += f"Consider adding: {', '.join(missing_keywords)}."
        else:
            feedback += "Your answer needs more work. Try reviewing the content and focusing on the main points. "
            if common_keywords:
                feedback += f"You were right to mention: {', '.join(common_keywords)}. "
            feedback += f"Important elements that were missing: {', '.join(missing_keywords)}."

        # Suggestion for review
        feedback += "\n\nSuggestion for review: "
        key_context_words = set(tokenize_and_stem(context)) - set(stopwords.words('english'))
        review_suggestions = list(key_context_words - user_keywords)[:3]
        feedback += f"Review the content related to {', '.join(review_suggestions)}."

        return feedback
    except Exception as e:
        print(f'Error generating detailed feedback: {str(e)}')
        return "Unable to generate detailed feedback at the moment."

def post_process_questions(questions):
    processed_questions = []
    for q in questions:
        question = q['question']
        context = q['context']
        
        # Verificar se a pergunta faz sentido e não é muito genérica
        if len(question.split()) > 5 and not question.lower().startswith(('what is', 'who is', 'when is')):
            processed_questions.append({"question": question, "context": context})
    
    return processed_questions

if __name__ == '__main__':
    app.run(port=port)