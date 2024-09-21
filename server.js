import express from 'express';
import ytdl from 'ytdl-core';
import ffmpeg from 'fluent-ffmpeg';
import { PassThrough } from 'stream';
import natural from 'natural';
import { HfInference } from '@huggingface/inference';
import vosk from 'vosk-browser';
import dotenv from 'dotenv';

const hf = process.env.HF_API_KEY;

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

const tokenizer = new natural.WordTokenizer();
const stemmer = natural.PorterStemmer;
import stopwords from './config/stopwords.js';
import { FEEDBACK_THRESHOLDS, VOSK_MODEL_PATH } from './config/constants.js';


app.use(express.json());

app.get('/', (req, res) => {
    res.send('Hello World!');
});

app.listen(port, () => {
    console.log(`Servidor rodando na porta ${port}`);
});

app.post('/transcribe', async (req, res) => {
    const { videoUrl } = req.body;
    console.time('total');

    if (!ytdl.validateURL(videoUrl)) {
        console.timeEnd('total');
        return res.status(400).json({ error: 'URL de vídeo inválida' });
    }

    try {
        console.log('iniciando download...');
        console.time('download');
        const info = await ytdl.getInfo(videoUrl);
        const format = ytdl.chooseFormat(info.formats, { quality: 'highestaudio' });

        if (!format) {
            console.timeEnd('download');
            console.timeEnd('total');
            return res.status(400).json({ error: 'Não foi possível encontrar um formato de áudio adequado' });
        }

        const audioStream = ytdl.downloadFromInfo(info, { format });
        const bufferStream = new PassThrough();
        console.timeEnd('download');
        console.log('iniciando processamento de áudio...');
        console.time('audio_processing');
        // Usando um buffer menor para otimizar o uso de memória e permitir que o ffmpeg inicie mais rápido
        ffmpeg(audioStream)
            .toFormat('wav')
            .audioFrequency(16000)
            .pipe(bufferStream, { end: true });

        let audioBuffer = [];
        bufferStream.on('data', (chunk) => {
            audioBuffer.push(chunk);
        });

        bufferStream.on('end', async () => {
            audioBuffer = Buffer.concat(audioBuffer);
            console.timeEnd('audio_processing');

            try {
                console.time('transcription');
                const model = new vosk.Model(process.env.VOSK_MODEL_PATH || VOSK_MODEL_PATH);
                const recognizer = new vosk.Recognizer({ model, sampleRate: 16000 });

                recognizer.acceptWaveform(audioBuffer);
                const result = recognizer.finalResult();
                const transcription = result.text;
                console.timeEnd('transcription');

                console.time('question_generation');
                const questions = await generateQuestions(transcription);
                console.timeEnd('question_generation');

                res.json({ transcription, questions });
                console.timeEnd('total');
            } catch (error) {
                console.error('Erro na transcrição ou geração de perguntas:', error);
                res.status(500).json({ error: 'Erro ao processar o áudio' });
                console.timeEnd('total');
            }
        });

        bufferStream.on('error', (error) => {
            console.error('Erro no stream de áudio:', error);
            res.status(500).json({ error: 'Erro ao processar o stream de áudio' });
            console.timeEnd('total');
        });

    } catch (error) {
        console.error('Erro ao obter informações do vídeo:', error);
        res.status(500).json({ error: 'Erro ao processar o vídeo', details: error.message });
        console.timeEnd('total');
    }
});

app.post('/answer', async (req, res) => {
    const { question, userAnswer, context } = req.body;

    try {
        const feedback = await evaluateAnswer(question, userAnswer, context);
        res.json({ feedback });
    } catch (error) {
        console.error('Erro ao avaliar resposta:', error);
        res.status(500).json({ error: 'Erro ao processar a resposta' });
    }
});

async function generateQuestions(text) {
    try {
        const tokens = tokenizer.tokenize(text);
        const filteredTokens = tokens.filter(token => !stopwords.includes(token.toLowerCase()));

        const frequency = {};
        filteredTokens.forEach(token => {
            frequency[token] = (frequency[token] || 0) + 1;
        });

        const keywords = Object.entries(frequency)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(entry => entry[0]);

        const questionPromises = keywords.map(keyword => generateQuestionForKeyword(keyword, text));
        const questions = await Promise.all(questionPromises);

        return questions.filter(question => question !== null);
    } catch (error) {
        console.error('Erro ao gerar perguntas:', error);
        return [];
    }
}

async function generateQuestionForKeyword(keyword, context) {
    try {
        const generator = await HfInference('text2text-generation', 'unicamp-dl/ptt5-base-portuguese-vocab');
        const prompt = `Gere uma pergunta em português usando a palavra-chave "${keyword}" no seguinte contexto: "${context}"`;
        const result = await generator(prompt, { max_length: 50 });
        return result[0].generated_text.trim();
    } catch (error) {
        console.error('Erro ao gerar pergunta:', error);
        return null;
    }
}

async function evaluateAnswer(question, userAnswer, context) {
    try {
        const contextTokens = tokenizeAndStem(context);
        const userAnswerTokens = tokenizeAndStem(userAnswer);

        const tfidf = new natural.TfIdf();
        tfidf.addDocument(contextTokens);
        const relevanceScore = calculateRelevance(tfidf, userAnswerTokens);

        const similarityScore = calculateJaccardSimilarity(new Set(contextTokens), new Set(userAnswerTokens));

        const feedback = await generateFeedback(relevanceScore, similarityScore, question, userAnswer, context);

        return feedback;
    } catch (error) {
        console.error('Erro ao avaliar resposta:', error);
        return "Desculpe, não foi possível avaliar a resposta no momento.";
    }
}

function tokenizeAndStem(text) {
    return tokenizer.tokenize(text.toLowerCase()).map(token => stemmer.stem(token));
}

function calculateRelevance(tfidf, tokens) {
    let totalScore = 0;
    tokens.forEach(token => {
        totalScore += tfidf.tfidf(token, 0);
    });
    return totalScore / tokens.length;
}

function calculateJaccardSimilarity(set1, set2) {
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    return intersection.size / union.size;
}

async function generateFeedback(relevanceScore, similarityScore, question, userAnswer, context) {
    const overallScore = (relevanceScore + similarityScore) / 2;
    let feedbackText = '';

    if (overallScore > FEEDBACK_THRESHOLDS.HIGH) {
        feedbackText = 'Sua resposta está muito boa e relevante. Parabéns!';
    } else if (overallScore > FEEDBACK_THRESHOLDS.MEDIUM) {
        feedbackText = 'Sua resposta está parcialmente correta. Há espaço para melhorias.';
    } else {
        feedbackText = 'Sua resposta precisa de mais trabalho. Tente revisar o conteúdo novamente.';
    }

    const detailedFeedback = await generateDetailedFeedback(question, userAnswer, context, overallScore);

    return `${feedbackText}\n\nFeedback detalhado: ${detailedFeedback}`;
}

async function generateDetailedFeedback(question, userAnswer, context, overallScore) {
    try {
        const generator = await HfInference('text2text-generation', 'unicamp-dl/ptt5-base-portuguese-vocab');
        const prompt = `
        Pergunta: "${question}"
        Resposta do usuário: "${userAnswer}"
        Contexto: "${context}"
        Pontuação geral: ${overallScore.toFixed(2)}
        
        Forneça um feedback detalhado e construtivo em português. Explique por que a resposta está correta, parcialmente correta ou incorreta. Sugira melhorias específicas e indique quais informações do contexto poderiam ter sido incluídas para uma resposta mais completa.
        `;
        const result = await generator(prompt, { max_length: 200 });
        return result[0].generated_text.trim();
    } catch (error) {
        console.error('Erro ao gerar feedback detalhado:', error);
        return "Não foi possível gerar um feedback detalhado no momento.";
    }
}
