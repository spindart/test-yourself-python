# Transcriber API

This project is a Flask API that allows you to transcribe audio from YouTube videos and generate questions based on the transcription. It utilizes several libraries, including `yt-dlp`, `Vosk`, `transformers`, and `nltk`.

## Features

- Transcription of audio from YouTube videos.
- Generation of questions from the transcription.
- Support for multiple audio formats.

## Prerequisites

Before running the project, ensure you have Python 3.7 or higher installed. You will also need the following libraries:

- Flask
- yt-dlp
- ffmpeg-python
- nltk
- sklearn
- transformers
- vosk
- python-dotenv
- pydub

## Installation

1. Clone the repository:

   ```bash
   git clone <https://github.com/spindart/test-yourself-python>
   cd <test-yourself-python>
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the required Vosk model and place it in the `vosk/models/` directory. You can use the `vosk-model-small-en-us-0.15`.

4. Create a `.env` file in the root of the project and define the `PORT` variable (optional):

   ```plaintext
   PORT=3000
   ```

## Usage

To start the server, run:

```bash
python server.py
```

The API will be available at `http://127.0.0.1:3000/`.

### Endpoints

#### `GET /`

Returns a greeting message.

#### `POST /transcribe`

Transcribes the audio from a YouTube video.

**Request Body:**

```json
{
    "videoUrl": "<VIDEO_URL>"
}
```

**Response:**

```json
{
    "transcription": "<TRANSCRIPTION_OF_AUDIO>"
}
```

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
