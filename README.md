# Arabic Pronunciation Checker

This Python script utilises a pre-trained Arabic speech recognition model to evaluate and score the pronunciation of spoken Arabic words.

## Overview

The script works by:

1.  **Loading Audio:** It loads an audio recording (`test_audio.wav`) and resamples it to 16,000 Hz, which is the required input rate for the model.
2.  **Speech Recognition:** It uses the `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` model from Hugging Face's `transformers` library to transcribe the spoken audio.
3.  **Text Normalisation:** It normalises both the predicted text and the reference word by:
    *   Stripping Arabic diacritics (Tashkeel).
    *   Standardising variations of Alef, Tah Marbuta, and Yah.
    *   Removing non-Arabic characters and extra spaces.
4.  **Evaluation & Scoring:** The script evaluates the pronunciation based on two factors:
    *   **Accuracy:** It calculates the Levenshtein distance and similarity ratio between the normalised prediction and a normalised "gold standard" reference word ("مرحبا" in the current code).
    *   **Clarity:** It calculates the average confidence score of the model's predictions for the valid characters spoken.
5.  **Final Grading:** A final score is generated, weighted 40% on accuracy (similarity) and 60% on clarity (confidence), and feedback is provided based on the score.

## Prerequisites

You need the following libraries installed:

```bash
pip install torch librosa transformers Levenshtein
```

You will also need `ffmpeg` installed on your system for `librosa` to effectively load audio files.

## Usage

1.  Ensure you have an audio file named `test_audio.wav` in the same directory as the script.
2.  Open `main_script.py` and modify the `reference_word` variable to match the word you are expecting to hear in the audio file.
3.  Run the script:

```bash
python main_script.py
```

## How the Scoring Works

The final pronunciation score is a combination of two metrics:

*   **Similarity (40%):** How close the transcribed text is to the reference word (using the Levenshtein ratio).
*   **Confidence (60%):** How confident the AI model was in its transcription. A clear voice leads to higher confidence.

The script then provides feedback:
*   **> 0.85:** Excellent pronunciation!
*   **> 0.60:** Good, but try to enunciate more clearly.
*   **<= 0.60:** Incorrect. Please try again.
