import torch
import re
import librosa
import Levenshtein
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

print("Loading model and processor...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# 1. Path to local recording
AUDIO_FILE = "test_audio.wav"

print(f"Loading and resampling local audio: {AUDIO_FILE}...")
# Resample audio to target (16,000 Hz)
speech_array, sampling_rate = librosa.load(AUDIO_FILE, sr=16_000)

# 2. Prepare audio for model
inputs = processor(
    speech_array, 
    sampling_rate=16_000, 
    return_tensors="pt", 
    padding=True
)

print("Running inference...")
with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

# 3. Decode prediction
predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentence = processor.batch_decode(predicted_ids)[0]

def normalize_arabic_text(text):
    # 1. Strip all Arabic diacritics (Tashkeel, including Tanween)
    # The unicode block \u064B to \u065F covers all standard short vowels and markers
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # 2. Normalize all forms of Alef (أ, إ, آ) to a bare Alef (ا)
    text = re.sub(r'[أإآ]', 'ا', text)
    
    # 3. Normalize all forms of Tah Marbuta (ة) to Ha (ه)
    # Handle cases where the model might confuse them at the end of words
    text = re.sub(r'ة', 'ه', text)
    
    # 4. Normalize all forms of Yeh (ى, ي) to a bare Yah (ي)
    text = re.sub(r'ى', 'ي', text)
    
    # 5. Remove any remaining non-Arabic characters (like punctuation or numbers)
    # Keep spaces and basic punctuation
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    
    # 6. Collapse multiple spaces into a single space
    text = re.sub(r'\s+', ' ', text)
    
    # 7. Strip leading/trailing spaces
    text = text.strip()
    
    return text

reference_word = "مرحبا" # Database's gold standard
prediction = normalize_arabic_text(predicted_sentence) # From normalization function

# Calculate Levenshtein distance
distance = Levenshtein.distance(reference_word, prediction)

# Calculate similarity ratio (0.0 to 1.0)
similarity = Levenshtein.ratio(reference_word, prediction)

# Convert raw logits to probabilities
probabilities = F.softmax(logits, dim=-1)

# Get probability of specific characters the model predicted
# Use torch.gather to pluck out the exact probabilities of our predicted_ids
confidence_scores = torch.gather(probabilities, 2, predicted_ids.unsqueeze(-1)).squeeze(-1)

# Remove "blank" or "padding" tokens (Wav2Vec2 uses ID 0 for blanks usually)
# Only average the confidence of the actual spoken letters
valid_scores = confidence_scores[predicted_ids != processor.tokenizer.pad_token_id]

# Calculate average confidence (0 to 1)
average_confidence = valid_scores.mean().item()

# Weighting: 40% Accuracy, 60% Clarity
final_pronunciation_score = (similarity * 0.4) + (average_confidence * 0.6)

print("-" * 50)
print("The AI heard you say:")
print(normalize_arabic_text(predicted_sentence))
print("-" * 50)
print(f"Levenshtein Distance: {distance}")
print(f"Similarity Ratio: {similarity:.2f}")
print(f"Average Confidence: {average_confidence:.2f}")
print(f"Final Grade: {final_pronunciation_score:.2f} / 100")

if final_pronunciation_score > 0.85:
    print("Feedback: Excellent pronunciation!")
elif final_pronunciation_score > 0.60:
    print("Feedback: Good, but try to enunciate more clearly.")
else:
    print("Feedback: Incorrect. Please try again.")