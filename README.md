

brew install llvm


https://github.com/idiap/coqui-ai-TTS

# Treinando YourTTS

uv run python -m vecl.yourtts.train_yourtts_custom

# Step 1: Gather and Prepare Pre-trained Components

- Since you've already set up your environment, let's focus on collecting the key pre-trained components you'll need:

- YourTTS Base Model: You can use the pre-trained YourTTS model from Coqui TTS o

- Speaker Encoder: from TTS.tts.utils.speakers import SpeakerManager

- Emotion Encoder: "alefiury/wav2vec2-xls-r-300m-pt-br-spontaneous-speech-emotion-recognition"

- Self-Supervised Speech Representations: Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

The paper uses Wav2Vec2 for content loss
Download a pre-trained Wav2Vec2 model for both English and Portuguese if available

Step 2: Data Collection and Preparation
You'll need data for both languages:

English Data:

VCTK dataset (mentioned in the paper)
LJSpeech for additional single speaker data


Brazilian Portuguese Data:

Consider using TTS-Portuguese-Corpus on GitHub
CETUC dataset if you can access it
M-AILABS Portuguese dataset


Emotional Speech Data:

Look for Portuguese emotional speech datasets
If unavailable, consider creating a small dataset with a native speaker
For English, use RAVDESS or EmoV-DB


Data Preprocessing:

Normalize audio (loudness normalization)
Remove silences
Prepare text normalization for both languages



Step 3: Adapt YourTTS for Cross-Lingual Transfer

Language Embeddings:

Create 2-dimensional language embeddings (one for English, one for Portuguese)
Integrate these with the text encoder


Text Processing:

Implement text normalization for Brazilian Portuguese
Decide whether to use graphemes or phonemes (paper mentions using raw text)
Create a simple mapping between Portuguese and English phonology if needed



Step 4: Implement the Proposed VECL-TTS Extensions

Multi-Style Controlling Block:

Integrate the emotion encoder with the base YourTTS architecture
Condition the encoder, decoder, and duration predictor with emotion embeddings


Loss Functions:

Implement Emotion Consistency Loss (ECL) as described in the paper
Implement Speaker Consistency Loss (SCL)
Implement Content Loss using Wav2Vec2 embeddings
Combine these with the existing MSE and KL divergence losses



Step 5: Training Strategy
Since you have computation constraints:

Incremental Training:

Start with a pre-trained YourTTS model
Fine-tune first on monolingual Portuguese data
Then train for cross-lingual capabilities
Finally, add emotion transfer


Reduced Model Size:

Consider using a smaller model configuration if needed
Reduce transformer layers if necessary
Start with fewer emotions to simplify the task


Efficient Training:

Use mixed precision training
Start with smaller batches
Implement gradient accumulation if needed



Step 6: Evaluation Framework

Objective Metrics:

Speaker similarity using cosine distance
Emotion similarity using cosine distance
MCD (Mel Cepstral Distortion) for speech quality


Subjective Evaluation:

Simple MOS tests for naturalness
ABX tests for emotion and speaker similarity



Prioritized Implementation Plan
Given your constraints, here's a prioritized implementation plan:

Week 1-2: Base System Setup

Get YourTTS running with pre-trained weights
Test it on English and Portuguese text (separately)
Set up data pipelines for both languages


Week 3-4: Add Speaker Encoder

Integrate a pre-trained speaker encoder
Implement speaker similarity loss
Test cross-lingual voice cloning without emotion


Week 5-6: Emotion Integration

Add the emotion encoder
Implement emotion consistency loss
Test with basic emotions (neutral, happy, sad)


Week 7-8: Content Loss and Refinement

Add Wav2Vec2-based content loss
Fine-tune the complete system
Conduct evaluations



Recommended Pre-trained Models

For YourTTS Base:

Use the multi-speaker YourTTS model from Coqui-TTS


For Speaker Encoder:

SpeechBrain's ECAPA-TDNN model pre-trained on VoxCeleb


For Emotion Encoder:

HuggingFace's Wav2Vec2 model fine-tuned for emotion recognition
Or use a simpler model like SER-FT if computation is limited


For Content Loss:

Facebook's Wav2Vec2-large-xlsr-53 (supports multiple languages including Portuguese)



By using these pre-trained components, you can significantly reduce the training time and computational requirements while still building a functional VECL-TTS system for Brazilian Portuguese and English.
Would you like me to elaborate on any specific part of this roadmap?