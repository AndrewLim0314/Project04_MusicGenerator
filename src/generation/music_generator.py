import torch
import numpy as np
from mido import Message, MidiFile, MidiTrack
from src.models.model import MusicNet

# Configurations
MODEL_PATH = "/Users/25lim/Project04_MusicGenerator/src/results/music_model_v2.pth"
MIDI_OUTPUT_PATH = "/Users/25lim/Project04_MusicGenerator/src/results/generated_song_v2.mid"
SEQ_LENGTH = 50  # Must match the seq_length during training
NUM_NOTES_TO_GENERATE = 100  # Number of notes to generate

# Load the trained model
def load_model(model_path, input_dim, output_dim):
    model = MusicNet(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

# Generate music
def generate_music(model, seed_sequence, num_notes, seq_length):
    generated_notes = seed_sequence[:]
    for _ in range(num_notes):
        # Prepare the input sequence
        input_tensor = torch.tensor(generated_notes[-seq_length:], dtype=torch.float32).unsqueeze(0)

        # Predict the next note
        with torch.no_grad():
            output = model(input_tensor)
            next_note = torch.argmax(output, dim=1).item()

        # Append the predicted note
        generated_notes.append(next_note)

    return generated_notes

# Convert notes to a MIDI file
def save_to_midi(generated_notes, output_path):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    for note in generated_notes:
        track.append(Message('note_on', note=note, velocity=64, time=200))
        track.append(Message('note_off', note=note, velocity=64, time=300))

    midi.save(output_path)
    print(f"Generated MIDI saved to {output_path}")

if __name__ == "__main__":
    # Load the model
    input_dim = SEQ_LENGTH  # Input dimension must match training
    output_dim = 87  # Set to the number of unique notes during training
    model = load_model(MODEL_PATH, input_dim, output_dim)

    # Define a seed sequence (use a predefined or random sequence)
    seed_sequence = np.random.randint(0, output_dim, size=SEQ_LENGTH).tolist()
    print(f"Seed sequence: {seed_sequence}")

    # Generate music
    generated_notes = generate_music(model, seed_sequence, NUM_NOTES_TO_GENERATE, SEQ_LENGTH)

    # Save to MIDI
    save_to_midi(generated_notes, MIDI_OUTPUT_PATH)
