import torch
import numpy as np
from mido import Message, MidiFile, MidiTrack
from src.models.model_v2 import EnhancedMusicGenerator  # Update model import



# Configurations
MODEL_PATH = "/Users/25lim/Project04_MusicGenerator/src/results/music_model_v3.pth"
MIDI_OUTPUT_PATH = "/Users/25lim/Project04_MusicGenerator/src/results/generated_song_v3.mid"
SEQ_LENGTH = 50  # Must match the seq_length during training
NUM_NOTES_TO_GENERATE = 100  # Number of notes to generate
NUM_DURATIONS = 32  # Number of unique durations (match model's duration_embedding)
NUM_VELOCITIES = 128  # Velocity range (match model's velocity_embedding)
NUM_INTERVALS = 126  # Matches interval_embedding size

# Load the trained model
def load_model(model_path, num_notes, num_durations, num_velocities, num_intervals, output_dim):
    # Use the correct parameters based on the saved model
    hidden_dim = 256  # Match the LSTM hidden size during training
    embed_dim = 64  # Embedding dimension
    num_layers = 3  # Match the number of LSTM layers from the training setup

    # Initialize the model
    model = EnhancedMusicGenerator(
        num_notes=num_notes,
        num_durations=num_durations,
        num_velocities=num_velocities,
        num_intervals=num_intervals,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        output_dim=output_dim,
        num_layers=num_layers
    )

    # Load the state dictionary
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

    # Load model weights (using strict=False if needed for partial matching)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print(f"Model loaded successfully from {model_path}")
    for name, param in model.state_dict().items():
        print(f"{name}: {param.shape}")
    return model


def generate_music(model, seed_notes, seed_durations, seed_velocities, seed_intervals, num_notes, seq_length):
    ...
    # Calculate intervals from seed notes
    seed_intervals = [seed_notes[i + 1] - seed_notes[i] for i in range(len(seed_notes) - 1)]
    seed_intervals = [interval + 65 for interval in seed_intervals]  # Shift up for model input
    seed_intervals = [max(0, min(interval, 125)) for interval in seed_intervals]  # Clamp to valid range

    # Pad intervals to match sequence length
    while len(seed_intervals) < seq_length:
        seed_intervals.insert(0, 65)  # Neutral interval as padding

    # Generated sequences initialized with seed data
    generated_notes = seed_notes[:]
    generated_durations = seed_durations[:]
    generated_velocities = seed_velocities[:]
    generated_intervals = seed_intervals[:]

    for _ in range(num_notes):
        # Prepare input tensors (last seq_length items)
        notes_tensor = torch.tensor([generated_notes[-seq_length:]], dtype=torch.long)
        durations_tensor = torch.tensor([generated_durations[-seq_length:]], dtype=torch.long)
        velocities_tensor = torch.tensor([generated_velocities[-seq_length:]], dtype=torch.long)
        intervals_tensor = torch.tensor([generated_intervals[-seq_length:]], dtype=torch.long)

        # Predict the next note
        with torch.no_grad():
            output = model(notes_tensor, durations_tensor, velocities_tensor, intervals_tensor)
            next_note = torch.argmax(output, dim=1).item()

        # Append the predicted note and constrain other features
        generated_notes.append(next_note)
        generated_durations.append(np.random.randint(1, NUM_DURATIONS))  # Random duration
        generated_velocities.append(np.random.randint(60, 100))  # Realistic velocity range

        # Calculate and append realistic interval
        new_interval = next_note - generated_notes[-2]  # Interval from last note
        new_interval_shifted = new_interval + 65  # Shift for model
        generated_intervals.append(max(0, min(new_interval_shifted, 125)))  # Clamp to valid range

    return generated_notes, generated_durations, generated_velocities, generated_intervals


# Convert generated notes to a MIDI file
def save_to_midi(generated_notes, generated_durations, generated_velocities, generated_intervals, output_path):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    # Denormalize notes: Shift back to MIDI range [21–102]
    denormalized_notes = [note + 21 for note in generated_notes]  # Add 21 to each note
    denormalized_intervals = [interval - 65 for interval in generated_intervals]  # Subtract 65 to restore intervals

    for note, duration, velocity, interval in zip(denormalized_notes, generated_durations, generated_velocities, denormalized_intervals):
        time = int(duration * 100)  # Scale duration to time units
        track.append(Message('note_on', note=note, velocity=velocity, time=time))
        track.append(Message('note_off', note=note, velocity=velocity, time=time))

    midi.save(output_path)
    print(f"Generated MIDI saved to {output_path}")


if __name__ == "__main__":
    # Model parameters (match the checkpoint)
    num_notes = 80       # Matches note vocabulary size
    num_durations = 32   # Matches duration embedding size
    num_velocities = 128 # Matches velocity embedding size
    num_intervals = 126  # Matches interval embedding size
    hidden_dim = 512     # Matches saved model
    embed_dim = 64       # Matches saved model
    output_dim = 80      # Matches num_notes

    # Load the model
    model = load_model(MODEL_PATH, num_notes, num_durations, num_velocities, num_intervals, output_dim)

    # Seed sequences
    seed_notes = np.random.randint(0, 80, size=SEQ_LENGTH).tolist()
    seed_durations = np.random.randint(0, 32, size=SEQ_LENGTH).tolist()
    seed_velocities = np.random.randint(0, 128, size=SEQ_LENGTH).tolist()
    # Calculate seed intervals from seed notes
    seed_intervals = [seed_notes[i + 1] - seed_notes[i] for i in range(len(seed_notes) - 1)]
    seed_intervals = [interval + 65 for interval in seed_intervals]  # Shift intervals to non-negative
    seed_intervals = [max(0, min(interval, 125)) for interval in seed_intervals]  # Clamp to valid range

    # Pad intervals to match SEQ_LENGTH
    while len(seed_intervals) < SEQ_LENGTH:
        seed_intervals.insert(0, 65)  # Neutral interval as padding

    print(f"Seed notes: {seed_notes}")
    print(f"Seed durations: {seed_durations}")
    print(f"Seed velocities: {seed_velocities}")
    print(f"Seed intervals: {seed_intervals}")

    # Generate music
    generated_notes, generated_durations, generated_velocities, generated_intervals = generate_music(
        model, seed_notes, seed_durations, seed_velocities, seed_intervals, NUM_NOTES_TO_GENERATE, SEQ_LENGTH
    )

    # Save to MIDI file
    save_to_midi(generated_notes, generated_durations, generated_velocities, generated_intervals, MIDI_OUTPUT_PATH)

