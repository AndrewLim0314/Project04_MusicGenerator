import os
import pandas as pd
import pickle as pkl

# Paths
raw_data_dir = '/Users/25lim/Project04_MusicGenerator/archive/midi/Bach'  # Path to Bach's MIDI CSV files
processed_data_dir = '/Users/25lim/Project04_MusicGenerator/src/data/processed'  # Path to save processed data

def preprocess_csv(file_path):
    """
    Preprocess a single CSV file into sequences, handling mixed types.
    """
    # Define expected column types
    dtype_map = {
        'numerator': 'Int64',  # Allow integers with nulls
        'denominator': 'Int64',
        'velocity': 'Int64',
        'note': 'Int64',
        'track': 'Int64',
        'key': 'str'  # Keys might be strings (e.g., 'C', 'G')
    }

    # Load CSV file with specified types
    data = pd.read_csv(file_path, dtype=dtype_map, low_memory=False)

    # Filter relevant rows
    relevant_types = ['note_on', 'note_off', 'set_tempo', 'time_signature', 'key_signature']
    filtered_data = data[data['type'].isin(relevant_types)].copy()

    # Convert ticks to seconds
    ticks_per_beat = 480
    tempo_events = filtered_data[filtered_data['type'] == 'set_tempo']
    tempo = tempo_events['tempo'].mean() if not tempo_events.empty else 500000
    time_per_tick = tempo / ticks_per_beat / 1e6
    filtered_data.loc[:, 'time_seconds'] = filtered_data['tick'] * time_per_tick

    # Retain all columns
    sequences = []
    for _, row in filtered_data.iterrows():
        event = {
            "time": row['time_seconds'],
            "type": row['type'],
            "note": row.get('note', None),
            "velocity": row.get('velocity', None),
            "tempo": row.get('tempo', None),
            "track": row.get('track', None),
            "key": row.get('key', None),
            "time_signature": (row.get('numerator', None), row.get('denominator', None)),
            "channel": row.get('channel', None),
            "program": row.get('program', None),
        }
        sequences.append(event)

    return sequences


# Process all CSV files in Bach's folder
print("Processing Bach's MIDI files...")
composer_sequences = []  # Collect all sequences

for file in os.listdir(raw_data_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(raw_data_dir, file)
        print(f"  Processing file: {file}")

        # Preprocess the file
        sequences = preprocess_csv(file_path)
        composer_sequences.append(sequences)

        # Save each song individually (optional)
        song_output_file = os.path.join(processed_data_dir, f"Bach_{file.split('.')[0]}.pkl")
        with open(song_output_file, 'wb') as f:
            pkl.dump(sequences, f)
        print(f"    Saved: {song_output_file}")

# Save all sequences for Bach into one file
bach_output_file = os.path.join(processed_data_dir, "Bach_all_songs.pkl")
with open(bach_output_file, 'wb') as f:
    pkl.dump(composer_sequences, f)
print(f"Saved all of Bach's sequences to: {bach_output_file}")
