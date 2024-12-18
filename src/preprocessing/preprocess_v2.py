import os
import pandas as pd
import pickle as pkl

# Paths
raw_data_dir = '/Users/25lim/Project04_MusicGenerator/archive/midi/Bach'  # Path to Bach's MIDI CSV files
processed_data_dir = '/Users/25lim/Project04_MusicGenerator/src/data/processed_v2'  # Path to save processed data

def preprocess_csv(file_path):
    """
    Preprocess a single CSV file into sequences, including pitch normalization and durations, and dynamically handling all headers.
    """
    # Define expected column types for known headers
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

    # Filter relevant rows (focus on musical content)
    relevant_types = ['note_on', 'note_off', 'set_tempo', 'time_signature', 'key_signature']
    filtered_data = data[data['type'].isin(relevant_types)].copy()

    # Normalize notes to piano range (MIDI pitches 21 to 108)
    if 'note' in filtered_data.columns:
        filtered_data = filtered_data[(filtered_data['note'] >= 21) & (filtered_data['note'] <= 108)]

    # Convert ticks to seconds
    ticks_per_beat = 480
    tempo_events = filtered_data[filtered_data['type'] == 'set_tempo']
    tempo = tempo_events['tempo'].mean() if not tempo_events.empty else 500000  # Default tempo: 500ms/quarter note
    time_per_tick = tempo / ticks_per_beat / 1e6
    filtered_data.loc[:, 'time_seconds'] = filtered_data['tick'] * time_per_tick

    # Calculate durations between notes
    filtered_data['duration_ticks'] = filtered_data['tick'].diff().fillna(0)  # Difference in ticks
    filtered_data['duration_seconds'] = filtered_data['duration_ticks'] * time_per_tick  # Convert ticks to seconds

    # Build sequences dynamically including all available headers
    sequences = []
    for _, row in filtered_data.iterrows():
        event = {col: row.get(col, None) for col in data.columns}  # Include all columns dynamically
        event["time"] = row['time_seconds']  # Ensure time in seconds is included
        event["duration"] = row.get('duration_seconds', None)  # Add duration (seconds)
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
bach_output_file = os.path.join(processed_data_dir, "Bach_all_songs_v2.pkl")
with open(bach_output_file, 'wb') as f:
    pkl.dump(composer_sequences, f)
print(f"Saved all of Bach's sequences to: {bach_output_file}")
