import torch
import torch.nn as nn

class EnhancedMusicGenerator(nn.Module):
    def __init__(self, num_notes, num_durations, num_velocities, num_intervals, hidden_dim=256, embed_dim=64, output_dim=80, num_layers=3):
        super(EnhancedMusicGenerator, self).__init__()

        # Embeddings for notes, durations, velocities, and intervals
        self.note_embedding = nn.Embedding(num_notes, embed_dim)
        self.duration_embedding = nn.Embedding(num_durations, embed_dim)
        self.velocity_embedding = nn.Embedding(num_velocities, embed_dim)
        self.interval_embedding = nn.Embedding(num_intervals, embed_dim)

        # LSTM: Match training settings
        self.lstm = nn.LSTM(embed_dim * 4, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3)

        # Fully connected layers: Match training checkpoint
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 4)  # Input: 256 → Output: 64
        self.fc2 = nn.Linear(hidden_dim // 4, output_dim)  # Input: 64 → Output: 80

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, notes, durations, velocities, intervals):
        # Embeddings
        note_embedded = self.note_embedding(notes)  # (batch_size, seq_length, embed_dim)
        duration_embedded = self.duration_embedding(durations)  # (batch_size, seq_length, embed_dim)
        velocity_embedded = self.velocity_embedding(velocities)  # (batch_size, seq_length, embed_dim)
        interval_embedded = self.interval_embedding(intervals)  # (batch_size, seq_length, embed_dim)

        # Combine embeddings
        x = torch.cat((note_embedded, duration_embedded, velocity_embedded, interval_embedded), dim=2)  # (batch_size, seq_length, embed_dim * 4)

        # LSTM forward pass
        out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_dim)

        # Take the last time step's output
        out = out[:, -1, :]  # (batch_size, hidden_dim)

        # Fully connected layers
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))  # (batch_size, hidden_dim // 4)
        out = self.fc2(out)  # (batch_size, output_dim)

        return out
