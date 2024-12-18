# Music Generator

This project generates classical music. It trains off of
every known song written and composed by Bach, then uses an
LSTM to train off of the songs. Then, I take the model and
run it through a music generating code that makes music
in the form of a MIDI file.

In detail, I have all of Bach's songs as CSV files. I preprocess
the data, and I filter rows based on notes, tempo, time signature,
and key signature. I also turn the CSV files in pickles for easier 
use. Then I use a Jupyter notebook to train the model. My model
gets notes, durations (how long the note is), velocities (loudness),
and intervals. It uses embeddings for these, and processes them
through a stacked LSTM to predict the next note in a sequence.
In the Jupyter notebook, I use torch, and I train in batch size 64
over 50 epochs, but implement early stopping to prevent overfitting.
Then once I save my weights, I run it through my music generator. The
output is a playable MIDI files containing the generated music.

## Things needed to run the project
Nothing unusual


## Sources Used

I used ChatGPT and Flint for helping me write the code and understand different
concepts

I got the dataset off of kaggle: 
https://www.kaggle.com/datasets/vincentloos/classical-music-midi-as-csv/data

## What I would do if I had more time

If I had more time, I would try retraining the model on specific
songs, such as violin songs or piano songs. The dataset is
very cluttered, and that led to my project not being able to
make connections as well. I also want to try training it without
notes, and only intervals, velocities, and note durations. 