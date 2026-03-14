import pandas as pd
import numpy as np
import tensorflow as tf

# --- 1. DATA PREPARATION (Your Code) ---

# Adjusted to a safer rate for RMSprop
learning_rate = 0.005 

print("Loading data...")
Data = pd.read_csv("StateNames.csv")

# Extracting the Names Column and lowercasing
names = Data["Name"].dropna().astype(str).tolist()
names = [str(name).lower() for name in names]

# Find all unique characters in the entire dataset
vocab = sorted(list(set("".join(names))))

# Create dictionaries to map characters to numbers and back
char_to_int = {char: i + 1 for i, char in enumerate(vocab)}
int_to_char = {i: char for char, i in char_to_int.items()}

vocab_size = len(vocab) + 1 # +1 for the padding zero
print(f"Total unique characters: {vocab_size}")

# Translate names to numbers
sequences = [[char_to_int[char] for char in name] for name in names]

# Find the longest name
max_length = max([len(seq) for seq in sequences])

# Pad all sequences
def pad_sequence(seq, max_len):
    return seq + [0] * (max_len - len(seq))

print("Padding sequences...")
padded_sequences = np.array([pad_sequence(seq, max_length) for seq in sequences])


# --- 2. DATA SLICING (The Memory-Safe Trick) ---

print("Slicing data for training...")
# X is everything EXCEPT the last character
X = padded_sequences[:, :-1]

# y is everything EXCEPT the first character (shifted by 1)
y = padded_sequences[:, 1:]


# --- 3. MODEL ARCHITECTURE ---

print("Building model...")
embedding_dim = 32
T = 1.5 # Temperature scaling to slow down softmax (your normalized arctan alternative)

model = tf.keras.Sequential([
    # Step 1: The Translation Layer
    tf.keras.layers.Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        mask_zero=True # Forces the model to ignore the padded 0s
    ),
    
    # Step 2: The GRU "Brain"
    # return_sequences=True is REQUIRED here so it outputs a prediction 
    # for every single letter in the X sequence, matching y.
    tf.keras.layers.GRU(units=128, return_sequences=True),
    
    # Step 3: Raw Output Layer
    tf.keras.layers.Dense(units=vocab_size, activation=None),
    
    # Step 4: Temperature Control & Softmax
    tf.keras.layers.Lambda(lambda x: x / T),
    tf.keras.layers.Activation('softmax')
])


# --- 4. COMPILATION & TRAINING ---

# Customizing the RMSprop optimizer with your learning rate
optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    # from_logits=False because we applied softmax in the step above
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.summary()

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_name_model.keras",
    save_best_only=True, # Only overwrites the file if the val_loss improves
    monitor="val_loss"
)

# Training Execution
# batch_size=2048 is huge, but necessary for 5.6M rows to train in a reasonable time. 
# If your computer throws an OutOfMemory (OOM) error, drop this to 1024 or 512.
print("Starting training...")
model.fit(
    X, 
    y, 
    epochs=5, 
    batch_size=2048, 
    validation_split=0.05,
    callbacks=[checkpoint_callback] # <--- The autosave trigger
)

# Saves the entire architecture, weights, and optimizer state to your hard drive
model.save("custom_name_generator.keras")
print("Model successfully saved!")











