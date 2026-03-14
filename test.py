import os
# These MUST be before importing tensorflow to silence the CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
import tensorflow as tf

import string
import time

print("1. Rebuilding vocabulary dictionary from CSV...")
Data = pd.read_csv("StateNames.csv")
names = Data["Name"].dropna().astype(str).tolist()
names = [str(name).lower() for name in names]

vocab = sorted(list(set("".join(names))))
char_to_int = {char: i + 1 for i, char in enumerate(vocab)}
int_to_char = {i: char for char, i in char_to_int.items()}
vocab_size = len(vocab) + 1 

print("2. Rebuilding architecture and loading weights...")
embedding_dim = 32
T = 1.5

# Rebuild the exact same model skeleton from your training file
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
    tf.keras.layers.GRU(units=128, return_sequences=True),
    tf.keras.layers.Dense(units=vocab_size, activation=None),
    tf.keras.layers.Lambda(lambda x: x / T),
    tf.keras.layers.Activation('softmax')
])

# Initialize the model by passing a dummy input shape so it knows how to size the layers
model.build(input_shape=(None, None))

# Load only the learned math (weights) directly into our fresh, bug-free skeleton
model.load_weights("custom_name_generator.keras")
print("Model ready!")

def generate_name(model, seed_text, max_gen_length=20):
    seed_text = seed_text.lower()
    input_seq = [char_to_int[char] for char in seed_text if char in char_to_int]
    
    if len(input_seq) != len(seed_text):
        return "Error: Seed contains unknown characters."

    generated_name = seed_text

    for _ in range(max_gen_length):
        current_sequence = np.array([input_seq])
        
        predicted_probs = model.predict(current_sequence, verbose=0)[0, -1, :]
        predicted_probs = predicted_probs / np.sum(predicted_probs)
        
        predicted_int = np.random.choice(range(vocab_size), p=predicted_probs)
        
        if predicted_int == 0:
            break
            
        predicted_char = int_to_char[predicted_int]
        generated_name += predicted_char
        input_seq.append(predicted_int)
        
    return generated_name.capitalize()


# --- Generate 20 Names ---
starting_seeds = [
    "Al", "Ma", "K", "Z", "Ry", 
    "El", "Ty", "J", "S", "Ch", 
    "L", "Br", "V", "Da", "Ni", 
    "Ta", "Ro", "G", "F", "Am"
]

print("\n--- Generating 20 Custom Names ---")
for seed in starting_seeds:
    new_name = generate_name(model, seed_text=seed)
    print(new_name)

def test_repetition_rate(model, num_trials=5):
    print(f"\n--- Starting Repetition Test ({num_trials} Trials) ---")
    print("Warning: This might take a few minutes depending on your CPU speed.\n")
    
    alphabet = list(string.ascii_uppercase)
    generations_to_repeat = []
    
    for trial in range(1, num_trials + 1):
        seen_names = set() # A set is mathematically the fastest way to check for duplicates
        count = 0
        start_time = time.time()
        
        while True:
            # Pick a random starting letter for each generation
            random_seed = np.random.choice(alphabet)
            
            # Generate the name using your existing function
            new_name = generate_name(model, seed_text=random_seed)
            count += 1
            
            # Check for a collision
            if new_name in seen_names:
                elapsed_time = round(time.time() - start_time, 2)
                print(f"Trial {trial}: Collision found! The model repeated '{new_name}' after {count} generations. ({elapsed_time} sec)")
                generations_to_repeat.append(count)
                break
                
            # If it's unique, add it to the set and keep going
            seen_names.add(new_name)
            
    # Calculate the final statistics
    average_reps = np.mean(generations_to_repeat)
    min_reps = np.min(generations_to_repeat)
    max_reps = np.max(generations_to_repeat)
    
    print("\n--- Final Results ---")
    print(f"Average generations before a repeat: {average_reps}")
    print(f"Shortest streak: {min_reps}")
    print(f"Longest streak: {max_reps}")
    
    return average_reps

# Run the test!
# We start with 5 trials so you aren't waiting an hour, but you can increase this later.
test_repetition_rate(model, num_trials=5)