import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import os
import random

# --- Assumed Imports from your environment ---
# Make sure these classes are defined or importable
# (Copied definitions below for completeness if needed)
try:
    from diffenvrnn1 import StandardGatedRNN, LoopEnvironment
except ImportError:
    print("Error: Could not import StandardGatedRNN or LoopEnvironment.")
    print("Please ensure diffenvrnn1.py is accessible.")
    # Add placeholder definitions if diffenvrnn1.py isn't available
    # class StandardGatedRNN(nn.Module): pass
    # class LoopEnvironment: pass
    exit()

# --- Configuration ---
# Paths (Adjust if necessary)
PRE_TRAINED_RNN_PATH = "diffenvrnn1.1/model.pth"
DECODER_BASE_PATH = "/nfs/nhome/live/jwhittington/yang_project/slot_decoders_hybrid"
# DATA_FILE_PATH = "diffenvrnn1.1/detailed_run_data_by_length.pkl" # For loading specific test set

# Model & Data Constants
LOOP_LENGTH_TO_ANALYZE = 14
HIDDEN_SIZE = 2048
VOCAB_SIZE = 16
OBSERVATION_BANK = list(range(VOCAB_SIZE))

# Evaluation Parameters
NUM_TEST_SEQS = 1000 # Number of sequences to generate for testing
TEST_SEQ_LENGTH = 150 # Length of sequences
EVAL_BATCH_SIZE = 512 # Batch size for evaluation
RANDOM_STATE = 42 # Use the same seed for consistency
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Decoder Model Definition (FIXED to match training script) ---
class DecoderMLP(nn.Sequential):
    """
    FIXED: Now inherits from nn.Sequential (not nn.Module with a 'layers' attribute)
    to match the state dict structure saved by the training script.
    
    The training script saves individual nn.Sequential modules with keys like:
    "0.weight", "2.weight", "4.weight", etc.
    
    This class structure matches that exactly.
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size)
        )

# --- Helper Functions ---
def load_rnn_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"RNN model not found at {path}")
    model = StandardGatedRNN(hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Successfully loaded RNN model from {path}")
    return model

def load_decoder_model(base_path, slot_k, device):
    path = os.path.join(base_path, f'slot_{slot_k:02d}', 'decoder.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Decoder model for slot {slot_k} not found at {path}")
    model = DecoderMLP(input_size=HIDDEN_SIZE, output_size=VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def generate_rnn_hidden_states(rnn_model, num_seqs, seq_len, loop_length, device):
    print(f"\nGenerating {num_seqs} test sequences (L={loop_length}, T={seq_len})...")
    env = LoopEnvironment(
        observation_bank=OBSERVATION_BANK,
        loop_lengths=[loop_length],
        velocities=[1] # Assuming velocity 1
    )
    loops_test = env.generate_unique_loops(num_seqs, [loop_length])
    
    # --- FIX 1: Capture ALL required tensors from generate_batch ---
    # We now need targets_test and loop_lengths_list
    obs_test, vels_test, targets_test, loop_lengths_list = env.generate_batch(
        loops_test, num_seqs, seq_len
    )

    # --- FIX 2: Move all tensors to the correct device ---
    obs_test = obs_test.to(device)
    vels_test = vels_test.to(device)
    targets_test = targets_test.to(device)
    # loop_lengths_list is a Python list, so it stays on the CPU

    print("Running RNN forward pass to get hidden states...")
    
    # --- FIX 3: Call the model ONCE with all 5 required arguments ---
    with torch.no_grad():
        # The model's forward pass expects vel_sequence as [batch, seq_len]
        # and it will handle the .float() and .unsqueeze(1) internally
        _, _, _, data_log = rnn_model(
            obs_test,          # 1. obs_sequence
            vels_test,         # 2. vel_sequence
            targets_test,      # 3. target_sequence
            loop_lengths_list, # 4. loop_lengths (as a list)
            collect_data=True  # 5. Set True to get data_log
        )

    print(f"RNN pass complete. Parsing {len(data_log)} data log entries...")

    # --- FIX 4: Reconstruct the hidden_states tensor from the data_log ---
    # data_log is a list of dicts, sorted by [timestep, batch_idx]
    # We need to build a [batch_size, seq_len, hidden_size] tensor
    
    # Get batch_size from obs_test, not num_seqs, in case it's smaller
    batch_size = obs_test.shape[0] 
    
    # Initialize an empty numpy array (it's faster to fill)
    hidden_states_array = np.zeros(
        (batch_size, seq_len, HIDDEN_SIZE), 
        dtype=np.float32
    )

    if not data_log:
        raise ValueError("RNN model returned an empty data_log. "
                         "Cannot retrieve hidden states. "
                         "Is `collect_data=True` working?")

    # Fill the array
    for record in data_log:
        b = record['batch_idx']
        t = record['timestep']
        # Ensure indices are within bounds (especially if batch_size != num_seqs)
        if b < batch_size and t < seq_len:
             hidden_states_array[b, t, :] = record['hidden_state']

    # Convert the completed array to a tensor
    hidden_states_tensor = torch.from_numpy(hidden_states_array)
    
    print("Hidden states tensor successfully reconstructed.")
    
    # Return observations (CPU) and hidden states (CPU)
    return obs_test.cpu(), hidden_states_tensor

# --- Main Evaluation ---
def evaluate_decoders(hidden_states, obs_sequences):
    # 3. Load and Evaluate Decoders
    print("\n--- Evaluating Decoders ---")
    results = {}

    for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1):
        print(f"Loading and evaluating Decoder D{k:02d}...")
        try:
            decoder_model = load_decoder_model(DECODER_BASE_PATH, k, DEVICE)
        except FileNotFoundError as e:
            print(e)
            continue

        total_correct = 0
        total_samples = 0

        # Prepare targets and inputs for this decoder
        inputs_list = []
        targets_list = []

        # Iterate through sequences and relevant timesteps
        # Decoder k predicts obs from t-(k-1) using h_t
        # Need t >= k-1 for the target obs to exist
        # Also, often analysis starts after first loop, t >= L-1
        start_t = LOOP_LENGTH_TO_ANALYZE - 1

        for t in range(start_t, TEST_SEQ_LENGTH):
            target_obs_idx = t - (k - 1)
            if target_obs_idx < 0: continue # Should not happen with start_t logic

            current_h_states = hidden_states[:, t, :] # [num_seqs, hidden_size]
            current_targets = obs_sequences[:, target_obs_idx] # [num_seqs]

            inputs_list.append(current_h_states)
            targets_list.append(current_targets)

        if not inputs_list:
            print(f"  No valid timesteps found for evaluation for slot {k}. Skipping.")
            continue

        # Concatenate inputs and targets across time for batching
        all_inputs = torch.cat(inputs_list, dim=0) # [num_samples, hidden_size]
        all_targets = torch.cat(targets_list, dim=0) # [num_samples]

        eval_dataset = TensorDataset(all_inputs, all_targets)
        eval_loader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE)

        with torch.no_grad():
            for batch_inputs, batch_targets in eval_loader:
                batch_inputs = batch_inputs.to(DEVICE)
                batch_targets = batch_targets.to(DEVICE)

                outputs = decoder_model(batch_inputs)
                _, predicted = torch.max(outputs.data, 1)

                total_samples += batch_targets.size(0)
                total_correct += (predicted == batch_targets).sum().item()

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        results[k] = accuracy
        print(f"  Decoder D{k:02d} Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")

    print("\n--- Summary ---")
    for k in sorted(results.keys()):
        print(f"Decoder D{k:02d}: {results[k]:.4f}")

def evaluate_observation_trace(hidden_states, obs_sequences, device):
    """
    Tests the "diagonal slice" hypothesis:
    Does h_t fed to D_{t+1} correctly predict the appropriate observation?
    
    Now testing only in the valid range (t >= L-1) where decoders were trained.
    """
    print("\n--- 💡 Evaluating Observation Trace (Test 2) ---")
    print(f"Testing diagonal pattern in valid range (t >= {LOOP_LENGTH_TO_ANALYZE-1})")
    
    results = {}
    
    # Start from where training data begins
    start_t = LOOP_LENGTH_TO_ANALYZE - 1  # = 13
    
    # Test for each position in the first loop (after the loop is established)
    for offset in range(LOOP_LENGTH_TO_ANALYZE):
        t = start_t + offset  # t = 13, 14, 15, ..., 26
        
        if t >= TEST_SEQ_LENGTH:
            break
        
        # Decoder index cycles through 1-14
        k = (offset % LOOP_LENGTH_TO_ANALYZE) + 1
        
        print(f"Testing: h_{t:02d} -> D_{k:02d} -> obs[{t-(k-1)}]")
        
        try:
            decoder_model = load_decoder_model(DECODER_BASE_PATH, k, device)
        except FileNotFoundError as e:
            print(f"  {e}")
            print(f"  Skipping t={t}, k={k}.")
            continue
        
        # The inputs are the hidden states at timestep t
        inputs_h_t = hidden_states[:, t, :]
        
        # The target is what D_k should predict: obs[t - (k-1)]
        target_idx = t - (k - 1)
        targets = obs_sequences[:, target_idx].to(device)
        
        # Use DataLoader to batch across the sequences
        eval_dataset = TensorDataset(inputs_h_t, targets)
        eval_loader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE)

        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in eval_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                outputs = decoder_model(batch_inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total_samples += batch_targets.size(0)
                total_correct += (predicted == batch_targets).sum().item()

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        results[t] = {'k': k, 'target_idx': target_idx, 'accuracy': accuracy}
        print(f"  Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")

    print("\n--- Trace Summary (Valid Range) ---")
    for t, info in sorted(results.items()):
        print(f"t={t:02d} (h_{t:02d} -> D_{info['k']:02d} -> obs[{info['target_idx']}]): {info['accuracy']:.4f}")

if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    print(f"Using device: {DEVICE}")

    # 1. Load RNN
    rnn_model = load_rnn_model(PRE_TRAINED_RNN_PATH, DEVICE)

    # 2. Generate Data and Hidden States using RNN (This is done only ONCE)
    obs_sequences, hidden_states = generate_rnn_hidden_states(
        rnn_model, NUM_TEST_SEQS, TEST_SEQ_LENGTH, LOOP_LENGTH_TO_ANALYZE, DEVICE
    )
    
    # 3. Run Test 1 (Original Evaluation)
    evaluate_decoders(hidden_states, obs_sequences) # Pass data in

    # 4. Run Test 2 (New Observation Trace)
    evaluate_observation_trace(hidden_states, obs_sequences, DEVICE) # Pass data in

    print("\nAll evaluations complete.")