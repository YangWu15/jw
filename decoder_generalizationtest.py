"""
HYBRID GENERALIZATION TEST SCRIPT
- Loads pre-trained slot decoders.
- Generates ALL test data ON-THE-FLY using the RNN model.
- No longer uses any static .pkl files for evaluation.
- Tests generalization by creating generators for L=7, 11, 14, 16.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- NEW: Imports from training script ---
import gc
import random
import time
from diffenvrnn1 import StandardGatedRNN, LoopEnvironment # Assumes this file is accessible

# --- Configuration ---
# The loop length that was USED FOR TRAINING
LOOP_LENGTH_TO_ANALYZE = 14 

# Loop lengths to TEST on
TEST_LOOP_LENGTHS = [7, 11, 14, 16]

# Constants from the original RNN model
HIDDEN_SIZE = 2048
VOCAB_SIZE = 16

# Decoder Hyperparameters
BATCH_SIZE = 512
RANDOM_STATE = 42

# Directory to LOAD models from and SAVE analysis to
BASE_OUTPUT_DIR = "slot_decoders_hybrid"

# --- NEW: Dynamic generation configuration ---
RNN_MODEL_PATH = "diffenvrnn1.1/model.pth"
FRESH_DATA_SEQ_LENGTH = 150
MAX_GENERATION_ATTEMPTS = 1000

# --- Analysis Configuration ---
TOP_PERCENT_NEURONS = 2.5

# --- REMOVED: load_and_prepare_data (no longer used) ---


# --- NEW: Pasted from slot_decoders_hybrid.py ---
class HybridOnDemandGenerator:
    """
    Generates loops on-demand with FULL vectorization.
    NO pre-generation, NO reuse - fresh data every call!
    Produces data for ALL decoders at once.
    """
    def __init__(self, loop_length, rnn_model, env, device, phase='test', seed=None):
        self.loop_length = loop_length
        self.rnn_model = rnn_model
        self.env = env
        self.device = device
        self.phase = phase
        
        # Track loops we've already used (as tuples for hashing)
        self.used_loops = set()
        
        # Statistics
        self.total_generated = 0
        self.failed_attempts = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def _generate_unique_loops_batch(self, batch_size):
        """
        Generate multiple unique loops at once.
        Returns list of loops or None if can't generate enough.
        """
        batch_loops = []
        attempts = 0
        
        while len(batch_loops) < batch_size and attempts < MAX_GENERATION_ATTEMPTS:
            # Generate more than needed to account for duplicates
            num_to_generate = min((batch_size - len(batch_loops)) * 2, 100)
            
            # VECTORIZED: Generate multiple loops at once
            candidate_loops = []
            for _ in range(num_to_generate):
                loop = list(np.random.choice(
                    self.env.observation_bank, 
                    size=self.loop_length, 
                    replace=False
                ))
                candidate_loops.append(loop)
            
            # Filter for uniqueness
            for loop in candidate_loops:
                loop_tuple = tuple(loop)
                if loop_tuple not in self.used_loops:
                    self.used_loops.add(loop_tuple)
                    batch_loops.append(loop)
                    if len(batch_loops) >= batch_size:
                        break
            
            attempts += num_to_generate
        
        if len(batch_loops) < batch_size:
            self.failed_attempts += 1
            
        self.total_generated += len(batch_loops)
        return batch_loops if batch_loops else None
    
    def sample_batch(self, batch_size):
        """
        Generate a batch of loops ON-DEMAND and process them through the RNN.
        Returns features and targets for ALL slots.
        CRITICAL: This generates FRESH data every call - no reuse!
        """
        # Generate fresh loops for this batch
        batch_loops = self._generate_unique_loops_batch(batch_size)
        
        if batch_loops is None or len(batch_loops) == 0:
            print(f"Warning: Could not generate unique loops for {self.phase} L={self.loop_length}")
            return None, None
        
        actual_batch_size = len(batch_loops)
        
        # Now process this batch through the RNN (only ONCE per call)
        with torch.no_grad():
            obs_seq, vel_seq, target_seq, loop_lengths = self.env.generate_batch(
                batch_loops, actual_batch_size, FRESH_DATA_SEQ_LENGTH
            )
            obs_seq = obs_seq.to(self.device)
            vel_seq = vel_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            
            _, _, _, data_log = self.rnn_model(
                obs_seq, vel_seq, target_seq, loop_lengths, collect_data=True
            )

        # ========================================
        # VECTORIZED target extraction!
        # ========================================
        
        # Filter valid timesteps
        valid_records = [r for r in data_log if r['timestep'] >= (self.loop_length - 1)]
        
        if len(valid_records) == 0:
            return None, None
        
        # Extract batch indices and timesteps
        batch_indices = torch.tensor([r['batch_idx'] for r in valid_records], 
                                     dtype=torch.long, device=self.device)
        timesteps = torch.tensor([r['timestep'] for r in valid_records], 
                                 dtype=torch.long, device=self.device)
        
        # Stack all hidden states at once
        hidden_states_np = np.stack([r['hidden_state'] for r in valid_records])
        X_batch = torch.from_numpy(hidden_states_np).float().to(self.device)
        
        # VECTORIZED target extraction - NO PYTHON LOOPS!
        k_offsets = torch.arange(0, self.loop_length, device=self.device)  # [0, 1, ..., L-1]
        target_timesteps = timesteps.unsqueeze(1) - k_offsets.unsqueeze(0)  # [N, L]
        
        batch_idx_expanded = batch_indices.unsqueeze(1).expand(-1, self.loop_length)  # [N, L]
        
        # Gather all targets at once for ALL slots!
        Y_batch = obs_seq[batch_idx_expanded, target_timesteps].long()  # [N, L]
        
        # Clean up
        del obs_seq, vel_seq, target_seq, data_log
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return X_batch, Y_batch
# --- END: Pasted Class ---


# --- Decoder Model Definition (FIXED to match training script) ---
class DecoderMLP(nn.Sequential):
    """
    FIXED: Now inherits from nn.Sequential (not nn.Module with a 'layers' attribute)
    to match the state dict structure saved by the training script.
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size)
        )

# --- Analysis Text/Plotting Functions (Unchanged) ---
def save_analysis_to_text(analysis_data: dict, shared_indices: set, save_path: str):
    print(f"\n--- Saving Detailed Analysis to {save_path} ---")
    with open(save_path, 'w') as f:
        f.write("--- Top Contributor Neurons by Slot Decoder ---\n\n")
        for slot_num in sorted(analysis_data.keys()):
            indices_set = analysis_data[slot_num]
            sorted_indices = sorted(list(indices_set))
            f.write(f"Slot {slot_num:02d}: {sorted_indices}\n")

        f.write("\n\n--- Shared Top Contributor Neurons (Overlap Across All Decoders) ---\n\n")
        sorted_shared = sorted(list(shared_indices))
        f.write(f"Found {len(sorted_shared)} shared neurons:\n")
        f.write(f"{sorted_shared}\n")
    print("Analysis text file saved successfully.")

def analyze_decoder_weights(model_path: str, top_percent: float) -> set:
    model = DecoderMLP(input_size=HIDDEN_SIZE, output_size=VOCAB_SIZE)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    if not (0 < top_percent <= 100):
        raise ValueError("top_percent must be between 0 and 100.")
    k = int(round(HIDDEN_SIZE * (top_percent / 100.0)))
    k = max(1, k)
    first_layer_weights = model[0].weight.data
    importances = torch.abs(first_layer_weights).sum(dim=0)
    top_indices = torch.topk(importances, k).indices
    return set(top_indices.cpu().numpy())

def run_and_plot_analysis(base_dir: str, num_slots: int, top_percent: float):
    print(f"\n--- 6. Analyzing Top {top_percent}% Contributing Neurons for {num_slots} Decoders ---")
    
    top_contributors_by_slot = {}
    for k in range(1, num_slots + 1):
        model_path = os.path.join(base_dir, f'slot_{k:02d}', 'decoder.pth')
        if os.path.exists(model_path):
            top_contributors_by_slot[k] = analyze_decoder_weights(model_path, top_percent)
        else:
            print(f"Warning: Model for slot {k} not found at {model_path}. Skipping.")
    
    if not top_contributors_by_slot:
        print("No decoder models found for analysis.")
        return

    print("\n--- Overlap Analysis Results ---")
    all_top_indices = [idx for indices_set in top_contributors_by_slot.values() for idx in indices_set]
    neuron_frequency = Counter(all_top_indices)
    num_top_neurons = len(next(iter(top_contributors_by_slot.values())))

    print(f"Top 15 most frequent contributor neurons (out of {HIDDEN_SIZE}):")
    for neuron_idx, count in neuron_frequency.most_common(15):
        print(f"   - Neuron {neuron_idx}: Appeared in {count}/{num_slots} decoders' top {top_percent}% ({num_top_neurons} neurons) list.")

    shared_indices = set.intersection(*top_contributors_by_slot.values())
    print(f"\nFound {len(shared_indices)} neurons that are top contributors for ALL {num_slots} decoders:")
    print(f"   {sorted(list(shared_indices))}")

    analysis_txt_path = os.path.join(base_dir, f'top_{top_percent}_neuron_analysis_L{LOOP_LENGTH_TO_ANALYZE}.txt')
    save_analysis_to_text(top_contributors_by_slot, shared_indices, analysis_txt_path)
    
    union_of_indices = sorted(list(set.union(*top_contributors_by_slot.values())))
    neuron_to_row_idx = {neuron_id: i for i, neuron_id in enumerate(union_of_indices)}
    heatmap_data = np.zeros((len(union_of_indices), num_slots))
    for slot_num, indices_set in top_contributors_by_slot.items():
        for neuron_id in indices_set:
            row_idx = neuron_to_row_idx[neuron_id]
            heatmap_data[row_idx, slot_num - 1] = 1

    row_sums = heatmap_data.sum(axis=1)
    sorted_row_indices = np.argsort(row_sums)[::-1]
    sorted_heatmap_data = heatmap_data[sorted_row_indices, :]
    sorted_neuron_labels = [union_of_indices[i] for i in sorted_row_indices]

    plt.figure(figsize=(12, 16))
    sns.heatmap(sorted_heatmap_data, cmap="viridis", cbar=False, yticklabels=sorted_neuron_labels)
    plt.title(f'Top {top_percent}% Contributing Neurons from RNN Hidden State (L={LOOP_LENGTH_TO_ANALYZE})')
    plt.xlabel('Slot Decoder (k)')
    plt.ylabel('RNN Neuron Index (Sorted by Frequency)')
    plt.xticks(ticks=np.arange(num_slots) + 0.5, labels=np.arange(1, num_slots + 1))
    
    n_ticks_to_show = 50
    tick_spacing = max(1, len(sorted_neuron_labels) // n_ticks_to_show)
    plt.yticks(
        ticks=np.arange(len(sorted_neuron_labels))[::tick_spacing] + 0.5,
        labels=sorted_neuron_labels[::tick_spacing],
        rotation=0
    )

    plot_path = os.path.join(base_dir, f'top_{top_percent}_neuron_overlap_heatmap_L{LOOP_LENGTH_TO_ANALYZE}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved neuron contribution heatmap to {plot_path}")

# --- REMOVED: evaluate_decoders_on_loop_length (logic moved to main) ---

# --- Generalization Plotting Functions (Unchanged) ---
def plot_and_save_generalization(results_by_length: dict, save_dir: str):
    """
    Plots the OVERALL (averaged) accuracy and loss across different test loop lengths.
    """
    print(f"\n--- 4. Plotting Overall Generalization Performance ---")
    
    loop_lengths = sorted(results_by_length.keys())
    if not loop_lengths:
        print("No generalization results to plot.")
        return

    accuracies = [results_by_length[L]['overall_acc'] for L in loop_lengths]
    losses = [results_by_length[L]['overall_loss'] for L in loop_lengths]
    
    x_labels = [f"L={L}" for L in loop_lengths]
    x_pos = np.arange(len(x_labels))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Test Loop Length')
    ax1.set_ylabel('Overall Avg. Accuracy', color=color)
    ax1.bar(x_pos - 0.2, accuracies, 0.4, label='Accuracy', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Overall Avg. Loss', color=color)
    ax2.bar(x_pos + 0.2, losses, 0.4, label='Loss', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.suptitle(f'Decoder Generalization Performance (Trained on L={LOOP_LENGTH_TO_ANALYZE})')
    fig.tight_layout()
    
    filepath = os.path.join(save_dir, f'generalization_performance_L{LOOP_LENGTH_TO_ANALYZE}_trained.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved generalization plot to {filepath}")

def save_generalization_results_to_text(results_by_length: dict, save_dir: str):
    """
    Saves the detailed generalization results (overall and per-slot) to a text file.
    """
    save_path = os.path.join(save_dir, f'generalization_results_L{LOOP_LENGTH_TO_ANALYZE}_trained.txt')
    print(f"\n--- 3. Saving Generalization Results to {save_path} ---")
    
    with open(save_path, 'w') as f:
        f.write(f"Generalization Results (Models Trained on L={LOOP_LENGTH_TO_ANALYZE})\n")
        f.write("="*60 + "\n\n")
        
        for l_test in sorted(results_by_length.keys()):
            results = results_by_length[l_test]
            f.write(f"--- Test Loop Length: L={l_test} ---\n")
            f.write(f"  Overall Avg. Accuracy: {results['overall_acc']:.4f}\n")
            f.write(f"  Overall Avg. Loss:     {results['overall_loss']:.4f}\n\n")
            f.write("  Per-Slot Performance:\n")
            
            if not results['per_slot_results']:
                f.write("    (No per-slot results)\n")
            else:
                for slot_res in results['per_slot_results']:
                    f.write(f"    - Slot {slot_res['slot']:02d}: Acc={slot_res['accuracy']:.4f}, Loss={slot_res['loss']:.4f}\n")
            f.write("\n" + "-"*60 + "\n\n")
    print("Generalization results text file saved successfully.")

def plot_and_save_per_slot_generalization(results_by_length: dict, save_dir: str):
    """
    Plots the per-slot accuracy for each test loop length on a single graph.
    """
    print(f"\n--- 5. Plotting Per-Slot Generalization Performance ---")
    
    loop_lengths = sorted(results_by_length.keys())
    if not loop_lengths:
        print("No generalization results to plot.")
        return

    plt.figure(figsize=(14, 8))
    
    max_slot = 0
    for l_test in loop_lengths:
        per_slot_data = results_by_length[l_test]['per_slot_results']
        if not per_slot_data:
            continue
        
        slots = [r['slot'] for r in per_slot_data]
        accuracies = [r['accuracy'] for r in per_slot_data]
        
        if slots:
            max_slot = max(max_slot, max(slots))
        
        plt.plot(slots, accuracies, '.-', label=f'Test on L={l_test}')

    plt.xlabel('Slot Number (k)')
    plt.ylabel('Decoder Accuracy')
    plt.title(f'Per-Slot Decoder Accuracy (Trained on L={LOOP_LENGTH_TO_ANALYZE})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    
    if max_slot > 0:
        # Use LOOP_LENGTH_TO_ANALYZE as the max, since L=14 defines the full x-axis
        ticks = np.arange(1, LOOP_LENGTH_TO_ANALYZE + 1)
        plt.xticks(ticks)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, f'generalization_per_slot_accuracy_L{LOOP_LENGTH_TO_ANALYZE}_trained.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved per-slot generalization plot to {filepath}")


# --- 5. Main Evaluation Script (HEAVILY MODIFIED) ---
def main():
    """
    Main function to orchestrate evaluation and analysis of PRE-TRAINED decoders
    using ON-THE-FLY data generation.
    """
    print("="*60)
    print("--- STARTING DECODER EVALUATION & ANALYSIS SCRIPT ---")
    print("--- (Using ON-THE-FLY Data Generation) ---")
    print(f"Models expected in: {BASE_OUTPUT_DIR}")
    print("="*60 + "\n")

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    criterion = nn.CrossEntropyLoss()
    
    # --- 1. Load RNN Model and Environment ---
    print(f"Loading RNN from {RNN_MODEL_PATH}...")
    rnn_model = StandardGatedRNN(hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE).to(device)
    rnn_model.load_state_dict(torch.load(RNN_MODEL_PATH, map_location=device))
    rnn_model.eval()
    print("✓ RNN loaded")
    
    env = LoopEnvironment(
        observation_bank=list(range(VOCAB_SIZE)),
        loop_lengths=TEST_LOOP_LENGTHS,
        velocities=[1]
    )
    print("✓ Environment created")

    # --- 2. Generalization Testing (Now uses live generation) ---
    print("\n" + "="*60)
    print("--- 2. STARTING GENERALIZATION TESTING ---")
    print(f"Models trained on L={LOOP_LENGTH_TO_ANALYZE} will be tested on {TEST_LOOP_LENGTHS}.")
    print("="*60 + "\n")

    generalization_results = {}

    for l_test in TEST_LOOP_LENGTHS:
        print(f"--- Evaluating decoders on L={l_test} data (Live Generation) ---")
        
        # Determine how many of our 14 decoders to test
        # e.g., for L=7, we only test decoders 1-7
        # e.g., for L=16, we test all 14 decoders
        num_slots_to_test = min(LOOP_LENGTH_TO_ANALYZE, l_test)

        # 1. Create a generator for this specific loop length
        test_gen = HybridOnDemandGenerator(
            l_test, rnn_model, env, device, 
            phase=f'test_L{l_test}', 
            seed=RANDOM_STATE + l_test # Use different seed per length
        )
        
        # 2. Generate a large batch of test data on-the-fly
        # (Increase BATCH_SIZE * N for a more stable evaluation)
        X_test, Y_test_all_slots = test_gen.sample_batch(BATCH_SIZE * 4)

        if X_test is None:
            print(f"Could not generate test data for L={l_test}. Skipping.")
            continue
            
        print(f"Generated {X_test.shape[0]} test samples for L={l_test}")

        # 3. Create a DataLoader for this test batch
        test_dataset = TensorDataset(X_test, Y_test_all_slots)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2)

        per_slot_results = []
        total_loss_all_slots = 0
        total_acc_all_slots = 0
        
        # 4. Evaluate each relevant slot decoder (1 to num_slots_to_test)
        for k in range(1, num_slots_to_test + 1):
            slot_index = k - 1
            model_path = os.path.join(BASE_OUTPUT_DIR, f'slot_{k:02d}', 'decoder.pth')
            
            if not os.path.exists(model_path):
                print(f"Warning: Model for slot {k} not found. Skipping.")
                continue
            
            model = DecoderMLP(input_size=HIDDEN_SIZE, output_size=VOCAB_SIZE).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            total_test_loss, correct_test, total_test = 0, 0, 0
            with torch.no_grad():
                for features, all_targets in test_loader:
                    # all_targets shape is [N, l_test]
                    # We correctly select the k-th target (index k-1)
                    targets = all_targets[:, slot_index].to(device)
                    features = features.to(device)
                    
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    total_test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += targets.size(0)
                    correct_test += (predicted == targets).sum().item()
            
            avg_test_loss = total_test_loss / len(test_loader)
            test_accuracy = correct_test / total_test
            per_slot_results.append({'slot': k, 'loss': avg_test_loss, 'accuracy': test_accuracy})
            total_loss_all_slots += avg_test_loss
            total_acc_all_slots += test_accuracy
            print(f"   L={l_test}, Slot={k}: Acc={test_accuracy:.4f}, Loss={avg_test_loss:.4f}")

        if not per_slot_results:
            print(f"No results for L={l_test}. Skipping.")
            continue
            
        # 5. Calculate and store overall results for this loop length
        num_results = len(per_slot_results)
        overall_avg_loss = total_loss_all_slots / num_results
        overall_avg_accuracy = total_acc_all_slots / num_results
        print(f"Overall for L={l_test}: Avg Acc={overall_avg_accuracy:.4f}, Avg Loss={overall_avg_loss:.4f}")
        
        generalization_results[l_test] = {
            'overall_loss': overall_avg_loss,
            'overall_acc': overall_avg_accuracy,
            'per_slot_results': per_slot_results
        }
        print("-" * 50 + "\n")


    # --- 3. Save Generalization Results to Text ---
    save_generalization_results_to_text(generalization_results, BASE_OUTPUT_DIR)

    # --- 4. Plot Overall Generalization Results ---
    plot_and_save_generalization(generalization_results, BASE_OUTPUT_DIR)
    
    # --- 5. Plot Per-Slot Generalization Results ---
    plot_and_save_per_slot_generalization(generalization_results, BASE_OUTPUT_DIR)

    # --- 6. Run Neuron Analysis (Unchanged) ---
    # This analysis is based on the L=14 decoders' weights, 
    # which is independent of data generation.
    run_and_plot_analysis(BASE_OUTPUT_DIR, LOOP_LENGTH_TO_ANALYZE, TOP_PERCENT_NEURONS)
    
    print("\n" + "="*60)
    print("--- EVALUATION & ANALYSIS COMPLETE ---")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()