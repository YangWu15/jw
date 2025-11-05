import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Configuration ---
# Path to the data file
DATA_FILE_PATH = "/nfs/nhome/live/jwhittington/yang_project/diffenvrnn1.1/detailed_run_data_by_length.pkl"

# The loop length that was USED FOR TRAINING
LOOP_LENGTH_TO_ANALYZE = 14 # This is the "training" loop length

# Loop lengths to TEST on
TEST_LOOP_LENGTHS = [7, 11, 14, 16]

# Constants from the original RNN model
HIDDEN_SIZE = 2048
VOCAB_SIZE = 16

# Decoder Hyperparameters (needed for data loading/model definition)
BATCH_SIZE = 512
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# Directory to LOAD models from and SAVE analysis to
BASE_OUTPUT_DIR = "slot_decoders_hybrid"

# --- Analysis Configuration ---
# Percentage of top contributing neurons to identify
TOP_PERCENT_NEURONS = 2.5

def load_and_prepare_data(file_path: str, loop_length: int) -> pd.DataFrame:
    """
    Loads data for a specific loop_length from the pickle file,
    creates shifted target columns, THEN filters for valid targets and t >= L-1.
    
    CRITICAL: Shifting must happen BEFORE filtering to preserve historical data!
    """
    print(f"--- Loading and Preparing Data for L={loop_length} ---")
    
    with open(file_path, 'rb') as f:
        all_data = pickle.load(f)

    if loop_length not in all_data:
        print(f"Warning: Loop length {loop_length} not found in data file.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data[loop_length])
    print(f"Loaded {len(df)} total records for L={loop_length}.")

    # CRITICAL FIX: Create shifted targets FIRST, using ALL data
    print(f"Creating target columns for each slot (1 to {loop_length})...")
    for k in range(1, loop_length + 1):
        shift_amount = k - 1
        # Shift within each batch BEFORE any filtering
        df[f'target_slot_{k}'] = (
            df.groupby('batch_idx')['obs'].shift(shift_amount)
        )
    
    # NOW filter: keep rows where t >= L-1 AND all targets are valid
    print(f"Filtering to keep timestep >= {loop_length - 1} with valid targets...")
    df_filtered = df[df['timestep'] >= (loop_length - 1)].copy()
    
    initial_rows = len(df_filtered)
    df_filtered.dropna(inplace=True)
    print(f"Removed {initial_rows - len(df_filtered)} rows with NaN targets after shifting.")
    
    # Convert targets to integers
    for k in range(1, loop_length + 1):
        df_filtered[f'target_slot_{k}'] = df_filtered[f'target_slot_{k}'].astype(np.int64)

    print(f"Final prepared dataset for L={loop_length} has {len(df_filtered)} samples.")
    
    # Verification
    if len(df_filtered) > 0:
        print("\n--- Verification Sample ---")
        # Check first batch
        first_batch = df_filtered[df_filtered['batch_idx'] == df_filtered['batch_idx'].min()].head(10)
        print("First batch (first 10 rows after filtering):")
        print(first_batch[['batch_idx', 'timestep', 'obs', 'target_slot_1', 'target_slot_5', 
                          f'target_slot_{loop_length}']].to_string(index=False))
        
        # Verify correctness
        print("\nVerifying temporal correctness:")
        for k in [1, 5, loop_length]:
            if f'target_slot_{k}' in first_batch.columns:
                shift_amount = k - 1
                # Check if target matches obs from shift_amount steps ago
                first_valid_idx = shift_amount
                if first_valid_idx < len(first_batch):
                    current_row = first_batch.iloc[first_valid_idx]
                    if shift_amount > 0:
                        target_row = first_batch.iloc[0]
                        expected = target_row['obs']
                        actual = current_row[f'target_slot_{k}']
                        status = "✓" if expected == actual else "✗"
                        print(f"  Slot {k:2d}: {status} (expected={expected}, actual={actual})")
    
    return df_filtered


# --- 2. Decoder Model Definition (FIXED to match training script) ---
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

# --- 3. Analysis Text/Plotting Functions (Unchanged) ---
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
    # Access first layer directly (index 0 in Sequential)
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

# --- 4. Generalization Evaluation Functions (Unchanged) ---
def evaluate_decoders_on_loop_length(
    test_loop_length: int,
    base_model_dir: str,
    num_trained_decoders: int,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """
    Loads ALL data for a specific loop length and evaluates the trained decoders.
    Used for L=7, 11, 16.
    """
    print(f"--- Evaluating decoders on L={test_loop_length} data (Full Dataset) ---")
    try:
        # 1. Load and prepare the test data
        df_test = load_and_prepare_data(DATA_FILE_PATH, test_loop_length)
        if df_test.empty:
            print(f"No data found for L={test_loop_length}, skipping evaluation.")
            return None
    except Exception as e:
        print(f"Error loading data for L={test_loop_length}: {e}. Skipping.")
        return None

    # 2. Determine number of slots to test
    num_slots_to_test = min(num_trained_decoders, test_loop_length)
    if num_slots_to_test == 0:
        print("No slots to test. Skipping.")
        return None
        
    print(f"Will test decoders 1 through {num_slots_to_test} on this data.")

    # 3. Prepare PyTorch data
    X_test_data = np.vstack(df_test['hidden_state'].values).astype(np.float32)
    target_cols = [f'target_slot_{k}' for k in range(1, test_loop_length + 1)]
    Y_test_data = df_test[target_cols].values

    test_dataset = TensorDataset(torch.from_numpy(X_test_data), torch.from_numpy(Y_test_data))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2) 

    per_slot_results = []
    total_loss_all_slots = 0
    total_acc_all_slots = 0

    # 4. Loop through each relevant decoder
    for k in range(1, num_slots_to_test + 1):
        slot_index = k - 1
        model_path = os.path.join(base_model_dir, f'slot_{k:02d}', 'decoder.pth')

        if not os.path.exists(model_path):
            print(f"Warning: Model for slot {k} not found at {model_path}. Skipping slot.")
            continue

        model = DecoderMLP(input_size=HIDDEN_SIZE, output_size=VOCAB_SIZE).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        total_test_loss, correct_test, total_test = 0, 0, 0
        with torch.no_grad():
            for features, all_targets in test_loader:
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
        
        print(f"   L={test_loop_length}, Slot={k}: Acc={test_accuracy:.4f}, Loss={avg_test_loss:.4f}")

    if not per_slot_results:
        print(f"No results generated for L={test_loop_length}.")
        return None

    # 5. Calculate overall averages
    overall_avg_loss = total_loss_all_slots / len(per_slot_results)
    overall_avg_accuracy = total_acc_all_slots / len(per_slot_results)

    print(f"Overall for L={test_loop_length}: Avg Acc={overall_avg_accuracy:.4f}, Avg Loss={overall_avg_loss:.4f}")
    
    return {
        'overall_loss': overall_avg_loss,
        'overall_acc': overall_avg_accuracy,
        'per_slot_results': per_slot_results
    }

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

# ******************************************************
# *** NEW FUNCTION: Save generalization results to text ***
# ******************************************************
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
        # Ensure x-ticks are integers from 1 to max_slot
        # Use LOOP_LENGTH_TO_ANALYZE as the max, since L=14 defines the full x-axis
        ticks = np.arange(1, LOOP_LENGTH_TO_ANALYZE + 1)
        plt.xticks(ticks)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, f'generalization_per_slot_accuracy_L{LOOP_LENGTH_TO_ANALYZE}_trained.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved per-slot generalization plot to {filepath}")


# --- 5. Main Evaluation Script ---
def main():
    """
    Main function to orchestrate evaluation and analysis of PRE-TRAINED decoders.
    """
    print("="*60)
    print("--- STARTING DECODER EVALUATION & ANALYSIS SCRIPT ---")
    print(f"Models expected in: {BASE_OUTPUT_DIR}")
    print("="*60 + "\n")

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    criterion = nn.CrossEntropyLoss()
    
    # --- 1. Re-create L=14 Test Split ---
    print(f"--- 1. Loading L={LOOP_LENGTH_TO_ANALYZE} data to re-create test split ---")
    prepared_df = load_and_prepare_data(DATA_FILE_PATH, LOOP_LENGTH_TO_ANALYZE)

    if prepared_df.empty:
        print(f"No data available for L={LOOP_LENGTH_TO_ANALYZE}. Exiting.")
        return

    X = np.vstack(prepared_df['hidden_state'].values).astype(np.float32)
    target_cols = [f'target_slot_{k}' for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1)]
    Y = prepared_df[target_cols].values

    _, X_test_L14, _, Y_test_L14 = train_test_split(
        X, Y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"Re-created L={LOOP_LENGTH_TO_ANALYZE} test set with {X_test_L14.shape[0]} samples.\n")
    del X, Y, prepared_df # Free up memory

    # --- 2. Generalization Testing ---
    print("\n" + "="*60)
    print("--- 2. STARTING GENERALIZATION TESTING ---")
    print(f"Models trained on L={LOOP_LENGTH_TO_ANALYZE} will be tested on {TEST_LOOP_LENGTHS}.")
    print("="*60 + "\n")

    generalization_results = {}

    for l_test in TEST_LOOP_LENGTHS:
        if l_test == LOOP_LENGTH_TO_ANALYZE:
            # Special case: Evaluate on the held-out L=14 test set
            print(f"--- Evaluating decoders on L={l_test} data (Held-out Test Set) ---")
            
            test_dataset_l14 = TensorDataset(torch.from_numpy(X_test_L14), torch.from_numpy(Y_test_L14))
            test_loader_l14 = DataLoader(test_dataset_l14, batch_size=BATCH_SIZE * 2)

            per_slot_results = []
            total_loss_all_slots = 0
            total_acc_all_slots = 0
            
            for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1):
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
                    for features, all_targets in test_loader_l14:
                        targets = all_targets[:, slot_index].to(device)
                        features = features.to(device)
                        
                        outputs = model(features)
                        loss = criterion(outputs, targets)
                        total_test_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total_test += targets.size(0)
                        correct_test += (predicted == targets).sum().item()
                
                avg_test_loss = total_test_loss / len(test_loader_l14)
                test_accuracy = correct_test / total_test
                per_slot_results.append({'slot': k, 'loss': avg_test_loss, 'accuracy': test_accuracy})
                total_loss_all_slots += avg_test_loss
                total_acc_all_slots += test_accuracy
                print(f"   L={l_test}, Slot={k}: Acc={test_accuracy:.4f}, Loss={avg_test_loss:.4f}")

            # Note: We divide by the number of *successful* results, not just LOOP_LENGTH_TO_ANALYZE
            # This avoids an error if a model file was missing
            num_results = len(per_slot_results) if per_slot_results else 1
            overall_avg_loss = total_loss_all_slots / num_results
            overall_avg_accuracy = total_acc_all_slots / num_results
            print(f"Overall for L={l_test}: Avg Acc={overall_avg_accuracy:.4f}, Avg Loss={overall_avg_loss:.4f}")
            
            generalization_results[l_test] = {
                'overall_loss': overall_avg_loss,
                'overall_acc': overall_avg_accuracy,
                'per_slot_results': per_slot_results
            }

        else:
            # For L=7, 11, 16, load all their data and test
            results = evaluate_decoders_on_loop_length(
                test_loop_length=l_test,
                base_model_dir=BASE_OUTPUT_DIR,
                num_trained_decoders=LOOP_LENGTH_TO_ANALYZE,
                criterion=criterion,
                device=device
            )
            if results:
                generalization_results[l_test] = results
        print("-" * 50 + "\n")

    # ****************************************************
    # *** MODIFIED SECTION: Call new save/plot functions ***
    # ****************************************************

    # --- 3. Save Generalization Results to Text ---
    save_generalization_results_to_text(generalization_results, BASE_OUTPUT_DIR)

    # --- 4. Plot Overall Generalization Results ---
    plot_and_save_generalization(generalization_results, BASE_OUTPUT_DIR)
    
    # --- 5. Plot Per-Slot Generalization Results ---
    plot_and_save_per_slot_generalization(generalization_results, BASE_OUTPUT_DIR)

    # --- 6. Run Neuron Analysis ---
    # This analysis is still based on the L=14 decoders
    run_and_plot_analysis(BASE_OUTPUT_DIR, LOOP_LENGTH_TO_ANALYZE, TOP_PERCENT_NEURONS)
    
    print("\n" + "="*60)
    print("--- EVALUATION & ANALYSIS COMPLETE ---")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()