"""
ULTIMATE OPTIMIZED causal manipulation script.

Combines ALL optimizations:
1. Train all 14 encoders SIMULTANEOUSLY (14x speedup)
2. Vectorized target extraction (10-50x speedup)
3. Batched RNN processing (20-30x speedup)
4. Fresh data every step (no reuse)

Includes TWO experiments:
- Lesion manipulation: Subtract encoder patterns, measure broken steps
- Swap manipulation: Swap patterns, use decoders to track changes

Expected total speedup: 50-100x faster than original!
Total runtime: ~5-10 minutes (vs 6-9 hours)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import pandas as pd
import numpy as np
import pickle
import os
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time

try:
    from diffenvrnn1 import StandardGatedRNN, LoopEnvironment
except ImportError:
    print("Could not import from 'diffenvrnn1.py'.") 
    exit()

# Paths
PRE_TRAINED_RNN_PATH = "diffenvrnn1.1/model.pth"
BASE_OUTPUT_DIR = "causal_manipulation_L14_ultimate2"

# --- Model & Data Constants ---
LOOP_LENGTH_TO_ANALYZE = 14
HIDDEN_SIZE = 2048
VOCAB_SIZE = 16
OBSERVATION_BANK = list(range(VOCAB_SIZE))

# --- Encoder Training Hyperparameters ---
ENCODER_LEARNING_RATE = 1e-4
ENCODER_BATCH_SIZE = 32
ENCODER_NUM_STEPS = 500  # Changed from NUM_EPOCHS - these are STEPS not epochs!
ENCODER_LOSS_THRESHOLD = 0.005
LOOPS_PER_LENGTH = 40000  

# --- Causal Manipulation Hyperparameters ---
NUM_MANIPULATION_SEQS = 500
MANIPULATION_SEQ_LENGTH = 150
T_INTERVENTION = LOOP_LENGTH_TO_ANALYZE * 2
NUM_SEQS_TO_LOG = 10 

TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiEncoderMLP(nn.Module):
    """
    Train ALL 14 slot encoders simultaneously!
    Each encoder has the same architecture, just different inputs/targets.
    """
    def __init__(self, vocab_size: int, hidden_size: int, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        
        # Separate encoder for each slot
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vocab_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, hidden_size)
            ) for _ in range(num_slots)
        ])
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_slots, vocab_size] - one-hot inputs for all slots
        Returns:
            [batch_size, num_slots, hidden_size] - predictions for all slots
        """
        # Stack outputs from all encoders
        # x[:, k, :] is the one-hot input for slot k
        outputs = torch.stack([
            self.encoders[k](x[:, k, :]) for k in range(self.num_slots)
        ], dim=1)
        return outputs
    
    def get_slot_encoder(self, slot_idx):
        """Extract a single slot's encoder for saving."""
        return self.encoders[slot_idx]


class UltimateBatchedDataset(Dataset):
    """
    ULTIMATE OPTIMIZED: Combines:
    1. Batched RNN processing (processes 32 loops at once)
    2. Vectorized target extraction (NO Python loops!)
    3. Generates data for ALL slots at once
    4. Fresh data every call (no pre-generation)
    """
    def __init__(self, loops: list, loop_length: int, 
                 average_h: np.ndarray, rnn_model: StandardGatedRNN, 
                 env: LoopEnvironment, device: torch.device,
                 rnn_batch_size: int = 32):
        
        self.loops = loops
        self.loop_length = loop_length
        self.rnn_batch_size = rnn_batch_size
        
        # Send average_h to device once
        self.average_h_tensor = torch.from_numpy(average_h).to(device) 
        self.rnn_model = rnn_model
        self.env = env
        self.device = device
        
        # Pre-organize loops into batches
        self.num_batches = (len(loops) + rnn_batch_size - 1) // rnn_batch_size
        
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, batch_idx: int):
        """
        CRITICAL OPTIMIZATIONS:
        1. Processes BATCH of loops through RNN at once
        2. Extracts targets for ALL slots using vectorization
        3. No Python loops!
        """
        # Get loops for this batch
        start_idx = batch_idx * self.rnn_batch_size
        end_idx = min(start_idx + self.rnn_batch_size, len(self.loops))
        batch_loops = self.loops[start_idx:end_idx]
        actual_batch_size = len(batch_loops)
        
        with torch.no_grad():
            # Process entire batch through RNN at once!
            obs_seq, vel_seq, target_seq, loop_lengths = self.env.generate_batch(
                batch_loops, actual_batch_size, MANIPULATION_SEQ_LENGTH
            )
            obs_seq = obs_seq.to(self.device)
            vel_seq = vel_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            
            _, _, _, data_log = self.rnn_model(
                obs_seq, vel_seq, target_seq, loop_lengths, collect_data=True
            )
        
        # ========================================
        # VECTORIZED extraction for ALL slots!
        # ========================================
        
        # Filter valid timesteps
        valid_records = [r for r in data_log if r['timestep'] >= (self.loop_length - 1)]
        
        if len(valid_records) == 0:
            # Return empty tensors
            empty_x = torch.empty(0, self.loop_length, VOCAB_SIZE, device=self.device)
            empty_y = torch.empty(0, self.loop_length, HIDDEN_SIZE, device=self.device)
            return empty_x, empty_y
        
        # Extract indices
        batch_indices = torch.tensor([r['batch_idx'] for r in valid_records], 
                                     dtype=torch.long, device=self.device)
        timesteps = torch.tensor([r['timestep'] for r in valid_records], 
                                dtype=torch.long, device=self.device)
        
        # Stack hidden states
        hidden_states_np = np.stack([r['hidden_state'] for r in valid_records])
        Y_h = torch.from_numpy(hidden_states_np).float().to(self.device)
        
        # ========================================
        # VECTORIZED target computation for ALL slots at once!
        # ========================================
        num_samples = len(valid_records)
        
        # For slot k, input is obs at t-(k-1)
        # We need to create inputs for ALL 14 slots at once
        # Shape will be: [num_samples, 14, vocab_size] (one-hot)
        
        # Create offset array for all slots: [0, 1, 2, ..., 13]
        slot_offsets = torch.arange(0, self.loop_length, device=self.device)  # [14]
        
        # For each sample and each slot, compute: t - (k-1) = t - k + 1
        # timesteps: [N]
        # slot_offsets: [14]
        # Broadcasting: [N, 1] - [1, 14] = [N, 14]
        input_timesteps = timesteps.unsqueeze(1) - slot_offsets.unsqueeze(0)  # [N, 14]
        
        # Expand batch indices for all slots
        batch_idx_expanded = batch_indices.unsqueeze(1).expand(-1, self.loop_length)  # [N, 14]
        
        # Gather observations for all slots at once!
        # obs_seq shape: [batch_size, seq_length]
        # We want: obs_seq[batch_idx, input_timestep] for each (sample, slot)
        X_obs_indices = obs_seq[batch_idx_expanded, input_timesteps].long()  # [N, 14]
        
        # Convert to one-hot: [N, 14, 16]
        X_one_hot = torch.nn.functional.one_hot(X_obs_indices, num_classes=VOCAB_SIZE).float()
        
        # Demean hidden states: [N, H] - [H] = [N, H]
        Y_demeaned = Y_h - self.average_h_tensor
        
        # Expand Y to match all slots: [N, 14, H]
        Y_all_slots = Y_demeaned.unsqueeze(1).expand(-1, self.loop_length, -1)
        
        # Clean up
        del obs_seq, vel_seq, target_seq, data_log, Y_h, X_obs_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return X_one_hot, Y_all_slots


def calculate_average_hidden_state_efficiently(
    rnn_model: StandardGatedRNN, 
    env: LoopEnvironment, 
    loops: list, 
    loop_length: int, 
    device: torch.device, 
    chunk_size: int = 32
) -> np.ndarray:
    """Calculate average hidden state (already optimized with chunking)."""
    print("\n--- Calculating Average Hidden State ---")
    rnn_model.eval()
    
    total_sum = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    total_count = 0
    
    num_loops = len(loops)
    for batch_start in range(0, num_loops, chunk_size):
        batch_end = min(batch_start + chunk_size, num_loops)
        batch_loops = loops[batch_start:batch_end]
        current_batch_size = len(batch_loops)
        
        with torch.no_grad():
            obs_seq, vel_seq, target_seq, loop_lengths_list = env.generate_batch(
                batch_loops, current_batch_size, MANIPULATION_SEQ_LENGTH
            )
            
            obs_seq = obs_seq.to(device)
            vel_seq = vel_seq.to(device)
            target_seq = target_seq.to(device)
            
            _, _, _, data_log = rnn_model(
                obs_seq, vel_seq, target_seq, loop_lengths_list, collect_data=True
            )
        
        h_states_chunk = []
        for record in data_log:
            if record['timestep'] >= (loop_length - 1):
                h_states_chunk.append(record['hidden_state'])
        
        if h_states_chunk:
            h_states_np = np.array(h_states_chunk, dtype=np.float32) 
            total_sum += np.sum(h_states_np, axis=0)
            total_count += h_states_np.shape[0]

        del obs_seq, vel_seq, target_seq, data_log, h_states_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if batch_end % (chunk_size * 20) == 0:
             print(f"  ...processed {batch_end}/{num_loops} loops")

    if total_count == 0:
        raise ValueError("No valid hidden states found.")
        
    average_h = (total_sum / total_count).astype(np.float32)
    print(f"--- Average calculated from {total_count} samples ---")
    return average_h


def train_all_encoders_simultaneously(
    train_dataset: UltimateBatchedDataset, 
    test_dataset: UltimateBatchedDataset,
    average_h: np.ndarray,
    output_dir: str
):
    """
    ULTIMATE: Train ALL 14 encoders at once!
    - Single model with 14 encoder heads
    - One forward/backward pass trains all encoders
    - Generates fresh data every step
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training ALL {LOOP_LENGTH_TO_ANALYZE} Encoders SIMULTANEOUSLY")
    print(f"{'='*60}")
    
    # Create DataLoaders (batch_size=1 because each item is already a batch)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)
    
    # Single model for all slots
    model = MultiEncoderMLP(VOCAB_SIZE, HIDDEN_SIZE, LOOP_LENGTH_TO_ANALYZE).to(device)
    criterion = nn.MSELoss(reduction='none')  
    optimizer = optim.Adam(model.parameters(), lr=ENCODER_LEARNING_RATE)

    history = {
        'train_loss': [],
        'test_loss': [],
        'step_times': [],
        'per_slot_train_loss': {k: [] for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1)},
        'per_slot_test_loss': {k: [] for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1)}
    }

    print(f"Training for {ENCODER_NUM_STEPS} steps")
    print(f"Each step generates fresh unique loops (no reuse!)")
    print(f"All {LOOP_LENGTH_TO_ANALYZE} encoders trained together (14x speedup!)")
    
    for step in range(ENCODER_NUM_STEPS):
        step_start = time.time()
        model.train()
        
        # Get one batch (fresh data generated on-demand!)
        try:
            features_chunk, targets_chunk = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            features_chunk, targets_chunk = next(train_iter)
        
        # Remove DataLoader's batch dimension
        # Shape: [1, N, 14, 16] → [N, 14, 16]
        features = features_chunk.squeeze(0)  # [N, 14, 16]
        targets = targets_chunk.squeeze(0)    # [N, 14, H]
        
        if features.shape[0] == 0:
            print(f"Warning: Empty batch at step {step}, skipping.")
            continue
        
        # Forward pass for ALL encoders
        optimizer.zero_grad()
        outputs = model(features)  # [N, 14, H]
        loss_per_sample_per_slot = criterion(outputs, targets).mean(dim=-1)  # [N, 14]
        loss = loss_per_sample_per_slot.mean()  # Scalar
        
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        history['train_loss'].append(train_loss)
        
        # NEW: Track per-slot training losses
        per_slot_losses = loss_per_sample_per_slot.mean(dim=0).detach().cpu().numpy()  # [14]
        for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1):
            history['per_slot_train_loss'][k].append(per_slot_losses[k-1])

        step_time = time.time() - step_start
        history['step_times'].append(step_time)
        
        # Evaluation
        if (step + 1) % 20 == 0 or step == ENCODER_NUM_STEPS - 1:
            model.eval()
            
            try:
                test_features_chunk, test_targets_chunk = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                test_features_chunk, test_targets_chunk = next(test_iter)
            
            test_features = test_features_chunk.squeeze(0)
            test_targets = test_targets_chunk.squeeze(0)
            
            with torch.no_grad():
                if test_features.shape[0] > 0:
                    test_outputs = model(test_features)
                    test_loss_per_sample_per_slot = criterion(test_outputs, test_targets).mean(dim=-1)
                    test_loss = test_loss_per_sample_per_slot.mean().item()
                    
                    # NEW: Track per-slot test losses
                    per_slot_test_losses = test_loss_per_sample_per_slot.mean(dim=0).cpu().numpy()
                    for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1):
                        history['per_slot_test_loss'][k].append(per_slot_test_losses[k-1])
                else:
                    test_loss = train_loss
                    for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1):
                        history['per_slot_test_loss'][k].append(per_slot_losses[k-1])
            history['test_loss'].append(test_loss)
            print(
                f"Step {step+1}/{ENCODER_NUM_STEPS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Test Loss: {test_loss:.6f} | "
                f"Time: {step_time:.3f}s"
            )

    final_loss = history['test_loss'][-1]
    avg_step_time = np.mean(history['step_times'])
    print(f"\n✓ Training Complete!")
    print(f"Final Test Loss: {final_loss:.6f}")
    print(f"Average step time: {avg_step_time:.3f}s")
    print(f"Total training time: {sum(history['step_times'])/60:.1f} minutes")

    # Save plots
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(history['train_loss'])), history['train_loss'], 
             label='Train Loss', alpha=0.7)
    eval_steps = [i for i in range(ENCODER_NUM_STEPS) 
                  if (i + 1) % 20 == 0 or i == ENCODER_NUM_STEPS - 1]
    plt.plot(eval_steps, history['test_loss'], 
             label='Test Loss', marker='o', linestyle='--')
    plt.title(f'All Encoders Training Loss (ULTIMATE - Simultaneous)')
    plt.xlabel('Training Step')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_loss_ultimate.png"))
    plt.close()

    # Save individual encoder models
    print(f"\nSaving individual slot encoders...")
    encoder_paths = {}
    for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1):
        slot_dir = os.path.join(output_dir, f"slot_{k:02d}")
        os.makedirs(slot_dir, exist_ok=True)
        
        encoder = model.get_slot_encoder(k - 1)
        model_path = os.path.join(slot_dir, 'encoder.pth')
        torch.save(encoder.state_dict(), model_path)
        
        avg_h_path = os.path.join(slot_dir, 'average_hidden_state.npy')
        np.save(avg_h_path, average_h)
        
        encoder_paths[k] = model_path
        
    # Save full model
    full_model_path = os.path.join(output_dir, 'multi_encoder_model.pth')
    torch.save(model.state_dict(), full_model_path)
    print(f"Saved full model to {full_model_path}")

    if final_loss > ENCODER_LOSS_THRESHOLD:
        print(f"WARNING: High final loss ({final_loss:.4f})!")
    
    print(f"\nSaving individual slot training curves...")
    
    for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1):
        slot_dir = os.path.join(output_dir, f"slot_{k:02d}")
        
        # Create individual training curve plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training loss
        ax.plot(range(len(history['per_slot_train_loss'][k])), 
                history['per_slot_train_loss'][k], 
                label=f'Slot {k} Train Loss', 
                color='blue', alpha=0.6, linewidth=1)
        
        # Plot test loss
        test_eval_steps = [i for i in range(ENCODER_NUM_STEPS) 
                          if (i + 1) % 20 == 0 or i == ENCODER_NUM_STEPS - 1]
        ax.plot(test_eval_steps, 
                history['per_slot_test_loss'][k], 
                label=f'Slot {k} Test Loss', 
                color='red', marker='o', linestyle='--', markersize=4)
        
        ax.set_title(f'Slot {k} Encoder Training Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add final loss annotation
        final_train_loss = history['per_slot_train_loss'][k][-1]
        final_test_loss = history['per_slot_test_loss'][k][-1]
        ax.text(0.02, 0.98, 
                f'Final Train Loss: {final_train_loss:.6f}\nFinal Test Loss: {final_test_loss:.6f}',
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        fig.tight_layout()
        plot_path = os.path.join(slot_dir, f'slot_{k:02d}_training_curve.png')
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        
        print(f"  Slot {k:02d}: training curve saved")

    return encoder_paths




def plot_temporal_accuracy(
    pred_classes_control: torch.Tensor,
    pred_classes_manipulated: torch.Tensor,
    targets: torch.Tensor,
    t_intervene: int,
    slot_k: int,
    output_path: str,
    broken_steps: list = None
):
    """Plots the average accuracy at each timestep after intervention.
    
    NEW: Marks broken steps with vertical lines!
    """
    seq_len = targets.shape[1]
    timesteps = np.arange(t_intervene, seq_len)
    
    acc_control_over_time = []
    acc_manip_over_time = []
    
    for t in timesteps:
        control_correct_t = (pred_classes_control[:, t] == targets[:, t])
        acc_control_t = control_correct_t.cpu().float().mean().item()
        acc_control_over_time.append(acc_control_t)
        
        manip_correct_t = (pred_classes_manipulated[:, t] == targets[:, t])
        acc_manip_t = manip_correct_t.cpu().float().mean().item()
        acc_manip_over_time.append(acc_manip_t)

    plt.figure(figsize=(14, 7))
    plt.plot(timesteps, acc_control_over_time, label='Control', marker='o', markersize=4, alpha=0.8)
    plt.plot(timesteps, acc_manip_over_time, label='Manipulated', marker='x', markersize=4, alpha=0.8)
    plt.axvline(x=t_intervene, color='r', linestyle='--', linewidth=2, label=f'Intervention at t={t_intervene}')
    
    # NEW: Mark broken steps with vertical lines!
    if broken_steps:
        for i, t_broken in enumerate(broken_steps):
            if t_broken < seq_len:
                # Only label first few to avoid cluttering legend
                label = f'Broken Step {i+1} (t={t_broken})' if i < 3 else None
                plt.axvline(x=t_broken, color='orange', linestyle=':', alpha=0.7, linewidth=1.5, label=label)
                # Add text annotation above the line
                if i < 10:  # Annotate first 10 broken steps
                    plt.text(t_broken, 0.95, f't={t_broken}', 
                            rotation=90, verticalalignment='top', 
                            fontsize=8, color='orange', alpha=0.8)
    
    plt.title(f'Temporal Accuracy for Slot {slot_k} Lesion', fontsize=16)
    plt.xlabel('Timestep (t)', fontsize=14)
    plt.ylabel('Average Accuracy', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def run_manipulation_for_slot(
    rnn_model,
    encoder_path: str,
    avg_h_path: str,
    slot_k: int,
    loops_test: list,
    obs_test: torch.Tensor,
    vels_test: torch.Tensor,
    targets_test: torch.Tensor,
    log_output_dir: str
):
    """Runs manipulation experiment for a single slot."""
    print(f"\n{'='*20} Running Manipulation for Slot {slot_k} {'='*20}")

    # Load encoder
    encoder = nn.Sequential(
        nn.Linear(VOCAB_SIZE, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, HIDDEN_SIZE)
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()
    rnn_model.eval()

    # Generate predictions
    with torch.no_grad():
        print("Running Control pass...")
        preds_control = forward_pass_with_intervention(rnn_model, obs_test, vels_test) 
        pred_classes_control = torch.argmax(preds_control, dim=-1)

        print(f"Running Manipulated pass (lesioning slot {slot_k})...")
        preds_manipulated = forward_pass_with_intervention(
            rnn_model, obs_test, vels_test,
            encoder=encoder,
            slot_to_manipulate=slot_k,
            t_intervene=T_INTERVENTION
        )
        pred_classes_manipulated = torch.argmax(preds_manipulated, dim=-1)

    # ============================================================
    # IMPORTANT: Calculate broken steps FIRST, before using them!
    # ============================================================
    all_broken_steps = []
    r, k, n = LOOP_LENGTH_TO_ANALYZE, slot_k, 1
    while True:
        t_broken = T_INTERVENTION + (n * r) - (k - 1)
        if t_broken >= MANIPULATION_SEQ_LENGTH:
            break
        if t_broken >= T_INTERVENTION:
             all_broken_steps.append(t_broken)
        n += 1
    
    # NOW we can use all_broken_steps in the step-by-step stats calculation
    # NEW: Calculate step-by-step average accuracy and loss
    step_by_step_stats = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    for t in range(MANIPULATION_SEQ_LENGTH):
        # Control statistics
        control_correct_t = (pred_classes_control[:, t] == targets_test[:, t])
        control_acc_t = control_correct_t.float().mean().item()
        control_loss_t = criterion(preds_control[:, t, :], targets_test[:, t]).mean().item()
        
        # Manipulated statistics
        manip_correct_t = (pred_classes_manipulated[:, t] == targets_test[:, t])
        manip_acc_t = manip_correct_t.float().mean().item()
        manip_loss_t = criterion(preds_manipulated[:, t, :], targets_test[:, t]).mean().item()
        
        # Check if this is a broken step (NOW all_broken_steps is defined!)
        is_broken = t in all_broken_steps
        
        step_by_step_stats.append({
            'timestep': t,
            'control_acc': control_acc_t,
            'control_loss': control_loss_t,
            'manip_acc': manip_acc_t,
            'manip_loss': manip_loss_t,
            'is_broken_step': is_broken,
            'acc_drop': control_acc_t - manip_acc_t
        })

    # IMPROVED: Detailed logging with step-by-step statistics
    log_file_path = os.path.join(log_output_dir, f"slot_{slot_k:02d}_detailed_log.txt")
    with open(log_file_path, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"Manipulation Log for Slot {slot_k}\n")
        f.write(f"{'='*70}\n")
        f.write(f"Intervention at t={T_INTERVENTION}\n")
        f.write(f"Loop Length: {LOOP_LENGTH_TO_ANALYZE}\n")
        f.write(f"Number of test sequences: {NUM_MANIPULATION_SEQS}\n")
        f.write(f"Sequence length: {MANIPULATION_SEQ_LENGTH}\n\n")
        
        # NEW: Step-by-step average accuracy and loss table
        f.write(f"{'='*70}\n")
        f.write(f"STEP-BY-STEP AVERAGE ACCURACY AND LOSS (ALL {NUM_MANIPULATION_SEQS} SEQUENCES)\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"{'t':>4} | {'Control':>8} | {'Manip':>8} | {'Drop':>7} | "
               f"{'C_Loss':>8} | {'M_Loss':>8} | {'Broken':>6}\n")
        f.write(f"{'-'*70}\n")
        
        for stat in step_by_step_stats:
            marker = "→" if stat['timestep'] == T_INTERVENTION else " "
            broken_marker = "✓" if stat['is_broken_step'] else ""
            
            f.write(f"{marker}{stat['timestep']:>3d} | "
                   f"{stat['control_acc']:>8.4f} | "
                   f"{stat['manip_acc']:>8.4f} | "
                   f"{stat['acc_drop']:>7.4f} | "
                   f"{stat['control_loss']:>8.4f} | "
                   f"{stat['manip_loss']:>8.4f} | "
                   f"{broken_marker:>6}\n")
        
        f.write(f"\n{'='*70}\n")
        f.write(f"BROKEN STEPS SUMMARY\n")
        f.write(f"{'='*70}\n")
        f.write(f"Expected broken steps: {len(all_broken_steps)}\n")
        f.write(f"Broken step timesteps: {all_broken_steps[:10]}")
        if len(all_broken_steps) > 10:
            f.write(f" ... (and {len(all_broken_steps) - 10} more)")
        f.write(f"\n\n")
        
        # Statistics for broken vs non-broken steps (post-intervention only)
        post_intervention_stats = [s for s in step_by_step_stats if s['timestep'] >= T_INTERVENTION]
        broken_stats = [s for s in post_intervention_stats if s['is_broken_step']]
        non_broken_stats = [s for s in post_intervention_stats if not s['is_broken_step']]
        
        if broken_stats:
            avg_acc_broken = np.mean([s['manip_acc'] for s in broken_stats])
            avg_drop_broken = np.mean([s['acc_drop'] for s in broken_stats])
            f.write(f"Average accuracy at broken steps: {avg_acc_broken:.4f}\n")
            f.write(f"Average accuracy drop at broken steps: {avg_drop_broken:.4f}\n\n")
        
        if non_broken_stats:
            avg_acc_non_broken = np.mean([s['manip_acc'] for s in non_broken_stats])
            avg_drop_non_broken = np.mean([s['acc_drop'] for s in non_broken_stats])
            f.write(f"Average accuracy at non-broken steps: {avg_acc_non_broken:.4f}\n")
            f.write(f"Average accuracy drop at non-broken steps: {avg_drop_non_broken:.4f}\n\n")
        
        f.write(f"\n{'='*70}\n")
        f.write(f"INDIVIDUAL SEQUENCE DETAILS (First {NUM_SEQS_TO_LOG} sequences)\n")
        f.write(f"{'='*70}\n\n")
        
        for i in range(NUM_SEQS_TO_LOG):
            f.write("="*70 + "\n")
            f.write(f"SEQUENCE {i} | Loop: {loops_test[i]}\n")
            f.write("-"*70 + "\n")
            
            for t in range(MANIPULATION_SEQ_LENGTH):
                obs_t = obs_test[i, t].item()
                target_t = targets_test[i, t].item()
                control_pred = pred_classes_control[i, t].item()
                manip_pred = pred_classes_manipulated[i, t].item()
                
                control_ok = "✅" if control_pred == target_t else "❌"
                manip_ok = "✅" if manip_pred == target_t else "❌"
                
                marker = "→" if t == T_INTERVENTION else " "
                broken_marker = " [BROKEN]" if t in all_broken_steps else ""
                
                f.write(f"{marker} t={t:3d} | obs={obs_t:2d} | target={target_t:2d} | "
                       f"control={control_pred:2d}{control_ok} | manip={manip_pred:2d}{manip_ok}{broken_marker}\n")
            f.write("\n")

    # NEW: Save CSV with step-by-step statistics
    csv_path = os.path.join(log_output_dir, f"slot_{slot_k:02d}_step_stats.csv")
    df_stats = pd.DataFrame(step_by_step_stats)
    df_stats.to_csv(csv_path, index=False)
    print(f"  Step-by-step statistics saved to {csv_path}")

    # Plot with broken steps marked
    plot_path = os.path.join(log_output_dir, f"slot_{slot_k:02d}_temporal_accuracy.png")
    plot_temporal_accuracy(
        pred_classes_control, pred_classes_manipulated, targets_test,
        T_INTERVENTION, slot_k, plot_path, broken_steps=all_broken_steps
    )

    # Calculate accuracies
    def calculate_accuracy_at_steps(pred_classes, targets, steps_list):
        if not steps_list:
            return np.nan
        mask = torch.zeros_like(targets, dtype=torch.bool)
        steps_tensor = torch.tensor(steps_list, dtype=torch.long)
        valid_steps = steps_tensor[steps_tensor < targets.shape[1]]
        if valid_steps.numel() == 0:
            return np.nan
        mask[:, valid_steps] = True
        correct_tensor = (pred_classes == targets) & mask
        total_steps = mask.sum().item()
        if total_steps == 0:
            return np.nan
        return correct_tensor.sum().item() / total_steps

    # Note: all_broken_steps was already calculated above!
    first_broken = all_broken_steps[:1]
    first_five_broken = all_broken_steps[:5]

    # Calculate 6 accuracies
    mask_post = torch.zeros_like(targets_test, dtype=torch.bool)
    mask_post[:, T_INTERVENTION:] = True
    
    acc_control = ((pred_classes_control == targets_test) & mask_post).sum().item() / mask_post.sum().item()
    acc_manip_overall = ((pred_classes_manipulated == targets_test) & mask_post).sum().item() / mask_post.sum().item()
    acc_manip_first_broken = calculate_accuracy_at_steps(pred_classes_manipulated, targets_test, first_broken)
    acc_manip_first_five = calculate_accuracy_at_steps(pred_classes_manipulated, targets_test, first_five_broken)
    acc_manip_all_broken = calculate_accuracy_at_steps(pred_classes_manipulated, targets_test, all_broken_steps)
    
    mask_broken = torch.zeros_like(targets_test, dtype=torch.bool)
    if all_broken_steps:
        valid_steps = torch.tensor(all_broken_steps, dtype=torch.long)
        valid_steps = valid_steps[valid_steps < targets_test.shape[1]]
        if valid_steps.numel() > 0:
            mask_broken[:, valid_steps] = True
    
    mask_not_broken = mask_post & (~mask_broken)
    total_not_broken = mask_not_broken.sum().item()
    acc_manip_not_broken = np.nan
    if total_not_broken > 0:
        acc_manip_not_broken = ((pred_classes_manipulated == targets_test) & mask_not_broken).sum().item() / total_not_broken

    print(f"Results: control={acc_control:.4f}, manip_overall={acc_manip_overall:.4f}")

    return (acc_control, acc_manip_overall, acc_manip_not_broken, 
            acc_manip_first_broken, acc_manip_first_five, acc_manip_all_broken)

def plot_final_results(results: dict, output_dir: str):
    """Plots aggregate summary with all 6 accuracies."""
    slots = sorted(results.keys())
    
    def nan_to_zero(l):
        return [0 if np.isnan(x) else x for x in l]

    acc_control = nan_to_zero([results[s][0] for s in slots])
    acc_manip_overall = nan_to_zero([results[s][1] for s in slots])
    acc_manip_not_broken = nan_to_zero([results[s][2] for s in slots])
    acc_manip_first_broken = nan_to_zero([results[s][3] for s in slots])
    acc_manip_first_five = nan_to_zero([results[s][4] for s in slots])
    acc_manip_all_broken = nan_to_zero([results[s][5] for s in slots])

    x = np.arange(len(slots))
    width = 0.14
    
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.bar(x - 2.5*width, acc_control, width, label='1. Control', color='royalblue')
    ax.bar(x - 1.5*width, acc_manip_overall, width, label='2. Manip Overall', color='skyblue')
    ax.bar(x - 0.5*width, acc_manip_not_broken, width, label='3. Not Broken', color='mediumseagreen')
    ax.bar(x + 0.5*width, acc_manip_first_broken, width, label='4. First Broken', color='gold')
    ax.bar(x + 1.5*width, acc_manip_first_five, width, label='5. First 5 Broken', color='darkorange')
    ax.bar(x + 2.5*width, acc_manip_all_broken, width, label='6. All Broken', color='firebrick')
    
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xlabel('Slot (k)', fontsize=14)
    ax.set_title(f'Effect of Slot-Specific Patterns (ULTIMATE) at t={T_INTERVENTION}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={s}" for s in slots])
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "AGGREGATE_accuracy_ultimate.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nFinal plot saved to {plot_path}")

class DecoderMLP(nn.Sequential):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size)
        )
    
    # Explicitly override forward to ensure it works
    def forward(self, x):
        return super().forward(x)

def forward_pass_with_intervention(
    rnn_model: StandardGatedRNN,
    obs_sequence: torch.Tensor,
    vel_sequence: torch.Tensor,
    encoder: nn.Module = None,
    slot_to_manipulate: int = -1,
    t_intervene: int = -1
) -> torch.Tensor:
    """Forward pass with optional intervention (already batched - efficient)."""
    batch_size, seq_len = obs_sequence.shape
    h_t = rnn_model.h_init.repeat(batch_size, 1)
    all_predictions = []

    for t in range(seq_len):
        obs_t = obs_sequence[:, t]
        vel_t = vel_sequence[:, t].float().unsqueeze(1)

        if t == t_intervene and encoder is not None and slot_to_manipulate > 0: 
            obs_idx_for_slot = t - (slot_to_manipulate - 1)
            
            if obs_idx_for_slot >= 0:
                obs_in_slot = obs_sequence[:, obs_idx_for_slot]
                obs_one_hot = torch.eye(VOCAB_SIZE, device=device)[obs_in_slot]
                demeaned_pattern = encoder(obs_one_hot) 
                h_t = h_t - demeaned_pattern
                
        obs_emb = rnn_model.embed_obs(obs_t)
        vel_emb = rnn_model.embed_vel(vel_t)
        sigma_t = rnn_model.gating_net(h_t, obs_emb, vel_emb)
        z_t = sigma_t * (h_t @ rnn_model.W_rec.T) + (1 - sigma_t) * (obs_emb + vel_emb) + rnn_model.b_rec
        normalized_z = rnn_model.norm_layer(z_t)
        h_t = torch.nn.functional.leaky_relu(normalized_z, negative_slope=0.01)
        
        prediction = h_t @ rnn_model.W_out.T
        all_predictions.append(prediction)

    return torch.stack(all_predictions, dim=1)


def forward_pass_with_swap_and_decode(
    rnn_model: StandardGatedRNN,
    obs_sequence: torch.Tensor,
    vel_sequence: torch.Tensor,
    loops_test: list,
    encoder: nn.Module,
    all_decoders: list,
    slot_to_manipulate: int,
    t_intervene: int,
) -> tuple:
    """
    CORRECTED Swap Manipulation:
    1. At t_intervene, subtract pattern for obs_x (from slot k)
    2. Add pattern for obs_y (NOT in the loop)
    3. Use ONLY decoder k for all t >= t_intervene to track obs_y
    """
    batch_size, seq_len = obs_sequence.shape
    h_t = rnn_model.h_init.repeat(batch_size, 1)
    
    all_rnn_predictions = []
    all_decoder_predictions = []
    
    obs_y_target = torch.zeros(batch_size, dtype=torch.long, device=obs_sequence.device)
    obs_x_removed = torch.zeros(batch_size, dtype=torch.long, device=obs_sequence.device)
    
    decoding_active = False
    decoder_for_tracking = None  # Will be set to decoder k
    
    for t in range(seq_len):
        obs_t = obs_sequence[:, t]
        vel_t = vel_sequence[:, t].float().unsqueeze(1)

        # Swap intervention
        if t == t_intervene:
            obs_idx_for_slot = t - (slot_to_manipulate - 1)

            if obs_idx_for_slot < 0:
                print(f"Warning: obs_idx_for_slot {obs_idx_for_slot} < 0. Skipping swap.")
            else:
                # Get obs_x and its pattern
                obs_x_in_slot = obs_sequence[:, obs_idx_for_slot]
                obs_x_one_hot = torch.eye(VOCAB_SIZE, device=obs_sequence.device)[obs_x_in_slot]
                obs_x_removed = obs_x_in_slot
                ''' ######################### NOT SHAM PART #################################
                # CORRECTED: Find obs_y NOT in each loop (vectorized)
                # Create a mask of valid observations for each sequence
                obs_y_batch = torch.zeros(batch_size, dtype=torch.long, device=obs_sequence.device)
                
                for i in range(batch_size):
                    current_loop = set(loops_test[i])
                    possible_obs = [o for o in OBSERVATION_BANK if o not in current_loop]
                    
                    if possible_obs:
                        obs_y_batch[i] = possible_obs[torch.randint(len(possible_obs), (1,)).item()]
                    else:
                        # Fallback if loop uses all observations
                        obs_y_batch[i] = torch.randint(VOCAB_SIZE, (1,)).item()
                
                obs_y_target = obs_y_batch
                obs_y_one_hot = torch.eye(VOCAB_SIZE, device=obs_sequence.device)[obs_y_batch]
                
                # Get patterns (encoder expects demeaned inputs if trained that way)
                # Encoder outputs demeaned patterns
                pattern_to_remove = encoder(obs_x_one_hot)  # This is already demeaned
                pattern_to_add = encoder(obs_y_one_hot)     # This is also demeaned
                
                # Apply swap
                h_t = h_t - pattern_to_remove + pattern_to_add
                '''
                #######################SHAAAAMMMM########################################
                obs_y_batch = obs_x_in_slot  # Use SAME observation
                obs_y_target = obs_y_batch  # This is now just for logging
                obs_x_removed = obs_x_in_slot

                # Calculate pattern (for verification, but net effect is zero)
                pattern = encoder(obs_x_one_hot)  
                h_t = h_t - pattern + pattern  # Net effect: no change
                # We can skip this line entirely since it does nothing
                #######################SHAAAAMMMM########################################
                # CORRECTED: Use ONLY decoder k for all future timesteps
                decoder_for_tracking = all_decoders[slot_to_manipulate - 1]
                decoding_active = True
        '''#######NOT SHAM
        # CORRECTED: Always use the same decoder k after intervention
        if decoding_active:
            with torch.no_grad():
                decode_logits = decoder_for_tracking(h_t)
                decode_pred_class = torch.argmax(decode_logits, dim=-1)
                
            all_decoder_predictions.append(decode_pred_class)
        '''
        #######################SHAAAAMMMM########################################
        if decoding_active:
            # For sham swap: track the ACTUAL observation k-1 steps back
            obs_idx_for_current_slot = t - (slot_to_manipulate - 1)
            
            # Only decode if we have valid history
            if obs_idx_for_current_slot >= 0:
                # Get the actual observation that should be in slot k
                actual_obs_in_slot = obs_sequence[:, obs_idx_for_current_slot]
                
                with torch.no_grad():
                    decode_logits = decoder_for_tracking(h_t)
                    decode_pred_class = torch.argmax(decode_logits, dim=-1)
                
                if t < T_INTERVENTION + 5:  # Only print first 5 timesteps after intervention
                    print(f"DEBUG t={t}, slot_k={slot_to_manipulate}, "
                        f"obs_idx={obs_idx_for_current_slot}, "
                        f"preds[:5]={decode_pred_class[:5].cpu().tolist()}, "
                        f"targets[:5]={actual_obs_in_slot[:5].cpu().tolist()}", 
                        flush=True)

                # Store both prediction and ground truth
                all_decoder_predictions.append((decode_pred_class, actual_obs_in_slot))
            else:
                # If we don't have valid history, store None
                all_decoder_predictions.append((None, None))
            #######################SHAAAAMMMM########################################

        # Standard RNN forward pass
        obs_emb = rnn_model.embed_obs(obs_t)
        vel_emb = rnn_model.embed_vel(vel_t)
        sigma_t = rnn_model.gating_net(h_t, obs_emb, vel_emb)
        z_t = sigma_t * (h_t @ rnn_model.W_rec.T) + (1 - sigma_t) * (obs_emb + vel_emb) + rnn_model.b_rec
        normalized_z = rnn_model.norm_layer(z_t)
        h_t = torch.nn.functional.leaky_relu(normalized_z, negative_slope=0.01)
        
        prediction = h_t @ rnn_model.W_out.T
        all_rnn_predictions.append(prediction)
    '''############NOT SHAM
    return (
        torch.stack(all_rnn_predictions, dim=1),
        torch.stack(all_decoder_predictions, dim=1) if all_decoder_predictions else None,
        obs_y_target,
        obs_x_removed
    )
    '''
###########################SHAM
    if all_decoder_predictions and all_decoder_predictions[0][0] is not None:
        decoder_preds = torch.stack([p[0] for p in all_decoder_predictions], dim=1)
        decoder_targets = torch.stack([p[1] for p in all_decoder_predictions], dim=1)
    else:
        decoder_preds = None
        decoder_targets = None

    return (
        torch.stack(all_rnn_predictions, dim=1),
        decoder_preds,  # Predictions
        decoder_targets,  # Now dynamic ground truth (not static obs_y)
        obs_x_removed
    )
###########################SHAM
def plot_swap_decode_accuracy(
    accuracy_over_time: np.ndarray,
    t_intervene: int,
    seq_len: int,
    slot_k: int,
    output_path: str
):
    """Plots decoder accuracy at predicting the added obs_y."""
    timesteps = np.arange(t_intervene, seq_len)
    
    if len(accuracy_over_time) != len(timesteps):
        min_len = min(len(accuracy_over_time), len(timesteps))
        timesteps = timesteps[:min_len]
        accuracy_over_time = accuracy_over_time[:min_len]

    decoder_labels = [f"D{((slot_k - 1 + i) % LOOP_LENGTH_TO_ANALYZE) + 1:02d}" 
                     for i in range(len(timesteps))]

    plt.figure(figsize=(20, 8))
    plt.plot(timesteps, accuracy_over_time, 
             label=f'obs_y Decode Accuracy (Slot {slot_k} Encoder)', 
             marker='o', markersize=5)
    
    for i in range(len(timesteps)):
        if i % (LOOP_LENGTH_TO_ANALYZE // 2) == 0:
             plt.text(timesteps[i], accuracy_over_time[i] + 0.02, 
                     decoder_labels[i], ha='center', size='x-small')

    plt.axvline(x=t_intervene, color='r', linestyle='--', 
                label=f'Swap Intervention at t={t_intervene}')
    plt.title(f'obs_y Decoding Accuracy After Slot {slot_k} "Swap" Manipulation', 
             fontsize=16)
    plt.xlabel('Timestep (t)', fontsize=14)
    plt.ylabel(f'Accuracy of Predicting Added obs_y', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.savefig(output_path)
    plt.close()


def run_swap_manipulation_for_slot(
    rnn_model: StandardGatedRNN,
    encoder_path: str,
    slot_k: int,
    loops_test: list,
    obs_test: torch.Tensor,
    vels_test: torch.Tensor,
    targets_test: torch.Tensor,
    all_decoders: list,
    log_output_dir: str
):
    """Runs the CORRECTED swap manipulation for slot k."""
    print(f"\n{'='*20} Running SWAP Manipulation for Slot {slot_k} {'='*20}")

    # Load encoder and average_h
    encoder = nn.Sequential(
        nn.Linear(VOCAB_SIZE, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, HIDDEN_SIZE)
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()

    rnn_model.eval()

    print(f"Running Swap/Decode pass (manipulating slot {slot_k} at t={T_INTERVENTION})...")
    print(f"Using ONLY Decoder D{slot_k:02d} for all post-intervention timesteps")
    '''not sham
    rnn_preds_manip, decoder_preds, obs_y_targets, obs_x_removed = \
        forward_pass_with_swap_and_decode(
            rnn_model, obs_test, vels_test, loops_test,
            encoder=encoder,
            all_decoders=all_decoders,
            slot_to_manipulate=slot_k,
            t_intervene=T_INTERVENTION,
        )
    '''
# ############sham
    rnn_preds_manip, decoder_preds, decoder_targets_dynamic, obs_x_removed = \
        forward_pass_with_swap_and_decode(
            rnn_model, obs_test, vels_test, loops_test,
            encoder=encoder,
            all_decoders=all_decoders,
            slot_to_manipulate=slot_k,
            t_intervene=T_INTERVENTION,
        )
##############sham

    rnn_pred_classes_manip = torch.argmax(rnn_preds_manip, dim=-1)

    # Analyze decoder accuracy
    #not sham######## correct_decodes = (decoder_preds == obs_y_targets.unsqueeze(1))
    correct_decodes = (decoder_preds == decoder_targets_dynamic) # THIS IS SHAM!!

    decoder_acc_over_time = correct_decodes.float().mean(dim=0).cpu().numpy()
    
    # Analyze RNN's task accuracy
    mask_post_intervention = torch.zeros_like(targets_test, dtype=torch.bool)
    mask_post_intervention[:, T_INTERVENTION:] = True
    
    correct_rnn_task = (rnn_pred_classes_manip == targets_test) & mask_post_intervention
    rnn_task_acc_total_post = correct_rnn_task.sum().item() / mask_post_intervention.sum().item()
    print(f"  RNN Task Accuracy (post-swap): {rnn_task_acc_total_post:.4f}")

    # Save detailed log
    log_file_path = os.path.join(log_output_dir, f"SWAP_slot_{slot_k:02d}_detailed_log.txt")
    
    with open(log_file_path, 'w') as f:
        f.write(f"SWAP Manipulation Log for Slot {slot_k}\n")
        f.write(f"Intervention at t={T_INTERVENTION}\n")
        f.write(f"RNN Task Accuracy (t>={T_INTERVENTION}): {rnn_task_acc_total_post:.4f}\n")
        f.write("="*70 + "\n")
        f.write("Decoder Accuracy by Timestep:\n")
        
        timesteps_log = np.arange(T_INTERVENTION, MANIPULATION_SEQ_LENGTH)
        for i in range(len(decoder_acc_over_time)):
            t = timesteps_log[i]
            decoder_k_used = ((slot_k - 1) + i) % LOOP_LENGTH_TO_ANALYZE + 1
            acc = decoder_acc_over_time[i]
            f.write(f"  t={t: <3} | Decoder: D{decoder_k_used:02d} | Accuracy: {acc:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"First {NUM_SEQS_TO_LOG} sequences:\n\n")
        '''################################NOT SHAM
        for i in range(NUM_SEQS_TO_LOG):
            f.write("="*70 + "\n")
            f.write(f"SEQUENCE {i} | Loop: {loops_test[i]}\n")
            f.write(f"Intervention: REMOVED obs_x={obs_x_removed[i].item()} (slot {slot_k}), "
                   f"ADDED obs_y={obs_y_targets[i].item()}\n")
            f.write("-"*70 + "\n")
            
            for j in range(MANIPULATION_SEQ_LENGTH - T_INTERVENTION):
                t = T_INTERVENTION + j
                if t >= MANIPULATION_SEQ_LENGTH: break
                
                decoder_k_used = ((slot_k - 1) + j) % LOOP_LENGTH_TO_ANALYZE + 1
                target_y = obs_y_targets[i].item()
                decoded_pred_y = decoder_preds[i, j].item()
                decode_ok = "✅" if target_y == decoded_pred_y else "❌"
                
                rnn_target = targets_test[i, t].item()
                rnn_pred = rnn_pred_classes_manip[i, t].item()
                rnn_ok = "✅" if rnn_target == rnn_pred else "❌"

                f.write(f"t={t:3d} | D{decoder_k_used:02d} | "
                       f"target_y={target_y:2d} pred_y={decoded_pred_y:2d}{decode_ok} | "
                       f"RNN: target={rnn_target:2d} pred={rnn_pred:2d}{rnn_ok}\n")
        '''

###################################SHAM#####################################################
        for i in range(NUM_SEQS_TO_LOG):
            f.write("="*70 + "\n")
            f.write(f"SEQUENCE {i} | Loop: {loops_test[i]}\n")
            f.write(f"SHAM SWAP: obs_x={obs_x_removed[i].item()} (slot {slot_k}) - NO NET CHANGE\n")
            f.write(f"Decoder tracking: obs at t-(k-1) where k={slot_k}\n")
            f.write("-"*70 + "\n")
            
            for j in range(MANIPULATION_SEQ_LENGTH - T_INTERVENTION):
                t = T_INTERVENTION + j
                if t >= MANIPULATION_SEQ_LENGTH: break
                
                decoder_k_used = ((slot_k - 1) + j) % LOOP_LENGTH_TO_ANALYZE + 1
                
                # Get the dynamic target (obs at t-(k-1))
                target_slot_obs = decoder_targets_dynamic[i, j].item()
                decoded_pred_y = decoder_preds[i, j].item()
                decode_ok = "✅" if target_slot_obs == decoded_pred_y else "❌"
                
                rnn_target = targets_test[i, t].item()
                rnn_pred = rnn_pred_classes_manip[i, t].item()
                rnn_ok = "✅" if rnn_target == rnn_pred else "❌"
                
                # Calculate which observation index we're tracking
                obs_idx_tracked = t - (slot_k - 1)

                f.write(f"t={t:3d} | D{decoder_k_used:02d} | "
                       f"target(t-{slot_k-1})={target_slot_obs:2d} pred={decoded_pred_y:2d}{decode_ok} | "
                       f"RNN: target={rnn_target:2d} pred={rnn_pred:2d}{rnn_ok}\n")
###################################SHAM#####################################################
    ''' ############################NOT SHAM########################
    # Save plot
    plot_path = os.path.join(log_output_dir, f"SWAP_slot_{slot_k:02d}_decoder_accuracy.png")
    plot_swap_decode_accuracy(
        decoder_acc_over_time, T_INTERVENTION, MANIPULATION_SEQ_LENGTH, slot_k, plot_path
    )
    '''
###################################SHAM#####################################################
    # Update plot title for sham swap
    plot_path = os.path.join(log_output_dir, f"SHAM_SWAP_slot_{slot_k:02d}_decoder_accuracy.png")
    plot_swap_decode_accuracy(
        decoder_acc_over_time, T_INTERVENTION, MANIPULATION_SEQ_LENGTH, slot_k, plot_path
    )
###################################SHAM#####################################################
    print(f"Saved SWAP results for slot {slot_k}")


def main():
    """Main orchestration with ULTIMATE optimizations."""
    # Create directories
    encoder_dir = os.path.join(BASE_OUTPUT_DIR, "encoders")
    results_dir = os.path.join(BASE_OUTPUT_DIR, "results")
    logs_dir = os.path.join(BASE_OUTPUT_DIR, "logs")
    
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("\n" + "="*60)
    print("ULTIMATE CAUSAL MANIPULATION")
    print("Optimizations:")
    print("  ✓ Train all 14 encoders simultaneously (14x speedup)")
    print("  ✓ Vectorized target extraction (10-50x speedup)")
    print("  ✓ Batched RNN processing (20-30x speedup)")
    print("  ✓ Fresh data every step (no reuse)")
    print("Expected: 5-10 min total (vs 6-9 hours original)")
    print("="*60 + "\n")

    # Load RNN
    print(f"Loading RNN from {PRE_TRAINED_RNN_PATH}...")
    if not os.path.exists(PRE_TRAINED_RNN_PATH):
        raise FileNotFoundError(f"RNN not found at {PRE_TRAINED_RNN_PATH}")
    
    rnn_model = StandardGatedRNN(hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE).to(device)
    rnn_model.load_state_dict(torch.load(PRE_TRAINED_RNN_PATH, map_location=device))
    rnn_model.eval()
    print("✓ RNN loaded")
    
    env = LoopEnvironment(
        observation_bank=OBSERVATION_BANK,
        loop_lengths=[LOOP_LENGTH_TO_ANALYZE],
        velocities=[1]
    )
    
    # Generate loops
    print(f"\nGenerating {LOOPS_PER_LENGTH} loops...")
    all_loops = env.generate_unique_loops(LOOPS_PER_LENGTH, [LOOP_LENGTH_TO_ANALYZE])
    print(f"Generated {len(all_loops)} unique loops")
    
    # Calculate average hidden state
    average_h = calculate_average_hidden_state_efficiently(
        rnn_model, env, all_loops, LOOP_LENGTH_TO_ANALYZE, device
    )
    
    # Split loops
    print(f"\nSplitting loops into train/test...")
    train_loops, test_loops = train_test_split(
        all_loops, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE
    )
    del all_loops
    print(f"Train: {len(train_loops)}, Test: {len(test_loops)}")
    
    # ========================================
    # ULTIMATE: Train ALL encoders simultaneously
    # ========================================
    print("\n" + "="*60)
    print("ENCODER TRAINING (ULTIMATE)")
    print("="*60)
    
    train_dataset = UltimateBatchedDataset(
        train_loops, LOOP_LENGTH_TO_ANALYZE, average_h, 
        rnn_model, env, device, rnn_batch_size=ENCODER_BATCH_SIZE
    )
    test_dataset = UltimateBatchedDataset(
        test_loops, LOOP_LENGTH_TO_ANALYZE, average_h, 
        rnn_model, env, device, rnn_batch_size=ENCODER_BATCH_SIZE
    )
    
    encoder_paths = train_all_encoders_simultaneously(
        train_dataset, test_dataset, average_h, encoder_dir
    )
    
    del train_loops, test_loops, train_dataset, test_dataset
    
    # ========================================
    # Run manipulation experiments
    # ========================================
    print("\n" + "="*60)
    print("MANIPULATION EXPERIMENTS")
    print("="*60)
    
    manip_env = LoopEnvironment(
        observation_bank=OBSERVATION_BANK,
        loop_lengths=[LOOP_LENGTH_TO_ANALYZE],
        velocities=[1]
    )

    print(f"\nGenerating test set ({NUM_MANIPULATION_SEQS} sequences)...")
    loops_test = manip_env.generate_unique_loops(NUM_MANIPULATION_SEQS, [LOOP_LENGTH_TO_ANALYZE])
    obs_test, vels_test, targets_test, _ = manip_env.generate_batch(
        loops_test, NUM_MANIPULATION_SEQS, MANIPULATION_SEQ_LENGTH
    )
    
    obs_test = obs_test.to(device)
    vels_test = vels_test.to(device)
    targets_test = targets_test.to(device)
    print("✓ Test set ready")

    all_results = {}
    summary_path = os.path.join(results_dir, "summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("Causal Manipulation Summary (ULTIMATE)\n")
        f.write(f"Loop Length: {LOOP_LENGTH_TO_ANALYZE}\n")
        f.write(f"Intervention: t={T_INTERVENTION}\n")
        f.write(f"Test Sequences: {NUM_MANIPULATION_SEQS}\n")
        f.write("-" * 50 + "\n\n")

        for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1):
            encoder_path = encoder_paths[k]
            slot_dir = os.path.dirname(encoder_path)
            avg_h_path = os.path.join(slot_dir, 'average_hidden_state.npy')
    
            accs = run_manipulation_for_slot(
                rnn_model, encoder_path, avg_h_path, k,
                loops_test, obs_test, vels_test, targets_test, logs_dir
            )
            
            all_results[k] = accs
            
            f.write(f"Slot {k:02d}:\n")
            f.write(f"  1. Control:         {accs[0]:.4f}\n")
            f.write(f"  2. Manip Overall:   {accs[1]:.4f}\n")
            f.write(f"  3. Not Broken:      {accs[2]:.4f}\n")
            f.write(f"  4. First Broken:    {accs[3]:.4f}\n")
            f.write(f"  5. First 5 Broken:  {accs[4]:.4f}\n")
            f.write(f"  6. All Broken:      {accs[5]:.4f}\n\n")
    
    print(f"\nSummary saved to {summary_path}")

    plot_final_results(all_results, results_dir)

    # ========================================
    # Run SWAP manipulation experiments
    # ========================================
    print("\n" + "="*60)
    print("SWAP/DECODE MANIPULATION EXPERIMENTS")
    print("="*60)
    
    swap_log_dir = os.path.join(BASE_OUTPUT_DIR, "swap_results")
    os.makedirs(swap_log_dir, exist_ok=True)
    
    # Load all decoders
    DECODER_BASE_PATH = "/nfs/nhome/live/jwhittington/yang_project/slot_decoders_hybrid"
    
    print(f"\nLoading all {LOOP_LENGTH_TO_ANALYZE} decoders from {DECODER_BASE_PATH}...")
    all_decoders_list = []
    try:
        for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1):
            decoder_path = os.path.join(DECODER_BASE_PATH, f'slot_{k:02d}', 'decoder.pth')
            if not os.path.exists(decoder_path):
                print(f"WARNING: Decoder not found at {decoder_path}")
                print("Skipping SWAP experiments.")
                all_decoders_list = None
                break
            decoder = DecoderMLP(input_size=HIDDEN_SIZE, output_size=VOCAB_SIZE).to(device)
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            decoder.eval()
            all_decoders_list.append(decoder)
        
        if all_decoders_list is not None:
            print(f"✓ Loaded {len(all_decoders_list)} decoders")
            
            for k in range(1, LOOP_LENGTH_TO_ANALYZE + 1):
                encoder_path = encoder_paths[k]

                run_swap_manipulation_for_slot(
                    rnn_model=rnn_model,
                    encoder_path=encoder_path,
                    slot_k=k,
                    loops_test=loops_test,
                    obs_test=obs_test,
                    vels_test=vels_test,
                    targets_test=targets_test,
                    all_decoders=all_decoders_list,
                    log_output_dir=swap_log_dir
                )
            
            print(f"\n✓ SWAP experiments complete")
            print(f"Results in: {swap_log_dir}")
        
    except Exception as e:
        print(f"Error loading decoders: {e}")
        print("Skipping SWAP experiments.")

    print("\n" + "="*60)
    print("✓ ALL EXPERIMENTS COMPLETE!")
    print(f"Lesion results: {results_dir}")
    print(f"Swap results: {swap_log_dir if all_decoders_list else 'N/A (decoders not found)'}")
    print(f"All outputs in: {BASE_OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
    main()