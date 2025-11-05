"""
for loop length = L
the step where the target obs is the first obs is True
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Dict
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from matplotlib.ticker import MaxNLocator
import numpy as np
import math
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- LoopEnvironment, GatingMLP, StandardGatedRNN classes are unchanged ---
class LoopEnvironment:
    def __init__(self,
                 observation_bank: List[int],
                 velocities: List[int],
                 loop_lengths: List[int],
                 behaviour_type: str = 'random'):

        self.observation_bank = observation_bank
        self.vocab_size = len(observation_bank)
        self.velocities = velocities
        self.behaviour_type = behaviour_type
        self.loop_lengths = loop_lengths

    def generate_unique_loops(self, num_loops: int, lengths: List[int]) -> List[List[int]]:
        generated_loops = set()
        # Ensure we only generate loops of the requested lengths
        allowed_lengths = [l for l in lengths if l in self.loop_lengths]
        if not allowed_lengths:
            return []
            
        attempts = 0
        # Increase max_attempts as finding unique loops can be hard
        max_attempts = num_loops * 500000 
        while len(generated_loops) < num_loops and attempts < max_attempts:
            length = random.choice(allowed_lengths)
            # Ensure loop elements are unique within the loop
            loop_tuple = tuple(np.random.choice(self.observation_bank, size=length, replace=False))
            generated_loops.add(loop_tuple)
            attempts += 1
        if len(generated_loops) < num_loops:
            print(f"Warning: Could only generate {len(generated_loops)} unique loops of lengths {allowed_lengths} after {max_attempts} attempts.")
        return [list(loop) for loop in generated_loops]

    def generate_batch(self, loops: List[List[int]], batch_size: int, seq_length: int) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        actual_batch_size = len(loops)
        obs_batch = np.zeros((actual_batch_size, seq_length), dtype=np.int64)
        vel_batch = np.zeros((actual_batch_size, seq_length), dtype=np.int64)
        target_batch = np.zeros((actual_batch_size, seq_length), dtype=np.int64)
        lengths_batch = []
        for i in range(actual_batch_size):
            loop = loops[i]
            loop_length = len(loop)
            lengths_batch.append(loop_length)
            current_pos_idx = 0 # starting each loop from the very first obs
            #current_pos_idx = np.random.randint(0, loop_length) # random starting obs
            for t in range(seq_length):
                obs_batch[i, t] = loop[current_pos_idx]
                velocity = 1 
                vel_batch[i, t] = velocity
                next_pos_idx = (current_pos_idx + velocity) % loop_length
                target_batch[i, t] = loop[next_pos_idx]
                current_pos_idx = next_pos_idx
        obs_tensor = torch.from_numpy(obs_batch).to(device)
        vel_tensor = torch.from_numpy(vel_batch).to(device)
        target_tensor = torch.from_numpy(target_batch).long().to(device)
        return obs_tensor, vel_tensor, target_tensor, lengths_batch

class GatingMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int = 512):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size * 3, intermediate_size)
        self.layer2 = nn.Linear(intermediate_size, hidden_size)
    def forward(self, h_t: torch.Tensor, obs_t_embedded: torch.Tensor, vel_t_embedded: torch.Tensor) -> torch.Tensor:
        mlp_input = torch.cat([h_t, obs_t_embedded, vel_t_embedded], dim=-1)
        hidden = F.relu(self.layer1(mlp_input))
        return torch.sigmoid(self.layer2(hidden))

def create_target_seen_mask(target_sequence: torch.Tensor, loop_lengths: List[int]) -> torch.Tensor:

    batch_size, seq_len = target_sequence.shape
    mask = torch.zeros_like(target_sequence, dtype=torch.float32)

    for i in range(batch_size):
        loop_len = loop_lengths[i]
        start_t = loop_len - 1 # say L=11, if loop_len -1, t=12 true (first time the target obs is first obs); if loop_len, t=12 false (first time seeing first obs again as true)
        if start_t < seq_len:
            mask[i, start_t:] = 1.0
    return mask

class StandardGatedRNN(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_obs = nn.Embedding(vocab_size, hidden_size)
        self.embed_vel = nn.Linear(1, hidden_size)
        self.norm_layer = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.gating_net = GatingMLP(hidden_size=hidden_size)
        self.W_rec = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_out = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.h_init = nn.Parameter(torch.empty(1, hidden_size))
        self.b_rec = nn.Parameter(torch.empty(hidden_size))
        self.initialize_weights()

    def initialize_weights(self):
        for module in [self.embed_vel, self.gating_net]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.embed_obs.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.W_out, nonlinearity='relu')
        nn.init.orthogonal_(self.W_rec)
        nn.init.normal_(self.h_init, mean=0.0, std=0.06)
        nn.init.constant_(self.b_rec, 0.1)

    def forward(self, obs_sequence: torch.Tensor, vel_sequence: torch.Tensor, target_sequence: torch.Tensor, 
                loop_lengths: List[int], collect_data: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = obs_sequence.shape
        h_t = self.h_init.repeat(batch_size, 1)
        all_predictions = []
        data_log = []

        for t in range(seq_len):
            obs_t = obs_sequence[:, t]
            vel_t = vel_sequence[:, t]
            obs_emb = self.embed_obs(obs_t)
            vel_t_tensor = vel_t.float().unsqueeze(1)
            vel_emb = self.embed_vel(vel_t_tensor)
            sigma_t = self.gating_net(h_t, obs_emb, vel_emb)
            z_t = sigma_t * (h_t @ self.W_rec.T) + (1 - sigma_t) * (obs_emb + vel_emb) + self.b_rec
            normalized_z = self.norm_layer(z_t)
            h_t = F.leaky_relu(normalized_z, negative_slope=0.01)
            prediction = h_t @ self.W_out.T
            all_predictions.append(prediction)

            if collect_data:
                pred_class = torch.argmax(prediction, dim=-1)
                target_t = target_sequence[:, t]
                is_correct = (pred_class == target_t)
                
                # Loop through each sequence in the batch
                for i in range(batch_size):
                    data_log.append({
                        'batch_idx': i,
                        'loop_length': loop_lengths[i],
                        'timestep': t,
                        'obs': obs_t[i].item(),
                        'vel': vel_t[i].item(),
                        'target': target_t[i].item(),
                        'prediction': pred_class[i].item(),
                        'is_correct': is_correct[i].item(),
                        # Detach and clone to avoid holding onto the computation graph
                        'hidden_state': h_t[i].detach().clone().cpu().numpy(),
                        'gate_value': sigma_t[i].detach().clone().cpu().numpy()
                    })

        predictions_tensor = torch.stack(all_predictions, dim=1)
        mask = create_target_seen_mask(target_sequence, loop_lengths)
        per_token_loss = F.cross_entropy(
            predictions_tensor.view(-1, self.vocab_size),
            target_sequence.view(-1),
            reduction='none'
        ).view(batch_size, seq_len)
        masked_loss = per_token_loss * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-8)
        return loss, predictions_tensor, mask, data_log

class DataPoolManager:
    def __init__(self, environment: LoopEnvironment, train_lengths: List[int], test_lengths: List[int], loops_per_length: int):
        print(f"\n--- Initializing DataPoolManager ---")
        self.env = environment
        self.observation_bank = set(self.env.observation_bank)
        
        # 1. Create separate pools for each length
        self.train_pools = defaultdict(list)
        self.test_pools = defaultdict(list)

        print("Generating training loops...")
        for length in train_lengths:
            loops = self.env.generate_unique_loops(loops_per_length, [length])
            self.train_pools[length] = loops
            print(f"  - Generated {len(loops)} loops of length {length} for training.")
            
        print("Generating testing loops...")
        for length in test_lengths:
            loops = self.env.generate_unique_loops(loops_per_length, [length])
            self.test_pools[length] = loops
            print(f"  - Generated {len(loops)} loops of length {length} for testing.")
        
        self.train_lengths = list(self.train_pools.keys())
        self.test_lengths = list(self.test_pools.keys())
        print("-" * 20)

    def _get_loops_from_pool(self, num_loops: int, pool_dict: Dict[int, List], allowed_lengths: List[int]) -> List[List[int]]:
        """Pulls loops from the specified pools and removes them."""
        batch_loops = []

        for _ in range(num_loops):
            # Pick a random length from the allowed lengths for this pool
            length = random.choice(allowed_lengths)
            
            # If the pool for this length is empty, try another one
            attempts = 0
            while not pool_dict[length] and attempts < len(allowed_lengths) * 2:
                print(f"Pool for length {length} is empty, trying another.")
                length = random.choice(allowed_lengths)
                attempts += 1
            
            # If we still couldn't find a non-empty pool, stop
            if not pool_dict[length]:
                print(f"Warning: All specified pools are empty. Returning {len(batch_loops)} loops.")
                break

            # Pop a loop to ensure it's removed
            loop = pool_dict[length].pop()
            batch_loops.append(loop)

        return batch_loops

    def _ensure_observation_coverage(self, batch_loops: List[List[int]], pool_dict: Dict[int, List], allowed_lengths: List[int]):
        
        # Use a while loop that re-calculates missing observations each iteration
        max_swaps = len(self.observation_bank) # Prevent infinite loops
        swaps = 0
        while swaps < max_swaps:
            seen_obs = set(obs for loop in batch_loops for obs in loop)
            missing_obs = self.observation_bank - seen_obs

            if not missing_obs:
                break # All observations are covered, we're done.

            # Pick one missing observation to find
            obs_to_find = missing_obs.pop()
            found_replacement = False
            
            # Search for a replacement loop
            for length in random.sample(allowed_lengths, len(allowed_lengths)):
                for i, loop in enumerate(pool_dict[length]):
                    if obs_to_find in loop:
                        # Perform the swap
                        swap_idx = random.randrange(len(batch_loops))
                        swapped_out_loop = batch_loops[swap_idx]
                        pool_dict[len(swapped_out_loop)].append(swapped_out_loop)
                        batch_loops[swap_idx] = pool_dict[length].pop(i)
                        
                        found_replacement = True
                        break
                if found_replacement:
                    break
            
            swaps += 1

            if not found_replacement:
                # If we couldn't find a replacement for this obs, no point continuing
                # Put the obs we took out back in the set to report it accurately
                missing_obs.add(obs_to_find)
                break
                
        final_seen_obs = set(obs for loop in batch_loops for obs in loop)
        if self.observation_bank - final_seen_obs:
            print(f"Warning: Could not ensure full coverage. Missing: {self.observation_bank - final_seen_obs}")

        return batch_loops


    def get_batch(self, batch_size: int, seq_length: int, pool_type: str) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        
        if pool_type == 'train':
            pool_dict, allowed_lengths = self.train_pools, self.train_lengths
        else: # 'test'
            pool_dict, allowed_lengths = self.test_pools, self.test_lengths
        
        if not any(pool_dict.values()):
            print(f"FATAL: {pool_type.capitalize()} pools are completely empty. Cannot generate a batch.")
            return None, None, None, None, None

        # 1. Get a preliminary batch of loops
        batch_loops = self._get_loops_from_pool(batch_size, pool_dict, allowed_lengths)

        if not batch_loops: # If we couldn't get any loops
            return None, None, None, None, None
            
        # 2. Ensure observation coverage
        batch_loops = self._ensure_observation_coverage(batch_loops, pool_dict, allowed_lengths)

        # In DataPoolManager.get_batch
        obs_tensor, vel_tensor, target_tensor, lengths_batch = self.env.generate_batch(batch_loops, len(batch_loops), seq_length)
        return obs_tensor, vel_tensor, target_tensor, lengths_batch, batch_loops

def evaluate(model, pool_manager, batch_size, seq_length, collect_data: bool = False):
    """
    Calculates accuracy and sorts the raw results into a dictionary keyed by loop length.
    """
    model.eval()
    with torch.no_grad():
        obs, vels, targets, loop_lengths, batch_loops_raw = pool_manager.get_batch(batch_size, seq_length, 'test')
        if obs is None:
            return 0.0, None, None
        
        # This `data` is the list of dictionaries we want to keep.
        _, predictions, mask, data = model(obs, vels, targets, loop_lengths, collect_data=collect_data)

        pred_classes = torch.argmax(predictions, dim=-1)
        correct_predictions = (pred_classes == targets).float()
        masked_correct = correct_predictions * mask
        accuracy = (masked_correct.sum() / (mask.sum() + 1e-8)).item()

        results_by_length = defaultdict(lambda: {'preds': [], 'loops': []})
        
        for i, length in enumerate(loop_lengths):
            results_by_length[length]['preds'].append(pred_classes[i].unsqueeze(0).cpu())
            results_by_length[length]['loops'].append(batch_loops_raw[i])

        final_results = {}

        for length, result_data in results_by_length.items():
            final_results[length] = (torch.cat(result_data['preds'], dim=0), result_data['loops'])

        return accuracy, final_results, data

def plot_hypothesis_distribution_bars(results_by_length, previous_results_by_length, epoch, hypothesized_timesteps, plot_dir="plots"):
    """
    Generates a SEPARATE bar chart plot for each loop length found in the batch.
    """
    os.makedirs(plot_dir, exist_ok=True)
    if previous_results_by_length is None:
        previous_results_by_length = {}

    for length, (preds_tensor, loops_list) in results_by_length.items():
        predictions_cpu = preds_tensor.numpy()
        obs_to_pos_maps = [{obs: i + 1 for i, obs in enumerate(loop)} for loop in loops_list]
        predictions_at_timestep = defaultdict(Counter)
        
        # --- MODIFIED ---
        # Remove the filter to analyze all available timesteps from t=0 onwards.
        timesteps_to_analyze = sorted([t for t in hypothesized_timesteps if t < predictions_cpu.shape[1]])

        for i in range(len(loops_list)):
            for t in timesteps_to_analyze:
                predicted_obs = predictions_cpu[i, t]
                predicted_pos = obs_to_pos_maps[i].get(predicted_obs, -1)
                predictions_at_timestep[t][predicted_pos] += 1
                
        prev_predictions_at_timestep = defaultdict(Counter)
        if length in previous_results_by_length:
            prev_preds_tensor, prev_loops_list = previous_results_by_length[length]
            prev_predictions_cpu = prev_preds_tensor.numpy()
            prev_obs_to_pos_maps = [{obs: i + 1 for i, obs in enumerate(loop)} for loop in prev_loops_list]
            for i in range(len(prev_loops_list)):
                for t in timesteps_to_analyze:
                    predicted_obs = prev_predictions_cpu[i, t]
                    predicted_pos = prev_obs_to_pos_maps[i].get(predicted_obs, -1)
                    prev_predictions_at_timestep[t][predicted_pos] += 1

        num_plots = len(timesteps_to_analyze)
        if num_plots == 0: continue
        
        cols = min(4, num_plots)
        rows = math.ceil(num_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)
        axes = axes.flatten()

        for i, t in enumerate(timesteps_to_analyze):
            ax = axes[i]
            position_counts = predictions_at_timestep[t]
            total_predictions = sum(position_counts.values()) if sum(position_counts.values()) > 0 else 1
            
            if position_counts:
                positions, counts = zip(*sorted(position_counts.items()))
                bars = ax.bar(positions, counts, color='teal', edgecolor='black')
                for bar in bars:
                    height = bar.get_height()
                    pos = bar.get_x() + bar.get_width() / 2.0
                    percentage = (height / total_predictions) * 100
                    ax.text(pos, height, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
                    prev_count = prev_predictions_at_timestep[t].get(int(pos), 0)
                    change = height - prev_count
                    if previous_results_by_length:
                        delta_str = f'({change:+d})'
                        y_offset = ax.get_ylim()[1] * 0.05 
                        ax.text(pos, height - y_offset, delta_str, ha='center', va='top', 
                                fontsize=8, color='green' if change >= 0 else 'red')

            # --- MODIFIED TITLE LOGIC ---
            if (t + 1) % length == 0 and t > 0:
                title = f"Predictions at t={t}\n(Correct: Pos 1 (Loop Closure!))"
            else:
                title = f"Predictions at t={t}"
            ax.set_title(title, fontsize=12)

            ax.set_xlabel("Predicted Position (1-based)")
            ax.set_ylabel("Count")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks(np.arange(-1, length + 2, 1))
            ax.tick_params(axis='x', labelsize=8, rotation=45)
            
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f"Epoch {epoch} | Predictions for Sequences of Length L={length}", fontsize=16, y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        
        filepath = os.path.join(plot_dir, f'hypothesis_barchart_epoch_{epoch:05d}_L_{length:02d}.png')
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Saved length-specific bar plot to {filepath}")

def plot_convergence_over_time(all_evaluation_results, epochs, hypothesized_timesteps, plot_dir="plots"):
    """
    Generates a SEPARATE convergence plot for each loop length found in the evaluation history,
    plotting all significant position predictions over time.
    """
    print("\n--- Generating Length-Specific Convergence Plots ---")
    os.makedirs(plot_dir, exist_ok=True)
    if not all_evaluation_results: return

    # Group evaluation results by loop length
    final_data = defaultdict(lambda: {'epoch_list': [], 'data_for_epochs': []})
    for i, epoch in enumerate(epochs):
        epoch_results_by_length = all_evaluation_results[i]
        for length, data in epoch_results_by_length.items():
            final_data[length]['epoch_list'].append(epoch)
            final_data[length]['data_for_epochs'].append(data)

    # Create a separate plot for each loop length
    for length, epoch_data in final_data.items():
        current_epochs = epoch_data['epoch_list']
        historical_data = epoch_data['data_for_epochs']
        
        # This dict will store the history of prediction fractions for each position at each timestep
        # Structure: {timestep: {position: [fraction_epoch1, fraction_epoch2, ...]} }
        convergence_data = defaultdict(lambda: defaultdict(list))
        
        timesteps_to_analyze = sorted([t for t in hypothesized_timesteps])

        # Process the data for each epoch to calculate prediction fractions
        for preds_tensor, loops_list in historical_data:
            predictions_cpu = preds_tensor.numpy()
            epoch_counts = defaultdict(Counter)
            obs_to_pos_maps = [{obs: i + 1 for i, obs in enumerate(loop)} for loop in loops_list]

            for i in range(len(loops_list)):
                for t in timesteps_to_analyze:
                    if t < predictions_cpu.shape[1]:
                        predicted_obs = predictions_cpu[i, t]
                        predicted_pos = obs_to_pos_maps[i].get(predicted_obs, -1) # -1 if not in loop
                        epoch_counts[t][predicted_pos] += 1
            
            # Convert counts to fractions for the current epoch
            for t in timesteps_to_analyze:
                total_preds = sum(epoch_counts[t].values())
                # Loop through all possible positions (1 to length) plus the 'not in loop' case (-1)
                for pos in range(-1, length + 2):
                    fraction = epoch_counts[t].get(pos, 0) / total_preds if total_preds > 0 else 0
                    convergence_data[t][pos].append(fraction)
        
        num_plots = len(timesteps_to_analyze)
        if num_plots == 0: continue

        cols = min(4, num_plots)
        rows = math.ceil(num_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), sharex=True, squeeze=False)
        axes = axes.flatten()

        for i, t in enumerate(timesteps_to_analyze):
            ax = axes[i]
            
            # --- MODIFICATION ---
            # Instead of picking the top 6, iterate through all positions.
            # Only plot lines for positions that were ever predicted more than 5% of the time
            # to avoid cluttering the plot with lines that are always zero.
            for pos, fractions in sorted(convergence_data[t].items()):
                if pos > 0 and any(f > 0.05 for f in fractions):
                    if len(fractions) == len(current_epochs):
                        ax.plot(current_epochs, fractions, marker='.', linestyle='-', label=f'Pos {pos}')
            
            # --- Title and Formatting ---
            if (t + 1) % length == 0 and t > 0:
                title = f"Predictions at t={t}\n(Correct: Pos 1 (Loop Closure!))"
            else:
                title = f"Predictions at t={t}"
            ax.set_title(title)

            ax.set_ylabel("Fraction of Guesses")
            ax.set_xlabel("Epoch")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend(loc='best', fontsize='small')
            ax.set_ylim(0, 1.05)

        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f"Convergence of Position Predictions for L={length}", fontsize=18, y=1.03)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        
        filepath = os.path.join(plot_dir, f'convergence_over_time_L_{length:02d}.png')
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Saved length-specific convergence plot to {filepath}")


def main():
    PLOT_DIR = "diffenvrnn1.1"
    os.makedirs(PLOT_DIR, exist_ok=True)
    total_records_collected = 0
    MAX_RECORDS_TO_COLLECT = 100000
    SEQ_LENGTH = 150
    ACCURACY_THRESHOLD = 0.92
    collection_triggered = False
    all_collected_data = defaultdict(list)
    observation_bank = list(range(16)) 
    vocab_size = len(observation_bank)
    hidden_size = 2048
    learning_rate = 5e-4
    num_epochs = 5000
    batch_size = 128 
    eval_interval = 40
    PLOT_HYPOTHESIS_INTERVAL = 40
    global_sequence_counter = 0

    np.random.seed(42); random.seed(42); torch.manual_seed(42)

    env = LoopEnvironment(
        observation_bank=observation_bank,
        loop_lengths=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        velocities=[1],
        behaviour_type='repeat'
    )
    
    train_lengths = [8, 9, 10, 12, 13, 14, 15]
    test_lengths = [7, 11, 14, 16]
    pool_manager = DataPoolManager(
        environment=env,
        train_lengths=train_lengths,
        test_lengths=test_lengths,
        loops_per_length=200000   
    )

    model = StandardGatedRNN(hidden_size=hidden_size, vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    history = {'train_loss': [], 'eval_acc': [], 'epochs': []}
    # Stores (predictions, loops) from each evaluation for final plotting
    all_evaluation_results = [] 
    previous_batch_results = None
    hypotheses_to_test = list(range(18)) 

    print(f"--- Starting Training (Dynamic Pools) ---")
    for epoch in range(num_epochs):
        model.train()
        
        obs, vels, targets, loop_lengths, _ = pool_manager.get_batch(batch_size, SEQ_LENGTH, 'train')
        if obs is None:
            print("Training pool exhausted. Stopping training.")
            break
        objective_loss, _, _, _ = model(obs, vels, targets, loop_lengths) 
        optimizer.zero_grad(); objective_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); scheduler.step()

        if epoch % eval_interval == 0 or epoch == num_epochs - 1:
                    
            # --- MODIFIED DATA COLLECTION LOGIC ---
            data = None # Ensure data is None by default
                    
            if not collection_triggered:
                # We're not collecting yet, just checking for the threshold
                eval_acc, batch_results, _ = evaluate(model, pool_manager, batch_size, SEQ_LENGTH, 
                                                            collect_data=False)
                if eval_acc > ACCURACY_THRESHOLD:
                    print(f"Accuracy {eval_acc:.3f} exceeded threshold of {ACCURACY_THRESHOLD}. Starting data collection on subsequent evaluations.")
                    collection_triggered = True
                    
            else: # collection_triggered is True
                # We're past the threshold, now check if we still have space to collect
                collect_now = (total_records_collected < MAX_RECORDS_TO_COLLECT)
                        
                eval_acc, batch_results, data = evaluate(model, pool_manager, batch_size, SEQ_LENGTH, 
                                                                collect_data=collect_now)
                                        
                if data: # 'data' will only be populated if collect_now was True
                    num_new_records = len(data)
                    
                    # Calculate how many records we can actually add
                    remaining_space = MAX_RECORDS_TO_COLLECT - total_records_collected
                    records_to_add = min(num_new_records, remaining_space)
                    
                    if records_to_add < num_new_records:
                        print(f"-> Collecting final {records_to_add} data points to reach limit.")
                        data_to_save = data[:records_to_add]
                    else:
                        print(f"-> Collected {records_to_add} data points this epoch.")
                        data_to_save = data

                    # 🔧 FIX: Create a mapping from old batch_idx to new unique sequence_id
                    # First, find all unique batch_idx values in this batch
                    unique_batch_indices = sorted(set(record['batch_idx'] for record in data_to_save))
                    
                    # Create mapping: old_batch_idx -> new_global_sequence_id
                    batch_idx_mapping = {
                        old_idx: global_sequence_counter + i 
                        for i, old_idx in enumerate(unique_batch_indices)
                    }
                    
                    # Update the global counter
                    global_sequence_counter += len(unique_batch_indices)

                    # Save the selected data with NEW unique sequence IDs
                    for record in data_to_save:
                        record['epoch'] = epoch
                        # 🔧 CRITICAL FIX: Replace batch_idx with globally unique ID
                        record['batch_idx'] = batch_idx_mapping[record['batch_idx']]
                        loop_len = record['loop_length']
                        all_collected_data[loop_len].append(record)
                    
                    total_records_collected += records_to_add

                    if total_records_collected >= MAX_RECORDS_TO_COLLECT:
                        print(f"-> Reached data collection limit of {MAX_RECORDS_TO_COLLECT} records. Stopping further collection.")

            if batch_results: # batch_results is now a dictionary like {7: (preds, loops), 11: ...}
                all_evaluation_results.append(batch_results)
                history['train_loss'].append(objective_loss.item())
                history['eval_acc'].append(eval_acc)
                history['epochs'].append(epoch)

                if epoch % PLOT_HYPOTHESIS_INTERVAL == 0 or epoch == num_epochs - 1:
                    plot_hypothesis_distribution_bars(
                        results_by_length=batch_results,
                        previous_results_by_length=previous_batch_results, 
                        epoch=epoch,
                        hypothesized_timesteps=hypotheses_to_test,
                        plot_dir=PLOT_DIR
                    )
                    previous_batch_results = batch_results 
            
            total_train_loops = sum(len(p) for p in pool_manager.train_pools.values())
            print(f"Epoch {epoch}/{num_epochs} | Loss: {objective_loss.item():.4f} | Eval Acc: {eval_acc:.3f} | Train Loops Left: {total_train_loops} | Records: {total_records_collected}/{MAX_RECORDS_TO_COLLECT}")

    print("\n--- Training Finished ---")
    model_save_path = os.path.join(PLOT_DIR, "model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model state dictionary saved to '{model_save_path}'")

    if all_collected_data:
        import pickle
        # The total count is now the sum of lengths of all lists in the dictionary
        total_records = sum(len(v) for v in all_collected_data.values())
        save_path = os.path.join(PLOT_DIR, "detailed_run_data_by_length.pkl")
        
        # You are now saving a dictionary, not a list
        with open(save_path, 'wb') as f:
            # We convert it back to a regular dict for saving, which is good practice
            pickle.dump(dict(all_collected_data), f) 
            
        print(f"Saved {total_records} detailed data points, sorted by loop length, to '{save_path}'")
    else:
        print("Data collection was not triggered or no data was collected.")

    sys.stdout.flush() 
    plot_convergence_over_time(
        all_evaluation_results=all_evaluation_results,
        epochs=history['epochs'],
        hypothesized_timesteps=hypotheses_to_test,
        plot_dir=PLOT_DIR
    )
    
    # --- Final Detailed Evaluation (unchanged logic) ---
    print("\n--- Detailed Final Evaluation on up to 10 Random Sequences ---")
    model.eval()
    with torch.no_grad():
        obs_final, vels_final, targets_final, loop_lengths_final, _ = pool_manager.get_batch(
            10, SEQ_LENGTH, 'test')
        
        if obs_final is None:
            print("Test pool is empty. Cannot perform final detailed evaluation.")
        else:
            _, predictions_final, mask_final, _ = model(obs_final, vels_final, targets_final, loop_lengths_final)
            pred_classes_final = torch.argmax(predictions_final, dim=-1)

            obs_cpu, vels_cpu = obs_final.cpu().numpy(), vels_final.cpu().numpy()
            targets_cpu, preds_cpu = targets_final.cpu().numpy(), pred_classes_final.cpu().numpy()
            mask_cpu = mask_final.cpu().numpy()
            for i in range(len(loop_lengths_final)):
                print(f"\n--- Sequence {i+1}/{len(loop_lengths_final)} (Actual Loop Length: {loop_lengths_final[i]}) ---")
                print(f"{'t':<4} | {'Obs':<5} | {'Vel':<5} | {'Target':<8} | {'Pred':<8} | {'Correct?':<10} | {'Target Seen Before?':<20}")
                print("-" * 85)
                
                for t in range(SEQ_LENGTH):
                    obs_t, vel_t = obs_cpu[i, t], vels_cpu[i, t]
                    target_t, pred_t = targets_cpu[i, t], preds_cpu[i, t]
                    
                    is_training_active = (mask_cpu[i, t] == 1.0) 
                    is_correct_str = "Y" if pred_t == target_t else "N"
                    print(f"{t:<4} | {obs_t:<5} | {vel_t:<5} | {target_t:<8} | {pred_t:<8} | {is_correct_str:<10} | {str(is_training_active):<20}")

    # --- Plot Training History (unchanged logic) ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Objective Loss', color='tab:red')
    ax1.plot(history['epochs'], history['train_loss'], 'r.-', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(history['epochs'], history['eval_acc'], 'b.-', label='Eval Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_ylim(0, 1.05)
    
    fig.tight_layout()
    plt.title('Training with Dynamic Pools (Without Replacement)')
    filepath = os.path.join(PLOT_DIR, 'training_progress_dynamic_pools.png')
    plt.savefig(filepath)
    plt.close()
    print(f"\nTraining plot saved to '{filepath}'")

if __name__ == "__main__":
    main()