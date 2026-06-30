"""
Wright-Fisher Simulation of Purging Lethal Recessive Mutations
During a Kākāpō-Like Population Bottleneck

This script simulates the purging of highly deleterious recessive mutations
during a severe population bottleneck, parameterized to reflect the demographic
history of the Kākāpō (Strigops habroptilus).

Author: Neo Ndlovu
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================================
# PARAMETERS
# ============================================================================

# Mutation rate (wild-type A -> deleterious recessive a)
MU = 1e-5  # per locus per generation

# Fitness values
FITNESS_AA = 1.0   # wild-type homozygote
FITNESS_Aa = 1.0   # heterozygote (completely recessive lethal)
FITNESS_aa = 0.0   # lethal homozygote

# Demographic parameters
N_ANCESTRAL = 10000       # ancestral population size
BURN_IN_GENERATIONS = 1000  # burn-in length (10N generations)
BOTTLENECK_GENERATIONS = 15   # duration of bottleneck
RECOVERY_GENERATIONS = 50     # post-bottleneck recovery period
N_RECOVERY = 250              # modern population size

# Bottleneck severities to test
BOTTLENECK_SIZES = [25, 50, 100]

# Number of independent replicate simulations per bottleneck size
N_REPLICATES = 10

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def initialize_population(N):
    """
    Initialize a population of N diploid individuals with zero deleterious alleles.
    
    Each individual is represented as an integer count of 'a' alleles:
        0 = AA (wild-type homozygote)
        1 = Aa (heterozygote)
        2 = aa (lethal homozygote)
    
    Parameters:
        N (int): Population size (number of diploid individuals).
    
    Returns:
        numpy.ndarray: Array of length N with genotype counts (initially all 0).
    """
    return np.zeros(N, dtype=int)


def mutation_step(population, mu):
    """
    Apply germline mutations to all alleles in the population.
    
    Each wild-type allele (A) can independently mutate to the deleterious
    allele (a) with probability mu. Since each diploid individual carries
    two alleles, we process the population as a pool of 2N alleles.
    
    Parameters:
        population (numpy.ndarray): Array of genotype counts (0, 1, or 2).
        mu (float): Per-allele mutation rate.
    
    Returns:
        numpy.ndarray: Updated population array after mutation.
    """
    N = len(population)
    # Count total number of 'a' alleles currently in the population
    total_a_alleles = np.sum(population)
    # Count total number of 'A' alleles
    total_A_alleles = 2 * N - total_a_alleles
    
    # Each A allele mutates to 'a' with probability mu
    new_mutations = np.random.binomial(total_A_alleles, mu)
    
    # Add new mutations to the existing a alleles
    new_total_a = total_a_alleles + new_mutations
    
    # Reconstruct genotype array from new allele count
    # We'll do this by creating an array of zeros and assigning alleles
    new_population = np.zeros(N, dtype=int)
    
    # Randomly distribute the 'a' alleles among the 2N allele positions
    alleles = np.zeros(2 * N, dtype=int)
    if new_total_a > 0:
        # Randomly choose which allele positions get the 'a' allele
        a_positions = np.random.choice(2 * N, size=new_total_a, replace=False)
        alleles[a_positions] = 1
    
    # Pair alleles to form genotypes
    for i in range(N):
        new_population[i] = alleles[2*i] + alleles[2*i + 1]
    
    return new_population


def selection_step(population):
    """
    Apply viability selection against aa homozygotes.
    
    Individuals with the aa genotype (count = 2) have fitness = 0 and are
    removed from the breeding population. AA and Aa individuals survive
    with equal probability (fitness = 1).
    
    Parameters:
        population (numpy.ndarray): Array of genotype counts before selection.
    
    Returns:
        numpy.ndarray: Array of surviving individuals' genotype counts.
    """
    # Only keep individuals with genotype count < 2 (i.e., not aa)
    survivors = population[population < 2]
    return survivors


def drift_and_reproduction(survivors, N_next):
    """
    Form the next generation through random mating and binomial sampling (drift).
    
    From the pool of surviving adults, we calculate the allele frequency of 'a'
    in the gamete pool. Then we draw 2N_next gametes with replacement to form
    N_next new diploid individuals. This binomial sampling introduces genetic
    drift due to finite population size.
    
    Parameters:
        survivors (numpy.ndarray): Array of surviving individuals' genotypes.
        N_next (int): Desired population size for the next generation.
    
    Returns:
        numpy.ndarray: New population array of length N_next.
    """
    if len(survivors) == 0:
        # If no individuals survived (possible in very small populations
        # with high load), return an empty population
        return np.zeros(N_next, dtype=int)
    
    # Calculate total number of 'a' alleles among survivors
    total_a = np.sum(survivors)
    total_alleles = 2 * len(survivors)
    
    # Frequency of 'a' allele in the gamete pool
    freq_a = total_a / total_alleles if total_alleles > 0 else 0.0
    
    # Binomial sampling: draw 2N_next gametes
    n_gametes = 2 * N_next
    a_alleles_in_next_gen = np.random.binomial(n_gametes, freq_a)
    
    # Create the new generation by randomly pairing gametes
    new_population = np.zeros(N_next, dtype=int)
    alleles = np.zeros(n_gametes, dtype=int)
    if a_alleles_in_next_gen > 0:
        a_positions = np.random.choice(n_gametes, size=a_alleles_in_next_gen, replace=False)
        alleles[a_positions] = 1
    
    for i in range(N_next):
        new_population[i] = alleles[2*i] + alleles[2*i + 1]
    
    return new_population


def calculate_lethal_equivalents(population):
    """
    Calculate the mean number of lethal equivalents per individual.
    
    For a fully recessive lethal, each 'a' allele contributes 0 in the
    heterozygous state and 1 in the homozygous state to the lethal equivalent
    count. This is equivalent to the sum of selection coefficients for all
    deleterious alleles carried.
    
    Parameters:
        population (numpy.ndarray): Array of genotype counts.
    
    Returns:
        float: Mean lethal equivalents per individual.
    """
    if len(population) == 0:
        return 0.0
    
    # Count how many individuals are aa homozygotes
    n_lethal_homozygotes = np.sum(population == 2)
    
    # Each aa homozygote carries 1 lethal equivalent (s = 1.0)
    total_lethal_equivalents = n_lethal_homozygotes  # since each aa = 1 lethal equivalent
    
    return total_lethal_equivalents / len(population)


def run_single_simulation(N_bottleneck):
    """
    Run one complete simulation of the demographic scenario.
    
    Parameters:
        N_bottleneck (int): Population size during the bottleneck phase.
    
    Returns:
        dict: Dictionary containing allele frequency trajectories and load measurements.
    """
    # --- Burn-in Phase ---
    pop = initialize_population(N_ANCESTRAL)
    
    # Store trajectory for analysis
    freq_trajectory = []
    load_trajectory = []
    
    for gen in range(BURN_IN_GENERATIONS):
        pop = mutation_step(pop, MU)
        pop = selection_step(pop)
        pop = drift_and_reproduction(pop, N_ANCESTRAL)
        
        # Record every 1000th generation during burn-in to keep memory manageable
        if gen % 1000 == 0:
            total_a = np.sum(pop)
            freq_a = total_a / (2 * len(pop)) if len(pop) > 0 else 0
            freq_trajectory.append((gen, freq_a))
            load_trajectory.append((gen, calculate_lethal_equivalents(pop)))
    
    # Record ancestral load at end of burn-in
    ancestral_load = calculate_lethal_equivalents(pop)
    total_a = np.sum(pop)
    ancestral_freq = total_a / (2 * len(pop)) if len(pop) > 0 else 0
    
    # --- Bottleneck Phase ---
    # Instantaneous crash to bottleneck size
    if len(pop) > N_bottleneck:
        pop = np.random.choice(pop, size=N_bottleneck, replace=False)
    
    for gen in range(BURN_IN_GENERATIONS, BURN_IN_GENERATIONS + BOTTLENECK_GENERATIONS):
        pop = mutation_step(pop, MU)
        pop = selection_step(pop)
        pop = drift_and_reproduction(pop, N_bottleneck)
        
        total_a = np.sum(pop)
        freq_a = total_a / (2 * len(pop)) if len(pop) > 0 else 0
        freq_trajectory.append((gen, freq_a))
        load_trajectory.append((gen, calculate_lethal_equivalents(pop)))
    
    # Record bottleneck-end load
    bottleneck_load = calculate_lethal_equivalents(pop)
    
    # --- Recovery Phase ---
    # Expand to recovery size
    N_current = len(pop)
    if N_current < N_RECOVERY and N_current > 0:
        # Expand population while preserving allele frequencies
        additional_needed = N_RECOVERY - N_current
        pop = np.concatenate([pop, np.random.choice(pop, size=additional_needed, replace=True)])
    
    for gen in range(BURN_IN_GENERATIONS + BOTTLENECK_GENERATIONS, 
                     BURN_IN_GENERATIONS + BOTTLENECK_GENERATIONS + RECOVERY_GENERATIONS):
        pop = mutation_step(pop, MU)
        pop = selection_step(pop)
        pop = drift_and_reproduction(pop, N_RECOVERY)
        
        total_a = np.sum(pop)
        freq_a = total_a / (2 * len(pop)) if len(pop) > 0 else 0
        freq_trajectory.append((gen, freq_a))
        load_trajectory.append((gen, calculate_lethal_equivalents(pop)))
    
    # Record modern load
    modern_load = calculate_lethal_equivalents(pop)
    total_a = np.sum(pop)
    modern_freq = total_a / (2 * len(pop)) if len(pop) > 0 else 0
    
    # Determine if the allele was purged (frequency = 0)
    purged = (modern_freq == 0.0)
    
    return {
        'ancestral_load': ancestral_load,
        'bottleneck_load': bottleneck_load,
        'modern_load': modern_load,
        'ancestral_freq': ancestral_freq,
        'modern_freq': modern_freq,
        'purged': purged,
        'freq_trajectory': freq_trajectory,
        'load_trajectory': load_trajectory
    }


# ============================================================================
# RUN SIMULATIONS
# ============================================================================

print("Starting Kākāpō Purging Simulations...")
print(f"Parameters: μ = {MU}, Burn-in = {BURN_IN_GENERATIONS} gens, "
      f"Bottleneck = {BOTTLENECK_GENERATIONS} gens, Recovery = {RECOVERY_GENERATIONS} gens")
print(f"Replicates per condition: {N_REPLICATES}")
print("-" * 60)

all_results = {}

for N_bn in BOTTLENECK_SIZES:
    print(f"Running simulations for bottleneck size N = {N_bn}...")
    
    results_list = []
    purged_count = 0
    
    for rep in range(N_REPLICATES):
        if rep % 100 == 0 and rep > 0:
            print(f"  Completed {rep}/{N_REPLICATES} replicates...")
        
        result = run_single_simulation(N_bn)
        results_list.append(result)
        if result['purged']:
            purged_count += 1
    
    all_results[N_bn] = results_list
    print(f"  Done. Purged in {purged_count}/{N_REPLICATES} replicates "
          f"({100 * purged_count / N_REPLICATES:.1f}%)")
    print("-" * 60)

print("All simulations complete!")


# Add this right after the simulation loop and before the plotting section
for N_bn in BOTTLENECK_SIZES:
    results = all_results[N_bn]
    ancestral = [r['ancestral_load'] for r in results]
    modern = [r['modern_load'] for r in results]
    print(f"\nN={N_bn}:")
    print(f"  Ancestral load - Mean: {np.mean(ancestral):.6f}, SD: {np.std(ancestral):.6f}")
    print(f"  Modern load    - Mean: {np.mean(modern):.6f}, SD: {np.std(modern):.6f}")
    print(f"  Individual values: {ancestral}")

# ============================================================================
# GENERATE FIGURES
# ============================================================================

# Set up the plotting style
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0

# --- Figure 1: Mean Lethal Equivalents Before vs After Bottleneck ---
fig1, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for idx, N_bn in enumerate(BOTTLENECK_SIZES):
    ax = axes[idx]
    results = all_results[N_bn]
    
    ancestral_loads = [r['ancestral_load'] for r in results]
    modern_loads = [r['modern_load'] for r in results]
    
    # Create grouped bar positions
    categories = ['Ancestral\n(Large Population)', 'Modern\n(Post-Bottleneck)']
    means = [np.mean(ancestral_loads), np.mean(modern_loads)]
    stds = [np.std(ancestral_loads), np.std(modern_loads)]
    
    x_pos = [0, 1]
    bars = ax.bar(x_pos, means, width=0.5, color=['#708090', '#008B8B'], 
                  edgecolor='black', linewidth=1.2, yerr=stds, capsize=8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(f'Bottleneck Size N = {N_bn}', fontweight='bold', fontsize=13)
    ax.set_ylabel('Mean Lethal Equivalents per Individual', fontsize=11)
    
    # Fix: Set a reasonable y-limit that handles zero values
    y_max = max(max(means) * 1.5, 0.001)
    ax.set_ylim(0, y_max)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_max * 0.02,
                f'{mean_val:.4f}', ha='center', fontweight='bold', fontsize=10)
    
    # Calculate percentage reduction safely
    if means[0] > 0:
        pct_reduction = 100 * (means[0] - means[1]) / means[0]
    else:
        pct_reduction = 0.0
    ax.text(0.5, 0.85, f'Reduction: {pct_reduction:.1f}%', transform=ax.transAxes,
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

fig1.suptitle('Purging of Lethal Load During a Simulated Kakapo Bottleneck\n'
              '(Mean ± 1 SD across replicates)',
              fontweight='bold', fontsize=14)
plt.tight_layout(pad=2.0)
plt.savefig('figure1_lethal_equivalents.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved as 'figure1_lethal_equivalents.png'")


# --- Figure 2: Proportion Purged vs Bottleneck Severity ---
fig2, ax = plt.subplots(figsize=(8, 5))

purged_proportions = []
for N_bn in BOTTLENECK_SIZES:
    results = all_results[N_bn]
    n_purged = sum(r['purged'] for r in results)
    purged_proportions.append(n_purged / len(results))

bars = ax.bar(range(len(BOTTLENECK_SIZES)), purged_proportions, 
              color='#008B8B', edgecolor='black', linewidth=1.2, width=0.5)

ax.set_xticks(range(len(BOTTLENECK_SIZES)))
ax.set_xticklabels([f'N = {n}' for n in BOTTLENECK_SIZES], fontsize=12)
ax.set_ylabel('Proportion of Replicates with Allele Purged', fontsize=11)
ax.set_xlabel('Bottleneck Population Size', fontsize=12)
ax.set_title('Probability of Complete Purging vs. Bottleneck Severity',
             fontweight='bold', fontsize=13)
ax.set_ylim(0, 1.05)

# Add percentage labels
for bar, prop in zip(bars, purged_proportions):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{prop:.1%}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout(pad=2.0)
plt.savefig('figure2_purging_probability.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved as 'figure2_purging_probability.png'")


# --- Figure 3: Example Allele Frequency Trajectories ---
fig3, axes = plt.subplots(1, 3, figsize=(18, 5))

# Pick one example replicate for each bottleneck size to show trajectory
for idx, N_bn in enumerate(BOTTLENECK_SIZES):
    ax = axes[idx]
    results = all_results[N_bn]
    
    # Show up to 5 example trajectories
    colors = ['#DC143C', '#4682B4', '#FF8C00', '#2E8B57', '#8B008B']
    n_to_plot = min(5, len(results))
    
    for rep_idx in range(n_to_plot):
        result = results[rep_idx]
        traj = result['freq_trajectory']
        gens, freqs = zip(*traj)
        ax.plot(gens, freqs, color=colors[rep_idx], alpha=0.7, linewidth=1.2,
                label=f'Replicate {rep_idx + 1}')
    
    # Add vertical lines to mark phase transitions
    ax.axvline(x=BURN_IN_GENERATIONS, color='gray', linestyle='--', 
               linewidth=1.0, alpha=0.7, label='Bottleneck Start')
    ax.axvline(x=BURN_IN_GENERATIONS + BOTTLENECK_GENERATIONS, color='gray', 
               linestyle=':', linewidth=1.0, alpha=0.7, label='Recovery Start')
    
    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Frequency of Lethal Allele (a)', fontsize=11)
    ax.set_title(f'Bottleneck N = {N_bn}', fontweight='bold', fontsize=13)
    ax.set_ylim(-0.01, None)
    ax.legend(fontsize=8, loc='upper right')

fig3.suptitle('Example Allele Frequency Trajectories Across Demographic Phases',
              fontweight='bold', fontsize=14)
plt.tight_layout(pad=2.0)
plt.savefig('figure3_frequency_trajectories.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved as 'figure3_frequency_trajectories.png'")


# --- Figure 4: Distribution of Modern Allele Frequencies ---
fig4, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, N_bn in enumerate(BOTTLENECK_SIZES):
    ax = axes[idx]
    results = all_results[N_bn]
    
    modern_freqs = [r['modern_freq'] for r in results]
    
    # Plot histogram, but separate the zero (purged) cases
    non_zero_freqs = [f for f in modern_freqs if f > 0]
    n_purged = sum(1 for f in modern_freqs if f == 0.0)
    
    if non_zero_freqs:
        ax.hist(non_zero_freqs, bins=20, color='#008B8B', edgecolor='black', 
                alpha=0.7, label=f'Non-zero (n={len(non_zero_freqs)})')
    
    # Add an annotation for purged cases
    if n_purged > 0:
        ax.annotate(f'Purged: {n_purged} replicates\n(frequency = 0)',
                    xy=(0.55, 0.85), xycoords='axes fraction',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.set_xlabel('Modern Allele Frequency', fontsize=11)
    ax.set_ylabel('Number of Replicates', fontsize=11)
    ax.set_title(f'Bottleneck N = {N_bn}', fontweight='bold', fontsize=13)

fig4.suptitle('Distribution of Lethal Allele Frequencies After Recovery',
              fontweight='bold', fontsize=14)
plt.tight_layout(pad=2.0)
plt.savefig('figure4_frequency_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved as 'figure4_frequency_distribution.png'")

print("\n" + "=" * 60)
print("All simulations and figures complete!")
print("=" * 60)