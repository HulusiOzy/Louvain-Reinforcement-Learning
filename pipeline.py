#!/usr/bin/env python3

"""
Full Louvain skill hierarchy pipeline.
STG -> Louvain clustering -> Option creation -> Offline policy training -> Online agent.
"""

import igraph as ig
import leidenalg as la
import networkx as nx

from agent import Option, PrimitiveOption, OptionTrainer, OptionAgent
from environment import GridWorld


# Step 1: Louvain clustering (matches louvain.py from the authors)

def apply_louvain(stg_nx, resolution=0.05, min_mean_cluster_size=4):
    """Apply Louvain to a networkx STG. Returns the nx graph with
    cluster-N vertex attributes, and the number of hierarchy levels."""

    # Convert networkx -> igraph.
    node_list = list(stg_nx.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in stg_nx.edges()]
    stg_ig = ig.Graph(n=len(node_list), edges=edges, directed=True)
    stg_ig.vs["name"] = node_list

    # Run Louvain via leidenalg.
    partition = la.RBConfigurationVertexPartition(stg_ig, resolution_parameter=resolution)
    optimiser = la.Optimiser()
    partition_agg = partition.aggregate_partition()

    level = 0
    levels_skipped = 0

    while optimiser.move_nodes(partition_agg) > 0:
        partition.from_coarse_partition(partition_agg)
        partition_agg = partition_agg.aggregate_partition()

        n_clusters = len(set(partition.membership))
        mean_size = len(node_list) / n_clusters

        if mean_size < min_mean_cluster_size:
            levels_skipped += 1
            continue

        # Store cluster membership on the networkx graph.
        for i, node in enumerate(node_list):
            stg_nx.nodes[node][f"cluster-{level}"] = partition.membership[i]

        level += 1

    print(f"Louvain: {level} hierarchy levels ({levels_skipped} skipped)")
    return stg_nx, level


# ---------------------------------------------------------------------------
# Step 2: Create options from cluster partitions
# ---------------------------------------------------------------------------

def create_options_from_clusters(stg_nx, num_levels):
    """For each level, for each pair of neighboring clusters,
    create an option that navigates from source to target cluster."""

    all_options = {}  # level -> list of options

    for lvl in range(num_levels):
        attr = f"cluster-{lvl}"
        options_at_level = []

        # Group nodes by cluster.
        clusters = {}
        for node, data in stg_nx.nodes(data=True):
            cid = data.get(attr)
            if cid is None:
                continue
            clusters.setdefault(cid, set()).add(node)

        # Find neighboring clusters (directed edges between them).
        neighbors = {}  # source_cid -> set of target_cids
        for u, v in stg_nx.edges():
            cu = stg_nx.nodes[u].get(attr)
            cv = stg_nx.nodes[v].get(attr)
            if cu is not None and cv is not None and cu != cv:
                neighbors.setdefault(cu, set()).add(cv)

        # Create one option per (source, target) cluster pair.
        for source_cid, target_cids in neighbors.items():
            for target_cid in target_cids:
                source_nodes = clusters[source_cid]
                target_nodes = clusters[target_cid]

                option = Option(
                    option_id=f"L{lvl}_{source_cid}_to_{target_cid}",
                    initiation_set=set(source_nodes),
                    termination_fn=lambda s, t=frozenset(target_nodes): 1.0 if s in t else 0.0,
                    level=lvl,
                )
                # Stash these for the trainer.
                option._source_states = set(source_nodes)
                option._target_states = set(target_nodes)

                options_at_level.append(option)

        all_options[lvl] = options_at_level
        print(f"  Level {lvl}: {len(clusters)} clusters, {len(options_at_level)} options")

    return all_options


# Step 3: Train option policies offline, level by level

def train_hierarchy(env, all_options, primitives, max_steps=100_000):
    """Train option policies level by level.
    Level 0 options use primitives. Level N options use level N-1 options."""

    trainer = OptionTrainer(env, max_steps=max_steps)

    for lvl in sorted(all_options.keys()):
        # Sub-options for this level.
        if lvl == 0:
            sub_options = list(primitives)
        else:
            sub_options = list(all_options[lvl - 1])

        print(f"  Training level {lvl}: {len(all_options[lvl])} options using {len(sub_options)} sub-options")

        for i, option in enumerate(all_options[lvl]):
            option.sub_options = sub_options
            trainer.train(
                option,
                sub_options,
                option._source_states,
                option._target_states,
                can_leave_initiation=False,
            )
            trained = sum(1 for v in option.trained_policy.values() if v is not None)
            if (i + 1) % 10 == 0 or i == len(all_options[lvl]) - 1:
                print(f"    [{i+1}/{len(all_options[lvl])}] {option.id}: policy covers {trained} states")


# Step 4: Run it

def main():
    grid_file = "./xu_four_rooms_brtl.txt"
    env = GridWorld(grid_file)

    print(f"Environment: {len(env.state_space)} states, start={env.initial_states}, goal={env.terminal_states}")

    # Build STG.
    stg = env.generate_stg(directed=True)
    print(f"STG: {stg.number_of_nodes()} nodes, {stg.number_of_edges()} edges")

    # Louvain clustering.
    print("\n--- Louvain Clustering ---")
    stg, num_levels = apply_louvain(stg, resolution=0.05, min_mean_cluster_size=4)

    # Create options.
    print("\n--- Creating Options ---")
    all_options = create_options_from_clusters(stg, num_levels)

    # Create primitives.
    primitives = [PrimitiveOption(a) for a in env.get_actions()]

    # Train option policies offline.
    print("\n--- Training Option Policies ---")
    train_hierarchy(env, all_options, primitives, max_steps=100_000)

    # Collect all options for the agent.
    all_opts_flat = list(primitives)
    for lvl in sorted(all_options.keys()):
        all_opts_flat.extend(all_options[lvl])
    env.options = all_opts_flat

    print(f"\n--- Running Agent ({len(all_opts_flat)} total options) ---")

    # Run Louvain agent.
    louvain_agent = OptionAgent(env, all_opts_flat, epsilon=0.1, macro_alpha=0.4, gamma=1.0)
    louvain_rewards = louvain_agent.run(num_epochs=60, epoch_length=100)

    # Run primitive-only baseline.
    env2 = GridWorld(grid_file)
    env2.options = list(primitives)
    prim_agent = OptionAgent(env2, list(primitives), epsilon=0.1, macro_alpha=0.4, gamma=1.0)
    prim_rewards = prim_agent.run(num_epochs=60, epoch_length=100)

    # Print comparison.
    print("\n--- Results (avg reward per 10-epoch block) ---")
    print(f"{'Epochs':<15} {'Louvain':>10} {'Primitive':>10}")
    for i in range(0, 60, 10):
        l_avg = sum(louvain_rewards[i:i+10]) / 10
        p_avg = sum(prim_rewards[i:i+10]) / 10
        print(f"{i}-{i+9:<11} {l_avg:>10.2f} {p_avg:>10.2f}")


if __name__ == "__main__":
    main()
