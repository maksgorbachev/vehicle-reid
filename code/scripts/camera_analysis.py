"""
Camera route graph analysis for VeRi-776 dataset.
Builds directed transition graph between cameras based on vehicle trajectories.
"""

import os
import re
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx


def parse_training_images(data_dir):
    """
    Parse training image filenames → dict: vehicle_id → sorted list of (frame, cam_id).
    Filename format: vehicleID_cameraID_frameID_misc.jpg
    Example: 0001_c001_00016450_0.jpg
    """
    name_train = os.path.join(data_dir, 'name_train.txt')
    if os.path.exists(name_train):
        with open(name_train, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]
    else:
        img_dir = os.path.join(data_dir, 'image_train')
        filenames = os.listdir(img_dir)

    pattern = re.compile(r'^(\d+)_c(\d+)_(\d+)_\d+\.jpg$')
    vehicle_cams = defaultdict(list)

    for fname in filenames:
        m = pattern.match(fname)
        if m:
            vid = int(m.group(1))
            cid = int(m.group(2))
            fid = int(m.group(3))
            vehicle_cams[vid].append((fid, cid))

    # Sort by frame timestamp → chronological camera sequence
    for vid in vehicle_cams:
        vehicle_cams[vid].sort(key=lambda x: x[0])

    return vehicle_cams


def build_transition_graph(vehicle_cams):
    """
    Build directed graph: edge cam_A → cam_B when vehicle seen on A then B (different cameras).
    Node weight = number of vehicles passing through.
    Edge weight = number of vehicle transitions.
    """
    edge_counts = defaultdict(int)
    node_counts = defaultdict(set)  # cam → set of vehicle IDs

    for vid, observations in vehicle_cams.items():
        # Get unique cameras in order of first appearance
        seen = []
        prev_cam = None
        for fid, cid in observations:
            if cid != prev_cam:
                seen.append(cid)
                prev_cam = cid
                node_counts[cid].add(vid)

        # Build transitions between consecutive distinct cameras
        for i in range(len(seen) - 1):
            a, b = seen[i], seen[i + 1]
            if a != b:
                edge_counts[(a, b)] += 1

    G = nx.DiGraph()

    for cam, vehicles in node_counts.items():
        G.add_node(cam, vehicle_count=len(vehicles))

    for (a, b), count in edge_counts.items():
        G.add_edge(a, b, weight=count)

    return G, edge_counts


def load_camera_distances(data_dir):
    """Load camera_Dist.txt — 20×20 distance matrix between cameras."""
    dist_file = os.path.join(data_dir, 'camera_Dist.txt')
    if not os.path.exists(dist_file):
        return None
    with open(dist_file, 'r') as f:
        rows = []
        for line in f:
            vals = list(map(int, line.split()))
            if vals:
                rows.append(vals)
    return np.array(rows) if rows else None


def visualize_camera_graph(G, output_path, title='Граф переходов транспортных средств между камерами'):
    """Visualize camera transition graph with node/edge weights."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Layout: circular or spring
    try:
        pos = nx.circular_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    node_counts = [G.nodes[n].get('vehicle_count', 1) for n in G.nodes()]
    max_count = max(node_counts) if node_counts else 1
    node_sizes = [300 + 1500 * (c / max_count) for c in node_counts]

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 4.5 * (w / max_w) for w in edge_weights]
    edge_alphas = [0.3 + 0.7 * (w / max_w) for w in edge_weights]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_counts,
        cmap='YlOrRd',
        alpha=0.9
    )

    # Draw labels
    labels = {n: f'C{n:02d}' for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9, font_weight='bold')

    # Draw edges with varying width/alpha
    for (u, v), w, width in zip(G.edges(), edge_weights, edge_widths):
        alpha = 0.3 + 0.7 * (w / max_w)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            width=width, alpha=alpha,
            edge_color='steelblue',
            arrows=True,
            arrowsize=15,
            connectionstyle='arc3,rad=0.1'
        )

    # Colorbar via scatter
    sm = plt.cm.ScalarMappable(
        cmap='YlOrRd',
        norm=plt.Normalize(vmin=min(node_counts), vmax=max(node_counts))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Число уникальных ТС через камеру', fontsize=10)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_top_transitions(edge_counts, output_path, top_n=20):
    """Bar chart of top N most frequent camera transitions."""
    sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [f'C{a:02d}→C{b:02d}' for (a, b), _ in sorted_edges]
    values = [v for _, v in sorted_edges]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(labels[::-1], values[::-1], color='steelblue', alpha=0.8)
    ax.set_xlabel('Число транспортных средств', fontsize=11)
    ax.set_title(f'Топ-{top_n} переходов между камерами (VeRi-776)', fontsize=12, fontweight='bold')
    ax.bar_label(bars, fmt='%d', padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_stats(G, vehicle_cams, edge_counts):
    """Print summary statistics."""
    print(f"\n=== Camera Route Analysis ===")
    print(f"Total vehicles:     {len(vehicle_cams)}")
    print(f"Total cameras:      {G.number_of_nodes()}")
    print(f"Unique transitions: {G.number_of_edges()}")
    print(f"Total transitions:  {sum(edge_counts.values())}")

    cam_visits = sorted(
        G.nodes(data=True),
        key=lambda x: x[1].get('vehicle_count', 0),
        reverse=True
    )
    print(f"\nTop cameras by vehicle count:")
    for cam, data in cam_visits[:5]:
        print(f"  C{cam:02d}: {data['vehicle_count']} vehicles")

    top_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop transitions:")
    for (a, b), count in top_edges:
        print(f"  C{a:02d} -> C{b:02d}: {count} vehicles")

    # Vehicles that pass through many cameras
    cam_counts_per_vehicle = [
        len(set(c for _, c in obs)) for obs in vehicle_cams.values()
    ]
    print(f"\nCameras per vehicle:")
    print(f"  Mean:  {np.mean(cam_counts_per_vehicle):.2f}")
    print(f"  Max:   {np.max(cam_counts_per_vehicle)}")
    print(f"  Min:   {np.min(cam_counts_per_vehicle)}")
    dist = defaultdict(int)
    for v in cam_counts_per_vehicle:
        dist[v] += 1
    print(f"  Distribution: {dict(sorted(dist.items()))}")


def main():
    parser = argparse.ArgumentParser(description='VeRi-776 camera route analysis')
    parser.add_argument('--data_dir', default='code/data/VeRi',
                        help='Path to VeRi dataset root')
    parser.add_argument('--output_dir', default='images',
                        help='Directory for output figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Parsing training images...")
    vehicle_cams = parse_training_images(args.data_dir)

    print("Building transition graph...")
    G, edge_counts = build_transition_graph(vehicle_cams)

    print_stats(G, vehicle_cams, edge_counts)

    print("\nGenerating visualizations...")
    visualize_camera_graph(
        G,
        os.path.join(args.output_dir, 'camera_route_graph.png')
    )
    visualize_top_transitions(
        edge_counts,
        os.path.join(args.output_dir, 'camera_transitions_top20.png')
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
