# -*- coding: utf-8 -*-
"""
Dislocation Structure Analysis from ParaDis Output

This script processes dislocation segment data from a ParaDis simulation output file
(e.g., 'visit*.seg'). It performs the following steps:
1.  Reads the segment data, including endpoint coordinates, Burgers vectors, and normal vectors.
2.  Traces the connectivity of these segments to reconstruct continuous dislocation lines.
3.  Optionally merges dislocation lines that are very close but not perfectly connected.
4.  Sorts the nodes within each reconstructed line to ensure a continuous path.
5.  Generates a 3D plot to visualize the final dislocation structure.
6.  Saves the processed line data to a Python pickle file for further analysis.
"""

import argparse
import pickle
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Set the backend for matplotlib to avoid GUI issues
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


def read_paradis_seg_file(filepath):
    """
    Reads a ParaDis segment file and extracts dislocation segment data.

    Each line in the file represents one dislocation segment and should contain
    at least 13 floating-point numbers:
    x1 y1 z1 x2 y2 z2 bx by bz nx ny nz burgers_magnitude

    Args:
        filepath (str): The path to the input '.seg' file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of segment endpoints of shape (N*2, 3), where N is
                          the number of segments.
            - np.ndarray: An array of normal vectors of shape (N*2, 3).
            - np.ndarray: An array of Burgers vector magnitudes of shape (N*2, 1).
    """
    segment_endpoints = []
    normal_vectors = []
    burgers_magnitudes = []

    print(f"Reading and parsing data from '{filepath}'...")
    with open(filepath, 'r') as f:
        for line in f:
            # Ignore empty lines or comment lines
            if not line.strip() or line.strip().startswith("#"):
                continue

            try:
                data = [float(x) for x in line.split()]

                # Ensure the line has enough data points
                if len(data) < 13:
                    continue

                start_pos = data[0:3]
                end_pos = data[3:6]
                # The normal vector is assumed to be at indices 9, 10, 11
                n_vector = data[9:12]
                # The last value is the Burgers vector magnitude
                b_mag = data[-1]

                segment_endpoints.extend([start_pos, end_pos])
                normal_vectors.extend([n_vector, n_vector])
                burgers_magnitudes.extend([[b_mag], [b_mag]])

            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line: '{line.strip()}'. Error: {e}")

    num_segments = len(segment_endpoints) // 2
    print(f"Successfully read {num_segments} segments.")

    return (np.array(segment_endpoints),
            np.array(normal_vectors),
            np.array(burgers_magnitudes))


def build_node_connectivity_map(segment_endpoints):
    """
    Builds a map from each unique node coordinate to the indices of the segments
    it belongs to. This is a crucial step for efficient line tracing.

    Args:
        segment_endpoints (np.ndarray): Array of segment endpoints (shape: N*2, 3).

    Returns:
        defaultdict: A dictionary where keys are node coordinate tuples and
                     values are lists of endpoint indices.
    """
    connectivity_map = defaultdict(list)
    for i, point in enumerate(segment_endpoints):
        # Using a tuple of coordinates as a dictionary key
        connectivity_map[tuple(point)].append(i)
    return connectivity_map


def trace_dislocation_lines(segment_endpoints, normal_vectors, burgers_magnitudes):
    """
    Traces continuous dislocation lines from a collection of individual segments.

    This function walks along connected segments to reconstruct the full dislocation
    line topology.

    Args:
        segment_endpoints (np.ndarray): Array of segment endpoints.
        normal_vectors (np.ndarray): Array of normal vectors for each endpoint.
        burgers_magnitudes (np.ndarray): Array of Burgers vector magnitudes.

    Returns:
        tuple: A tuple containing dictionaries for the reconstructed lines:
            - dict: `dislocation_lines` {line_id: np.ndarray of node coordinates}
            - dict: `line_normals` {line_id: np.ndarray of normal vectors}
            - dict: `line_burgers` {line_id: np.ndarray of Burgers magnitudes}
    """
    print("Tracing dislocation lines from segments...")
    num_segments = len(segment_endpoints) // 2

    # Use a set for efficient checking of processed segments
    processed_segment_indices = set()

    # Build a map for quick lookup of connected nodes
    connectivity_map = build_node_connectivity_map(segment_endpoints)

    dislocation_lines = {}
    line_normals = {}
    line_burgers = {}
    line_id_counter = 0

    for seg_idx in range(num_segments):
        if seg_idx in processed_segment_indices:
            continue

        # Start a new dislocation line
        processed_segment_indices.add(seg_idx)

        # The first segment's endpoints (indices in the main array)
        start_node_idx = seg_idx * 2
        end_node_idx = start_node_idx + 1

        # Initialize the line with the starting segment's nodes
        current_line_nodes = [segment_endpoints[start_node_idx], segment_endpoints[end_node_idx]]
        current_line_normals = [normal_vectors[start_node_idx], normal_vectors[end_node_idx]]
        current_line_burgers = [burgers_magnitudes[start_node_idx], burgers_magnitudes[end_node_idx]]

        # --- Trace backwards from the start node of the initial segment ---
        current_node_idx = start_node_idx
        while True:
            node_coord = tuple(segment_endpoints[current_node_idx])
            neighbor_node_indices = connectivity_map[node_coord]

            # A junction has >2 connections; a free end has 1. We only trace along paths with 2.
            if len(neighbor_node_indices) != 2:
                break

            # Find the neighbor that is not part of the current segment
            neighbor_idx = -1
            for idx in neighbor_node_indices:
                # Find the segment index this node belongs to
                neighbor_seg_idx = idx // 2
                if neighbor_seg_idx not in processed_segment_indices:
                    neighbor_idx = idx
                    break

            if neighbor_idx == -1:
                break  # No unprocessed neighbor found

            # Add the neighbor segment to our processed list
            neighbor_seg_idx = neighbor_idx // 2
            processed_segment_indices.add(neighbor_seg_idx)

            # Find the other end of the neighbor segment to continue tracing
            if neighbor_idx % 2 == 0:  # It's a start node
                next_node_idx = neighbor_idx + 1
            else:  # It's an end node
                next_node_idx = neighbor_idx - 1

            # Prepend the new node to the line (since we are tracing backwards)
            current_line_nodes.insert(0, segment_endpoints[next_node_idx])
            current_line_normals.insert(0, normal_vectors[next_node_idx])
            current_line_burgers.insert(0, burgers_magnitudes[next_node_idx])

            current_node_idx = next_node_idx

        # --- Trace forwards from the end node of the initial segment ---
        current_node_idx = end_node_idx
        while True:
            node_coord = tuple(segment_endpoints[current_node_idx])
            neighbor_node_indices = connectivity_map[node_coord]

            if len(neighbor_node_indices) != 2:
                break

            neighbor_idx = -1
            for idx in neighbor_node_indices:
                neighbor_seg_idx = idx // 2
                if neighbor_seg_idx not in processed_segment_indices:
                    neighbor_idx = idx
                    break

            if neighbor_idx == -1:
                break

            neighbor_seg_idx = neighbor_idx // 2
            processed_segment_indices.add(neighbor_seg_idx)

            if neighbor_idx % 2 == 0:
                next_node_idx = neighbor_idx + 1
            else:
                next_node_idx = neighbor_idx - 1

            # Append the new node to the line
            current_line_nodes.append(segment_endpoints[next_node_idx])
            current_line_normals.append(normal_vectors[next_node_idx])
            current_line_burgers.append(burgers_magnitudes[next_node_idx])

            current_node_idx = next_node_idx

        # Store the completed line
        dislocation_lines[line_id_counter] = np.array(current_line_nodes)
        line_normals[line_id_counter] = np.array(current_line_normals)
        line_burgers[line_id_counter] = np.array(current_line_burgers)
        line_id_counter += 1

    print(f"Tracing complete. Found {len(dislocation_lines)} dislocation lines.")
    return dislocation_lines, line_normals, line_burgers


def sort_nodes_on_line(nodes):
    """
    Sorts a list of nodes to form a continuous path using a nearest-neighbor approach.

    Args:
        nodes (np.ndarray): An array of 3D node coordinates.

    Returns:
        np.ndarray: The sorted array of nodes.
    """
    if len(nodes) < 2:
        return nodes

    # Start from the first node
    sorted_nodes = [nodes[0]]
    remaining_nodes = list(nodes[1:])

    while remaining_nodes:
        last_node = sorted_nodes[-1]
        # Compute distances from the last sorted node to all remaining nodes
        distances = np.linalg.norm(np.array(remaining_nodes) - last_node, axis=1)
        nearest_index = np.argmin(distances)

        # Append the nearest node and remove it from the remaining list
        sorted_nodes.append(remaining_nodes.pop(nearest_index))

    return np.array(sorted_nodes)


def post_process_lines(lines):
    """
    Cleans up the reconstructed lines by removing duplicate nodes and sorting them.

    This is especially important after merging lines, where node order can be jumbled.

    Args:
        lines (dict): Dictionary of dislocation lines.

    Returns:
        dict: The cleaned and sorted dictionary of lines.
    """
    print("Post-processing lines: removing duplicates and sorting nodes...")
    processed_lines = {}
    for key, nodes in lines.items():
        # Find unique nodes in the line
        unique_nodes, indices = np.unique(nodes, axis=0, return_index=True)

        # Sort the unique nodes to ensure a continuous path
        sorted_unique_nodes = sort_nodes_on_line(unique_nodes)

        processed_lines[key] = sorted_unique_nodes

    return processed_lines


def plot_dislocations(dislocation_lines, box_lims=None):
    """
    Creates a 3D plot of the dislocation lines.

    Args:
        dislocation_lines (dict): A dictionary where keys are line IDs and values
                                  are numpy arrays of node coordinates.
        box_lims (list or tuple, optional): The [min, max] limits for the plot axes.
                                            Defaults to None (autoscale).
    """
    print("Generating 3D plot of the dislocation structure...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for line_id, nodes in dislocation_lines.items():
        if nodes.shape[0] > 1:  # Only plot lines with at least one segment
            ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], marker='o', markersize=2, linestyle='-')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Reconstructed 3D Dislocation Structure')

    if box_lims:
        ax.set_xlim(box_lims)
        ax.set_ylim(box_lims)
        ax.set_zlim(box_lims)

    # Improve viewing angle and layout
    ax.view_init(elev=20., azim=-35)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to orchestrate the dislocation analysis workflow."""

    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Process ParaDis dislocation segment files to reconstruct and visualize dislocation lines."
    )
    parser.add_argument(
        '-i', '--input',
        default="visit0002.seg",
        help="Path to the input ParaDis segment file (e.g., 'visit0002.seg')."
    )
    parser.add_argument(
        '-o', '--output',
        default="dislocation_lines.pkl",
        help="Path to the output pickle file to save processed line data."
    )
    parser.add_argument(
        '--plot-lims',
        type=float,
        nargs=2,
        default=[-5000, 5000],
        help="Plot axis limits, e.g., '--plot-lims -5000 5000'."
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help="Disable the 3D plot generation."
    )
    args = parser.parse_args()

    # --- Step 1: Read and Parse Segment Data ---
    segment_endpoints, normal_vectors, burgers_magnitudes = read_paradis_seg_file(args.input)

    if segment_endpoints.size == 0:
        print("No data was read from the input file. Exiting.")
        return

    # --- Step 2: Trace Continuous Dislocation Lines ---
    dislocation_lines, line_normals, line_burgers = trace_dislocation_lines(
        segment_endpoints, normal_vectors, burgers_magnitudes
    )

    # --- Step 3: Post-process Lines (Remove duplicates and sort) ---
    # The new tracing method produces sorted nodes, but we can still clean up
    # any potential duplicates from imperfect data.
    final_lines = post_process_lines(dislocation_lines)

    # --- Analysis & Filtering (Example from original code) ---
    # This section filters for lines with a specific normal vector near the origin.
    # It's a specific analysis task that can be adapted as needed.
    specific_normal_vec = np.array([-0.5773503, -0.5773503, 0.5773503])
    keys_with_specific_normal = []
    for key, nodes in final_lines.items():
        # Check if the normal vector for this line matches
        # We check the first node's normal, assuming it's constant along the line
        if np.allclose(line_normals[key][0], specific_normal_vec):
            # Check if the line passes close to the origin
            if np.abs(np.dot(nodes[0, :], specific_normal_vec)) < 100:
                keys_with_specific_normal.append(key)
    print(f"Found {len(keys_with_specific_normal)} lines matching specific filter criteria.")

    # --- Step 4: Visualize the Dislocation Structure ---
    if not args.no_plot:
        plot_dislocations(final_lines, box_lims=args.plot_lims)

    # --- Step 5: Save the Processed Data ---
    lines_info = {
        'lines': final_lines,
        'normals': line_normals,
        'burgers': line_burgers
    }
    with open(args.output, "wb") as f:
        pickle.dump(lines_info, f)
    print(f"Processed dislocation data successfully saved to '{args.output}'.")


if __name__ == "__main__":
    main()
