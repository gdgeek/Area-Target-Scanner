#!/usr/bin/env python3
"""Analyze and visualize the scan result from data/scan_20260317_132224."""

import json
import os
import numpy as np
import trimesh
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SCAN_DIR = "data/scan_20260317_132224/scan_20260317_132224"
OUT_DIR = "data/scan_analysis"
os.makedirs(OUT_DIR, exist_ok=True)


def load_intrinsics():
    with open(os.path.join(SCAN_DIR, "intrinsics.json")) as f:
        return json.load(f)


def load_poses():
    with open(os.path.join(SCAN_DIR, "poses.json")) as f:
        return json.load(f)


def load_mesh():
    return trimesh.load(
        os.path.join(SCAN_DIR, "model.obj"),
        process=False,
    )


def load_pointcloud():
    """Load PLY point cloud manually (simple ASCII parser)."""
    path = os.path.join(SCAN_DIR, "pointcloud.ply")
    points, colors = [], []
    in_data = False
    with open(path) as f:
        for line in f:
            if line.strip() == "end_header":
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.array(points), np.array(colors)


def analyze_mesh(mesh):
    print("=" * 60)
    print("MESH ANALYSIS")
    print("=" * 60)
    print(f"  Vertices:       {len(mesh.vertices):,}")
    print(f"  Faces:          {len(mesh.faces):,}")
    bb = mesh.bounds
    dims = bb[1] - bb[0]
    print(f"  Bounding box:   [{bb[0][0]:.2f}, {bb[0][1]:.2f}, {bb[0][2]:.2f}]")
    print(f"                  [{bb[1][0]:.2f}, {bb[1][1]:.2f}, {bb[1][2]:.2f}]")
    print(f"  Dimensions:     {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} meters")
    print(f"  Surface area:   {mesh.area:.2f} m²")
    if mesh.is_watertight:
        print(f"  Volume:         {mesh.volume:.4f} m³")
    else:
        print(f"  Volume:         N/A (mesh is not watertight)")
    print(f"  Watertight:     {mesh.is_watertight}")

    # Edge length stats
    edges = mesh.edges_unique_length
    print(f"  Edge lengths:   min={edges.min():.6f}, max={edges.max():.4f}, mean={edges.mean():.4f}")
    return dims


def analyze_pointcloud(points, colors):
    print("\n" + "=" * 60)
    print("POINT CLOUD ANALYSIS")
    print("=" * 60)
    print(f"  Points:         {len(points):,}")
    bb_min, bb_max = points.min(axis=0), points.max(axis=0)
    dims = bb_max - bb_min
    print(f"  Bounding box:   [{bb_min[0]:.2f}, {bb_min[1]:.2f}, {bb_min[2]:.2f}]")
    print(f"                  [{bb_max[0]:.2f}, {bb_max[1]:.2f}, {bb_max[2]:.2f}]")
    print(f"  Dimensions:     {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} meters")
    print(f"  Color range:    R[{colors[:,0].min()}-{colors[:,0].max()}] "
          f"G[{colors[:,1].min()}-{colors[:,1].max()}] "
          f"B[{colors[:,2].min()}-{colors[:,2].max()}]")
    has_color = np.any(colors > 0)
    print(f"  Has color data: {has_color}")


def analyze_poses(poses_data):
    frames = poses_data["frames"]
    print("\n" + "=" * 60)
    print("CAMERA POSES ANALYSIS")
    print("=" * 60)
    print(f"  Total frames:   {len(frames)}")
    print(f"  Time span:      {frames[0]['timestamp']:.2f}s - {frames[-1]['timestamp']:.2f}s "
          f"({frames[-1]['timestamp'] - frames[0]['timestamp']:.2f}s)")

    positions = []
    for frame in frames:
        t = np.array(frame["transform"]).reshape(4, 4).T  # column-major to row-major
        pos = t[:3, 3]
        positions.append(pos)
    positions = np.array(positions)

    total_dist = 0
    for i in range(1, len(positions)):
        total_dist += np.linalg.norm(positions[i] - positions[i - 1])

    print(f"  Camera travel:  {total_dist:.3f} meters")
    print(f"  Start position: [{positions[0][0]:.4f}, {positions[0][1]:.4f}, {positions[0][2]:.4f}]")
    print(f"  End position:   [{positions[-1][0]:.4f}, {positions[-1][1]:.4f}, {positions[-1][2]:.4f}]")
    return positions


def analyze_texture():
    tex_path = os.path.join(SCAN_DIR, "texture.jpg")
    img = Image.open(tex_path)
    print("\n" + "=" * 60)
    print("TEXTURE ANALYSIS")
    print("=" * 60)
    print(f"  Resolution:     {img.size[0]} x {img.size[1]}")
    print(f"  Mode:           {img.mode}")
    arr = np.array(img)
    print(f"  Mean intensity: R={arr[:,:,0].mean():.1f} G={arr[:,:,1].mean():.1f} B={arr[:,:,2].mean():.1f}")


def analyze_intrinsics(intr):
    print("\n" + "=" * 60)
    print("CAMERA INTRINSICS")
    print("=" * 60)
    print(f"  Resolution:     {intr['width']} x {intr['height']}")
    print(f"  Focal length:   fx={intr['fx']:.2f}, fy={intr['fy']:.2f}")
    print(f"  Principal pt:   cx={intr['cx']:.2f}, cy={intr['cy']:.2f}")
    fov_h = 2 * np.degrees(np.arctan(intr['width'] / (2 * intr['fx'])))
    fov_v = 2 * np.degrees(np.arctan(intr['height'] / (2 * intr['fy'])))
    print(f"  FOV:            H={fov_h:.1f}°, V={fov_v:.1f}°")


def plot_mesh_views(mesh, cam_positions):
    """Render top-down and side views of the mesh with camera trajectory."""
    fig = plt.figure(figsize=(18, 12))

    verts = mesh.vertices
    # Subsample for plotting
    idx = np.random.choice(len(verts), min(20000, len(verts)), replace=False)
    sv = verts[idx]

    # Top-down view (XZ plane)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(sv[:, 0], sv[:, 2], s=0.1, c='gray', alpha=0.3)
    ax1.plot(cam_positions[:, 0], cam_positions[:, 2], 'r.-', markersize=4, linewidth=1)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('Top-Down View (XZ) + Camera Path')
    ax1.set_aspect('equal')
    ax1.legend(['Camera path', 'Mesh vertices'], fontsize=8)

    # Side view (XY plane)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(sv[:, 0], sv[:, 1], s=0.1, c='gray', alpha=0.3)
    ax2.plot(cam_positions[:, 0], cam_positions[:, 1], 'r.-', markersize=4, linewidth=1)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Side View (XY) + Camera Path')
    ax2.set_aspect('equal')

    # 3D view
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(sv[:, 0], sv[:, 2], sv[:, 1], s=0.1, c='gray', alpha=0.2)
    ax3.plot(cam_positions[:, 0], cam_positions[:, 2], cam_positions[:, 1],
             'r.-', markersize=4, linewidth=1.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_zlabel('Y')
    ax3.set_title('3D View: Mesh + Camera Trajectory')

    # Face area distribution
    ax4 = fig.add_subplot(2, 2, 4)
    areas = mesh.area_faces
    ax4.hist(areas, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    ax4.set_xlabel('Face Area (m²)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Face Area Distribution (mean={areas.mean():.6f} m²)')
    ax4.set_yscale('log')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "mesh_overview.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved: {out}")


def plot_texture_and_frames():
    """Show texture atlas and sample captured frames."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Texture atlas
    tex = Image.open(os.path.join(SCAN_DIR, "texture.jpg"))
    axes[0][0].imshow(tex)
    axes[0][0].set_title("Texture Atlas (4096x4096)")
    axes[0][0].axis('off')

    # Sample frames
    frame_indices = [0, 4, 8, 12, 1, 6, 10, 15]
    for i, fi in enumerate(frame_indices):
        if i == 0:
            continue  # skip first slot (texture)
        row, col = divmod(i, 4)
        img = Image.open(os.path.join(SCAN_DIR, f"images/frame_{fi:04d}.jpg"))
        axes[row][col].imshow(img)
        axes[row][col].set_title(f"Frame {fi:04d}")
        axes[row][col].axis('off')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "texture_and_frames.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")


def plot_camera_trajectory(positions, poses_data):
    """Detailed camera trajectory analysis."""
    frames = poses_data["frames"]
    timestamps = [f["timestamp"] for f in frames]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # XYZ over time
    ax = axes[0][0]
    ax.plot(timestamps, positions[:, 0], 'r-', label='X')
    ax.plot(timestamps, positions[:, 1], 'g-', label='Y')
    ax.plot(timestamps, positions[:, 2], 'b-', label='Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Camera Position Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Inter-frame distance
    ax = axes[0][1]
    dists = [np.linalg.norm(positions[i] - positions[i-1]) for i in range(1, len(positions))]
    ax.bar(range(1, len(positions)), dists, color='steelblue')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Inter-Frame Camera Movement')
    ax.grid(True, alpha=0.3)

    # Camera orientation (forward vector)
    ax = axes[1][0]
    for i, frame in enumerate(frames):
        t = np.array(frame["transform"]).reshape(4, 4).T
        fwd = t[:3, 2]  # Z axis = forward
        ax.arrow(positions[i, 0], positions[i, 2],
                 fwd[0]*0.05, fwd[2]*0.05,
                 head_width=0.003, color=plt.cm.viridis(i / len(frames)))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Camera Positions + Viewing Directions (top-down)')
    ax.set_aspect('equal')

    # Frame timestamps
    ax = axes[1][1]
    dt = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    ax.bar(range(1, len(timestamps)), dt, color='coral')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Δt (s)')
    ax.set_title(f'Frame Intervals (mean={np.mean(dt):.3f}s ≈ {1/np.mean(dt):.1f} fps)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "camera_trajectory.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    print("Loading scan data from:", SCAN_DIR)
    print()

    # Load data
    intr = load_intrinsics()
    poses_data = load_poses()
    mesh = load_mesh()
    points, colors = load_pointcloud()

    # Analysis
    analyze_intrinsics(intr)
    analyze_mesh(mesh)
    analyze_pointcloud(points, colors)
    cam_positions = analyze_poses(poses_data)
    analyze_texture()

    # Visualizations
    print("\nGenerating visualizations...")
    plot_mesh_views(mesh, cam_positions)
    plot_texture_and_frames()
    plot_camera_trajectory(cam_positions, poses_data)

    print("\n" + "=" * 60)
    print(f"All analysis images saved to: {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
