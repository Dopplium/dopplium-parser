"""
Example script demonstrating how to parse Dopplium Detections files.

This script shows how to:
1. Parse a detections binary file
2. Access detection data and headers
3. Filter detections by various criteria
4. Get statistics about the detections
5. Visualize detections
"""

import numpy as np
import matplotlib.pyplot as plt
from python_parser import (
    parse_dopplium_detections,
    filter_detections_by_range,
    filter_detections_by_velocity,
    filter_detections_by_amplitude,
    get_detection_statistics,
)


def main():
    # --- Parse detections file ---
    detections, headers = parse_dopplium_detections(
        "PATH_TO_DETECTIONS.bin",
        max_batches=None,  # Read all batches (or specify a number)
        verbose=True
    )
    
    print(f"\n=== Detections Parsed ===")
    print(f"Total detections: {len(detections)}")
    print(f"Detection fields: {list(detections.dtype.names)}")
    
    # --- Access individual fields ---
    ranges = detections['range']
    velocities = detections['velocity']
    azimuths = detections['azimuth']
    elevations = detections['elevation']
    amplitudes = detections['amplitude']
    
    print(f"\nRange: min={ranges.min():.2f}, max={ranges.max():.2f} m")
    print(f"Velocity: min={velocities.min():.2f}, max={velocities.max():.2f} m/s")
    print(f"Azimuth: min={azimuths.min():.2f}, max={azimuths.max():.2f} deg")
    print(f"Elevation: min={elevations.min():.2f}, max={elevations.max():.2f} deg")
    print(f"Amplitude: min={amplitudes.min():.2f}, max={amplitudes.max():.2f}")
    
    # --- Get detection statistics ---
    stats = get_detection_statistics(detections)
    print(f"\n=== Detection Statistics ===")
    print(f"Total detections: {stats['count']}")
    print(f"Unique batches: {stats['batches']['unique_batches']}")
    print(f"Range stats: min={stats['range']['min']:.2f}, max={stats['range']['max']:.2f}, "
          f"mean={stats['range']['mean']:.2f}, std={stats['range']['std']:.2f}")
    print(f"Velocity stats: min={stats['velocity']['min']:.2f}, max={stats['velocity']['max']:.2f}, "
          f"mean={stats['velocity']['mean']:.2f}, std={stats['velocity']['std']:.2f}")
    
    # --- Filter detections ---
    # Filter by range (e.g., detections between 10m and 100m)
    filtered_range = filter_detections_by_range(detections, 10.0, 100.0)
    print(f"\nDetections in range [10, 100] m: {len(filtered_range)}")
    
    # Filter by velocity (e.g., moving targets: |v| > 1 m/s)
    moving = filter_detections_by_velocity(detections, -100.0, -1.0)
    moving = np.concatenate([moving, filter_detections_by_velocity(detections, 1.0, 100.0)])
    print(f"Moving detections (|v| > 1 m/s): {len(moving)}")
    
    # Filter by amplitude threshold
    strong_detections = filter_detections_by_amplitude(detections, 20.0)
    print(f"Strong detections (amplitude > 20): {len(strong_detections)}")
    
    # --- Combined filtering ---
    # Get strong, close, moving targets
    close = filter_detections_by_range(detections, 0.0, 50.0)
    close_strong = filter_detections_by_amplitude(close, 15.0)
    print(f"Close and strong detections: {len(close_strong)}")
    
    # --- Visualize detections ---
    if len(detections) > 0:
        visualize_detections(detections)
    
    # --- Access batch headers ---
    print(f"\n=== Batch Information ===")
    print(f"Total batches: {len(headers['batch'])}")
    for i, batch in enumerate(headers['batch'][:5]):  # Show first 5 batches
        print(f"Batch {i}: seq={batch.sequence_number}, detections={batch.detection_count}, "
              f"timestamp={batch.timestamp_utc_ticks}")


def visualize_detections(detections: np.ndarray):
    """
    Create visualizations of detection data.
    
    Parameters:
    -----------
    detections : np.ndarray
        Structured array from parse_dopplium_detections
    """
    fig = plt.figure(figsize=(15, 10))
    
    # --- 1. Range-Doppler plot ---
    ax1 = fig.add_subplot(2, 3, 1)
    scatter1 = ax1.scatter(
        detections['velocity'], 
        detections['range'],
        c=detections['amplitude'],
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    ax1.set_xlabel('Velocity (m/s)')
    ax1.set_ylabel('Range (m)')
    ax1.set_title('Range-Doppler Map')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Amplitude')
    
    # --- 2. Range-Azimuth plot (top view) ---
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(
        detections['azimuth'], 
        detections['range'],
        c=detections['amplitude'],
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    ax2.set_xlabel('Azimuth (deg)')
    ax2.set_ylabel('Range (m)')
    ax2.set_title('Range-Azimuth Map (Top View)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Amplitude')
    
    # --- 3. Range-Elevation plot (side view) ---
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(
        detections['elevation'], 
        detections['range'],
        c=detections['amplitude'],
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    ax3.set_xlabel('Elevation (deg)')
    ax3.set_ylabel('Range (m)')
    ax3.set_title('Range-Elevation Map (Side View)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Amplitude')
    
    # --- 4. Range histogram ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(detections['range'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_xlabel('Range (m)')
    ax4.set_ylabel('Count')
    ax4.set_title('Range Distribution')
    ax4.grid(True, alpha=0.3)
    
    # --- 5. Velocity histogram ---
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(detections['velocity'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax5.set_xlabel('Velocity (m/s)')
    ax5.set_ylabel('Count')
    ax5.set_title('Velocity Distribution')
    ax5.grid(True, alpha=0.3)
    
    # --- 6. Amplitude histogram ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(detections['amplitude'], bins=50, alpha=0.7, color='red', edgecolor='black')
    ax6.set_xlabel('Amplitude')
    ax6.set_ylabel('Count')
    ax6.set_title('Amplitude Distribution')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

