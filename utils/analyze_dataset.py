import os
import numpy as np

def analyze():
    root_dir = './data/npy'
    if not os.path.exists(root_dir):
        print(f"Error: {root_dir} not found.")
        return

    all_offsets = []
    global_spans_x = []
    global_spans_y = []
    global_spans_z = []

    file_count = 0
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.npy'):
                path = os.path.join(root, f)
                data = np.load(path) # (frames, joints, 3)
                file_count += 1

                # 1. Local range (relative to root joint index 0)
                root_joint = data[:, 0:1, :] # (frames, 1, 3)
                offsets = data - root_joint   # (frames, joints, 3)
                all_offsets.append(np.abs(offsets).max())

                # 2. Global spans
                spans = data.max(axis=(0,1)) - data.min(axis=(0,1))
                global_spans_x.append(spans[0])
                global_spans_y.append(spans[1])
                global_spans_z.append(spans[2])

    if file_count == 0:
        print("No .npy files found.")
        return

    print(f"--- Analysis Result (Total {file_count} files) ---")
    print(f"Max Local Offset (from Hip): {max(all_offsets):.2f} mm")
    print(f"Recommended Static Range: +/- {int(max(all_offsets) * 1.1)} mm")
    print(f"Max Global Span X: {max(global_spans_x):.2f} mm")
    print(f"Max Global Span Y: {max(global_spans_y):.2f} mm")
    print(f"Max Global Span Z: {max(global_spans_z):.2f} mm")

if __name__ == "__main__":
    analyze()
