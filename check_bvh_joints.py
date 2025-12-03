import sys
import re

def get_all_joints(bvh_file):
    joints = []
    with open(bvh_file, 'r') as f:
        for line in f:
            if "MOTION" in line:
                break
            match = re.search(r'^\s*(ROOT|JOINT)\s+(\w+)', line)
            if match:
                joints.append(match.group(2))
    return joints

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_bvh_joints.py <bvh_file>")
        sys.exit(1)
    
    bvh_file = sys.argv[1]
    joints = get_all_joints(bvh_file)
    print(f"Found {len(joints)} joints:")
    for j in joints:
        print(j)


