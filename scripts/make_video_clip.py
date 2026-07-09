import argparse

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source video: {args.src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        args.dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    count = 0
    while count < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        count += 1

    writer.release()
    cap.release()
    print(f"written {count} frames -> {args.dst}")


if __name__ == "__main__":
    main()
