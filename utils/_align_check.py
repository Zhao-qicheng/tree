import json, numpy as np, os, cv2

def get_main_range(parts):
    ranges = [(p['Range']['Start'], p['Range']['End']) for p in parts]
    lengths = [e - s for s, e in ranges]
    idx = int(np.argmax(lengths))
    return ranges[idx][0] - 1, ranges[idx][1] - 1

def get_time_range(markers):
    s0, e0 = get_main_range(markers[0]['Parts'])
    t_start, t_end = s0, e0
    for m in markers:
        ts, te = get_main_range(m['Parts'])
        t_start = max(t_start, ts)
        t_end   = min(t_end, te)
    return t_start, t_end

results = []
skaters = ['Skater_A', 'Skater_B', 'Skater_C', 'Skater_D']
jumps   = ['Axel', 'Comb', 'Flip', 'Loop', 'Lutz', 'Salchow', 'Toeloop']

BASE = r'E:\八叉树\data'

for skater in skaters:
    for jump in jumps:
        json_dir = os.path.join(BASE, 'json', skater, jump)
        npy_dir  = os.path.join(BASE, 'npy',  skater, jump)
        vid_dir  = os.path.join(BASE, 'video', skater.lower(), 'cam_1')
        if not os.path.isdir(json_dir):
            continue
        for fname in sorted(os.listdir(json_dir)):
            if not fname.endswith('.json'):
                continue
            stem     = fname[:-5]
            npy_path = os.path.join(npy_dir, stem + '.npy')
            vid_path = os.path.join(vid_dir,  stem + '.mp4')
            if not os.path.exists(npy_path):
                continue

            with open(os.path.join(json_dir, fname)) as f:
                jdata = json.load(f)

            freq   = jdata['Timebase']['Frequency']
            tb_end = jdata['Timebase']['Range']['End']
            ts, te = get_time_range(jdata['Markers'])

            d      = np.load(npy_path)
            npy_fc = d.shape[0]

            vid_fc = -1
            if os.path.exists(vid_path):
                cap    = cv2.VideoCapture(vid_path)
                vid_fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

            calc = te - ts
            ok   = (calc == npy_fc)

            results.append({
                'file':   f'{skater}/{jump}/{stem}',
                'vid':    vid_fc,
                'npy':    npy_fc,
                'ts':     ts,
                'te':     te,
                'calc':   calc,
                'freq':   freq,
                'tb_end': tb_end,
                'ok':     ok,
            })

# ---- 打印报告 ----
header = f"{'File':<38} {'VID':>5} {'NPY':>5} {'ts':>5} {'te':>5} {'calc':>5} {'FPS':>5} {'Match':>5}"
print(header)
print('-' * len(header))

for r in results:
    match_str = 'OK' if r['ok'] else 'MISMATCH'
    print(f"{r['file']:<38} {r['vid']:>5} {r['npy']:>5} {r['ts']:>5} {r['te']:>5} {r['calc']:>5} {r['freq']:>5.0f} {match_str:>5}")

bad     = [r for r in results if not r['ok']]
offsets = [r['ts'] for r in results]
print()
print(f"Total={len(results)}, Mismatch={len(bad)}")
print(f"ts(video_start_offset) range: {min(offsets)} ~ {max(offsets)}")
print(f"ts == 0  count: {sum(1 for x in offsets if x == 0)}")
print(f"ts > 0   count: {sum(1 for x in offsets if x > 0)}")
print()
print("=> npy frame i  corresponds to  video frame:  ts + i  (0-indexed)")
print("=> video frame  corresponds to  npy frame:    video_frame - ts  (if in range)")
