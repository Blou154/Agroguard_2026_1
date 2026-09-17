from pathlib import Path
import csv, cv2

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data" / "validation" / "validation_cut_plan.csv"
SRC = ROOT / "data" / "validation" / "source"
OUT = ROOT / "data" / "validation" / "clips"

def t(s):
    p=[float(x) for x in s.split(":")]
    return p[0]*60+p[1] if len(p)==2 else p[0]*3600+p[1]*60+p[2]

def cut(src, out, a, b):
    cap=cv2.VideoCapture(str(src))
    if not cap.isOpened(): raise RuntimeError(f"No se pudo abrir {src}")
    fps=cap.get(cv2.CAP_PROP_FPS)
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sf=round(a*fps); ef=round(b*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    out.parent.mkdir(parents=True, exist_ok=True)
    wr=cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    n=0
    while sf+n < ef:
        ok, frame=cap.read()
        if not ok: break
        wr.write(frame); n+=1
    cap.release(); wr.release()
    return n/fps

def main():
    with PLAN.open(encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    print(f"Se generarán {len(rows)} clips.")
    ok=0
    for i,r in enumerate(rows,1):
        src=SRC/r["area"]/r["source_video"]
        out=OUT/r["area"]/(r["clip_id"]+".mp4")
        try:
            d=cut(src,out,t(r["start"]),t(r["end"]))
            print(f"[{i:02d}/{len(rows)}] OK {r['clip_id']} ({d:.2f}s)")
            ok+=1
        except Exception as e:
            print(f"[{i:02d}/{len(rows)}] ERROR {r['clip_id']}: {e}")
    print(f"\nGenerados: {ok}/{len(rows)}")

if __name__=="__main__":
    main()