import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path,
                    help="JSON 파일들이 들어있는 폴더 경로")
    args = ap.parse_args()

    loss_sum = 0.0
    loss_cnt = 0
    dur_sum = 0.0
    dur_cnt = 0

    json_files = sorted([p for p in args.results.iterdir() if p.suffix.lower() == ".json"])
    if not json_files:
        print(f"[WARN] {args.results} 폴더에 json 파일이 없습니다.")
        return

    for p in json_files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[SKIP] {p.name}: JSON 로드 실패 ({e})")
            continue

        # 1) 리스트 형태: "losses", "durations"
        if "losses" in data and isinstance(data["losses"], list):
            for v in data["losses"]:
                loss_sum += float(v)
                loss_cnt += 1
        if "durations" in data and isinstance(data["durations"], list):
            for v in data["durations"]:
                dur_sum += float(v)
                dur_cnt += 1

        # 2) 단일 값: "loss", "duration"
        if "loss" in data and not isinstance(data.get("losses"), list):
            loss_sum += float(data["loss"])
            loss_cnt += 1
        if "duration" in data and not isinstance(data.get("durations"), list):
            dur_sum += float(data["duration"])
            dur_cnt += 1

    if loss_cnt == 0 and dur_cnt == 0:
        print("[ERROR] 유효한 loss/duration 값을 찾지 못했습니다.")
        return

    if loss_cnt > 0:
        print(f"Average loss     = {loss_sum / loss_cnt:.6f}  (N={loss_cnt})")
    else:
        print("Average loss     = N/A")

    if dur_cnt > 0:
        print(f"Average duration = {dur_sum / dur_cnt:.6f}  (N={dur_cnt})")
    else:
        print("Average duration = N/A")

if __name__ == "__main__":
    main()