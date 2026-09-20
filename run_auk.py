#!/usr/bin/env python3
"""按轨单逐段驱动 AuK（MLX 后端）生成语音，再按顺序拼成整条。

为什么必须分段
    AuK 单次的预算是「参考音频 + 生成目标 ≤ 30 秒」。一段 178 字的稿子读出来
    大约 32 秒，加上参考音频就超了，所以长稿必须切开。这里每段单独起一个进程，
    让 MLX 在一次生成结束后把内存自然释放掉，避免几段连着跑把内存顶满。

为什么时长要自己给
    AuK 的输出时长由 gen_seconds 硬控，给多少就出多少。这正是「按脚本站位」
    需要的性质：想让这一句占 9.5 秒，它就读 9.5 秒。
    反过来，这个值同时也决定了语速，给宽了语速就慢。经验值约 0.19 秒/字。

用法
    export AUK_REPO=/path/to/AuK          # 官方仓库，需切到 feat/mlx-apple-silicon 分支
    python3 run_auk.py \
        --segments segments.json \
        --ref ref_8s.wav \
        --outdir ./out \
        --variant flash

    只想要一个「多长都行」的快速试听，把 --variant 换成 flash 即可（4 步定长）。
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import wave

# AuK 单次预算：参考音频 + 生成目标，合计不超过 30 秒。留一点余量。
SEGMENT_BUDGET_SECONDS = 30.0
DEFAULT_GAP_SECONDS = 0.12
DEFAULT_INSTRUCTION = 'Say the following with the same voice: "{text}"'


def resolve_repo(cli_value):
    """官方 AuK 仓库的本地路径。优先命令行，其次环境变量，最后给个默认。"""
    if cli_value:
        return pathlib.Path(cli_value).expanduser()
    env = os.environ.get("AUK_REPO")
    if env:
        return pathlib.Path(env).expanduser()
    return pathlib.Path.home() / "Models/AuK/repo"


def run_segment(py, repo, text, gen_seconds, ref, out, variant, nfe, bits, cfg, seed, instruct):
    cmd = [
        str(py), "-m", "auk_mlx.cli",
        "--instruction", instruct.format(text=text),
        "--audio", str(ref),
        "--gen_seconds", f"{gen_seconds:.2f}",
        "--output", str(out),
        "--seed", str(seed),
    ]
    if variant == "flash":
        # Flash 是 4 步蒸馏，固定步数、CFG 关闭，传 --nfe 无效。
        cmd.append("--flash")
    else:
        cmd += ["--nfe", str(nfe)]
    if bits:
        cmd += ["--bits", str(bits)]
    if cfg is not None:
        cmd += ["--cfg", str(cfg)]

    # 干净环境，避免宿主 shell 里其它 PYTHONPATH / 虚拟环境影响子进程。
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(pathlib.Path.home()),
        "PYTHONPATH": str(repo / "src"),
        "PYTHONUNBUFFERED": "1",
    }
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(repo), env=env, capture_output=True, text=True)
    dt = time.time() - t0
    print(f"--- 第 {out.stem} 段  rc={p.returncode}  耗时 {dt:.1f}s")
    for line in (p.stdout or "").strip().splitlines()[-3:]:
        print(f"    {line}")
    if p.returncode != 0:
        for line in (p.stderr or "").strip().splitlines()[-12:]:
            print(f"    ! {line}")
        return None, dt
    return out, dt


def concat(paths, dst, gap):
    """按顺序拼接，段间插一小段静音。用 wave 直接拼，不依赖 ffmpeg。"""
    frames, sr, ch, sw = [], None, None, None
    for p in paths:
        with wave.open(str(p), "rb") as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            frames.append(w.readframes(w.getnframes()))
    silence = b"\x00" * (int(sr * gap) * ch * sw)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        for i, f in enumerate(frames):
            w.writeframes(f)
            if i != len(frames) - 1:
                w.writeframes(silence)
    return sr


def main():
    ap = argparse.ArgumentParser(description="AuK 分段生成 + 拼接")
    ap.add_argument("--repo", default=None,
                    help="官方 AuK 仓库路径（默认取环境变量 AUK_REPO，再默认 ~/Models/AuK/repo）")
    ap.add_argument("--segments", default="segments.json", help="轨单 JSON")
    ap.add_argument("--ref", default=None, help="参考音频；不传则用轨单里的 ref_audio")
    ap.add_argument("--outdir", default="./out", help="成品输出目录")
    ap.add_argument("--work", default=None, help="分段中间件目录（默认 outdir/_segments）")
    ap.add_argument("--variant", choices=["base", "flash"], default="flash",
                    help="flash=4 步蒸馏（快）；base=32 步（质量优先，慢约 4 倍）")
    ap.add_argument("--nfe", type=int, default=32, help="base 的采样步数")
    ap.add_argument("--bits", type=int, default=8,
                    help="量化位宽。官方结论：用 8，别用 4（4-bit 中文发音会退化）")
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--gap", type=float, default=DEFAULT_GAP_SECONDS, help="段间静音秒数")
    ap.add_argument("--tag", default=None, help="产物文件名前缀（默认用 variant）")
    ap.add_argument("--instruct", default=None,
                    help="覆盖指令模板，用 {text} 占位。默认用官方的零样本模板。"
                         "注意：这个模板换一个字都会出问题，别随手改。")
    args = ap.parse_args()

    repo = resolve_repo(args.repo)
    py = repo / ".venv/bin/python"
    if not py.exists():
        sys.exit(f"找不到 {py}。请先按 AuK 官方 docs/MLX.md 建好 .venv，"
                 f"或用 --repo 指定仓库路径。")

    spec = json.loads(pathlib.Path(args.segments).read_text())
    ref = pathlib.Path(args.ref).expanduser() if args.ref else pathlib.Path(spec["ref_audio"])
    if not ref.exists():
        sys.exit(f"找不到参考音频 {ref}")

    tag = args.tag or args.variant
    outdir = pathlib.Path(args.outdir).expanduser()
    work = pathlib.Path(args.work).expanduser() if args.work else outdir / "_segments"
    outdir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    segs = spec["segments"]
    print(f"变体={args.variant}  bits={args.bits}  段数={len(segs)}")
    print(f"参考音频={ref.name}  输出={outdir}")

    made, report, total = [], [], 0.0
    for s in segs:
        if s["gen_seconds"] > SEGMENT_BUDGET_SECONDS:
            print(f"!! 第 {s['id']} 段给了 {s['gen_seconds']}s，超过单次预算，请切得更细")
            break
        out = work / f"{tag}_seg{s['id']}.wav"
        got, dt = run_segment(py, repo, s["text"], s["gen_seconds"], ref, out,
                              args.variant, args.nfe, args.bits, args.cfg,
                              args.seed, args.instruct or DEFAULT_INSTRUCTION)
        total += dt
        report.append({"id": s["id"], "chars": s.get("chars"), "gen_seconds": s["gen_seconds"],
                       "wall": round(dt, 1), "ok": got is not None})
        if got:
            made.append(got)
        else:
            print(f"!! 第 {s['id']} 段失败，停止")
            break

    if not made:
        sys.exit("没有任何一段生成成功。")

    final = outdir / f"auk_{tag}.wav"
    sr = concat(made, final, args.gap)
    with wave.open(str(final), "rb") as w:
        dur = w.getnframes() / w.getframerate()

    (outdir / f"{tag}_report.json").write_text(json.dumps(
        {"variant": args.variant, "nfe": args.nfe, "bits": args.bits, "cfg": args.cfg,
         "seed": args.seed, "segments": report, "total_wall": round(total, 1),
         "final_dur": round(dur, 2), "rtf": round(total / dur, 2) if dur else None,
         "final": str(final)}, ensure_ascii=False, indent=2))

    print(f"\n成品: {final}")
    print(f"时长 {dur:.2f}s @ {sr}Hz   生成总耗时 {total:.1f}s   RTF={total / dur:.2f}")
    print("（RTF 是「生成耗时 ÷ 音频时长」，小于 1 就是比实时快）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
