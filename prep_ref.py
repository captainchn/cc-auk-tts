#!/usr/bin/env python3
"""从一段自己的录音里裁出干净的参考音频，并打印每段的音质特征。

为什么需要这一步
    AuK 单次预算只有 30 秒（参考音频 + 生成目标）。拿一整条几十分钟的录音
    直接当参考是不行的，必须切出一小段。同时，参考音频的质量会直接决定
    生成结果 —— 参考里有什么毛病，产物里就会有，而且会被放大。

    所以裁之前先看一眼质量特征：峰值、RMS、削波样本数、静音比例。
    选一段没有削波、静音少的。

用法
    # 先看整条录音，再看几个候选段的特征
    python3 prep_ref.py --src 我的录音.wav --out ./ref --scan 11.5-19.5 22.0-30.0 27.0-35.0

    # 也可以只裁一段
    python3 prep_ref.py --src 我的录音.wav --out ./ref --cut 11.5-19.5

输入要求
    16bit PCM wav。不是 wav 的话，先用 ffmpeg 转：
        ffmpeg -i input.m4a -ac 1 -ar 48000 -c:a pcm_s16le output.wav
"""
import argparse
import pathlib
import sys
import wave

import numpy as np


def read_wav(p):
    with wave.open(str(p), "rb") as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        sys.exit(f"{p} 不是 16bit，实际 {sw * 8}bit。先用 ffmpeg 转成 pcm_s16le。")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(p, x, sr):
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.clip(x, -32768, 32767).astype("<i2").tobytes())


def features(seg, sr):
    peak = float(np.max(np.abs(seg)) / 32768.0)
    rms = float(np.sqrt(np.mean((seg / 32768.0) ** 2)))
    clip = int(np.sum(np.abs(seg) >= 32700))
    frame = int(0.02 * sr)
    nf = len(seg) // frame
    silent = 0.0
    if nf:
        fr = np.abs(seg[: nf * frame].reshape(nf, frame)).mean(axis=1) / 32768.0
        silent = float(np.mean(fr < 0.01))
    return peak, rms, clip, silent


def parse_range(s):
    a, b = s.split("-")
    return s, float(a), float(b)


def main():
    ap = argparse.ArgumentParser(description="裁参考音频 + 打印质量特征")
    ap.add_argument("--src", required=True, help="源录音（16bit wav）")
    ap.add_argument("--out", default="./ref", help="参考音频输出目录")
    ap.add_argument("--scan", nargs="*", default=[],
                    help="候选区间，形如 11.5-19.5（秒）。会打印特征并写出文件")
    ap.add_argument("--cut", default=None, help="只裁一个区间，形如 11.5-19.5")
    args = ap.parse_args()

    src = pathlib.Path(args.src).expanduser()
    out = pathlib.Path(args.out).expanduser()
    x, sr = read_wav(src)
    print(f"源: {src.name}  采样率 {sr}Hz  时长 {len(x) / sr:.2f}s")

    cands = []
    if args.cut:
        cands.append(parse_range(args.cut))
    for s in args.scan:
        cands.append(parse_range(s))
    if not cands:
        sys.exit("请用 --scan 给候选区间，或用 --cut 给一个区间。")

    print(f"\n{'区间':>16s}  {'时长':>6s}  {'峰值':>6s}  {'RMS':>7s}  {'削波':>6s}  {'静音':>6s}")
    for tag, a, b in cands:
        i, j = int(a * sr), int(b * sr)
        seg = x[i:j]
        if len(seg) == 0:
            print(f"{tag:>16s}  （区间超出音频长度）")
            continue
        peak, rms, clip, silent = features(seg, sr)
        print(f"{tag:>16s}  {len(seg) / sr:5.2f}s  {peak:6.3f}  {rms:7.4f}  "
              f"{clip:6d}  {silent * 100:5.1f}%")
        p = out / f"{tag.replace('.', '_').replace('-', '-')}.wav"
        write_wav(p, seg, sr)
        print(f"{'':>16s}  -> {p}  {p.stat().st_size / 1024:.0f} KB")

    print("\n挑选标准：削波 0、静音比例低、时长 8 到 20 秒、一句话说完不中断。")
    print("参考音频总时长超过 20 秒意义不大：单次预算只有 30 秒，留给生成目标的空间会被挤掉。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
