#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_shengbte_control.py —— 从 POSCAR 自动生成 ShengBTE CONTROL。

用法:
    python3 gen_shengbte_control.py POSCAR [-o CONTROL] [--ngrid 12 12 12]
        [--scell 2 2 2] [--T 300] [--scalebroad 1.0] [--nonanalytic]

自动推导（杜绝手写错误）:
  - lattvec: POSCAR 晶格(含缩放), lfactor=0.1 (单位 nm; lattvec 给 A 时 lfactor 必须 0.1, 否则晶格放大 10 倍)
  - elements/types: 按 POSCAR 元素行顺序; types = 1-based 元素序号
    (ShengBTE 约定 types 是 elements 列表里的位置, 不是原子序数!)
  - positions: POSCAR Direct 分数坐标 (ShengBTE 用分数坐标)
  - scell: 超胞倍数 (需与 fc2/fc3 拟合一致, 默认 2 2 2)
  - ngrid: q 网格 (默认 12 12 12)
"""
import argparse
import numpy as np


def parse_poscar(path):
    with open(path) as f:
        lines = f.read().splitlines()
    nonempty = [l for l in lines if l.strip()]
    scale = float(nonempty[1])
    lat = np.array([[float(x) for x in nonempty[2 + i].split()] for i in range(3)])
    if abs(scale - 1.0) > 1e-8:
        lat *= scale
    # VASP5 species line (line index 5 in nonempty list)
    line5 = nonempty[5].split()
    counts = [int(x) for x in nonempty[6].split()]
    species = [s for s in line5 if not s.replace('.', '').replace('-', '').isdigit()]
    if len(species) != len(counts):
        raise SystemExit("需 VASP5 POSCAR（含元素符号行）；请补元素行")
    natom = sum(counts)
    # 定位坐标起始行
    coord_start = None
    cart = False
    for k in range(7, min(14, len(nonempty))):
        low = nonempty[k].strip().lower()
        if low.startswith('direct') or low.startswith('cartesian'):
            coord_start = k + 1
            cart = low.startswith('cartesian')
            break
    if coord_start is None:
        # VASP4 无 Direct 行？VASP5 一定有。保守回退：species+counts 后即坐标
        coord_start = 8 if len(species) else 7
    frac = np.array([[float(x) for x in nonempty[coord_start + i].split()[:3]]
                     for i in range(natom)])
    if cart:
        frac = frac @ np.linalg.inv(lat)
    return lat, species, counts, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poscar")
    ap.add_argument("-o", "--output", default="CONTROL")
    ap.add_argument("--ngrid", nargs=3, type=int, default=[12, 12, 12])
    ap.add_argument("--scell", nargs=3, type=int, default=[2, 2, 2])
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--scalebroad", type=float, default=1.0)
    ap.add_argument("--nonanalytic", action="store_true")
    args = ap.parse_args()

    lat, species, counts, frac = parse_poscar(args.poscar)
    natom = frac.shape[0]
    nsp = len(species)
    types = []
    for si, n in enumerate(counts):
        types += [si + 1] * n
    out = ["&allocations",
           "    nelements=%d" % nsp,
           "    natoms=%d" % natom,
           "    ngrid(:)=%d %d %d" % tuple(args.ngrid),
           "&end",
           "&crystal",
           "    lfactor=0.100000"]
    for i in range(3):
        out.append("    lattvec(:,%d)=%s %s %s"
                   % (i + 1, *["%.15f" % v for v in lat[i]]))
    out.append("    elements= %s" % " ".join('"%s"' % s for s in species))
    out.append("    types= %s" % " ".join(str(t) for t in types))
    for i in range(natom):
        out.append("    positions(:,%d)=%.15f %.15f %.15f"
                   % (i + 1, *[float(v) for v in frac[i]]))
    out.append("    scell(:)=%d %d %d" % tuple(args.scell))
    out.append("&end")
    out.append("&parameters")
    out.append("    T=%.6f" % args.T)
    out.append("    scalebroad=%.6f" % args.scalebroad)
    out.append("&end")
    out.append("&flags")
    out.append("    nonanalytic=%s" % (".TRUE." if args.nonanalytic else ".FALSE."))
    out.append("    convergence=.FALSE.")  # RTA 非迭代; 默认迭代(Ind)对 62 原子 186 模式会死循环
    out.append("    nanowires=.FALSE.")
    out.append("&end")
    with open(args.output, "w") as f:
        f.write("\n".join(out) + "\n")
    print("CONTROL -> %s (natoms=%d nelements=%d ngrid=%dx%dx%d scell=%dx%dx%d)"
          % (args.output, natom, nsp, *args.ngrid, *args.scell))


if __name__ == "__main__":
    main()
