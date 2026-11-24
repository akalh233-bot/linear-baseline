# -*- coding: utf-8 -*-
"""Shared label-free genomic & Micro-C 3D feature builders for packaging full-bio tables.

Pure-Python (numpy + pandas + stdlib only; no hicstraw/cooler dependency):
  - ``load_genomic_intervals``: hg38 protein-coding TSSs + ENCODE cCRE v3 midpoints
  - ``compute_raw_extra``: unused cell-type-measured columns from EPCrisprBenchmark TSV
  - ``compute_genomic_ctx``: exact pair-level intervening TSS/cCRE counts, local density,
    and genome-wide power-law cCRE competition (gamma=0.87, Fulco et al. 2019)
  - ``extract_microc_contacts``: VC_SQRT-balanced K562 Micro-C 3D contact (5/10/25 kb,
    3x3 smoothed + distance-stratified O/E) from Juicer .hic v9 files
"""
from __future__ import annotations

import gzip
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class HicV9Reader:
    """Minimal pure-Python Juicer .hic v9 intrachromosomal contact reader."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.f = open(self.path, "rb")
        self._parse_header()

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> "HicV9Reader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _i32(self) -> int:
        return struct.unpack("<i", self.f.read(4))[0]

    def _i64(self) -> int:
        return struct.unpack("<q", self.f.read(8))[0]

    def _f32(self) -> float:
        return struct.unpack("<f", self.f.read(4))[0]

    def _str(self) -> str:
        buf = bytearray()
        while True:
            b = self.f.read(1)
            if not b or b == b"\x00":
                break
            buf.extend(b)
        return buf.decode("ascii", "replace")

    def _parse_header(self) -> None:
        magic = self.f.read(4)
        if magic != b"HIC\x00":
            raise ValueError(f"Not a .hic file: {magic!r}")
        self.version = self._i32()
        if self.version != 9:
            raise ValueError(f"Expected .hic v9, got v{self.version}")
        self.footer_pos = self._i64()
        self.genome = self._str()
        _nvi_pos = self._i64()
        _nvi_len = self._i64()
        n_attrs = self._i32()
        for _ in range(n_attrs):
            self._str()
            self._str()
        n_chroms = self._i32()
        self.chrom_idx: Dict[str, int] = {}
        for i in range(n_chroms):
            name = self._str()
            _size = self._i64()
            self.chrom_idx[name] = i
            if name.startswith("chr"):
                self.chrom_idx[name[3:]] = i
            else:
                self.chrom_idx["chr" + name] = i
        n_bp_res = self._i32()
        self.bp_resolutions = [self._i32() for _ in range(n_bp_res)]
        n_frag_res = self._i32()
        _frag = [self._i32() for _ in range(n_frag_res)]

        self.f.seek(self.footer_pos)
        _n_bytes_v5 = self._i64()
        n_entries = self._i32()
        self.master_index: Dict[str, Tuple[int, int]] = {}
        for _ in range(n_entries):
            key = self._str()
            fpos = self._i64()
            size = self._i32()
            self.master_index[key] = (fpos, size)

    def read_intra_contacts(self, chrom_name: str, resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ci = self.chrom_idx.get(chrom_name)
        if ci is None:
            return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
        key = f"{ci}_{ci}"
        if key not in self.master_index:
            return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
        fpos, _ = self.master_index[key]
        self.f.seek(fpos)
        _c1, _c2, n_res = self._i32(), self._i32(), self._i32()
        block_meta = None
        for _ in range(n_res):
            unit = self._str()
            _res_idx = self._i32()
            for _ in range(4):
                self._f32()
            bin_size = self._i32()
            _bbc, _bcc = self._i32(), self._i32()
            n_blocks = self._i32()
            blocks = [(self._i32(), self._i64(), self._i32()) for _ in range(n_blocks)]
            if unit == "BP" and bin_size == resolution:
                block_meta = blocks
                break
        if block_meta is None:
            return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)

        b1_list, b2_list, c_list = [], [], []
        for _bid, bpos, bsize in block_meta:
            self.f.seek(bpos)
            raw = zlib.decompress(self.f.read(bsize))
            mv = memoryview(raw)
            pos = 0
            _n_rec = struct.unpack_from("<i", mv, pos)[0]
            pos += 4
            bin_x_offset, bin_y_offset = struct.unpack_from("<ii", mv, pos)
            pos += 8
            use_float, use_int_x, use_int_y, block_type = struct.unpack_from("<bbbb", mv, pos)
            pos += 4
            if block_type == 1:
                if use_int_y:
                    row_count = struct.unpack_from("<i", mv, pos)[0]
                    pos += 4
                else:
                    row_count = struct.unpack_from("<h", mv, pos)[0]
                    pos += 2
                for _ in range(row_count):
                    if use_int_y:
                        dy = struct.unpack_from("<i", mv, pos)[0]
                        pos += 4
                    else:
                        dy = struct.unpack_from("<h", mv, pos)[0]
                        pos += 2
                    bin_y = bin_y_offset + dy
                    if use_int_x:
                        col_count = struct.unpack_from("<i", mv, pos)[0]
                        pos += 4
                    else:
                        col_count = struct.unpack_from("<h", mv, pos)[0]
                        pos += 2
                    if col_count == 0:
                        continue
                    dx_dt = "<i4" if use_int_x else "<i2"
                    c_dt = "<f4" if use_float else "<i2"
                    dt = np.dtype([("dx", dx_dt), ("c", c_dt)])
                    rec = np.frombuffer(mv[pos:pos + col_count * dt.itemsize], dtype=dt)
                    pos += col_count * dt.itemsize
                    b1_list.append(bin_x_offset + rec["dx"].astype(np.int32))
                    b2_list.append(np.full(col_count, bin_y, dtype=np.int32))
                    c_list.append(rec["c"].astype(np.float32))
            elif block_type == 2:
                n_pts = struct.unpack_from("<i", mv, pos)[0]
                pos += 4
                w = struct.unpack_from("<h", mv, pos)[0]
                pos += 2
                if use_float:
                    vals = np.frombuffer(mv[pos:pos + n_pts * 4], dtype="<f4")
                    mask = ~np.isnan(vals) & (vals > 0)
                else:
                    vals = np.frombuffer(mv[pos:pos + n_pts * 2], dtype="<i2")
                    mask = vals != -32768
                idx = np.nonzero(mask)[0]
                if len(idx) > 0:
                    b1_list.append(bin_x_offset + (idx % w).astype(np.int32))
                    b2_list.append(bin_y_offset + (idx // w).astype(np.int32))
                    c_list.append(vals[idx].astype(np.float32))
        if not b1_list:
            return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
        return np.concatenate(b1_list), np.concatenate(b2_list), np.concatenate(c_list)


def load_genomic_intervals(tss_npz: str | Path, ccre_bed_gz: str | Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load sorted hg38 protein-coding TSS positions and ENCODE cCRE v3 midpoints."""
    z = np.load(tss_npz, allow_pickle=True)
    tss_by_chr: Dict[str, List[int]] = {}
    for ch, pos in zip(z["chrom"].astype(str), z["pos"].astype(np.int64)):
        c = ch if ch.startswith("chr") else "chr" + ch
        tss_by_chr.setdefault(c, []).append(int(pos))
    tss_map = {c: np.sort(np.unique(v)) for c, v in tss_by_chr.items()}

    ccre_by_chr: Dict[str, List[int]] = {}
    with gzip.open(ccre_bed_gz, "rt") as handle:
        for line in handle:
            parts = line.split("\t", 4)
            ccre_by_chr.setdefault(parts[0], []).append((int(parts[1]) + int(parts[2])) // 2)
    ccre_map = {c: np.sort(np.asarray(v, dtype=np.int64)) for c, v in ccre_by_chr.items()}
    return tss_map, ccre_map


def compute_raw_extra(raw_df: pd.DataFrame) -> np.ndarray:
    """Extract 5 label-free columns already present in EPCrisprBenchmark TSVs.

    Columns (in order):
      0: dhs_pct        DHS.percentile (within-cell-type quantile, ChIP-free)
      1: is_upstream    distanceToTSS < 0 (ChIP-free)
      2: log_width      log1p(chromEnd - chromStart) (ChIP-free)
      3: k27_exp        log1p(H3K27ac.RPM.expandedRegion) (ChIP mark)
      4: k27_exp_pct    H3K27ac.expandedRegion.percentile (ChIP mark)
    """
    dhs_pct = pd.to_numeric(raw_df["DHS.percentile"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    is_up = (raw_df["distanceToTSS"].to_numpy(np.float64) < 0).astype(np.float64)
    width = np.maximum(raw_df["chromEnd"].to_numpy(np.float64) - raw_df["chromStart"].to_numpy(np.float64), 1.0)
    k27_exp = np.log1p(np.clip(pd.to_numeric(raw_df["H3K27ac.RPM.expandedRegion"], errors="coerce").fillna(0.0).to_numpy(np.float64), 0.0, None))
    k27_exp_pct = pd.to_numeric(raw_df["H3K27ac.expandedRegion.percentile"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    return np.column_stack([dhs_pct, is_up, np.log1p(width), k27_exp, k27_exp_pct]).astype(np.float32)


def compute_genomic_ctx(
    raw_df: pd.DataFrame,
    tss_by_chr: Dict[str, np.ndarray],
    ccre_by_chr: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute 7 pure hg38 genomic-topology features (zero ChIP, zero labels, screen-independent).

    Columns (in order):
      0: log_int_tss       log1p(protein-coding TSSs strictly between CRE mid and target TSS)
      1: log_int_ccre      log1p(ENCODE cCRE v3 elements strictly between CRE mid and target TSS)
      2: log_ccre_5kb      log1p(cCREs within +/-5 kb of CRE mid)
      3: log_ccre_50kb     log1p(cCREs within +/-50 kb of CRE mid)
      4: log_tss_50kb      log1p(protein-coding TSSs within +/-50 kb of CRE mid)
      5: log_gw_ccre_sum   log(sum (|d_e'| + 5000)^(-0.87) over all cCREs within +/-500 kb of TSS)
      6: log_gw_ccre_share log(own_contact / (sum_contact + own_contact))
    """
    n = len(raw_df)
    chrom = raw_df["chrom"].astype(str).to_numpy()
    cre_mid = (raw_df["chromStart"].to_numpy(np.int64) + raw_df["chromEnd"].to_numpy(np.int64)) // 2
    tss_pos = raw_df["startTSS"].to_numpy(np.int64)
    dist = np.abs(raw_df["distanceToTSS"].to_numpy(np.float64))

    n_int_tss = np.zeros(n, dtype=np.float64)
    n_int_ccre = np.zeros(n, dtype=np.float64)
    n_ccre_5kb = np.zeros(n, dtype=np.float64)
    n_ccre_50kb = np.zeros(n, dtype=np.float64)
    n_tss_50kb = np.zeros(n, dtype=np.float64)
    sum_c = np.full(n, 1e-6, dtype=np.float64)

    for c in np.unique(chrom):
        m = chrom == c
        lo = np.minimum(cre_mid[m], tss_pos[m])
        hi = np.maximum(cre_mid[m], tss_pos[m])
        if c in tss_by_chr:
            arr = tss_by_chr[c]
            a = np.searchsorted(arr, lo + 500, side="left")
            b = np.searchsorted(arr, np.maximum(lo + 500, hi - 500), side="right")
            n_int_tss[m] = np.maximum(b - a, 0)
            a2 = np.searchsorted(arr, cre_mid[m] - 50_000, side="left")
            b2 = np.searchsorted(arr, cre_mid[m] + 50_000, side="right")
            n_tss_50kb[m] = np.maximum(b2 - a2, 0)
        if c in ccre_by_chr:
            arr = ccre_by_chr[c]
            a = np.searchsorted(arr, lo + 500, side="left")
            b = np.searchsorted(arr, np.maximum(lo + 500, hi - 500), side="right")
            n_int_ccre[m] = np.maximum(b - a, 0)
            n_ccre_5kb[m] = np.searchsorted(arr, cre_mid[m] + 5_000, side="right") - np.searchsorted(arr, cre_mid[m] - 5_000, side="left")
            n_ccre_50kb[m] = np.searchsorted(arr, cre_mid[m] + 50_000, side="right") - np.searchsorted(arr, cre_mid[m] - 50_000, side="left")
            # Genome-wide power-law cCRE competition per unique TSS
            uniq_t = np.unique(tss_pos[m])
            lo_t = np.searchsorted(arr, uniq_t - 500_000, side="left")
            hi_t = np.searchsorted(arr, uniq_t + 500_000, side="right")
            t_cache = {}
            for tv, ia, ib in zip(uniq_t, lo_t, hi_t):
                if ib > ia:
                    dd = np.abs(arr[ia:ib] - tv).astype(np.float64)
                    t_cache[int(tv)] = float(np.sum((dd + 5000.0) ** (-0.87)))
                else:
                    t_cache[int(tv)] = 1e-6
            sum_c[m] = np.array([t_cache[int(tv)] for tv in tss_pos[m]], dtype=np.float64)

    own_c = (dist + 5000.0) ** (-0.87)
    share = own_c / (sum_c + own_c)
    return np.column_stack([
        np.log1p(n_int_tss),
        np.log1p(n_int_ccre),
        np.log1p(n_ccre_5kb),
        np.log1p(n_ccre_50kb),
        np.log1p(n_tss_50kb),
        np.log(np.maximum(sum_c, 1e-9)),
        np.log(np.clip(share, 1e-6, 1.0)),
    ]).astype(np.float32)


def extract_microc_contacts(raw_df: pd.DataFrame, hic_path: str | Path) -> np.ndarray:
    """Extract 4 VC_SQRT-balanced 3D contact features (3x3 smoothed) from a .hic v9 file.

    Columns (in order):
      0: mc_5kb       log1p(balanced contact at 5 kb)
      1: mc_10kb      log1p(balanced contact at 10 kb)
      2: mc_10kb_oe   log1p(observed / distance-expected at 10 kb)
      3: mc_25kb_oe   log1p(observed / distance-expected at 25 kb)
    """
    n = len(raw_df)
    out = np.zeros((n, 4), dtype=np.float64)
    chrom = raw_df["chrom"].astype(str).to_numpy()
    cre_mid = (raw_df["chromStart"].to_numpy(np.int64) + raw_df["chromEnd"].to_numpy(np.int64)) // 2
    tss_pos = raw_df["startTSS"].to_numpy(np.int64)

    with HicV9Reader(hic_path) as hr:
        for ch in sorted(np.unique(chrom)):
            m = chrom == ch
            for res, col_raw, col_oe in [(5000, 0, None), (10000, 1, 2), (25000, None, 3)]:
                b1, b2, c = hr.read_intra_contacts(ch, res)
                if len(c) == 0:
                    continue
                lo = np.minimum(b1, b2)
                hi = np.maximum(b1, b2)
                max_bin = int(hi.max()) + 2
                row_sum = np.zeros(max_bin, dtype=np.float64)
                np.add.at(row_sum, lo, c)
                np.add.at(row_sum, hi[lo != hi], c[lo != hi])
                med_rs = float(np.median(row_sum[row_sum > 0]))
                norm_vec = np.where(row_sum > 0, np.sqrt(row_sum / med_rs), 1.0)
                c_bal = c / (norm_vec[lo] * norm_vec[hi])

                max_d_bin = int(5_000_000 // res) + 2
                d_bin = hi - lo
                in_rng = d_bin < max_d_bin
                exp_sum = np.bincount(d_bin[in_rng], weights=c_bal[in_rng], minlength=max_d_bin)
                n_poss = np.maximum(max_bin - np.arange(max_d_bin), 1)
                exp_mean = exp_sum / n_poss
                floor = float(np.percentile(exp_mean[exp_mean > 0], 5)) if (exp_mean > 0).any() else 1e-4
                exp_mean = np.maximum(exp_mean, floor)

                keys = (lo.astype(np.int64) << 32) | hi.astype(np.int64)
                order = np.argsort(keys)
                keys_s = keys[order]
                c_bal_s = c_bal[order]

                qb1 = cre_mid[m] // res
                qb2 = tss_pos[m] // res
                acc = np.zeros(len(qb1), dtype=np.float64)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        x = qb1 + dx
                        y = qb2 + dy
                        ql = np.minimum(x, y)
                        qh = np.maximum(x, y)
                        qk = (ql.astype(np.int64) << 32) | qh.astype(np.int64)
                        pos = np.searchsorted(keys_s, qk)
                        valid = (pos < len(keys_s)) & (ql >= 0)
                        hit = np.zeros(len(qk), dtype=bool)
                        hit[valid] = keys_s[pos[valid]] == qk[valid]
                        w = 2.0 if (dx == 0 and dy == 0) else 1.0
                        acc[hit] += w * c_bal_s[pos[hit]]
                val = acc / 10.0
                if col_raw is not None:
                    out[m, col_raw] = np.log1p(val)
                if col_oe is not None:
                    qd = np.clip(np.abs(qb2 - qb1), 0, max_d_bin - 1)
                    out[m, col_oe] = np.log1p(val / exp_mean[qd])
    return out.astype(np.float32)
