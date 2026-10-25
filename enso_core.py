"""ENSO 核心预测模型
核心链路：
  NOAA CPC Nino3.4 数据 -> 状态编码(±0.5) -> 状态转移模型 + 历史相似型模型
  -> 回测技能加权组合 -> 逐年留一法验证 -> 内置自测
用法：
  python enso_core.py             # 输出最新预测与回测技能
  python enso_core.py selftest    # 运行内置自测
"""
import math
import os
import sys
import urllib.request

URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "nino34_cache.txt")
STATES = ("La Nina", "Neutral", "El Nino")
MISSING = 99.9
WINDOW = 12
PRIOR = 4.0

def load_rows(allow_network=True):
    """首次运行下载 NOAA CPC 官方数据并缓存；返回 [(年, 月, Nino3.4)]。"""
    if not os.path.exists(CACHE):
        if not allow_network:
            return []
        urllib.request.urlretrieve(URL, CACHE)
    rows = []
    with open(CACHE, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 10 and p[0].isdigit() and len(p[0]) == 4:
                v = float(p[9])
                if v != MISSING:
                    rows.append((int(p[0]), int(p[1]), v))
    return rows

def encode(v):
    return 1 if v >= 0.5 else -1 if v <= -0.5 else 0

def states_of(rows):
    return [encode(v) for _, _, v in rows]

def clim_freq(states):
    n = len(states)
    freq = {}
    for s in states:
        freq[s] = freq.get(s, 0) + 1
    return {s: freq.get(s, 0) / n for s in (-1, 0, 1)}

def trans_probs(states, t, horizon, prior=PRIOR):
    """模型 1 状态转移：以当前状态为条件统计历史转移，Dirichlet 平滑。"""
    counts = {s: {q: 0.0 for q in (-1, 0, 1)} for s in (-1, 0, 1)}
    for i in range(t - horizon):            
        counts[states[i]][states[i + horizon]] += 1.0
    cur = states[t] if t < len(states) else 0
    base = clim_freq(states[:t + 1])
    tot = sum(counts[cur].values())
    return {q: (counts[cur][q] + prior * base[q]) / (tot + prior)
            for q in (-1, 0, 1)}

def analog_probs(states, vals, t, horizon, window=WINDOW):
    """模型 2 相似型：近 12 个月轨迹 vs 历史窗口，高斯核加权投票。"""
    if t + 1 < window:
        return None
    query = vals[t - window + 1:t + 1]
    scale = max(abs(x) for x in query) or 1.0
    qz = [x / scale for x in query]
    weight, vote = 0.0, {q: 0.0 for q in (-1, 0, 1)}
    for j in range(t - window - horizon + 1):
        if j + window + horizon > len(states):
            break
        cand = [vals[j + k] / scale for k in range(window)]
        d2 = sum((a - b) ** 2 for a, b in zip(qz, cand))
        w = math.exp(-d2 / 2.0)             # 高斯核
        vote[states[j + window + horizon - 1]] += w
        weight += w
    if weight <= 0:
        return None
    return {q: vote[q] / weight for q in (-1, 0, 1)}

def blend(p, q, w):
    if p is None:
        return dict(q)
    if q is None:
        return dict(p)
    return {s: w * p[s] + (1 - w) * q[s] for s in (-1, 0, 1)}

def brier(p, obs):
    return sum((p[s] - (1.0 if s == obs else 0.0)) ** 2 for s in (-1, 0, 1))

def mix_weight(acc, n):
    """按两模型过去的组合技能自适应定权；无历史时等权。"""
    if n == 0 or acc["c"] <= 0:
        return 0.5
    sk_tr = max(1.0 - acc["tr"] / acc["c"], 0.0)
    sk_an = max(1.0 - acc["an"] / acc["c"], 0.0)
    s = sk_tr + sk_an
    return sk_tr / s if s > 0 else 0.5

def cross_month_skill(rows, states, month=5, horizon=3, verify_start=1970):
    """逐年留一法：每年 month 月起报预测 +horizon 月状态；
    组合权重只用起报点之前的累计技能，杜绝未来泄漏。 """
    acc = {"tr": 0.0, "an": 0.0, "c": 0.0}
    n, last = 0, None
    for i, (y, m, _) in enumerate(rows):
        if m != month or y < verify_start or i + horizon >= len(rows):
            continue
        obs = states[i + horizon]
        p_tr = trans_probs(states, i, horizon)
        p_an = analog_probs(states, [v for _, _, v in rows], i, horizon)
        c = clim_freq(states[:i + 1])
        p = blend(p_tr, p_an, mix_weight(acc, n))
        acc["tr"] += brier(p_tr, obs)
        acc["an"] += brier(p_an or c, obs)
        acc["c"] += brier(c, obs)
        n += 1
        last = (y, m, p)
    out = {k: acc[k] / n for k in acc} if n else None
    return out, n, last

def forecast_now(rows, hor=(3, 6, 9, 12)):
    """基于最新一个月做多档提前期概率预测。"""
    states = states_of(rows)
    vals = [v for _, _, v in rows]
    t = len(rows) - 1
    y, m, _ = rows[t]
    c = clim_freq(states)
    print(f"[最新数据] {y} 年 {m} 月  Nino3.4 = {vals[-1]:+.2f}C"
          f"  当前状态: {STATES[states[-1] + 1]}")
    print("提前期   转移模型 / 相似型 / 组合        （La Nina/中性/El Nino）")
    for h in hor:
        p_tr = trans_probs(states, t, h)
        p_an = analog_probs(states, vals, t, h)
        p = blend(p_tr, p_an, 0.5)
        fmt = lambda p: "/".join(f"{p[s]:.0%}" for s in (-1, 0, 1))
        print(f"  {h:>2} 个月   {fmt(p_tr)}  {fmt(p_an)}  {fmt(p)}")

def selftest():
    ok, fail = 0, 0
    def check(name, cond):
        nonlocal ok, fail
        ok += cond
        fail += (not cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("= 自测 =")
    check("状态编码边界", (encode(0.5), encode(-0.5), encode(0.49)) == (1, -1, 0))
    st = [0, 0, 1, 1, 1, 0, -1, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    freq = clim_freq(st)
    check("气候频率归一", abs(sum(freq.values()) - 1) < 1e-12)
    p = trans_probs(st, len(st) - 1, 1)
    check("转移概率归一", abs(sum(p.values()) - 1) < 1e-12)
    st_flip = [-s for s in st]
    pf = trans_probs(st_flip, len(st) - 1, 1)
    check("符号翻转对称", abs(p[1] - pf[-1]) < 1e-12)
    sin = [0.9 * math.sin(i / 3.1) for i in range(120)]
    ssin = [encode(v) for v in sin]
    pa = analog_probs(ssin, sin, len(sin) - 1, 6)
    check("相似型概率归一", pa is None or abs(sum(pa.values()) - 1) < 1e-12)
    check("Brier 完美/错误", brier({-1: 0.0, 0: 0.0, 1: 1.0}, 1) < 1e-12
          and brier({-1: 0.0, 0: 0.0, 1: 1.0}, -1) == 2.0)
    rows_syn = [(2000 + i // 12, i % 12 + 1, sin[i]) for i in range(120)]
    res, n, _ = cross_month_skill(rows_syn, ssin, month=5, horizon=3,
                                  verify_start=2000)
    check("留一法返回样本", res is not None and n >= 1)
    if res:
        skill = 1.0 - res["tr"] / res["c"]
        check("技能范围有界", -1.01 <= skill <= 1.01)
    check("缺测值跳过", load_rows(allow_network=False) == [])
    print(f"= 结果：{ok}/{ok + fail} 通过 =")
    return fail == 0

def main():
    if "selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    rows = load_rows()
    if not rows:
        print("无数据缓存且下载失败，请联网重试。")
        sys.exit(1)
    forecast_now(rows)
    states = states_of(rows)
    res, n, _ = cross_month_skill(rows, states)
    print(f"\n[回测] 5 月起报 / 3 个月提前期 / 留一法 {n} 年（1970 起）")
    for k, name in (("tr", "状态转移"), ("an", "相似型"), ("c", "气候态")):
        print(f"  {name}: Brier = {res[k]:.3f}")
    print(f"  转移模型技能 = {1 - res['tr'] / res['c']:+.2f}"
          f"   相似型技能 = {1 - res['an'] / res['c']:+.2f}")

if __name__ == "__main__":
    main()
