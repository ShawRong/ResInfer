'Implementation details.'

from decimal import Decimal
from functools import lru_cache
import math, re

FLOATS = {
    "atcoder:abc324_f": "1e-9",
    "atcoder:abc327_e": "1e-6",
    "atcoder:abc315_f": "1e-5",
    "atcoder:abc385_f": "1e-9",
    "atcoder:abc314_e": "1e-5",
    "atcoder:abc350_e": "1e-6",
}
CONSTRUCT = {
    "atcoder:abc366_g",
    "atcoder:abc398_f",
    "atcoder:abc396_e",
    "atcoder:abc315_e",
    "atcoder:arc191_c",
    "atcoder:abc333_e",
    "atcoder:arc190_a",
    "atcoder:abc343_e",
    "atcoder:abc397_d",
    "atcoder:abc326_d",
    "atcoder:arc195_c",
    "atcoder:abc373_g",
    "atcoder:arc183_d",
    "atcoder:arc181_c",
    "atcoder:abc363_f",
    "atcoder:arc185_c",
    "atcoder:arc188_c",
}
TASKS = set(FLOATS) | CONSTRUCT


def ints(s):
    return list(map(int, s.split()))


def need(ok, msg="invalid output"):
    if not ok:
        raise ValueError(msg)


@lru_cache(20000)
def factors(n):
    out = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            while n % d == 0:
                n //= d
        d = 3 if d == 2 else d + 2
    if n > 1:
        out.append(n)
    return out


def parity_consistent(n, edges):
    g = [[] for _ in range(n)]
    for a, b, c in edges:
        g[a].append((b, c))
        g[b].append((a, c))
    val = [None] * n
    for start in range(n):
        if val[start] is not None:
            continue
        val[start] = 0
        st = [start]
        while st:
            a = st.pop()
            for b, c in g[a]:
                v = val[a] ^ c
                if val[b] is None:
                    val[b] = v
                    st.append(b)
                elif val[b] != v:
                    return False
    return True


def coverage_optimum(n, intervals):
    if any(l == 1 and r == n for l, r in intervals):
        return 1
    if len(intervals) < 2:
        return -1
    # Two outside operations cover all iff their intervals are disjoint.
    if min(r for l, r in intervals) < max(l for l, r in intervals):
        return 2
    # An inside A plus outside B covers all iff B is a subset of A.
    hi = 0
    for l, r in sorted(intervals, key=lambda z: (z[0], -z[1])):
        if hi >= r:
            return 2
        hi = max(hi, r)
    left = max((r for l, r in intervals if l == 1), default=0)
    right = min((l for l, r in intervals if r == n), default=n + 2)
    if left and right <= left + 1:
        return 2
    # With no containment/disjoint pair, take outside of the two extreme
    # intervals and inside any third; their common intersection is covered.
    return 3 if len(intervals) >= 3 else -1


def matching_tree_score(inp, out):
    """Validate leaf removals by alternating paths in current unique matching.
    HLD plus lazy bit-flip segment tree avoids quadratic path traversal.
    """
    d = ints(inp)
    n = d[0]
    edges = list(zip(d[1::2], d[2::2]))
    o = ints(out)
    need(len(o) == n and set(o) == set(range(1, n + 1)))
    g = [[] for _ in range(n)]
    initial = set()
    for j, (a, b) in enumerate(edges):
        a -= 1
        b -= 1
        g[a].append(b)
        g[b].append(a)
        if j < n // 2:
            initial.add((min(a, b), max(a, b)))
    parent = [-1] * n
    depth = [0] * n
    order = [0]
    for a in order:
        for b in g[a]:
            if b != parent[a]:
                parent[b] = a
                depth[b] = depth[a] + 1
                order.append(b)
    size = [1] * n
    heavy = [-1] * n
    for a in reversed(order):
        for b in g[a]:
            if parent[b] == a:
                size[a] += size[b]
                if heavy[a] < 0 or size[b] > size[heavy[a]]:
                    heavy[a] = b
    head = [0] * n
    pos = [0] * n
    values = [0] * n
    idx = 0
    stack = [(0, 0)]
    while stack:
        a, h = stack.pop()
        while a >= 0:
            head[a] = h
            pos[a] = idx
            values[idx] = int((min(a, parent[a]), max(a, parent[a])) in initial)
            idx += 1
            for b in g[a]:
                if parent[b] == a and b != heavy[a]:
                    stack.append((b, b))
            a = heavy[a]
    tree = [0] * (4 * n)
    lazy = bytearray(4 * n)

    def build(k, l, r):
        if r - l == 1:
            tree[k] = values[l]
            return
        m = (l + r) // 2
        build(k * 2, l, m)
        build(k * 2 + 1, m, r)
        tree[k] = tree[k * 2] + tree[k * 2 + 1]

    build(1, 0, n)

    def op(k, l, r, ql, qr, flip):
        if qr <= l or r <= ql:
            return 0
        if ql <= l and r <= qr:
            old = tree[k]
            if flip:
                tree[k] = r - l - tree[k]
                lazy[k] ^= 1
            return old
        m = (l + r) // 2
        if lazy[k]:
            for z, length in [(k * 2, m - l), (k * 2 + 1, r - m)]:
                tree[z] = length - tree[z]
                lazy[z] ^= 1
            lazy[k] = 0
        v = op(k * 2, l, m, ql, qr, flip) + op(k * 2 + 1, m, r, ql, qr, flip)
        tree[k] = tree[k * 2] + tree[k * 2 + 1]
        return v

    degree = list(map(len, g))
    alive = bytearray([1]) * n
    score = 0
    for a, b in zip(o[::2], o[1::2]):
        a -= 1
        b -= 1
        need(
            alive[a] and alive[b] and degree[a] == degree[b] == 1, "not current leaves"
        )
        u, v = a, b
        segments = []
        distance = 0
        while head[u] != head[v]:
            if depth[head[u]] < depth[head[v]]:
                u, v = v, u
            lo, hi = pos[head[u]], pos[u] + 1
            segments.append((lo, hi))
            distance += hi - lo
            u = parent[head[u]]
        if depth[u] > depth[v]:
            u, v = v, u
        lo, hi = pos[u] + 1, pos[v] + 1
        if lo < hi:
            segments.append((lo, hi))
            distance += hi - lo
        used = sum(op(1, 0, n, l, r, False) for l, r in segments)
        need(
            distance % 2 == 1 and used == (distance + 1) // 2,
            "remaining tree has no perfect matching",
        )
        for l, r in segments:
            op(1, 0, n, l, r, True)
        score += distance
        alive[a] = alive[b] = 0
        for x in [a, b]:
            for y in g[x]:
                if alive[y]:
                    degree[y] -= 1
            degree[x] = 0
    return score


def check(key, inp, pred, ref):
    try:
        _check(key, inp, pred, ref)
        return True, "valid"
    except (ValueError, IndexError, StopIteration, ArithmeticError) as e:
        return False, str(e)


def _check(key, inp, pred, ref):
    if key in FLOATS:
        a, b = pred.split(), ref.split()
        need(len(a) == len(b) == 1)
        x, y = Decimal(a[0]), Decimal(b[0])
        need(x.is_finite() and y.is_finite())
        tol = Decimal(FLOATS[key])
        need(abs(x - y) <= tol * max(Decimal(1), abs(y)), "outside task tolerance")
        return
    p = pred.split()
    r = ref.split()
    need(bool(p), "empty")
    if key == "atcoder:abc398_f":
        s = inp.strip()
        need(
            len(p) == 1
            and p[0].startswith(s)
            and p[0] == p[0][::-1]
            and len(p[0]) == len(r[0])
        )
        return
    if key == "atcoder:abc326_d":
        n, first_rows, first_cols = inp.split()
        n = int(n)
        if p == ["No"]:
            need(r == ["No"], "reference existence")
            return
        need(p[0] == "Yes" and len(p) == n + 1)
        grid = p[1:]
        need(all(len(s) == n and set(s) <= set(".ABC") for s in grid))
        cols = ["".join(grid[j][k] for j in range(n)) for k in range(n)]
        for lines, first in [(grid, first_rows), (cols, first_cols)]:
            for line, expected in zip(lines, first):
                line = line.replace(".", "")
                need(sorted(line) == list("ABC") and line[0] == expected)
        return
    d = ints(inp)
    if key == "atcoder:abc315_e":
        n = d[0]
        g = [[] for _ in range(n)]
        k = 1
        for a in range(n):
            c = d[k]
            g[a] = [x - 1 for x in d[k + 1 : k + 1 + c]]
            k += c + 1
        needed = set()
        stack = [0]
        while stack:
            for b in g[stack.pop()]:
                if b not in needed:
                    needed.add(b)
                    stack.append(b)
        o = [x - 1 for x in ints(pred)]
        need(len(o) == len(needed) and set(o) == needed)
        done = set()
        for a in o:
            need(all(b in done for b in g[a]), "prerequisite order")
            done.add(a)
    elif key == "atcoder:abc363_f":
        need(len(p) == 1)
        s = p[0]
        if s == "-1":
            need(r == ["-1"])
            return
        need(
            1 <= len(s) <= 1000
            and re.fullmatch("[1-9]+(?:\\*[1-9]+)*", s)
            and s == s[::-1]
        )
        need(math.prod(map(int, s.split("*"))) == d[0], "wrong product")
    elif key == "atcoder:arc183_d":
        need(
            matching_tree_score(inp, pred) == matching_tree_score(inp, ref),
            "not reference optimum",
        )
    elif key == "atcoder:arc185_c":
        if p == ["-1"]:
            need(r == ["-1"])
            return
        o = ints(pred)
        n, x = d[:2]
        need(len(o) == 3 and 1 <= o[0] < o[1] < o[2] <= n)
        need(sum(d[1 + j] for j in o) == x)
    elif key == "atcoder:abc397_d":
        n = d[0]
        if p == ["-1"]:
            t = 1
            while t * t * t < n:
                if n % t == 0:
                    z = n // t - t * t
                    if z > 0 and z % 3 == 0:
                        disc = t * t + 4 * (z // 3)
                        sq = math.isqrt(disc)
                        need(
                            not (sq * sq == disc and sq > t and (sq - t) % 2 == 0),
                            "positive solution exists",
                        )
                t += 1
        else:
            o = ints(pred)
            need(
                len(o) == 2 and min(o) > 0 and o[0] ** 3 - o[1] ** 3 == n,
                "invalid positive cubes",
            )
    elif key == "atcoder:abc396_e":
        n, m = d[:2]
        edges = [(d[k] - 1, d[k + 1] - 1, d[k + 2]) for k in range(2, len(d), 3)]
        g = [[] for _ in range(n)]
        for a, b, z in edges:
            g[a].append((b, z))
            g[b].append((a, z))
        val = [None] * n
        optimal = 0
        possible = True
        for a in range(n):
            if val[a] is not None:
                continue
            val[a] = 0
            comp = [a]
            for u in comp:
                for v, z in g[u]:
                    w = val[u] ^ z
                    if val[v] is None:
                        val[v] = w
                        comp.append(v)
                    elif val[v] != w:
                        possible = False
            for bit in range(30):
                c = sum((val[v] >> bit) & 1 for v in comp)
                optimal += min(c, len(comp) - c) * (1 << bit)
        if p == ["-1"]:
            need(not possible)
            return
        o = ints(pred)
        need(possible and len(o) == n and min(o) >= 0 and sum(o) == optimal)
        need(all(o[a] ^ o[b] == z for a, b, z in edges))
    elif key == "atcoder:arc191_c":
        o = ints(pred)
        need(len(o) == 2 * d[0])
        for n, a, m in zip(d[1:], o[::2], o[1::2]):
            need(1 <= a <= 10**18 and 1 <= m <= 10**18)
            need(
                pow(a, n, m) == 1 % m
                and all(pow(a, n // q, m) != 1 % m for q in factors(n)),
                "wrong multiplicative order",
            )
    elif key == "atcoder:arc195_c":
        # A No answer uses the reference existence decision; Yes is checked geometrically.
        it = iter(p)
        rt = iter(r)
        for red, blue in zip(d[1::2], d[2::2]):
            status = next(it)
            rs = next(rt)
            cnt = red + blue
            if rs == "Yes":
                for _ in range(3 * cnt):
                    next(rt)
            if status == "No":
                need(rs == "No")
                continue
            need(status == "Yes")
            pts = [(next(it), int(next(it)), int(next(it))) for _ in range(cnt)]
            need(
                sum(t == "R" for t, x, y in pts) == red
                and sum(t == "B" for t, x, y in pts) == blue
            )
            need(
                len({(x, y) for t, x, y in pts}) == cnt
                and all(1 <= x <= 10**9 and 1 <= y <= 10**9 for t, x, y in pts)
            )
            for (t, x, y), (_, xx, yy) in zip(pts, pts[1:] + pts[:1]):
                dx, dy = abs(x - xx), abs(y - yy)
                need(dx + dy == 1 if t == "R" else dx == dy == 1, "illegal move")
        need(next(it, None) is None)
    elif key == "atcoder:abc333_e":
        n = d[0]
        events = list(zip(d[1::2], d[2::2]))
        o = ints(pred)
        required = {}
        chosen = [0] * n
        for j in range(n - 1, -1, -1):
            t, x = events[j]
            if t == 2:
                required[x] = required.get(x, 0) + 1
            elif required.get(x, 0) > 0:
                chosen[j] = 1
                required[x] -= 1
        possible = not any(required.values())
        cur = 0
        optimum = 0
        for j, (t, x) in enumerate(events):
            cur += chosen[j] if t == 1 else -1
            optimum = max(optimum, cur)
        if o == [-1]:
            need(not possible)
            return
        need(possible and len(o) == 1 + sum(t == 1 for t, x in events))
        need(o[0] == optimum, "not optimal")
        bag = {}
        num = 0
        peak = 0
        k = 1
        for t, x in events:
            if t == 1:
                v = o[k]
                k += 1
                need(v in [0, 1])
                bag[x] = bag.get(x, 0) + v
                num += v
                peak = max(peak, num)
            else:
                need(bag.get(x, 0) > 0, "missing potion")
                bag[x] -= 1
                num -= 1
        need(peak == o[0])
    elif key == "atcoder:abc366_g":
        if p == ["No"]:
            need(r == ["No"])
            return
        n, m = d[:2]
        need(p[0] == "Yes" and len(p) == n + 1)
        o = list(map(int, p[1:]))
        need(all(1 <= v < 2**60 for v in o))
        xs = [0] * n
        for a, b in zip(d[2::2], d[3::2]):
            xs[a - 1] ^= o[b - 1]
            xs[b - 1] ^= o[a - 1]
        need(not any(xs), "neighbor xor nonzero")
    elif key == "atcoder:abc373_g":
        if p == ["-1"]:
            need(r == ["-1"])
            return
        n = d[0]
        o = ints(pred)
        need(len(o) == n and set(o) == set(range(1, n + 1)))
        pts = list(zip(d[1::2], d[2::2]))
        segs = [(pts[j], pts[n + o[j] - 1]) for j in range(n)]

        def orient(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        for j, (a, b) in enumerate(segs):
            for c, e in segs[:j]:
                need(
                    not (
                        orient(a, b, c) * orient(a, b, e) < 0
                        and orient(c, e, a) * orient(c, e, b) < 0
                    ),
                    "crossing segments",
                )
    elif key == "atcoder:arc181_c":
        n = d[0]
        need(len(p) == n and all(len(z) == n and set(z) <= set("01") for z in p))
        cols = ["".join(p[j][k] for j in range(n)) for k in range(n)]
        rows = [p[j - 1] for j in d[1 : 1 + n]]
        cs = [cols[j - 1] for j in d[1 + n :]]
        need(
            all(a < b for a, b in zip(rows, rows[1:]))
            and all(a < b for a, b in zip(cs, cs[1:]))
        )
    elif key == "atcoder:arc188_c":
        n, m = d[:2]
        e = [(d[k] - 1, d[k + 1] - 1, d[k + 2]) for k in range(2, len(d), 3)]
        if p == ["-1"]:
            need(not parity_consistent(2 * n, [(a + n, b, c) for a, b, c in e]))
            return
        need(len(p) == 1 and len(p[0]) == n and set(p[0]) <= set("01"))
        s = list(map(int, p[0]))
        need(
            parity_consistent(n, [(a, b, c ^ s[a]) for a, b, c in e]),
            "inconsistent testimonies",
        )
    elif key == "atcoder:arc190_a":
        n, m = d[:2]
        o = ints(pred)
        optimal = coverage_optimum(n, list(zip(d[2::2], d[3::2])))
        if o == [-1]:
            need(optimal == -1)
            return
        need(
            len(o) == m + 1 and o[0] == optimal and o[0] == sum(v != 0 for v in o[1:]),
            "not independent optimum",
        )
        segments = []
        for l, h, v in zip(d[2::2], d[3::2], o[1:]):
            need(v in [0, 1, 2])
            if v == 1:
                segments.append((l, h))
            if v == 2:
                if l > 1:
                    segments.append((1, l - 1))
                if h < n:
                    segments.append((h + 1, n))
        covered = 0
        for l, h in sorted(segments):
            need(l <= covered + 1, "uncovered interval")
            covered = max(covered, h)
        need(covered == n)
    elif key == "atcoder:abc343_e":
        if p == ["No"]:
            need(r == ["No"])
            return
        need(p[0] == "Yes" and len(p) == 10)
        o = list(map(int, p[1:]))
        need(all(abs(x) <= 100 for x in o))
        pts = [o[k : k + 3] for k in [0, 3, 6]]

        def volume(ps):
            return math.prod(
                max(0, min(z[k] + 7 for z in ps) - max(z[k] for z in ps))
                for k in range(3)
            )

        triple = volume(pts)
        pair = sum(volume([pts[a], pts[b]]) for a, b in [(0, 1), (0, 2), (1, 2)])
        need(
            [1029 - 2 * pair + 3 * triple, pair - 3 * triple, triple] == d,
            "incorrect volumes",
        )
    else:
        raise ValueError("unsupported task")
