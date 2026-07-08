from std.memory import alloc, UnsafePointer
from std.math import round
from std.collections import InlineArray
from hope import COLOR_DIM

# ==========================================
# Grid substrate for content-addressed reads (Rung CF)
# ==========================================
# Per-grid REPRESENTATIONS a written content-fetch layer reads through learned
# per-key actions: connected components (with size/bbox/colour), global content
# registers (largest/smallest/unique/majority nonbg colour) and their bbox
# anchors, nearest-nonbg colour+distance (multi-source BFS), and first-nonbg
# ray casts. This is SUBSTRATE in the factor_scan sense — representations a
# learned read operates over, not hand-coded transforms; the learned part
# (which view, which action per relational key) lives in the memory
# (ContentFetchComposedMemory). Border/wrap conventions mirror the Python scan
# exactly (rays clamped at the border, anchor displacement toroidal) — the
# 22/146 coverage evidence was measured under precisely these conventions.
#
# The struct is computed once per grid buffer (never inside an ES hot loop:
# the content layer is written closed-form and applied only at scoring time),
# so per-construction allocation is fine. It does NOT own the grid buffer it
# reads (`q`); __del__ frees only the derived arrays.

comptime SUB_DIST_CAP = 5  # nearest-nonbg BFS cap (matches the scan)
comptime SUB_N_VIEWS = 15  # fetch views, enumerated below
comptime SUB_REL_K = 8  # rel-bucket range per view (key = is_bg*8 + rel)

# View ids (write and apply must agree):
#   0..3   ray up / down / left / right (first nonbg strictly along the ray)
#   4      nearest nonbg (8-connected BFS colour; rel = capped distance)
#   5..8   registers: largest / smallest / unique / majority nonbg colour
#   9..12  anchor displacement: largest+ / largest- / unique+ / unique-
#          (toroidal read at (r ± r0, c ± c0), bbox corner of the register comp)
#   13..14 object-local bbox mirror: horizontal / vertical


struct GridSubstrate(Movable):
    var q: UnsafePointer[Float32, MutAnyOrigin]  # NOT owned
    var rows: Int
    var cols: Int
    var bg: Int
    var comp: UnsafePointer[Int, MutAnyOrigin]  # component id per cell
    var comp_col: UnsafePointer[Int, MutAnyOrigin]  # per component
    var comp_size: UnsafePointer[Int, MutAnyOrigin]
    var bbox: UnsafePointer[Int, MutAnyOrigin]  # r0,r1,c0,c1 per component
    var n_comp: Int
    var near_col: UnsafePointer[Int, MutAnyOrigin]  # -1 = unreached
    var near_d: UnsafePointer[Int, MutAnyOrigin]  # capped at SUB_DIST_CAP
    var ray: UnsafePointer[Int, MutAnyOrigin]  # 4 planes u,d,l,r; -1 = none
    var large_col: Int  # register colours; -1 = absent
    var small_col: Int
    var uniq_col: Int
    var major_col: Int
    var anc_large_r: Int  # bbox top-left anchors; -1 = absent
    var anc_large_c: Int
    var anc_uniq_r: Int
    var anc_uniq_c: Int

    def __init__(
        out self,
        q: UnsafePointer[Float32, MutAnyOrigin],
        rows: Int,
        cols: Int,
    ):
        var n = rows * cols
        self.q = q
        self.rows = rows
        self.cols = cols
        self.comp = alloc[Int](n)
        self.comp_col = alloc[Int](n)
        self.comp_size = alloc[Int](n)
        self.bbox = alloc[Int](4 * n)
        self.near_col = alloc[Int](n)
        self.near_d = alloc[Int](n)
        self.ray = alloc[Int](4 * n)

        # ---- plurality background colour ----
        var hist = alloc[Int](COLOR_DIM)
        for s in range(COLOR_DIM):
            hist[s] = 0
        for k in range(n):
            var v = Int(round(q[k]))
            if v >= 0 and v < COLOR_DIM:
                hist[v] += 1
        self.bg = 0
        for s in range(COLOR_DIM):
            if hist[s] > hist[self.bg]:
                self.bg = s

        # ---- 4-connected same-colour components over ALL cells ----
        for k in range(n):
            self.comp[k] = -1
        var stack = alloc[Int](n)
        self.n_comp = 0
        for k0 in range(n):
            if self.comp[k0] != -1:
                continue
            var cid = self.n_comp
            self.n_comp += 1
            var col = Int(round(q[k0]))
            self.comp_col[cid] = col
            self.comp_size[cid] = 0
            self.bbox[4 * cid + 0] = k0 // cols
            self.bbox[4 * cid + 1] = k0 // cols
            self.bbox[4 * cid + 2] = k0 % cols
            self.bbox[4 * cid + 3] = k0 % cols
            var top = 0
            stack[top] = k0
            top += 1
            self.comp[k0] = cid
            while top > 0:
                top -= 1
                var k = stack[top]
                var r = k // cols
                var c = k % cols
                self.comp_size[cid] += 1
                if r < self.bbox[4 * cid + 0]:
                    self.bbox[4 * cid + 0] = r
                if r > self.bbox[4 * cid + 1]:
                    self.bbox[4 * cid + 1] = r
                if c < self.bbox[4 * cid + 2]:
                    self.bbox[4 * cid + 2] = c
                if c > self.bbox[4 * cid + 3]:
                    self.bbox[4 * cid + 3] = c
                for d in range(4):
                    var rr = r
                    var cc = c
                    if d == 0:
                        rr -= 1
                    elif d == 1:
                        rr += 1
                    elif d == 2:
                        cc -= 1
                    else:
                        cc += 1
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                        continue
                    var kk = rr * cols + cc
                    if self.comp[kk] == -1 and Int(round(q[kk])) == col:
                        self.comp[kk] = cid
                        stack[top] = kk
                        top += 1

        # ---- registers + anchors from the nonbg components ----
        var large_i = -1
        var small_i = -1
        # per-colour count of nonbg components (for the unique register)
        var col_comp_n = alloc[Int](COLOR_DIM)
        for s in range(COLOR_DIM):
            col_comp_n[s] = 0
        for i in range(self.n_comp):
            if self.comp_col[i] == self.bg:
                continue
            var cc2 = self.comp_col[i]
            if cc2 >= 0 and cc2 < COLOR_DIM:
                col_comp_n[cc2] += 1
            if large_i < 0 or self.comp_size[i] > self.comp_size[large_i]:
                large_i = i
            if small_i < 0 or self.comp_size[i] < self.comp_size[small_i]:
                small_i = i
        self.large_col = self.comp_col[large_i] if large_i >= 0 else -1
        self.small_col = self.comp_col[small_i] if small_i >= 0 else -1
        # unique register: exactly one colour whose nonbg component count is 1
        self.uniq_col = -1
        var n_uniq = 0
        for s in range(COLOR_DIM):
            if col_comp_n[s] == 1:
                n_uniq += 1
                self.uniq_col = s
        if n_uniq != 1:
            self.uniq_col = -1
        # majority nonbg colour by CELL count
        self.major_col = -1
        var best_cells = 0
        for s in range(COLOR_DIM):
            if s != self.bg and hist[s] > best_cells:
                best_cells = hist[s]
                self.major_col = s
        # anchors: bbox top-left of the largest / unique-colour component
        self.anc_large_r = self.bbox[4 * large_i + 0] if large_i >= 0 else -1
        self.anc_large_c = self.bbox[4 * large_i + 2] if large_i >= 0 else -1
        self.anc_uniq_r = -1
        self.anc_uniq_c = -1
        if self.uniq_col >= 0:
            for i in range(self.n_comp):
                if self.comp_col[i] == self.uniq_col:
                    self.anc_uniq_r = self.bbox[4 * i + 0]
                    self.anc_uniq_c = self.bbox[4 * i + 2]
                    break

        # ---- nearest-nonbg colour + distance (multi-source BFS, 8-conn) ----
        var queue = alloc[Int](n)
        var head = 0
        var tail = 0
        for k in range(n):
            if Int(round(q[k])) != self.bg:
                self.near_col[k] = Int(round(q[k]))
                self.near_d[k] = 0
                queue[tail] = k
                tail += 1
            else:
                self.near_col[k] = -1
                self.near_d[k] = SUB_DIST_CAP
        while head < tail:
            var k = queue[head]
            head += 1
            if self.near_d[k] + 1 >= SUB_DIST_CAP:
                continue
            var r = k // cols
            var c = k % cols
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    var rr = r + dr
                    var cc = c + dc
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                        continue
                    var kk = rr * cols + cc
                    if self.near_d[kk] > self.near_d[k] + 1:
                        self.near_d[kk] = self.near_d[k] + 1
                        self.near_col[kk] = self.near_col[k]
                        queue[tail] = kk
                        tail += 1

        # ---- first nonbg strictly along each ray (clamped at the border) ----
        for c in range(cols):
            var last = -1
            for r in range(rows):
                self.ray[0 * n + r * cols + c] = last  # up
                var v = Int(round(q[r * cols + c]))
                if v != self.bg:
                    last = v
            last = -1
            for r in range(rows - 1, -1, -1):
                self.ray[1 * n + r * cols + c] = last  # down
                var v = Int(round(q[r * cols + c]))
                if v != self.bg:
                    last = v
        for r in range(rows):
            var last = -1
            for c in range(cols):
                self.ray[2 * n + r * cols + c] = last  # left
                var v = Int(round(q[r * cols + c]))
                if v != self.bg:
                    last = v
            last = -1
            for c in range(cols - 1, -1, -1):
                self.ray[3 * n + r * cols + c] = last  # right
                var v = Int(round(q[r * cols + c]))
                if v != self.bg:
                    last = v

        hist.free()
        stack.free()
        col_comp_n.free()
        queue.free()

    def __del__(deinit self):
        # self.q is not owned; free only the derived arrays.
        self.comp.free()
        self.comp_col.free()
        self.comp_size.free()
        self.bbox.free()
        self.near_col.free()
        self.near_d.free()
        self.ray.free()

    def _anchor_read(self, r: Int, c: Int, ar: Int, ac: Int, s: Int) -> Int:
        # Toroidal read at (r + s*ar, c + s*ac) — the scan's fetch_anchor.
        var rr = (r + s * ar + self.rows) % self.rows
        var cc = (c + s * ac + self.cols) % self.cols
        return Int(round(self.q[rr * self.cols + cc]))

    def fetch(self, view: Int, r: Int, c: Int) -> InlineArray[Int, 2]:
        """[rel_bucket, fetched_value] for a view at (r, c); fetched -1 = none.

        rel buckets (all < SUB_REL_K), mirroring the scan's relational keys:
        rays: 0 = no hit / 1 = hit; nearest: capped distance 0..5;
        registers / anchors / obj-local: 0 = absent, 1 = fetched != centre,
        2 = fetched == centre.
        """
        var n = self.rows * self.cols
        var k = r * self.cols + c
        var ctr = Int(round(self.q[k]))
        var f = -1
        var rel = 0
        if view < 4:  # rays u/d/l/r
            f = self.ray[view * n + k]
            rel = 1 if f >= 0 else 0
        elif view == 4:  # nearest nonbg
            f = self.near_col[k]
            rel = self.near_d[k]
        elif view < 9:  # registers
            if view == 5:
                f = self.large_col
            elif view == 6:
                f = self.small_col
            elif view == 7:
                f = self.uniq_col
            else:
                f = self.major_col
            rel = 0 if f < 0 else (2 if f == ctr else 1)
        elif view < 13:  # anchor displacement large+/large-/uniq+/uniq-
            var ar = self.anc_large_r if view < 11 else self.anc_uniq_r
            var ac = self.anc_large_c if view < 11 else self.anc_uniq_c
            var sgn = 1 if (view == 9 or view == 11) else -1
            if ar >= 0:
                f = self._anchor_read(r, c, ar, ac, sgn)
                rel = 2 if f == ctr else 1
        else:  # object-local bbox mirror h/v
            var i = self.comp[k]
            var mr = r
            var mc = c
            if view == 13:
                mc = self.bbox[4 * i + 2] + self.bbox[4 * i + 3] - c
            else:
                mr = self.bbox[4 * i + 0] + self.bbox[4 * i + 1] - r
            f = Int(round(self.q[mr * self.cols + mc]))
            rel = 2 if f == ctr else 1
        var out = InlineArray[Int, 2](fill=0)
        out[0] = rel
        out[1] = f
        return out
