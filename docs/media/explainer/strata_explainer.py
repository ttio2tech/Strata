"""How Strata works - an explainer animation (Manim Community Edition).

Render (1080p, 30 fps):
    manim -r 1920,1080 --frame_rate 30 strata_explainer.py StrataExplainer

Text only (no LaTeX needed). Numbers are from docs/paper/Strata-Paper.pdf
(RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5, Q2_0 at 4K context).
"""

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, ORIGIN,
    AnimationGroup, Create, FadeIn, FadeOut, LaggedStart, Line, Rectangle,
    RoundedRectangle, Scene, Square, Succession, Text, Transform, UpdateFromAlphaFunc,
    Group, ValueTracker, VGroup, VMobject, Wait, always_redraw, config, linear, smooth,
    there_and_back, DashedVMobject, Write, GrowFromEdge, Dot,
)

config.background_color = "#0e1116"
Text.set_default(font="Inter")

# palette
C_GPU = "#7ac943"
C_CPU = "#f59e42"
C_RAM = "#4f9dff"
C_SSD = "#b38cff"
C_PCIE = "#9aa4b2"
C_TEXT = "#e8ecf1"
C_DIM = "#8b95a3"
C_WEIGHT = "#6ec1ff"   # weights moving
C_ID = "#ffd84d"       # expert ids (doorbell)
C_RESULT = "#5eea8a"   # results
C_NGRAM = "#ff6fd8"    # n-gram rows
C_HIT = "#5eea8a"
C_MISS = "#ff7b54"

Y_BUS = 0.85  # height of the PCIe / memory bus


def P(x, y):
    return np.array([x, y, 0.0])


def box(w, h, color, fill=0.10, r=0.12, sw=2.5):
    return RoundedRectangle(width=w, height=h, corner_radius=r, stroke_color=color,
                            stroke_width=sw, fill_color=color, fill_opacity=fill)


def label(s, size=18, color=C_TEXT, weight="NORMAL"):
    return Text(s, font_size=size, color=color, weight=weight)


def grid(rows, cols, size, buff, color, fill=0.55):
    g = VGroup(*[Square(side_length=size, stroke_width=0, fill_color=color, fill_opacity=fill)
                 for _ in range(rows * cols)])
    g.arrange_in_grid(rows=rows, cols=cols, buff=buff)
    return g


class StrataExplainer(Scene):
    # ------------------------------------------------------------------ helpers
    def flow(self, points, n=6, color=C_WEIGHT, radius=0.055, spread=0.7):
        """Dots travelling along a polyline; returns an animation (dots removed at the end)."""
        path = VMobject()
        path.set_points_as_corners([np.array(p, dtype=float) for p in points])
        dots = VGroup(*[Dot(path.get_start(), radius=radius, color=color) for _ in range(n)])
        dots.set_z_index(20)
        for d in dots:
            d.set_opacity(0)
        self.add(dots)

        def upd(m, a):
            for i, d in enumerate(m):
                t = a * (1 + spread) - spread * (i / max(n - 1, 1))
                if 0.0 <= t <= 1.0:
                    d.move_to(path.point_from_proportion(t))
                    d.set_opacity(1)
                else:
                    d.set_opacity(0)

        return UpdateFromAlphaFunc(dots, upd, rate_func=linear, remover=True)

    def hold_caption(self):
        """Keep the current caption up long enough to read (~4 words a second)."""
        if self.caption is not None:
            left = self.caption_until - self.renderer.time
            if left > 0.05:
                self.wait(left)

    def say(self, s, run_time=0.5):
        self.hold_caption()
        self.caption_until = self.renderer.time + 1.2 + 0.26 * len(s.split())
        new = label(s, size=22, color=C_TEXT)
        if new.width > 13.2:
            new.scale_to_fit_width(13.2)
        new.move_to(P(0, -3.62))
        if self.caption is None:
            self.play(FadeIn(new, shift=UP * 0.1), run_time=run_time)
        else:
            self.play(FadeOut(self.caption, shift=UP * 0.1), FadeIn(new, shift=UP * 0.1),
                      run_time=run_time)
        self.caption = new

    def pulse(self, mob, color, run_time=1.0):
        return mob.animate(rate_func=there_and_back, run_time=run_time).set_fill(color, opacity=0.55)

    # ------------------------------------------------------------------ paths
    def ram_to_cpu(self, sq):
        c = sq.get_center()
        return [c, P(c[0], Y_BUS) if abs(c[1] - Y_BUS) > 0.01 else c, P(3.25, Y_BUS), P(1.9, Y_BUS)]

    def ram_to_gpu_cores(self, sq):
        return self.ram_to_cpu(sq)[:-1] + [P(0.95, Y_BUS), P(-0.75, Y_BUS), P(-4.25, Y_BUS), P(-5.4, Y_BUS)]

    def ram_to_vram(self, target):
        src = self.ram_grid[60].get_center()
        t = target.get_center()
        return [src, P(src[0], Y_BUS), P(3.25, Y_BUS), P(0.95, Y_BUS), P(-0.95, Y_BUS), P(t[0], Y_BUS), t]

    # ------------------------------------------------------------------ scene
    def construct(self):
        self.caption = None
        self.caption_until = 0.0
        self.intro()
        self.build_pc()
        self.load_model()
        self.one_layer()
        self.guess_and_check()
        self.adaptive_cache()
        self.balance()
        self.prompt_reading()
        self.outro()

    # 1 ----------------------------------------------------------------------
    def intro(self):
        t = label("How Strata works", size=56, weight="BOLD")
        s = label("a 125-billion-parameter AI model on one gaming PC", size=28, color=C_DIM)
        VGroup(t, s).arrange(DOWN, buff=0.35).move_to(P(0, 1.2))
        self.play(Write(t), run_time=1.2)
        self.play(FadeIn(s, shift=UP * 0.2))

        # the problem: model vs GPU memory
        unit = 0.13  # scene units per GB
        model_bar = Rectangle(width=66 * unit, height=0.55, stroke_width=0,
                              fill_color=C_MISS, fill_opacity=0.85)
        gpu_bar = Rectangle(width=12 * unit, height=0.55, stroke_width=0,
                            fill_color=C_GPU, fill_opacity=0.85)
        model_bar.move_to(P(-4.29 + 33 * unit, -0.6))
        gpu_bar.move_to(P(-4.29 + 6 * unit, -1.5))
        ml = label("model file  66 GB", size=22).next_to(model_bar, UP, buff=0.12).align_to(model_bar, LEFT)
        gl = label("graphics card memory  12 GB", size=22).next_to(gpu_bar, UP, buff=0.12).align_to(gpu_bar, LEFT)
        self.play(GrowFromEdge(model_bar, LEFT), FadeIn(ml), run_time=1.0)
        self.play(GrowFromEdge(gpu_bar, LEFT), FadeIn(gl), run_time=0.8)
        x = label("5.5x too big for the GPU", size=30, color=C_MISS, weight="BOLD").move_to(P(0, -2.6))
        self.play(FadeIn(x, scale=1.2))
        self.wait(1.2)
        self.play(FadeOut(VGroup(model_bar, gpu_bar, ml, gl, x, s)), run_time=0.6)
        title = label("How Strata works", size=30, weight="BOLD").move_to(P(0, 3.6))
        self.play(Transform(t, title))
        self.title = t
        self.say("The trick: use the whole PC - GPU, CPU, RAM, SSD and the PCIe link - at the same time.")
        self.wait(1.2)

    # 2 ----------------------------------------------------------------------
    def build_pc(self):
        case = RoundedRectangle(width=13.4, height=6.2, corner_radius=0.25, stroke_color="#3a4250",
                                stroke_width=2, fill_color="#151a22", fill_opacity=1).move_to(P(0, -0.15))
        case_l = label("Your PC", size=18, color=C_DIM).move_to(P(-6.05, 3.12))
        self.play(FadeIn(case), FadeIn(case_l), run_time=0.7)

        # GPU card
        card = box(5.6, 3.7, C_GPU, fill=0.06).move_to(P(-3.55, 0.85))
        card_l = label("GPU  ·  RTX 5070", size=17, color=C_GPU, weight="BOLD").move_to(P(-5.15, 2.47))
        cores = box(1.5, 1.7, C_GPU, fill=0.18).move_to(P(-5.4, Y_BUS))
        cores_l = label("GPU\ncores", size=16, color=C_TEXT).move_to(cores)
        vram = box(3.3, 3.1, C_GPU, fill=0.04, sw=1.5).move_to(P(-2.6, 0.7))
        vram_l = label("VRAM  12 GB  ·  672 GB/s", size=14, color=C_GPU).move_to(P(-2.6, 2.03))
        self.say("GPU: the fastest memory (VRAM, 672 GB/s) - but only 12 GB of it.")
        self.play(Create(card), FadeIn(card_l), run_time=0.8)
        self.play(FadeIn(cores), FadeIn(cores_l), Create(vram), FadeIn(vram_l), run_time=0.8)

        # VRAM contents (empty for now)
        dense = Rectangle(width=2.9, height=0.36, stroke_color=C_WEIGHT, stroke_width=1.5,
                          fill_color=C_WEIGHT, fill_opacity=0.0).move_to(P(-2.6, 1.62))
        dense_l = label("always-used weights  3.5 GB", size=12).move_to(dense)
        cache_l = label("expert cache  ~4,500 experts", size=12, color=C_DIM).move_to(P(-2.6, 1.24))
        cache = grid(5, 11, 0.16, 0.07, C_RAM, fill=0.0).move_to(P(-2.6, 0.55))
        for s in cache:
            s.set_stroke(C_RAM, width=0.8, opacity=0.5)
        mtp = box(1.35, 0.42, C_ID, fill=0.0, r=0.06, sw=1.2).move_to(P(-3.45, -0.5))
        mtp_l = label("draft layer (MTP)", size=10).move_to(mtp)
        kv = box(1.35, 0.42, C_DIM, fill=0.0, r=0.06, sw=1.2).move_to(P(-1.75, -0.5))
        kv_l = label("KV cache + state", size=10).move_to(kv)
        self.play(FadeIn(VGroup(dense, dense_l, cache_l, cache, mtp, mtp_l, kv, kv_l)), run_time=0.6)

        # CPU
        cpu = box(1.9, 1.6, C_CPU, fill=0.12).move_to(P(1.9, Y_BUS))
        cpu_l = label("CPU\n6 cores", size=16, color=C_TEXT).move_to(cpu)
        cores6 = VGroup(*[Square(0.16, stroke_width=0, fill_color=C_CPU, fill_opacity=0.8) for _ in range(6)])
        cores6.arrange_in_grid(rows=1, cols=6, buff=0.06).next_to(cpu_l, DOWN, buff=0.12)

        # RAM
        ram = box(3.0, 3.7, C_RAM, fill=0.06).move_to(P(4.75, 0.85))
        ram_l = label("System RAM  64 GB\n~45 GB/s", size=14, color=C_RAM).move_to(P(4.75, 2.38))
        ram_grid = grid(11, 14, 0.15, 0.045, C_RAM, fill=0.0).move_to(P(4.75, 0.72))
        for s in ram_grid:
            s.set_stroke(C_RAM, width=0.6, opacity=0.4)
        ram_cap = label("all 24,576 experts  ·  34 GB", size=12, color=C_DIM).move_to(P(4.75, -0.78))
        membus = Line(P(2.85, Y_BUS), P(3.25, Y_BUS), stroke_color=C_RAM, stroke_width=8)

        self.say("RAM: 64 GB - big enough for the whole model, but ~15x slower than VRAM.")
        self.play(Create(ram), FadeIn(ram_l), run_time=0.7)
        self.play(FadeIn(ram_grid), FadeIn(ram_cap), run_time=0.6)
        self.say("CPU: six cores sitting right next to the RAM.")
        self.play(FadeIn(cpu), FadeIn(cpu_l), FadeIn(cores6), Create(membus), run_time=0.8)

        # PCIe
        pcie = Line(P(-0.75, Y_BUS), P(0.95, Y_BUS), stroke_color=C_PCIE, stroke_width=10)
        pcie_l = label("PCIe 4.0 x16\n26 GB/s", size=13, color=C_PCIE).move_to(P(0.1, 1.42))
        self.say("PCIe: the narrow link between the GPU and the rest of the PC (26 GB/s).")
        self.play(Create(pcie), FadeIn(pcie_l), run_time=0.8)

        # SSD
        ssd = box(2.6, 1.1, C_SSD, fill=0.10).move_to(P(1.9, -2.2))
        ssd_l = label("NVMe SSD", size=16, color=C_SSD, weight="BOLD").move_to(P(1.9, -2.0))
        ssd_c = label("n-gram table 28.8 GB", size=12, color=C_DIM).move_to(P(1.9, -2.35))
        nvme = Line(P(1.9, -1.65), P(1.9, 0.05), stroke_color=C_SSD, stroke_width=5)
        self.say("SSD: the biggest and slowest tier.")
        self.play(FadeIn(ssd), FadeIn(ssd_l), FadeIn(ssd_c), Create(nvme), run_time=0.8)

        # bottom-left info panel and legend
        panel = box(5.6, 1.65, C_DIM, fill=0.03, sw=1).move_to(P(-3.55, -2.1))
        legend_items = [(C_WEIGHT, "weights"), (C_ID, "expert ids"), (C_RESULT, "results"), (C_NGRAM, "n-gram rows")]
        legend = VGroup()
        for col, name in legend_items:
            legend.add(VGroup(Dot(radius=0.07, color=col), label(name, size=13, color=C_DIM)).arrange(RIGHT, buff=0.12))
        legend.arrange_in_grid(rows=2, cols=2, buff=(0.35, 0.22), col_alignments="ll").move_to(P(4.85, -2.1))
        self.play(FadeIn(panel), FadeIn(legend), run_time=0.5)

        self.case = case
        self.cores, self.vram, self.dense, self.cache = cores, vram, dense, cache
        self.mtp = mtp
        self.cpu, self.cores6 = cpu, cores6
        self.ram, self.ram_grid = ram, ram_grid
        self.pcie, self.ssd = pcie, ssd
        self.panel = panel
        self.wait(0.5)

    # 3 ----------------------------------------------------------------------
    def load_model(self):
        self.say("The model is a team of 24,576 small 'experts' - but each token uses only 10 per layer (2%).")
        rng = np.random.default_rng(3)
        picks = rng.choice(len(self.ram_grid), 10, replace=False)
        self.ram_grid.set_fill(C_RAM, opacity=0.18)
        self.play(*[self.ram_grid[i].animate.set_fill(C_ID, opacity=0.95) for i in picks], run_time=0.8)
        self.wait(1.0)
        self.play(*[self.ram_grid[i].animate.set_fill(C_RAM, opacity=0.18) for i in picks], run_time=0.5)
        self.ram_grid.set_fill(C_RAM, opacity=0.0)

        # SSD -> RAM : every expert into RAM
        self.say("At start-up every expert is loaded from the SSD into RAM and pinned there.")
        path = [P(1.9, -1.65), P(1.9, Y_BUS), P(3.25, Y_BUS), P(4.75, Y_BUS)]
        order = sorted(range(len(self.ram_grid)), key=lambda i: (-self.ram_grid[i].get_center()[1], self.ram_grid[i].get_center()[0]))
        fills = [self.ram_grid[i].animate.set_fill(C_RAM, opacity=0.6) for i in order]
        self.play(self.flow(path, n=30, color=C_WEIGHT, spread=2.5),
                  LaggedStart(*fills, lag_ratio=0.02), run_time=3.0)

        # RAM -> VRAM : always-used weights + hot experts
        self.say("The always-used weights and the ~4,500 busiest experts are copied onto the GPU.")
        self.play(self.flow(self.ram_to_vram(self.dense), n=14, color=C_WEIGHT, spread=1.5),
                  self.dense.animate.set_fill(C_WEIGHT, opacity=0.35), run_time=1.6)
        cache_fills = [s.animate.set_fill(C_RAM, opacity=0.75) for s in self.cache]
        self.play(self.flow(self.ram_to_vram(self.cache), n=20, color=C_WEIGHT, spread=1.8),
                  LaggedStart(*cache_fills, lag_ratio=0.03),
                  self.mtp.animate.set_fill(C_ID, opacity=0.25), run_time=2.2)
        self.say("The 28.8 GB n-gram table stays on the SSD: each token needs only 16 small rows of it.")
        self.play(self.ssd.animate(rate_func=there_and_back).set_fill(C_SSD, opacity=0.4), run_time=1.0)
        self.wait(0.6)

    # 4 ----------------------------------------------------------------------
    def layer_counter(self, tracker):
        return always_redraw(lambda: label(f"layer {int(round(tracker.get_value()))} / 48", size=24,
                                           color=C_TEXT, weight="BOLD").move_to(P(-4.9, -1.65)))

    def one_layer(self):
        self.say("Every token goes through 48 layers. Here is one of them, slowed down.")
        lt = ValueTracker(1)
        counter = self.layer_counter(lt)
        ms = label("one layer = 0.6 ms", size=15, color=C_DIM).move_to(P(-4.9, -2.15))
        self.play(FadeIn(counter), FadeIn(ms))
        self.lt, self.counter, self.layer_ms = lt, counter, ms

        # n-gram rows, a step ahead
        self.say("1. The SSD delivers this token's n-gram rows, one step ahead of when they are needed.")
        self.play(self.flow([P(1.9, -1.65), P(1.9, Y_BUS), P(3.25, Y_BUS), P(4.0, Y_BUS)],
                            n=6, color=C_NGRAM, radius=0.05), run_time=1.6)

        # mixer + router on the GPU
        self.say("2. The GPU runs the attention/mixer and the router, reading the always-used weights from VRAM.")
        dense_path = [self.dense.get_center(), P(-4.7, 1.62), P(-5.4, Y_BUS)]
        self.play(self.flow(dense_path, n=10, color=C_WEIGHT),
                  self.pulse(self.cores, C_GPU, run_time=1.6), run_time=1.6)

        # router picks 10: 7 hits, 3 misses
        self.say("3. The router picks 10 experts: 7 are already in VRAM (hits), 3 are only in RAM (misses).")
        hits = [3, 9, 14, 20, 27, 38, 46]
        misses = [17, 70, 121]
        hit_sq = [self.cache[i] for i in hits]
        miss_sq = [self.ram_grid[i] for i in misses]
        self.play(*[s.animate.set_fill(C_HIT, opacity=1) for s in hit_sq],
                  *[s.animate.set_fill(C_MISS, opacity=1) for s in miss_sq], run_time=0.8)
        miss_l = label("3 misses", size=14, color=C_MISS).move_to(P(4.75, 2.0))
        self.play(FadeIn(miss_l), run_time=0.4)
        self.wait(0.6)

        # doorbell
        self.say("4. The GPU writes the 10 expert ids to a 'doorbell' in shared memory - the CPU is waiting on it.")
        bell = label("doorbell: 10 ids", size=13, color=C_ID).move_to(P(0.1, 0.3))
        self.play(self.flow([P(-5.4, Y_BUS), P(-0.75, Y_BUS), P(0.95, Y_BUS), P(1.9, Y_BUS)],
                            n=10, color=C_ID, radius=0.05, spread=0.35),
                  FadeIn(bell), run_time=1.4)
        self.play(self.pulse(self.cpu, C_CPU, run_time=0.5))

        # everybody at once
        self.say("5. All at once: the GPU computes its 7 hits, 1 miss is copied over PCIe (DMA), the CPU computes 2 in RAM.")
        dma_l = label("DMA", size=13, color=C_WEIGHT).move_to(P(0.1, 0.3))
        hit_flows = [self.flow([s.get_center(), P(-5.4, Y_BUS)], n=4, color=C_WEIGHT, spread=0.9) for s in hit_sq]
        dma_flow = self.flow(self.ram_to_gpu_cores(miss_sq[0]), n=8, color=C_WEIGHT, spread=0.9)
        cpu_flows = [self.flow(self.ram_to_cpu(s), n=8, color=C_WEIGHT, spread=0.9) for s in miss_sq[1:]]
        self.play(FadeOut(bell), FadeIn(dma_l), run_time=0.3)
        self.play(*hit_flows, dma_flow, *cpu_flows,
                  self.pulse(self.cores, C_GPU, run_time=3.0),
                  self.pulse(self.cpu, C_CPU, run_time=3.0),
                  *[c.animate(rate_func=there_and_back, run_time=3.0).set_fill(C_TEXT, opacity=1) for c in self.cores6],
                  run_time=3.0)

        # results back
        self.say("6. The CPU's results go back to the GPU, which adds everything up - and the next layer starts.")
        self.play(FadeOut(dma_l), run_time=0.2)
        self.play(self.flow([P(1.9, Y_BUS), P(0.95, Y_BUS), P(-0.75, Y_BUS), P(-5.4, Y_BUS)],
                            n=8, color=C_RESULT, spread=0.5), run_time=1.4)
        self.play(*[s.animate.set_fill(C_RAM, opacity=0.75) for s in hit_sq],
                  *[s.animate.set_fill(C_RAM, opacity=0.6) for s in miss_sq],
                  FadeOut(miss_l), self.lt.animate.set_value(2), run_time=0.6)

        # fast-forward through the rest of the layers
        self.say("Repeat for all 48 layers. The whole pass is pre-recorded as one CUDA graph, so nothing waits on the driver.")
        rng = np.random.default_rng(7)
        for k in range(4):
            hs = [self.cache[i] for i in rng.choice(len(self.cache), 7, replace=False)]
            ms_ = [self.ram_grid[i] for i in rng.choice(len(self.ram_grid), 3, replace=False)]
            self.play(*[s.animate.set_fill(C_HIT, opacity=1) for s in hs],
                      *[s.animate.set_fill(C_MISS, opacity=1) for s in ms_], run_time=0.2)
            self.play(*[self.flow([s.get_center(), P(-5.4, Y_BUS)], n=3, color=C_WEIGHT) for s in hs],
                      self.flow(self.ram_to_gpu_cores(ms_[0]), n=4, color=C_WEIGHT),
                      *[self.flow(self.ram_to_cpu(s), n=4, color=C_WEIGHT) for s in ms_[1:]],
                      self.lt.animate(rate_func=linear).set_value(2 + (k + 1) * 11.5),
                      run_time=0.7)
            self.play(*[s.animate.set_fill(C_RAM, opacity=0.75) for s in hs],
                      *[s.animate.set_fill(C_RAM, opacity=0.6) for s in ms_], run_time=0.15)
        self.lt.set_value(48)
        out = self.token_box("cat", C_RESULT).move_to(P(-2.2, -1.9))
        out_l = label("next token!", size=15, color=C_RESULT).next_to(out, RIGHT, buff=0.2)
        self.play(FadeIn(out, scale=1.3), FadeIn(out_l), run_time=0.6)
        self.wait(0.8)
        self.play(FadeOut(VGroup(out, out_l, self.counter, self.layer_ms)), run_time=0.4)

    # 5 ----------------------------------------------------------------------
    def token_box(self, word, color, dashed=False):
        r = RoundedRectangle(width=0.95, height=0.48, corner_radius=0.08, stroke_color=color,
                             stroke_width=2, fill_color=color, fill_opacity=0.12)
        if dashed:
            r = VGroup(DashedVMobject(r, num_dashes=18), r.copy().set_stroke(opacity=0).set_fill(opacity=0.06))
        t = label(word, size=17, color=C_TEXT)
        return VGroup(r, t)

    def guess_and_check(self):
        self.say("Guess, then check: the small draft layer (MTP) on the GPU guesses the next 3 tokens cheaply.")
        known = VGroup(self.token_box("The", C_DIM), self.token_box("cat", C_DIM))
        drafts = VGroup(self.token_box("sat", C_ID, True), self.token_box("on", C_ID, True),
                        self.token_box("a", C_ID, True))
        row = VGroup(*known, *drafts).arrange(RIGHT, buff=0.12).move_to(P(-3.55, -1.75))
        self.play(FadeIn(known), run_time=0.4)
        self.play(self.pulse(self.mtp, C_ID, run_time=1.0),
                  self.flow([self.mtp.get_center(), P(-3.45, -1.2), drafts[0].get_center()], n=3, color=C_ID),
                  run_time=1.0)
        self.play(LaggedStart(*[FadeIn(d, shift=RIGHT * 0.2) for d in drafts], lag_ratio=0.3), run_time=0.9)

        self.say("The last token + 3 guesses go through the 48 layers together: the weights are read once for 4 tokens.")
        win = RoundedRectangle(width=4 * 0.95 + 3 * 0.12 + 0.2, height=0.7, corner_radius=0.1,
                               stroke_color=C_TEXT, stroke_width=1.5).move_to(VGroup(known[1], drafts).get_center())
        win_l = label("one verify window", size=13, color=C_DIM).next_to(win, DOWN, buff=0.06)
        self.play(Create(win), FadeIn(win_l), run_time=0.6)
        dense_path = [self.dense.get_center(), P(-4.7, 1.62), P(-5.4, Y_BUS)]
        self.play(self.flow(dense_path, n=12, color=C_WEIGHT),
                  self.pulse(self.cores, C_GPU, run_time=2.0),
                  self.flow(self.ram_to_cpu(self.ram_grid[40]), n=6, color=C_WEIGHT),
                  self.flow(self.ram_to_cpu(self.ram_grid[95]), n=6, color=C_WEIGHT),
                  self.pulse(self.cpu, C_CPU, run_time=2.0), run_time=2.0)

        self.say("Keep every guess the full model agrees with, plus one token of its own. The text is identical.")
        marks = VGroup(
            Text("✓", font="DejaVu Sans", font_size=24, color=C_HIT).next_to(drafts[0], UP, buff=0.08),
            Text("✓", font="DejaVu Sans", font_size=24, color=C_HIT).next_to(drafts[1], UP, buff=0.08),
            Text("✗", font="DejaVu Sans", font_size=24, color=C_MISS).next_to(drafts[2], UP, buff=0.08),
        )
        self.play(LaggedStart(*[FadeIn(m, scale=1.5) for m in marks], lag_ratio=0.4), run_time=1.0)
        fixed = self.token_box("the", C_RESULT).move_to(drafts[2])
        self.play(drafts[0][0].animate.set_stroke(C_RESULT), drafts[1][0].animate.set_stroke(C_RESULT),
                  Transform(drafts[2], fixed), FadeOut(marks[2]), run_time=0.8)
        gain = label("3 new tokens from one pass  ·  ~3.2 on average  ·  1.6-1.8x faster", size=14,
                     color=C_RESULT).move_to(P(-3.55, -2.65))
        self.play(FadeIn(gain), run_time=0.5)
        self.wait(1.8)
        self.play(FadeOut(VGroup(known, drafts, win, win_l, marks[:2], gain)), run_time=0.5)

    # 6 ----------------------------------------------------------------------
    def adaptive_cache(self):
        self.say("The expert cache learns the conversation: every 4 rounds, up to 96 of its hot experts move into VRAM.")
        ht = ValueTracker(50)
        hr = always_redraw(lambda: label(f"served from VRAM: {int(round(ht.get_value()))}%", size=22,
                                         color=C_HIT, weight="BOLD").move_to(P(-3.55, -1.75)))
        sub = label("profile from other prompts: 50%  ->  adapted to this chat: 72%", size=13,
                    color=C_DIM).move_to(P(-3.55, -2.35))
        self.play(FadeIn(hr), FadeIn(sub), run_time=0.5)
        rng = np.random.default_rng(11)
        hot = [self.ram_grid[i] for i in rng.choice(len(self.ram_grid), 8, replace=False)]
        cold = [self.cache[i] for i in rng.choice(len(self.cache), 8, replace=False)]
        self.play(*[s.animate.set_fill(C_MISS, opacity=1) for s in hot],
                  *[s.animate.set_fill(C_RAM, opacity=0.15) for s in cold], run_time=0.7)
        flows = [self.flow(self.ram_to_cpu(h)[:-1] + [P(0.95, Y_BUS), P(-0.95, Y_BUS), P(c.get_center()[0], Y_BUS), c.get_center()],
                           n=3, color=C_WEIGHT, spread=0.5) for h, c in zip(hot, cold)]
        self.play(*flows, *[c.animate.set_fill(C_MISS, opacity=1) for c in cold],
                  ht.animate.set_value(72), run_time=2.2)
        self.play(*[s.animate.set_fill(C_RAM, opacity=0.6) for s in hot],
                  *[s.animate.set_fill(C_RAM, opacity=0.75) for s in cold], run_time=0.5)
        self.say("Copies happen while the GPU is drafting, and a new expert is used only once it has landed - nobody waits.")
        self.wait(2.0)
        self.play(FadeOut(hr), FadeOut(sub), run_time=0.4)

    # 7 ----------------------------------------------------------------------
    def balance(self):
        self.say("One round (a verify window, Q2_0 at 4K context) takes ~34 ms. GPU and CPU each need about half.")
        total = 34.3
        width = 5.0
        x0 = -3.55 - width / 2
        parts = [("GPU", 14.6, C_GPU), ("CPU experts", 13.4, C_CPU), ("draft", 2.0, C_ID), ("other", 4.3, C_DIM)]
        bars = VGroup()
        labels = VGroup()
        x = x0
        for name, ms, col in parts:
            w = width * ms / total
            b = Rectangle(width=w, height=0.42, stroke_width=0, fill_color=col, fill_opacity=0.85)
            b.move_to(P(x + w / 2, -1.85))
            bars.add(b)
            if ms > 3:
                labels.add(label(f"{name}\n{ms} ms", size=12, color=C_TEXT).next_to(b, DOWN, buff=0.08))
            x += w
        self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.6), run_time=1.6)
        self.play(FadeIn(labels), run_time=0.5)
        self.wait(1.5)
        self.say("The engine is balanced: no single upgrade doubles it - more VRAM helps most (more experts fit).")
        self.wait(2.2)
        self.play(FadeOut(bars), FadeOut(labels), run_time=0.4)

    # 8 ----------------------------------------------------------------------
    def prompt_reading(self):
        self.say("Reading a prompt: 2,048 tokens at a time. Now every expert is busy, so the GPU computes them all.")
        chunk = Rectangle(width=5.0, height=0.36, stroke_color=C_TEXT, stroke_width=1.2).move_to(P(-3.55, -1.75))
        fill = Rectangle(width=5.0, height=0.36, stroke_width=0, fill_color=C_WEIGHT, fill_opacity=0.6)
        fill.move_to(chunk).stretch_to_fit_width(0.01).align_to(chunk, LEFT)
        cl = label("2,048-token chunk", size=13, color=C_DIM).next_to(chunk, DOWN, buff=0.1)
        self.play(FadeIn(chunk), FadeIn(cl), run_time=0.4)
        self.say("Missing experts stream from RAM over PCIe while the GPU works on the previous ones.")
        src = [self.ram_grid[i] for i in (5, 33, 61, 88, 112, 140)]
        flows = [self.flow(self.ram_to_gpu_cores(s), n=10, color=C_WEIGHT, spread=1.4) for s in src]
        target = fill.copy().stretch_to_fit_width(5.0).align_to(chunk, LEFT)
        self.play(LaggedStart(*flows, lag_ratio=0.15),
                  self.pulse(self.cores, C_GPU, run_time=3.5),
                  Transform(fill, target, rate_func=linear), run_time=3.5)
        self.wait(0.5)
        self.play(FadeOut(VGroup(chunk, fill, cl)), run_time=0.4)

    # 9 ----------------------------------------------------------------------
    def outro(self):
        self.say("Every part of the PC does the job it is best at - at the same time.")
        lines = [
            ("GPU", "always-used weights + the busiest experts, the draft layer", C_GPU),
            ("CPU", "computes the missed experts right next to the RAM", C_CPU),
            ("RAM", "holds all 24,576 experts (pinned)", C_RAM),
            ("PCIe", "doorbells, DMA copies, results, cache swaps", C_PCIE),
            ("SSD", "the n-gram table - 16 small rows per token", C_SSD),
        ]
        rows = VGroup()
        for name, txt, col in lines:
            rows.add(VGroup(label(name, size=22, color=col, weight="BOLD"), label(txt, size=20)).arrange(RIGHT, buff=0.3))
        rows.arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        names_w = max(r[0].width for r in rows)
        for r in rows:
            r[1].next_to(r[0], RIGHT, buff=0.3 + names_w - r[0].width)
        rows.move_to(P(0, 0.6))
        result = label("~95 tokens/s on an RTX 5070 + 64 GB RAM", size=34, color=C_RESULT, weight="BOLD")
        result.next_to(rows, DOWN, buff=0.6)
        everything = Group(*[m for m in self.mobjects if m not in (self.title, self.caption)])
        self.play(FadeOut(everything), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.3) for r in rows], lag_ratio=0.25), run_time=2.0)
        self.play(FadeIn(result, scale=1.15), run_time=0.8)
        self.hold_caption()
        self.wait(2.5)
        self.play(FadeOut(VGroup(rows, result, self.caption, self.title)), run_time=1.0)
