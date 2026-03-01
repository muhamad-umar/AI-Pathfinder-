"""
Dynamic Pathfinding Agent — Tkinter Edition
Shows g(n), h(n), f(n) values on each cell + full metrics panel
"""

import tkinter as tk
import math
import heapq
import random
import time
import threading

# ──────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────
ROWS      = 15
COLS      = 20
CELL_SIZE = 48   # larger so values fit inside cells

# ──────────────────────────────────────────
#  COLOURS
# ──────────────────────────────────────────
BG_MAIN  = "#0F172A"
BG_PANEL = "#1E293B"
BG_DARK  = "#0D1424"
TEXT_HI  = "#E2E8F0"
TEXT_LO  = "#64748B"
ACCENT   = "#38BDF8"
ACCENT2  = "#F43F5E"
ACCENT3  = "#FACC15"

COLOR = {
    "empty"    : "#1E293B",
    "wall"     : "#020617",
    "start"    : "#16A34A",
    "goal"     : "#BE123C",
    "frontier" : "#78350F",   # dark amber bg
    "visited"  : "#1E3A5F",   # dark blue bg
    "path"     : "#065F46",   # dark teal bg
    "agent"    : "#38BDF8",
}
# Text colours drawn on cells
TXT = {
    "frontier" : "#FCD34D",
    "visited"  : "#93C5FD",
    "path"     : "#6EE7B7",
    "start"    : "#BBF7D0",
    "goal"     : "#FCA5A5",
    "empty"    : "#475569",
    "wall"     : "#1E293B",
}

# ──────────────────────────────────────────
#  HEURISTICS
# ──────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ──────────────────────────────────────────
#  GRID MODEL
# ──────────────────────────────────────────
class Grid:
    def __init__(self, rows, cols):
        self.rows  = rows
        self.cols  = cols
        self.walls = set()
        self.start = (0, 0)
        self.goal  = (rows-1, cols-1)

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbours(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if self.in_bounds(nr, nc) and (nr,nc) not in self.walls:
                yield (nr, nc)

    def random_maze(self, density=0.30):
        self.walls.clear()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) not in (self.start, self.goal):
                    if random.random() < density:
                        self.walls.add((r,c))

# ──────────────────────────────────────────
#  SEARCH ALGORITHMS
#  Each step yields node scores so we can
#  display g / h / f on the canvas.
# ──────────────────────────────────────────
def gbfs(grid, heuristic_fn):
    start, goal = grid.start, grid.goal
    h  = lambda n: heuristic_fn(n, goal)
    scores = {start: {'g': 0, 'h': round(h(start),2), 'f': round(h(start),2)}}

    open_set  = []
    heapq.heappush(open_set, (h(start), start))
    came_from = {start: None}
    visited   = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        yield {
            'type'    : 'visit',
            'node'    : current,
            'frontier': {n for _,n in open_set},
            'visited' : visited,
            'scores'  : scores,
            'path'    : None,
        }
        if current == goal:
            path = []
            n = current
            while n is not None:
                path.append(n); n = came_from[n]
            path.reverse()
            yield {'type':'done','path':path,'visited':visited,
                   'frontier':set(),'scores':scores}
            return

        for nb in grid.neighbours(*current):
            if nb not in came_from:
                came_from[nb] = current
                hn = round(h(nb), 2)
                scores[nb] = {'g': '—', 'h': hn, 'f': hn}
                heapq.heappush(open_set, (hn, nb))

    yield {'type':'no_path','path':None,'visited':visited,
           'frontier':set(),'scores':scores}


def astar(grid, heuristic_fn):
    start, goal = grid.start, grid.goal
    h  = lambda n: heuristic_fn(n, goal)
    h0 = round(h(start), 2)
    scores = {start: {'g': 0, 'h': h0, 'f': h0}}

    open_set  = []
    heapq.heappush(open_set, (h0, 0, start))
    came_from = {start: None}
    g_score   = {start: 0}
    visited   = set()

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        yield {
            'type'    : 'visit',
            'node'    : current,
            'frontier': {n for _,_,n in open_set},
            'visited' : visited,
            'scores'  : scores,
            'path'    : None,
        }
        if current == goal:
            path = []
            n = current
            while n is not None:
                path.append(n); n = came_from[n]
            path.reverse()
            yield {'type':'done','path':path,'visited':visited,
                   'frontier':set(),'scores':scores}
            return

        for nb in grid.neighbours(*current):
            tg = g_score[current] + 1
            if nb not in g_score or tg < g_score[nb]:
                g_score[nb]   = tg
                came_from[nb] = current
                hn = round(h(nb), 2)
                fn = round(tg + hn, 2)
                scores[nb] = {'g': tg, 'h': hn, 'f': fn}
                heapq.heappush(open_set, (fn, tg, nb))

    yield {'type':'no_path','path':None,'visited':visited,
           'frontier':set(),'scores':scores}

# ──────────────────────────────────────────
#  APPLICATION
# ──────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        root.title("Dynamic Pathfinding Agent")
        root.configure(bg=BG_MAIN)
        root.resizable(True, True)

        # State
        self.rows      = ROWS
        self.cols      = COLS
        self.cell_size = CELL_SIZE
        self.grid      = Grid(self.rows, self.cols)

        self.algorithm   = tk.StringVar(value="A*")
        self.heuristic   = tk.StringVar(value="Manhattan")
        self.edit_mode   = tk.StringVar(value="Wall")
        self.speed_var   = tk.IntVar(value=20)
        self.rows_var    = tk.IntVar(value=ROWS)
        self.cols_var    = tk.IntVar(value=COLS)
        self.dens_var    = tk.IntVar(value=28)
        self.dynamic_var = tk.BooleanVar(value=False)
        self.show_scores = tk.BooleanVar(value=True)

        self.visited   = set()
        self.frontier  = set()
        self.path      = []
        self.scores    = {}          # node -> {'g','h','f'}
        self.no_path   = False
        self.running   = False
        self.paused    = False
        self.replaying = False
        self.agent_pos = None
        self.agent_step= 0

        self.replan_count  = 0
        self.nodes_visited = 0
        self.path_cost     = 0
        self.exec_time_ms  = 0.0
        self._start_time   = 0.0
        self._search_gen   = None
        self._stop_flag    = False

        # step log for the calculation panel
        self.step_log   = []         # list of strings
        self.log_limit  = 200

        self._build_gui()
        self._draw_grid()

    # ─────────────────────────────────────
    #  GUI
    # ─────────────────────────────────────
    def _build_gui(self):
        # ── Title bar
        hdr = tk.Frame(self.root, bg=BG_MAIN)
        hdr.pack(fill=tk.X, padx=16, pady=(12,4))
        tk.Label(hdr, text="DYNAMIC PATHFINDING AGENT",
                 fg=ACCENT, bg=BG_MAIN,
                 font=("Consolas",15,"bold")).pack(side=tk.LEFT)
        tk.Label(hdr, text="  GBFS · A* · Real-time Re-planning",
                 fg=TEXT_LO, bg=BG_MAIN,
                 font=("Consolas",9)).pack(side=tk.LEFT)

        # ── Three-column layout
        body = tk.Frame(self.root, bg=BG_MAIN)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # LEFT: calculation / step log panel
        self._build_left_panel(body)

        # CENTRE: grid canvas with scrollbars
        centre = tk.Frame(body, bg=BG_DARK)
        centre.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        hbar = tk.Scrollbar(centre, orient=tk.HORIZONTAL,
                            bg=BG_PANEL, troughcolor=BG_DARK, width=10, relief=tk.FLAT)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = tk.Scrollbar(centre, orient=tk.VERTICAL,
                            bg=BG_PANEL, troughcolor=BG_DARK, width=10, relief=tk.FLAT)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(centre, bg=BG_MAIN, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4,0), pady=4)
        hbar.config(command=self.canvas.xview)
        vbar.config(command=self.canvas.yview)

        self.canvas.bind("<Button-1>",         self._on_lclick)
        self.canvas.bind("<B1-Motion>",        self._on_ldrag)
        self.canvas.bind("<Button-3>",         self._on_rclick)
        self.canvas.bind("<B3-Motion>",        self._on_rdrag)
        self.canvas.bind("<MouseWheel>",       self._on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.canvas.bind("<Button-4>",         lambda e: self.canvas.yview_scroll(-1,"units"))
        self.canvas.bind("<Button-5>",         lambda e: self.canvas.yview_scroll(1,"units"))

        # RIGHT: controls + metrics
        self._build_right_panel(body)

        # ── Legend
        self._build_legend()

    # ── LEFT PANEL ───────────────────────
    def _build_left_panel(self, parent):
        pnl = tk.Frame(parent, bg=BG_PANEL, width=230)
        pnl.pack(side=tk.LEFT, fill=tk.Y, padx=(0,6))
        pnl.pack_propagate(False)

        self._sec(pnl, "CALCULATIONS")

        # Formula display
        form = tk.Frame(pnl, bg=BG_DARK)
        form.pack(fill=tk.X, padx=8, pady=(4,6))

        self.lbl_formula = tk.Label(form,
            text="f(n) = g(n) + h(n)",
            fg=ACCENT3, bg=BG_DARK,
            font=("Consolas",10,"bold"), pady=4)
        self.lbl_formula.pack(fill=tk.X)

        self.lbl_current_g = self._calc_row(form, "g(n)  path cost", "—", ACCENT)
        self.lbl_current_h = self._calc_row(form, "h(n)  heuristic", "—", "#F472B6")
        self.lbl_current_f = self._calc_row(form, "f(n)  total",     "—", ACCENT3)
        self.lbl_cur_node  = self._calc_row(form, "Node (r,c)",       "—", TEXT_HI)

        # Score table for selected/last node
        self._sec(pnl, "STEP LOG")
        log_frame = tk.Frame(pnl, bg=BG_DARK)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.log_text = tk.Text(log_frame,
            bg=BG_DARK, fg="#94A3B8",
            font=("Consolas",8),
            relief=tk.FLAT, state=tk.DISABLED,
            wrap=tk.NONE)
        sb = tk.Scrollbar(log_frame, command=self.log_text.yview,
                          bg=BG_PANEL, troughcolor=BG_DARK,
                          width=8, relief=tk.FLAT)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Tag colours
        self.log_text.tag_config("hdr",    foreground=ACCENT,   font=("Consolas",8,"bold"))
        self.log_text.tag_config("visit",  foreground="#93C5FD")
        self.log_text.tag_config("path",   foreground="#6EE7B7", font=("Consolas",8,"bold"))
        self.log_text.tag_config("done",   foreground=ACCENT3,   font=("Consolas",8,"bold"))
        self.log_text.tag_config("replan", foreground=ACCENT2,   font=("Consolas",8,"bold"))

    def _calc_row(self, parent, label, value, fg):
        f = tk.Frame(parent, bg=BG_DARK)
        f.pack(fill=tk.X, padx=6, pady=1)
        tk.Label(f, text=label, fg=TEXT_LO, bg=BG_DARK,
                 font=("Consolas",8), anchor="w").pack(side=tk.LEFT)
        lbl = tk.Label(f, text=value, fg=fg, bg=BG_DARK,
                       font=("Consolas",10,"bold"), anchor="e")
        lbl.pack(side=tk.RIGHT)
        return lbl

    # ── RIGHT PANEL (scrollable) ─────────
    def _build_right_panel(self, parent):
        # Outer container
        outer = tk.Frame(parent, bg=BG_PANEL, width=220)
        outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(6,0))
        outer.pack_propagate(False)

        # Scrollbar
        sb = tk.Scrollbar(outer, orient=tk.VERTICAL,
                          bg=BG_PANEL, troughcolor=BG_DARK, width=8, relief=tk.FLAT)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Canvas that holds the inner frame
        scroll_canvas = tk.Canvas(outer, bg=BG_PANEL, highlightthickness=0,
                                  yscrollcommand=sb.set)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=scroll_canvas.yview)

        # Inner frame — all widgets go here
        pnl = tk.Frame(scroll_canvas, bg=BG_PANEL)
        win_id = scroll_canvas.create_window((0,0), window=pnl, anchor="nw")

        # Resize inner frame width when canvas resizes
        def _on_canvas_resize(e):
            scroll_canvas.itemconfig(win_id, width=e.width)
        scroll_canvas.bind("<Configure>", _on_canvas_resize)

        # Update scrollregion when inner frame changes size
        def _on_frame_resize(e):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        pnl.bind("<Configure>", _on_frame_resize)

        # Mouse-wheel support on sidebar
        def _wheel(e):
            scroll_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        scroll_canvas.bind("<MouseWheel>", _wheel)
        pnl.bind("<MouseWheel>", _wheel)

        self._sec(pnl, "GRID CONFIG")
        self._slider(pnl, "Rows",     self.rows_var,  5, 40)
        self._slider(pnl, "Cols",     self.cols_var,  5, 60)
        self._slider(pnl, "Density%", self.dens_var,  0, 75)
        self._slider(pnl, "Speed",    self.speed_var, 1, 80)

        bf = tk.Frame(pnl, bg=BG_PANEL)
        bf.pack(fill=tk.X, padx=8, pady=(6,0))
        self._btn(bf,"Gen Maze",  self._generate_maze, ACCENT, BG_MAIN).pack(side=tk.LEFT,expand=True,fill=tk.X,padx=(0,2))
        self._btn(bf,"Clear",     self._clear_grid,   BG_DARK, TEXT_HI).pack(side=tk.LEFT,expand=True,fill=tk.X)

        self._sec(pnl, "ALGORITHM")
        af = tk.Frame(pnl, bg=BG_PANEL); af.pack(fill=tk.X, padx=8, pady=2)
        for alg in ("GBFS","A*"):
            tk.Radiobutton(af, text=alg, variable=self.algorithm, value=alg,
                bg=BG_PANEL, fg=TEXT_HI, selectcolor=BG_DARK,
                activebackground=BG_PANEL, activeforeground=ACCENT,
                font=("Consolas",9), command=self._on_alg_change
            ).pack(side=tk.LEFT, padx=8)

        self._sec(pnl, "HEURISTIC")
        hf = tk.Frame(pnl, bg=BG_PANEL); hf.pack(fill=tk.X, padx=8, pady=2)
        for h in ("Manhattan","Euclidean"):
            tk.Radiobutton(hf, text=h, variable=self.heuristic, value=h,
                bg=BG_PANEL, fg=TEXT_HI, selectcolor=BG_DARK,
                activebackground=BG_PANEL, activeforeground=ACCENT,
                font=("Consolas",9)
            ).pack(side=tk.LEFT, padx=4)

        self._sec(pnl, "EDIT MODE")
        ef = tk.Frame(pnl, bg=BG_PANEL); ef.pack(fill=tk.X, padx=8, pady=2)
        for m in ("Wall","Start","Goal"):
            tk.Radiobutton(ef, text=m, variable=self.edit_mode, value=m,
                bg=BG_PANEL, fg=TEXT_HI, selectcolor=BG_DARK,
                activebackground=BG_PANEL, activeforeground=ACCENT,
                font=("Consolas",9)
            ).pack(side=tk.LEFT, padx=3)

        self._sec(pnl, "DISPLAY")
        df = tk.Frame(pnl, bg=BG_PANEL); df.pack(fill=tk.X, padx=8, pady=2)
        tk.Checkbutton(df, text="Show g/h/f on cells",
            variable=self.show_scores,
            bg=BG_PANEL, fg=TEXT_HI, selectcolor=BG_DARK,
            activebackground=BG_PANEL, activeforeground=ACCENT,
            font=("Consolas",9)
        ).pack(anchor="w")

        self._sec(pnl, "CONTROL")
        for text, cmd, bg, fg in [
            ("▶  Start Search",   self._start_search,  ACCENT,  BG_MAIN),
            ("⏸  Pause/Resume",   self._toggle_pause,  BG_DARK, TEXT_HI),
            ("⏹  Stop & Reset",   self._stop_reset,    ACCENT2, TEXT_HI),
            ("↺  Replay Agent",   self._replay_agent,  ACCENT3, BG_MAIN),
        ]:
            self._btn(pnl, text, cmd, bg, fg).pack(fill=tk.X, padx=8, pady=2)

        self._sec(pnl, "DYNAMIC MODE")
        dk = tk.Frame(pnl, bg=BG_PANEL); dk.pack(fill=tk.X, padx=8, pady=2)
        tk.Checkbutton(dk, text="Enable obstacle spawning",
            variable=self.dynamic_var,
            bg=BG_PANEL, fg=TEXT_HI, selectcolor=BG_DARK,
            activebackground=BG_PANEL, activeforeground=ACCENT,
            font=("Consolas",9)
        ).pack(anchor="w")

        # ── METRICS PANEL
        self._sec(pnl, "METRICS")
        mf = tk.Frame(pnl, bg=BG_DARK)
        mf.pack(fill=tk.X, padx=8, pady=4)

        self.m_status   = self._metric(mf, "Status",         "IDLE",   TEXT_HI)
        self.m_algo     = self._metric(mf, "Algorithm",      "—",      ACCENT)
        self.m_heur     = self._metric(mf, "Heuristic",      "—",      ACCENT)
        self.m_visited  = self._metric(mf, "Nodes Visited",  "0",      "#93C5FD")
        self.m_frontier = self._metric(mf, "Frontier Size",  "0",      ACCENT3)
        self.m_path_len = self._metric(mf, "Path Length",    "0",      "#6EE7B7")
        self.m_cost     = self._metric(mf, "Path Cost",      "0",      "#6EE7B7")
        self.m_time     = self._metric(mf, "Time (ms)",      "0.00",   "#F9A8D4")
        self.m_replans  = self._metric(mf, "Re-plans",       "0",      ACCENT2)

        # Path cost formula note
        note = tk.Frame(pnl, bg=BG_DARK)
        note.pack(fill=tk.X, padx=8, pady=(0,6))
        tk.Label(note,
            text="Path Cost = steps × 1\n(uniform grid, 4-direction)",
            fg=TEXT_LO, bg=BG_DARK, font=("Consolas",7),
            justify=tk.LEFT
        ).pack(anchor="w", padx=4, pady=2)

    def _sec(self, parent, text):
        f = tk.Frame(parent, bg=BG_PANEL)
        f.pack(fill=tk.X, padx=8, pady=(8,2))
        tk.Label(f, text=text, fg=ACCENT, bg=BG_PANEL,
                 font=("Consolas",8,"bold")).pack(side=tk.LEFT)
        tk.Frame(f, bg=TEXT_LO, height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6,0))

    def _slider(self, parent, text, var, lo, hi):
        f = tk.Frame(parent, bg=BG_PANEL)
        f.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(f, text=text, fg=TEXT_HI, bg=BG_PANEL,
                 font=("Consolas",9), width=9, anchor="w").pack(side=tk.LEFT)
        tk.Label(f, textvariable=var, fg=ACCENT, bg=BG_PANEL,
                 font=("Consolas",9), width=3).pack(side=tk.RIGHT)
        tk.Scale(f, from_=lo, to=hi, orient=tk.HORIZONTAL, variable=var,
                 showvalue=False, bg=BG_PANEL, fg=ACCENT,
                 troughcolor=BG_DARK, highlightthickness=0,
                 bd=0, length=90).pack(side=tk.RIGHT)

    def _btn(self, parent, text, cmd, bg=ACCENT, fg=BG_MAIN):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg, font=("Consolas",9,"bold"),
                         relief=tk.FLAT, activebackground=BG_DARK,
                         activeforeground=TEXT_HI, cursor="hand2", pady=4)

    def _metric(self, parent, label, value, fg):
        f = tk.Frame(parent, bg=BG_DARK)
        f.pack(fill=tk.X, padx=4, pady=1)
        tk.Label(f, text=label+":", fg=TEXT_LO, bg=BG_DARK,
                 font=("Consolas",8), anchor="w", width=14).pack(side=tk.LEFT)
        lbl = tk.Label(f, text=value, fg=fg, bg=BG_DARK,
                       font=("Consolas",9,"bold"), anchor="e")
        lbl.pack(side=tk.RIGHT)
        return lbl

    def _build_legend(self):
        leg = tk.Frame(self.root, bg=BG_MAIN)
        leg.pack(fill=tk.X, padx=16, pady=(2,8))
        for col, txt in [
            (COLOR["start"],    "Start (S)"),
            (COLOR["goal"],     "Goal (G)"),
            (COLOR["frontier"], "Frontier"),
            (COLOR["visited"],  "Visited"),
            (COLOR["path"],     "Path"),
            (COLOR["wall"],     "Wall"),
            (COLOR["agent"],    "Agent"),
        ]:
            f = tk.Frame(leg, bg=BG_MAIN); f.pack(side=tk.LEFT, padx=6)
            tk.Label(f, bg=col, width=2, height=1).pack(side=tk.LEFT)
            tk.Label(f, text=txt, bg=BG_MAIN, fg=TEXT_LO,
                     font=("Consolas",8)).pack(side=tk.LEFT, padx=2)

    # ─────────────────────────────────────
    #  DRAWING
    # ─────────────────────────────────────
    def _draw_grid(self):
        self.canvas.delete("all")
        w = self.cols * self.cell_size
        h = self.rows * self.cell_size
        self.canvas.config(scrollregion=(0, 0, w, h))
        cs  = self.cell_size
        pad = 1
        show = self.show_scores.get() and cs >= 36

        path_set = set(self.path)

        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = c*cs, r*cs
                x2, y2 = x1+cs, y1+cs
                cx, cy  = x1+cs//2, y1+cs//2
                node    = (r, c)

                # Background colour
                if node in self.grid.walls:       bg = COLOR["wall"]
                elif node == self.grid.start:     bg = COLOR["start"]
                elif node == self.grid.goal:      bg = COLOR["goal"]
                elif node in path_set:            bg = COLOR["path"]
                elif node in self.frontier:       bg = COLOR["frontier"]
                elif node in self.visited:        bg = COLOR["visited"]
                else:                             bg = COLOR["empty"]

                self.canvas.create_rectangle(
                    x1+pad, y1+pad, x2-pad, y2-pad,
                    fill=bg, outline="", width=0)

                # ── Text overlay
                if node == self.grid.start:
                    self.canvas.create_text(cx, cy, text="S",
                        fill=TXT["start"], font=("Consolas", cs//3, "bold"))

                elif node == self.grid.goal:
                    self.canvas.create_text(cx, cy, text="G",
                        fill=TXT["goal"], font=("Consolas", cs//3, "bold"))

                elif node in self.grid.walls:
                    pass   # no text on walls

                elif show and node in self.scores:
                    sc  = self.scores[node]
                    g   = sc['g']
                    h_  = sc['h']
                    f_  = sc['f']
                    alg = self.algorithm.get()

                    if node in path_set:
                        tc = TXT["path"]
                    elif node in self.frontier:
                        tc = TXT["frontier"]
                    else:
                        tc = TXT["visited"]

                    fnt_s = max(7, cs//5)
                    fnt_b = max(7, cs//4)

                    if alg == "A*":
                        # top: h(n)
                        self.canvas.create_text(cx, y1+5,
                            text=f"h: {h_}", fill="#F9A8D4",
                            font=("Consolas", fnt_s), anchor="n")
                        # middle: f(n) big
                        self.canvas.create_text(cx, cy,
                            text=str(f_), fill=ACCENT3,
                            font=("Consolas", fnt_b, "bold"), anchor="center")
                        # bottom: g(n)
                        self.canvas.create_text(cx, y2-5,
                            text=f"g: {g}", fill=tc,
                            font=("Consolas", fnt_s), anchor="s")
                    else:
                        # GBFS: just h in centre
                        self.canvas.create_text(cx, y1+4,
                            text=f"h:{h_}", fill="#F9A8D4",
                            font=("Consolas", fnt_s), anchor="n")
                        self.canvas.create_text(cx, cy,
                            text=str(h_), fill=ACCENT3,
                            font=("Consolas", fnt_b, "bold"))

                elif not show:
                    # small cells: just coordinates or nothing
                    pass

        # ── Agent dot
        if self.agent_pos:
            r, c = self.agent_pos
            x1, y1 = c*cs, r*cs
            acx, acy = x1+cs//2, y1+cs//2
            rad = max(5, cs//3)
            self.canvas.create_oval(acx-rad, acy-rad, acx+rad, acy+rad,
                fill=COLOR["agent"], outline="#fff", width=2)
            self.canvas.create_text(acx, acy, text="A",
                fill=BG_MAIN, font=("Consolas", max(7,cs//4), "bold"))

    # ─────────────────────────────────────
    #  CANVAS MOUSE EVENTS
    # ─────────────────────────────────────
    def _cell(self, e):
        # Adjust for canvas scroll offset
        x = self.canvas.canvasx(e.x)
        y = self.canvas.canvasy(e.y)
        c = int(x // self.cell_size)
        r = int(y // self.cell_size)
        return (r,c) if 0<=r<self.rows and 0<=c<self.cols else None

    def _edit(self, cell):
        if not cell or self.running or self.replaying: return
        m = self.edit_mode.get()
        if m == "Wall":
            if cell not in (self.grid.start, self.grid.goal):
                self.grid.walls.add(cell); self._draw_grid()
        elif m == "Start":
            if cell not in self.grid.walls and cell != self.grid.goal:
                self.grid.start = cell; self._draw_grid()
        elif m == "Goal":
            if cell not in self.grid.walls and cell != self.grid.start:
                self.grid.goal = cell; self._draw_grid()

    def _erase(self, cell):
        if cell and not self.running and not self.replaying:
            self.grid.walls.discard(cell); self._draw_grid()

    def _on_mousewheel(self, e):
        if e.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif e.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1*(e.delta/120)), "units")

    def _on_shift_mousewheel(self, e):
        self.canvas.xview_scroll(int(-1*(e.delta/120)), "units")

    def _on_lclick(self, e): self._edit(self._cell(e))
    def _on_ldrag(self,  e): self._edit(self._cell(e))
    def _on_rclick(self, e): self._erase(self._cell(e))
    def _on_rdrag(self,  e): self._erase(self._cell(e))

    # ─────────────────────────────────────
    #  GRID COMMANDS
    # ─────────────────────────────────────
    def _generate_maze(self):
        if self.running: return
        self._stop_reset()
        self.rows = self.rows_var.get()
        self.cols = self.cols_var.get()
        self.grid = Grid(self.rows, self.cols)
        self.grid.random_maze(self.dens_var.get()/100)
        self._update_scroll_region()
        self._draw_grid()

    def _clear_grid(self):
        if self.running: return
        self._stop_reset()
        self.rows = self.rows_var.get()
        self.cols = self.cols_var.get()
        self.grid = Grid(self.rows, self.cols)
        self._update_scroll_region()
        self._draw_grid()

    def _update_scroll_region(self):
        w = self.cols * self.cell_size
        h = self.rows * self.cell_size
        self.canvas.config(scrollregion=(0, 0, w, h))

    def _on_alg_change(self):
        alg = self.algorithm.get()
        txt = "f(n) = g(n) + h(n)" if alg == "A*" else "f(n) = h(n)"
        self.lbl_formula.config(text=txt)

    # ─────────────────────────────────────
    #  SEARCH
    # ─────────────────────────────────────
    def _hfn(self):
        return manhattan if self.heuristic.get()=="Manhattan" else euclidean

    def _start_search(self):
        if self.running: return
        self._begin_search(from_node=None)

    def _begin_search(self, from_node=None):
        self.visited   = set()
        self.frontier  = set()
        self.path      = []
        self.scores    = {}
        self.no_path   = False
        self.nodes_visited = 0
        self.path_cost = 0
        self.exec_time_ms = 0.0
        self.replaying = False
        self.agent_pos = None
        self._stop_flag= False
        self.step_log  = []

        src = self.grid
        if from_node:
            src = Grid(self.rows, self.cols)
            src.walls = self.grid.walls.copy()
            src.start = from_node
            src.goal  = self.grid.goal

        fn = gbfs if self.algorithm.get()=="GBFS" else astar
        self._search_gen = fn(src, self._hfn())
        self.running     = True
        self.paused      = False
        self._start_time = time.perf_counter()

        self.m_status.config(text="SEARCHING…", fg=ACCENT3)
        self.m_algo.config(text=self.algorithm.get())
        self.m_heur.config(text=self.heuristic.get()[:3])

        # log header
        alg = self.algorithm.get()
        heur = self.heuristic.get()
        self._log(f"{'─'*26}\n{alg} + {heur}\nStart:{src.start}  Goal:{src.goal}\n{'─'*26}\n", "hdr")

        threading.Thread(target=self._search_loop, daemon=True).start()

    def _search_loop(self):
        while self.running and not self._stop_flag:
            if self.paused:
                time.sleep(0.05); continue
            try:
                state = next(self._search_gen)
            except StopIteration:
                self.running = False; break

            self.visited  = state['visited']
            self.frontier = state['frontier']
            self.scores   = state.get('scores', {})
            self.nodes_visited = len(self.visited)

            # update calc panel for current node
            node = state.get('node')
            if node and node in self.scores:
                sc = self.scores[node]
                self.root.after(0, self._update_calc_panel, node, sc)
                # append to step log
                alg = self.algorithm.get()
                if alg == "A*":
                    line = (f"({node[0]},{node[1]})  "
                            f"g={sc['g']}  h={sc['h']}  f={sc['f']}\n")
                else:
                    line = f"({node[0]},{node[1]})  h={sc['h']}\n"
                self.root.after(0, self._log, line, "visit")

            self.root.after(0, self._draw_grid)
            self.root.after(0, self._refresh_metrics)

            if state['type'] in ('done','no_path'):
                elapsed = time.perf_counter() - self._start_time
                self.exec_time_ms = elapsed * 1000
                self.running = False

                if state['type'] == 'done':
                    self.path      = state['path']
                    self.path_cost = len(self.path)-1
                    self.root.after(0, self._draw_grid)
                    self.root.after(0, self._on_done)
                else:
                    self.no_path = True
                    self.root.after(0, self._on_no_path)
                break

            delay = max(0.005, 1.0/max(1, self.speed_var.get()))
            time.sleep(delay)

    def _on_done(self):
        self.m_status.config(text="PATH FOUND ✓", fg="#4ADE80")
        self._refresh_metrics()
        # log path
        path_str = " → ".join(f"({r},{c})" for r,c in self.path)
        self._log(f"\n✓ PATH (cost={self.path_cost}):\n{path_str}\n", "done")
        if self.dynamic_var.get():
            self.root.after(400, self._replay_agent)

    def _on_no_path(self):
        self.m_status.config(text="NO PATH FOUND", fg=ACCENT2)
        self._log("\n✗ No path found.\n", "done")
        self._refresh_metrics()

    def _toggle_pause(self):
        if self.running or self.paused:
            self.paused = not self.paused
            self.m_status.config(text="PAUSED" if self.paused else "SEARCHING…",
                                 fg=TEXT_HI if self.paused else ACCENT3)

    def _stop_reset(self):
        self._stop_flag    = True
        self.running       = False
        self.paused        = False
        self.replaying     = False
        self.visited       = set()
        self.frontier      = set()
        self.path          = []
        self.scores        = {}
        self.no_path       = False
        self.agent_pos     = None
        self.agent_step    = 0
        self.replan_count  = 0
        self.nodes_visited = 0
        self.path_cost     = 0
        self.exec_time_ms  = 0.0
        self.step_log      = []
        self.m_status.config(text="IDLE", fg=TEXT_HI)
        self._reset_calc_panel()
        self._refresh_metrics()
        self._clear_log()
        self._draw_grid()

    # ─────────────────────────────────────
    #  AGENT REPLAY
    # ─────────────────────────────────────
    def _replay_agent(self):
        if not self.path or self.running: return
        self.replaying  = True
        self.agent_step = 0
        self.agent_pos  = self.path[0]
        self.replan_count = 0
        self.m_status.config(text="REPLAYING…", fg=ACCENT)
        self._tick_replay()

    def _tick_replay(self):
        if not self.replaying: return
        if self.running:
            self.root.after(100, self._tick_replay); return

        if self.dynamic_var.get():
            self._maybe_spawn()

        if not self.replaying: return   # re-plan triggered

        self.agent_step += 1
        if self.agent_step >= len(self.path):
            self.agent_pos = self.grid.goal
            self.replaying = False
            self._draw_grid()
            self.m_status.config(text="ARRIVED ✓", fg="#4ADE80")
            return

        self.agent_pos = self.path[self.agent_step]
        self._draw_grid()
        ms = max(50, 1000//max(1, self.speed_var.get()))
        self.root.after(ms, self._tick_replay)

    def _maybe_spawn(self):
        if random.random() > 0.04: return
        for _ in range(30):
            r = random.randint(0, self.rows-1)
            c = random.randint(0, self.cols-1)
            node = (r,c)
            if node in (self.grid.start, self.grid.goal): continue
            if node in self.grid.walls: continue
            if node == self.agent_pos: continue
            self.grid.walls.add(node)
            remaining = set(self.path[self.agent_step:])
            if node in remaining:
                self.replan_count += 1
                self._log(f"\n⚠ Re-plan #{self.replan_count} from {self.agent_pos}\n","replan")
                self.replaying = False
                self._begin_search(from_node=self.agent_pos)
            return

    # ─────────────────────────────────────
    #  METRICS / CALC PANEL UPDATES
    # ─────────────────────────────────────
    def _refresh_metrics(self):
        self.m_visited.config(text=str(self.nodes_visited))
        self.m_frontier.config(text=str(len(self.frontier)))
        self.m_path_len.config(text=str(len(self.path)) if self.path else "0")
        self.m_cost.config(text=str(self.path_cost))
        self.m_time.config(text=f"{self.exec_time_ms:.2f}")
        self.m_replans.config(text=str(self.replan_count))

    def _update_calc_panel(self, node, sc):
        self.lbl_cur_node.config(text=f"({node[0]},{node[1]})")
        g = sc['g']; h = sc['h']; f = sc['f']
        self.lbl_current_g.config(text=str(g))
        self.lbl_current_h.config(text=str(h))
        self.lbl_current_f.config(text=str(f))

    def _reset_calc_panel(self):
        for lbl in (self.lbl_cur_node, self.lbl_current_g,
                    self.lbl_current_h, self.lbl_current_f):
            lbl.config(text="—")

    def _log(self, text, tag="visit"):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text, tag)
        # trim if too long
        lines = int(self.log_text.index(tk.END).split('.')[0])
        if lines > self.log_limit:
            self.log_text.delete("1.0", f"{lines-self.log_limit}.0")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)


# ──────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()