import tkinter as tk
import time
import random
from collections import deque
import heapq

# ──────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────
ROWS           = 10
COLS           = 10
CELL_SIZE      = 45
STEP_DELAY     = 0.05   # Speed of visualization

# ──────────────────────────────────────────
#  COLORS (Midnight Blue Theme)
# ──────────────────────────────────────────
BG_MAIN   = "#0F172A"       # Midnight Blue
BG_PANEL  = "#1E293B"       # Panel Blue
TEXT_HIGH = "#E2E8F0"       # White-ish
TEXT_LOW  = "#94A3B8"       # Grey
ACCENT    = "#38BDF8"       # Sky Blue

COLOR = {
    "empty"        : "#334155",  # Slate
    "wall"         : "#020617",  # Static Wall
    "start"        : "#22C55E",  # Green
    "target"       : "#F43F5E",  # Pink/Red
    "frontier"     : "#60A5FA",  # Blue (Scanning)
    "explored"     : "#475569",  # Dark Blue (Visited)
    "path"         : "#FACC15",  # Yellow (Final Path)
    
    # Bidirectional specific
    "fwd_frontier" : "#60A5FA",
    "bwd_frontier" : "#F472B6",
    "fwd_explored" : "#334155",
    "bwd_explored" : "#881337",
    "meet"         : "#FB923C",
}

# ──────────────────────────────────────────
#  GRID & MOVEMENT
# ──────────────────────────────────────────
BASE_GRID = [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
]

START  = (0, 0)
TARGET = (9, 9) # Initial default

# Global Grid State
grid = [row[:] for row in BASE_GRID]

# ──────────────────────────────────────────
#  RESTRICTED 6-DIRECTION MOVEMENT (Main Diagonal Only)
# ──────────────────────────────────────────
# Order: Up, Right, Bottom, Bottom-Right, Left, Top-Left
DIRECTIONS = [
    (-1, 0),   # 1. Up
    ( 0, 1),   # 2. Right
    ( 1, 0),   # 3. Bottom
    ( 1, 1),   # 4. Bottom-Right (Diagonal)
    ( 0, -1),  # 5. Left
    (-1, -1)   # 6. Top-Left (Diagonal)
]

# Only Bottom-Right and Top-Left are diagonals now
DIAG_PAIRS = {(1, 1), (-1, -1)}

# ──────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────

def randomize_target():
    """Pick a random empty spot for the Target."""
    global TARGET
    while True:
        r = random.randint(0, ROWS - 1)
        c = random.randint(0, COLS - 1)
        # Ensure it's not a wall and not the start node
        if grid[r][c] == 0 and (r, c) != START:
            TARGET = (r, c)
            return

def reset_grid():
    global grid
    grid = [row[:] for row in BASE_GRID]
    
    # CHECKBOX LOGIC: Only randomize if the box is checked
    if random_target_var.get():
        randomize_target()

def get_neighbors(row, col):
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        if 0 <= r < ROWS and 0 <= c < COLS and grid[r][c] == 0:
            cost = 1.414 if (dr, dc) in DIAG_PAIRS else 1.0
            yield r, c, cost

# ──────────────────────────────────────────
#  DRAWING ENGINE
# ──────────────────────────────────────────

def draw_grid(canvas, frontier=frozenset(), explored=frozenset(), 
              path=frozenset(), status=""):
    canvas.delete("all")
    
    # Background
    canvas.create_rectangle(0, 0, COLS * CELL_SIZE, ROWS * CELL_SIZE, fill=BG_MAIN, outline="")

    for row in range(ROWS):
        for col in range(COLS):
            x1, y1 = col * CELL_SIZE, row * CELL_SIZE
            x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
            cell = (row, col)

            # Determine Color
            fill_col = COLOR["empty"]
            if grid[row][col] == 1: fill_col = COLOR["wall"]
            
            if cell in path:     fill_col = COLOR["path"]
            elif cell in explored: fill_col = COLOR["explored"]
            elif cell in frontier: fill_col = COLOR["frontier"]
            
            if cell == START:    fill_col = COLOR["start"]
            elif cell == TARGET: fill_col = COLOR["target"]

            # Draw Tile
            pad = 1
            canvas.create_rectangle(x1 + pad, y1 + pad, x2 - pad, y2 - pad,
                                    fill=fill_col, outline="", width=0)

            # Text Labels
            if cell == START:
                canvas.create_text(x1 + CELL_SIZE/2, y1 + CELL_SIZE/2,
                                   text="S", fill="#000", font=("Arial", 12, "bold"))
            elif cell == TARGET:
                canvas.create_text(x1 + CELL_SIZE/2, y1 + CELL_SIZE/2,
                                   text="T", fill="#FFF", font=("Arial", 12, "bold"))

    # Status Bar
    canvas.create_text(COLS * CELL_SIZE // 2, ROWS * CELL_SIZE + 15,
                       text=status, fill=ACCENT, font=("Segoe UI", 10, "bold"))

def draw_grid_bidir(canvas, fwd_frontier=frozenset(), bwd_frontier=frozenset(),
                    fwd_explored=frozenset(), bwd_explored=frozenset(),
                    path=frozenset(), meet=None, status=""):
    canvas.delete("all")
    canvas.create_rectangle(0, 0, COLS * CELL_SIZE, ROWS * CELL_SIZE, fill=BG_MAIN, outline="")

    for row in range(ROWS):
        for col in range(COLS):
            x1, y1 = col * CELL_SIZE, row * CELL_SIZE
            x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
            cell = (row, col)

            fill_col = COLOR["empty"]
            if grid[row][col] == 1: fill_col = COLOR["wall"]
            
            if cell in path:     fill_col = COLOR["path"]
            elif cell == meet:   fill_col = COLOR["meet"]
            elif cell in fwd_explored and cell in bwd_explored: fill_col = COLOR["meet"]
            elif cell in fwd_explored: fill_col = COLOR["fwd_explored"]
            elif cell in bwd_explored: fill_col = COLOR["bwd_explored"]
            elif cell in fwd_frontier: fill_col = COLOR["fwd_frontier"]
            elif cell in bwd_frontier: fill_col = COLOR["bwd_frontier"]
            
            if cell == START:    fill_col = COLOR["start"]
            elif cell == TARGET: fill_col = COLOR["target"]

            pad = 1
            canvas.create_rectangle(x1 + pad, y1 + pad, x2 - pad, y2 - pad,
                                    fill=fill_col, outline="", width=0)

            if cell == START:
                canvas.create_text(x1 + CELL_SIZE/2, y1 + CELL_SIZE/2,
                                   text="S", fill="#000", font=("Arial", 12, "bold"))
            elif cell == TARGET:
                canvas.create_text(x1 + CELL_SIZE/2, y1 + CELL_SIZE/2,
                                   text="T", fill="#FFF", font=("Arial", 12, "bold"))

    canvas.create_text(COLS * CELL_SIZE // 2, ROWS * CELL_SIZE + 15,
                       text=status, fill=ACCENT, font=("Segoe UI", 10, "bold"))

# ──────────────────────────────────────────
#  SEARCH ALGORITHMS
# ──────────────────────────────────────────

def bfs(canvas):
    queue = deque([[START]])
    explored = set()
    in_queue = {START}
    
    while queue:
        path = queue.popleft()
        current = path[-1]
        in_queue.discard(current)

        if current in explored: continue
        explored.add(current)

        draw_grid(canvas, frontier=in_queue.copy(), explored=explored, status=f"BFS: Exploring {current}")
        canvas.update()
        time.sleep(STEP_DELAY)

        if current == TARGET:
            draw_grid(canvas, path=set(path), status=f"BFS: Path Found! Length {len(path)}")
            return

        r, c = current
        for nr, nc, _ in get_neighbors(r, c):
            if (nr, nc) not in explored and (nr, nc) not in in_queue:
                queue.append(path + [(nr, nc)])
                in_queue.add((nr, nc))
    draw_grid(canvas, status="BFS: No Path")

def dfs(canvas):
    stack = [[START]]
    explored = set()
    in_stack = {START}

    while stack:
        path = stack.pop()
        current = path[-1]
        in_stack.discard(current)

        if current in explored: continue
        explored.add(current)

        draw_grid(canvas, frontier=in_stack.copy(), explored=explored, status=f"DFS: Exploring {current}")
        canvas.update()
        time.sleep(STEP_DELAY)

        if current == TARGET:
            draw_grid(canvas, path=set(path), status=f"DFS: Path Found! Length {len(path)}")
            return

        r, c = current
        for nr, nc, _ in get_neighbors(r, c):
            if (nr, nc) not in explored and (nr, nc) not in in_stack:
                stack.append(path + [(nr, nc)])
                in_stack.add((nr, nc))
    draw_grid(canvas, status="DFS: No Path")

def ucs(canvas):
    pq = [(0.0, 0, [START])]
    explored = set()
    best_cost = {START: 0.0}
    counter = 0

    while pq:
        cost, _, path = heapq.heappop(pq)
        current = path[-1]

        if current in explored: continue
        explored.add(current)

        frontier = {x[2][-1] for x in pq}
        draw_grid(canvas, frontier=frontier, explored=explored, status=f"UCS: Cost {cost:.2f}")
        canvas.update()
        time.sleep(STEP_DELAY)

        if current == TARGET:
            draw_grid(canvas, path=set(path), status=f"UCS: Path Found! Cost {cost:.2f}")
            return

        r, c = current
        for nr, nc, move_cost in get_neighbors(r, c):
            new_cost = cost + move_cost
            if (nr, nc) not in explored:
                if new_cost < best_cost.get((nr, nc), float('inf')):
                    best_cost[(nr, nc)] = new_cost
                    counter += 1
                    heapq.heappush(pq, (new_cost, counter, path + [(nr, nc)]))
    draw_grid(canvas, status="UCS: No Path")

def dls(canvas, limit):
    stack = [([START], 0)]
    explored = set()
    in_stack = {START}

    while stack:
        path, depth = stack.pop()
        current = path[-1]
        in_stack.discard(current)

        if current in explored and len(path) > 1: continue 
        explored.add(current)

        draw_grid(canvas, frontier=in_stack.copy(), explored=explored, status=f"DLS: Depth {depth}/{limit}")
        canvas.update()
        time.sleep(STEP_DELAY)

        if current == TARGET:
            draw_grid(canvas, path=set(path), status=f"DLS: Found at Depth {depth}")
            return

        if depth >= limit: continue

        r, c = current
        for nr, nc, _ in get_neighbors(r, c):
            if (nr, nc) not in explored:
                stack.append((path + [(nr, nc)], depth + 1))
                in_stack.add((nr, nc))
    draw_grid(canvas, status=f"DLS: Not found in limit {limit}")

def iddfs(canvas):
    for limit in range(ROWS * COLS):
        draw_grid(canvas, status=f"IDDFS: Restarting limit {limit}...")
        canvas.update()
        time.sleep(STEP_DELAY)
        
        stack = [([START], 0)]
        explored = set()
        in_stack = {START}
        
        while stack:
            path, depth = stack.pop()
            current = path[-1]
            in_stack.discard(current)

            if current in explored and len(path) > 1: continue
            explored.add(current)

            draw_grid(canvas, frontier=in_stack.copy(), explored=explored, status=f"IDDFS (Lim {limit}): {current}")
            canvas.update()
            time.sleep(STEP_DELAY)

            if current == TARGET:
                draw_grid(canvas, path=set(path), status=f"IDDFS: Found! Limit {limit}")
                return

            if depth < limit:
                r, c = current
                for nr, nc, _ in get_neighbors(r, c):
                    if (nr, nc) not in explored:
                        stack.append((path + [(nr, nc)], depth + 1))
                        in_stack.add((nr, nc))
    draw_grid(canvas, status="IDDFS: No Path")

def bidirectional(canvas):
    fq = deque([START]); bq = deque([TARGET])
    ff = {START}; bf = {TARGET}
    fe = {START: None}; be = {TARGET: None}
    meet = None

    while fq or bq:
        if fq:
            curr = fq.popleft(); ff.discard(curr)
            draw_grid_bidir(canvas, ff, bf, set(fe), set(be), status=f"BiDir: FWD {curr}")
            canvas.update(); time.sleep(STEP_DELAY)
            
            if curr in be: meet = curr; break
            
            r, c = curr
            for nr, nc, _ in get_neighbors(r, c):
                if (nr, nc) not in fe:
                    fe[(nr, nc)] = curr; ff.add((nr, nc)); fq.append((nr, nc))
        
        if bq:
            curr = bq.popleft(); bf.discard(curr)
            draw_grid_bidir(canvas, ff, bf, set(fe), set(be), status=f"BiDir: BWD {curr}")
            canvas.update(); time.sleep(STEP_DELAY)
            
            if curr in fe: meet = curr; break
            
            r, c = curr
            for nr, nc, _ in get_neighbors(r, c):
                if (nr, nc) not in be:
                    be[(nr, nc)] = curr; bf.add((nr, nc)); bq.append((nr, nc))
    
    if meet:
        path = []
        n = meet
        while n: path.append(n); n = fe.get(n)
        path.reverse()
        n = be.get(meet)
        while n: path.append(n); n = be.get(n)
        final_path = []
        [final_path.append(x) for x in path if x not in final_path]
        draw_grid_bidir(canvas, set(fe), set(be), path=set(final_path), meet=meet, status="BiDir: Connected!")
    else:
        draw_grid_bidir(canvas, set(fe), set(be), status="BiDir: No Path")

# ──────────────────────────────────────────
#  RUN BUTTON CALLBACK
# ──────────────────────────────────────────

def run_btn_click():
    reset_grid() # Checks the box status inside here
    draw_grid(canvas, status="Initializing...")
    canvas.update()
    run_btn.config(state=tk.DISABLED, bg=BG_PANEL)
    
    algo = algo_var.get()
    
    if algo == "BFS": bfs(canvas)
    elif algo == "DFS": dfs(canvas)
    elif algo == "UCS": ucs(canvas)
    elif algo == "IDDFS": iddfs(canvas)
    elif algo == "Bidir": bidirectional(canvas)
    elif algo == "DLS":
        try: dls(canvas, int(depth_var.get()))
        except: pass
    
    run_btn.config(state=tk.NORMAL, bg=ACCENT)

# ──────────────────────────────────────────
#  GUI SETUP
# ──────────────────────────────────────────

root = tk.Tk()
root.title("GOOD PERFORMANCE TIME APP")
root.configure(bg=BG_MAIN)
root.resizable(False, False)

# Header
header = tk.Frame(root, bg=BG_MAIN)
header.pack(pady=(15, 5))
tk.Label(header, text="GOOD PERFORMANCE TIME APP", 
         fg=ACCENT, bg=BG_MAIN, font=("Impact", 18, "italic")).pack()
tk.Label(header, text="UNINFORMED SEARCH VISUALIZATION", 
         fg=TEXT_LOW, bg=BG_MAIN, font=("Segoe UI", 9)).pack()

# Canvas
canvas = tk.Canvas(root, width=COLS*CELL_SIZE, height=ROWS*CELL_SIZE + 30,
                   bg=BG_MAIN, highlightthickness=0)
canvas.pack(padx=20, pady=5)

# Controls
ctrl = tk.Frame(root, bg=BG_PANEL, padx=15, pady=10)
ctrl.pack(fill=tk.X, padx=20, pady=(0, 20))

# Algorithm Menu
f1 = tk.Frame(ctrl, bg=BG_PANEL)
f1.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
tk.Label(f1, text="STRATEGY", fg=TEXT_LOW, bg=BG_PANEL, font=("Arial", 7, "bold")).pack(anchor="w")
algo_var = tk.StringVar(value="BFS")
om = tk.OptionMenu(f1, algo_var, "BFS", "DFS", "UCS", "DLS", "IDDFS", "Bidir")
om.config(bg=BG_MAIN, fg=TEXT_HIGH, bd=0, highlightthickness=0)
om["menu"].config(bg=BG_PANEL, fg=TEXT_HIGH)
om.pack(fill=tk.X)

# Depth Input
f2 = tk.Frame(ctrl, bg=BG_PANEL)
f2.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
tk.Label(f2, text="DLS LIMIT", fg=TEXT_LOW, bg=BG_PANEL, font=("Arial", 7, "bold")).pack(anchor="w")
depth_var = tk.StringVar(value="15")
tk.Entry(f2, textvariable=depth_var, width=5, bg=BG_MAIN, fg=TEXT_HIGH, 
         relief=tk.FLAT, justify="center").pack()

# Random Target Checkbox (NEW)
f3 = tk.Frame(ctrl, bg=BG_PANEL)
f3.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
tk.Label(f3, text="OPTIONS", fg=TEXT_LOW, bg=BG_PANEL, font=("Arial", 7, "bold")).pack(anchor="w")
random_target_var = tk.BooleanVar(value=False) # Unchecked by default
cb = tk.Checkbutton(f3, text="Random Target", variable=random_target_var, 
                    bg=BG_PANEL, fg=TEXT_HIGH, selectcolor=BG_MAIN, activebackground=BG_PANEL, activeforeground=TEXT_HIGH)
cb.pack()

# Run Button
run_btn = tk.Button(ctrl, text="RUN SEARCH", command=run_btn_click,
                    bg=ACCENT, fg=BG_MAIN, font=("Segoe UI", 10, "bold"),
                    activebackground=TEXT_HIGH, relief=tk.FLAT)
run_btn.pack(side=tk.RIGHT, padx=(10, 0))

# Legend
leg = tk.Frame(root, bg=BG_MAIN)
leg.pack(pady=(0, 10))
def add_leg(col, txt):
    f = tk.Frame(leg, bg=BG_MAIN); f.pack(side=tk.LEFT, padx=5)
    tk.Label(f, bg=col, width=1, height=1).pack(side=tk.LEFT)
    tk.Label(f, text=txt, bg=BG_MAIN, fg=TEXT_LOW, font=("Arial", 8)).pack(side=tk.LEFT, padx=2)

add_leg(COLOR["start"], "Start")
add_leg(COLOR["target"], "Target")
add_leg(COLOR["frontier"], "Frontier")
add_leg(COLOR["explored"], "Explored")
add_leg(COLOR["path"], "Path")

# Init
reset_grid()
draw_grid(canvas, status="Ready to Start")

root.mainloop()