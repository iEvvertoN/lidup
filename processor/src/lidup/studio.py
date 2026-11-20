from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

from ._rust_bridge import rust_available

# Paleta Editorial Preto & Branco Ink
BG_OBSIDIAN = "#09090B"
BG_SURFACE = "#111114"
BG_CARD = "#18181C"
BG_INPUT = "#101014"
BG_HOVER = "#23232A"
BORDER_INK = "#2B2B34"
TEXT_WHITE = "#FAFAFA"
TEXT_SILVER = "#A1A1AA"
TEXT_MUTED = "#71717A"
BTN_WHITE = "#FAFAFA"
BTN_WHITE_HOVER = "#E4E4E7"
BTN_WHITE_TEXT = "#09090B"

_MEASURE_RE = re.compile(
    r"\[lidup\]\s+Trecho\s+(\d+)\s+\(#\d+\s*->\s*#\d+\):\s*"
    r"3D\s*=\s*([0-9.]+)\s*m\s*\(([0-9.]+)\s*cm\)\s*\|\s*"
    r"Horiz\s*=\s*([0-9.]+)\s*m\s*\|\s*dV\s*=\s*([+-]?[0-9.]+)\s*m\s*\|\s*"
    r"Incl\s*=\s*([0-9.]+)%"
)
_LEVEL_CUT_RE = re.compile(r"Corte(?:\s*\(cut\))?:\s*([0-9.,]+)\s*m", re.IGNORECASE)
_LEVEL_FILL_RE = re.compile(r"Aterro(?:\s*\(fill\))?:\s*([0-9.,]+)\s*m", re.IGNORECASE)
_LEVEL_Z_RE = re.compile(r"Cota[^:]*:\s*([+-]?[0-9.,]+)\s*m", re.IGNORECASE)


def discover_workspace_items(base_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Localiza capturas brutas (`lidup_bundle_v1`) e pastas processadas no diretório base."""
    root = Path(base_dir)
    bundles: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    if not root.exists() or not root.is_dir():
        return {"bundles": bundles, "outputs": outputs}

    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower(), reverse=True):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        manifest = entry / "manifest.json"
        frames_csv = entry / "frames.csv"
        if manifest.exists() and frames_csv.exists():
            try:
                meta = json.loads(manifest.read_text(encoding="utf-8"))
                if meta.get("schema_version") == "lidup_bundle_v1":
                    with frames_csv.open("r", encoding="utf-8", errors="ignore") as fh:
                        n_frames = max(0, sum(1 for _ in fh) - 1)
                    bundles.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "sector": str(meta.get("sector_name", "")),
                            "frames": n_frames,
                        }
                    )
                    continue
            except Exception:
                pass

        artifacts = {
            "obj_uv": (entry / "object_textured.obj").exists(),
            "mesh": (entry / "object_mesh_local.ply").exists(),
            "cloud": (entry / "cloud_local.ply").exists() or (entry / "cloud_georef.ply").exists(),
            "mdt": (entry / "mdt.tif").exists(),
            "svg": (entry / "floorplan.svg").exists() or (entry / "ponding_map.svg").exists() or (entry / "clearance_map.svg").exists(),
            "dxf": (entry / "floorplan.dxf").exists(),
            "flatness": (entry / "floor_flatness.ply").exists(),
            "room_json": (entry / "room_report.json").exists(),
            "report_json": (entry / "report.json").exists(),
            "sim": (entry / "diff_heatmap.ply").exists() or (entry / "ponding_map.ply").exists() or (entry / "clearance_map.ply").exists(),
        }
        if any(artifacts.values()):
            outputs.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "artifacts": artifacts,
                }
            )

    return {"bundles": bundles, "outputs": outputs}


def parse_measure_output(stdout_text: str) -> list[dict[str, str]]:
    """Extrai tabela de medições da saída de `lidup view --measure`."""
    rows: list[dict[str, str]] = []
    for line in stdout_text.splitlines():
        m = _MEASURE_RE.search(line)
        if m:
            rows.append(
                {
                    "segment": m.group(1),
                    "dist_m": m.group(2),
                    "dist_cm": m.group(3),
                    "horiz_m": m.group(4),
                    "dv_m": m.group(5),
                    "slope_pct": m.group(6),
                }
            )
    return rows


def _preferred_pythonw() -> str:
    exe = Path(sys.executable)
    if sys.platform.startswith("win"):
        pw = exe.with_name("pythonw.exe")
        if pw.exists():
            return str(pw)
    return str(exe)


def _apply_dark_titlebar(win: tk.Tk) -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        value = ctypes.c_int(1)
        # DWMWA_USE_IMMERSIVE_DARK_MODE (Windows 10/11)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(value), ctypes.sizeof(value)
        )
    except Exception:
        pass


class LidUpStudioApp:
    """Central desktop LidUp em estética monocromática Preto & Branco Ink."""

    def __init__(self, root: tk.Tk, workspace_dir: Path | None = None) -> None:
        self.root = root
        self.root.title("LidUp Studio — Engenharia Espacial LiDAR")
        self.root.geometry("1240x820")
        self.root.minsize(1040, 700)
        self.root.configure(bg=BG_OBSIDIAN)
        _apply_dark_titlebar(self.root)

        self.workspace_dir = workspace_dir or Path.cwd()
        self.active_proc: subprocess.Popen[str] | None = None
        self.worker_thread: threading.Thread | None = None
        self.discovery_data: dict[str, list[dict[str, Any]]] = {"bundles": [], "outputs": []}

        # Variáveis de estado
        self.var_workspace = tk.StringVar(value=str(self.workspace_dir))
        self.var_bundle_path = tk.StringVar(value="")
        self.var_out_dir = tk.StringVar(value="")
        self.var_view_target = tk.StringVar(value="")
        self.var_room_target = tk.StringVar(value="")
        self.var_room_out = tk.StringVar(value="")
        self.var_status = tk.StringVar(value="PRONTO")

        # Parâmetros de reconstrução
        self.var_tsdf = tk.BooleanVar(value=True)
        self.var_uv = tk.BooleanVar(value=True)
        self.var_mvs = tk.BooleanVar(value=True)
        self.var_photo_opt = tk.BooleanVar(value=False)
        self.var_hires_color = tk.BooleanVar(value=True)
        self.var_pose_refine = tk.BooleanVar(value=True)
        self.var_with_3dgs = tk.BooleanVar(value=False)
        self.var_upsample = tk.StringVar(value="2")
        self.var_tsdf_voxel = tk.StringVar(value="0.008")
        self.var_voxel_size = tk.StringVar(value="0.03")
        self.var_max_depth = tk.StringVar(value="4.0")
        self.var_min_conf = tk.StringVar(value="2")

        # Parâmetros do visor
        self.var_no_color = tk.BooleanVar(value=False)
        self.var_show_axes = tk.BooleanVar(value=False)
        self.var_view_tsdf = tk.BooleanVar(value=False)
        self.var_point_size = tk.StringVar(value="3.5")

        # Parâmetros de engenharia
        self.var_up_axis = tk.StringVar(value="auto")
        self.var_lvl_mdt = tk.StringVar(value="")
        self.var_lvl_min_x = tk.StringVar(value="-2.0")
        self.var_lvl_min_y = tk.StringVar(value="-2.0")
        self.var_lvl_max_x = tk.StringVar(value="2.0")
        self.var_lvl_max_y = tk.StringVar(value="2.0")
        self.var_lvl_local = tk.BooleanVar(value=True)
        self.var_lvl_target_z = tk.StringVar(value="")

        # Parâmetros de simulações 4D & física
        self.var_diff_before = tk.StringVar(value="")
        self.var_diff_after = tk.StringVar(value="")
        self.var_diff_tol = tk.StringVar(value="15.0")
        self.var_diff_align = tk.BooleanVar(value=True)
        self.var_diff_out = tk.StringVar(value="")

        self.var_drain_target = tk.StringVar(value="")
        self.var_drain_cell = tk.StringVar(value="0.04")
        self.var_drain_out = tk.StringVar(value="")

        self.var_clear_target = tk.StringVar(value="")
        self.var_clear_w = tk.StringVar(value="0.80")
        self.var_clear_d = tk.StringVar(value="1.20")
        self.var_clear_h = tk.StringVar(value="1.80")
        self.var_clear_out = tk.StringVar(value="")

        self.kpi_labels: dict[str, tk.StringVar] = {
            "dims": tk.StringVar(value="—"),
            "area": tk.StringVar(value="—"),
            "ceiling": tk.StringVar(value="—"),
            "walls": tk.StringVar(value="—"),
            "flat_rms": tk.StringVar(value="—"),
            "plumb": tk.StringVar(value="—"),
            "earthwork": tk.StringVar(value="Aguardando simulação de platô"),
            "diff_summary": tk.StringVar(value="Aguardando comparação Antes × Depois"),
            "drain_summary": tk.StringVar(value="Aguardando simulação hidrológica"),
            "clear_summary": tk.StringVar(value="Aguardando simulação logística / CFTV"),
            "vol_summary": tk.StringVar(value="Aguardando cubagem / análise de vidros"),
        }

        self._setup_styles()
        self._build_layout()
        self.refresh_workspace()

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(".", background=BG_OBSIDIAN, foreground=TEXT_WHITE, font=("Segoe UI", 10))
        style.configure("Surface.TFrame", background=BG_SURFACE)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("Obsidian.TFrame", background=BG_OBSIDIAN)

        style.configure("Header.TLabel", background=BG_SURFACE, foreground=TEXT_WHITE, font=("Segoe UI", 13, "bold"))
        style.configure("Subtle.TLabel", background=BG_SURFACE, foreground=TEXT_SILVER, font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background=BG_CARD, foreground=TEXT_WHITE, font=("Segoe UI", 10, "bold"))
        style.configure("CardText.TLabel", background=BG_CARD, foreground=TEXT_SILVER, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=BG_CARD, foreground=TEXT_MUTED, font=("Segoe UI", 9))
        style.configure("KpiVal.TLabel", background=BG_CARD, foreground=TEXT_WHITE, font=("Consolas", 14, "bold"))
        style.configure("KpiKey.TLabel", background=BG_CARD, foreground=TEXT_MUTED, font=("Segoe UI", 8, "bold"))

        style.configure(
            "Ink.TEntry",
            fieldbackground=BG_INPUT,
            foreground=TEXT_WHITE,
            insertcolor=TEXT_WHITE,
            bordercolor=BORDER_INK,
            lightcolor=BORDER_INK,
            darkcolor=BORDER_INK,
            padding=(8, 6),
        )
        style.configure(
            "Ink.TCombobox",
            fieldbackground=BG_INPUT,
            background=BG_CARD,
            foreground=TEXT_WHITE,
            arrowcolor=TEXT_WHITE,
            bordercolor=BORDER_INK,
            padding=(6, 4),
        )
        style.map("Ink.TCombobox", fieldbackground=[("readonly", BG_INPUT)], selectbackground=[("readonly", BG_HOVER)])

        style.configure(
            "Ink.TCheckbutton",
            background=BG_CARD,
            foreground=TEXT_WHITE,
            indicatorbackground=BG_INPUT,
            indicatorcolor=TEXT_WHITE,
            font=("Segoe UI", 9),
        )
        style.map("Ink.TCheckbutton", background=[("active", BG_CARD)], foreground=[("active", TEXT_WHITE)])

        # Notebook tabs monocromáticos
        style.configure("Ink.TNotebook", background=BG_OBSIDIAN, borderwidth=0, tabmargins=[0, 0, 0, 8])
        style.configure(
            "Ink.TNotebook.Tab",
            background=BG_SURFACE,
            foreground=TEXT_SILVER,
            padding=(18, 9),
            borderwidth=0,
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Ink.TNotebook.Tab",
            background=[("selected", TEXT_WHITE), ("active", BG_HOVER)],
            foreground=[("selected", BG_OBSIDIAN), ("active", TEXT_WHITE)],
        )

        # Treeview tabela de medições
        style.configure(
            "Ink.Treeview",
            background=BG_INPUT,
            foreground=TEXT_WHITE,
            fieldbackground=BG_INPUT,
            bordercolor=BORDER_INK,
            rowheight=26,
            font=("Consolas", 9),
        )
        style.configure(
            "Ink.Treeview.Heading",
            background=BG_CARD,
            foreground=TEXT_SILVER,
            relief="flat",
            font=("Segoe UI", 8, "bold"),
        )
        style.map("Ink.Treeview", background=[("selected", BG_HOVER)], foreground=[("selected", TEXT_WHITE)])

    def _make_primary_button(self, parent: tk.Widget, text: str, command: Any) -> tk.Button:
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=BTN_WHITE,
            fg=BTN_WHITE_TEXT,
            activebackground=BTN_WHITE_HOVER,
            activeforeground=BTN_WHITE_TEXT,
            relief="flat",
            bd=0,
            padx=16,
            pady=9,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        btn.bind("<Enter>", lambda _e: btn.configure(bg=BTN_WHITE_HOVER))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=BTN_WHITE))
        return btn

    def _make_outline_button(self, parent: tk.Widget, text: str, command: Any) -> tk.Button:
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=BG_CARD,
            fg=TEXT_WHITE,
            activebackground=BG_HOVER,
            activeforeground=TEXT_WHITE,
            highlightbackground=BORDER_INK,
            highlightthickness=1,
            relief="flat",
            bd=0,
            padx=12,
            pady=7,
            font=("Segoe UI", 9),
            cursor="hand2",
        )
        btn.bind("<Enter>", lambda _e: btn.configure(bg=BG_HOVER))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=BG_CARD))
        return btn

    def _make_card(self, parent: tk.Widget, title: str) -> ttk.Frame:
        outer = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER_INK, highlightthickness=1)
        outer.pack(fill="x", padx=2, pady=(0, 12))
        hdr = tk.Frame(outer, bg=BG_CARD)
        hdr.pack(fill="x", padx=14, pady=(11, 6))
        tk.Label(hdr, text=title, bg=BG_CARD, fg=TEXT_WHITE, font=("Segoe UI", 9, "bold")).pack(side="left")
        body = ttk.Frame(outer, style="Card.TFrame")
        body.pack(fill="x", padx=14, pady=(0, 12))
        return body

    def _build_layout(self) -> None:
        # 1. Barra Superior
        topbar = tk.Frame(self.root, bg=BG_SURFACE, height=54, highlightbackground=BORDER_INK, highlightthickness=1)
        topbar.pack(side="top", fill="x")
        topbar.pack_propagate(False)

        left_brand = tk.Frame(topbar, bg=BG_SURFACE)
        left_brand.pack(side="left", padx=18)
        tk.Label(left_brand, text="LIDUP", bg=TEXT_WHITE, fg=BG_OBSIDIAN, font=("Segoe UI", 10, "bold"), padx=7, pady=2).pack(side="left")
        tk.Label(left_brand, text="STUDIO", bg=BG_SURFACE, fg=TEXT_WHITE, font=("Segoe UI", 11, "bold"), padx=8).pack(side="left")

        rust_ok = rust_available()
        rust_txt = "MOTOR RUST: ATIVO (ABI v3)" if rust_ok else "MOTOR PYTHON FALLBACK"
        tk.Label(
            left_brand,
            text=rust_txt,
            bg=BG_CARD,
            fg=TEXT_WHITE if rust_ok else TEXT_SILVER,
            font=("Consolas", 8, "bold"),
            padx=9,
            pady=3,
            highlightbackground=BORDER_INK,
            highlightthickness=1,
        ).pack(side="left", padx=10)

        right_ws = tk.Frame(topbar, bg=BG_SURFACE)
        right_ws.pack(side="right", padx=16)
        tk.Label(right_ws, textvariable=self.var_workspace, bg=BG_SURFACE, fg=TEXT_SILVER, font=("Consolas", 8)).pack(side="left", padx=(0, 10))
        self._make_outline_button(right_ws, "Pasta Base...", self._pick_workspace).pack(side="left", padx=(0, 6))
        self._make_outline_button(right_ws, "Atualizar", self.refresh_workspace).pack(side="left")

        # 2. Drawer de Console Inferior (sem CMD externo!)
        console_frame = tk.Frame(self.root, bg=BG_SURFACE, height=150, highlightbackground=BORDER_INK, highlightthickness=1)
        console_frame.pack(side="bottom", fill="x")
        console_frame.pack_propagate(False)

        con_hdr = tk.Frame(console_frame, bg=BG_SURFACE)
        con_hdr.pack(fill="x", padx=14, pady=(6, 4))
        tk.Label(con_hdr, text="CONSOLE DE EXECUÇÃO", bg=BG_SURFACE, fg=TEXT_SILVER, font=("Segoe UI", 8, "bold")).pack(side="left")
        tk.Label(
            con_hdr,
            textvariable=self.var_status,
            bg=BG_CARD,
            fg=TEXT_WHITE,
            font=("Consolas", 8, "bold"),
            padx=8,
            pady=2,
            highlightbackground=BORDER_INK,
            highlightthickness=1,
        ).pack(side="left", padx=10)
        self._make_outline_button(con_hdr, "Limpar Log", self._clear_log).pack(side="right")

        self.log_text = tk.Text(
            console_frame,
            bg=BG_INPUT,
            fg=TEXT_SILVER,
            insertbackground=TEXT_WHITE,
            relief="flat",
            bd=0,
            font=("Consolas", 9),
            wrap="word",
            padx=12,
            pady=6,
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        # 3. Corpo Principal (Sidebar + Notebook)
        main_body = ttk.Frame(self.root, style="Obsidian.TFrame")
        main_body.pack(fill="both", expand=True, padx=14, pady=12)

        sidebar = tk.Frame(main_body, bg=BG_SURFACE, width=285, highlightbackground=BORDER_INK, highlightthickness=1)
        sidebar.pack(side="left", fill="y", padx=(0, 14))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        content = ttk.Frame(main_body, style="Obsidian.TFrame")
        content.pack(side="left", fill="both", expand=True)

        self.notebook = ttk.Notebook(content, style="Ink.TNotebook")
        self.notebook.pack(fill="both", expand=True)

        tab_proc = ttk.Frame(self.notebook, style="Obsidian.TFrame")
        tab_view = ttk.Frame(self.notebook, style="Obsidian.TFrame")
        tab_eng = ttk.Frame(self.notebook, style="Obsidian.TFrame")
        tab_sim = ttk.Frame(self.notebook, style="Obsidian.TFrame")

        self.notebook.add(tab_proc, text="01  RECONSTRUÇÃO")
        self.notebook.add(tab_view, text="02  VISOR & RÉGUA 3D")
        self.notebook.add(tab_eng, text="03  ENGENHARIA & PLANTAS")
        self.notebook.add(tab_sim, text="04  SIMULAÇÕES & 4D")

        self._build_tab_process(tab_proc)
        self._build_tab_view(tab_view)
        self._build_tab_engineering(tab_eng)
        self._build_tab_simulations(tab_sim)

    def _build_sidebar(self, sidebar: tk.Frame) -> None:
        tk.Label(sidebar, text="CAPTURAS DO IPHONE", bg=BG_SURFACE, fg=TEXT_SILVER, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=14, pady=(14, 6))
        self.list_bundles = tk.Listbox(
            sidebar,
            bg=BG_INPUT,
            fg=TEXT_WHITE,
            selectbackground=TEXT_WHITE,
            selectforeground=BG_OBSIDIAN,
            highlightbackground=BORDER_INK,
            highlightthickness=1,
            relief="flat",
            font=("Consolas", 9),
            height=7,
            exportselection=False,
        )
        self.list_bundles.pack(fill="x", padx=12)
        self.list_bundles.bind("<<ListboxSelect>>", self._on_select_bundle)

        tk.Label(sidebar, text="SAÍDAS & MODELOS 3D", bg=BG_SURFACE, fg=TEXT_SILVER, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=14, pady=(16, 6))
        self.list_outputs = tk.Listbox(
            sidebar,
            bg=BG_INPUT,
            fg=TEXT_WHITE,
            selectbackground=TEXT_WHITE,
            selectforeground=BG_OBSIDIAN,
            highlightbackground=BORDER_INK,
            highlightthickness=1,
            relief="flat",
            font=("Consolas", 9),
            height=9,
            exportselection=False,
        )
        self.list_outputs.pack(fill="both", expand=True, padx=12)
        self.list_outputs.bind("<<ListboxSelect>>", self._on_select_output)

        btn_row = tk.Frame(sidebar, bg=BG_SURFACE)
        btn_row.pack(fill="x", padx=12, pady=12)
        self._make_outline_button(btn_row, "Abrir Arquivo .ply/.obj/.laz", self._pick_single_file).pack(fill="x")

    def _build_tab_process(self, parent: ttk.Frame) -> None:
        c_src = self._make_card(parent, "ORIGEM & SAÍDA DA RECONSTRUÇÃO")
        c_src.columnconfigure(1, weight=1)
        ttk.Label(c_src, text="Pacote Bruto:", style="CardText.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(c_src, textvariable=self.var_bundle_path, style="Ink.TEntry").grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        self._make_outline_button(c_src, "Pasta...", self._pick_bundle_dir).grid(row=0, column=2, pady=4)

        ttk.Label(c_src, text="Pasta de Saída:", style="CardText.TLabel").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(c_src, textvariable=self.var_out_dir, style="Ink.TEntry").grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        self._make_outline_button(c_src, "Destino...", self._pick_out_dir).grid(row=1, column=2, pady=4)

        c_pre = self._make_card(parent, "PRESETS RÁPIDOS (1 CLIQUE)")
        pre_row = tk.Frame(c_pre, bg=BG_CARD)
        pre_row.pack(fill="x", pady=2)
        self._make_outline_button(pre_row, "Objeto 360° (TSDF + UV 4K + MVS)", lambda: self._apply_preset("object")).pack(side="left", padx=(0, 8))
        self._make_outline_button(pre_row, "Ambiente / Galpão (TSDF + Planta)", lambda: self._apply_preset("room")).pack(side="left", padx=(0, 8))
        self._make_outline_button(pre_row, "Terreno Rural (MDT + Curvas)", lambda: self._apply_preset("terrain")).pack(side="left", padx=(0, 8))
        self._make_outline_button(pre_row, "Nuvem Rápida", lambda: self._apply_preset("fast")).pack(side="left")

        c_opt = self._make_card(parent, "MOTOR GEOMÉTRICO & PARÂMETROS")
        for col in range(4):
            c_opt.columnconfigure(col, weight=1)

        toggles = [
            ("Malha Sólida TSDF (--tsdf)", self.var_tsdf),
            ("Textura UV 4K (.obj)", self.var_uv),
            ("MVS Micro-relevos (Rust)", self.var_mvs),
            ("Otimização Fotométrica", self.var_photo_opt),
            ("Cor Fotos 12 MP", self.var_hires_color),
            ("PoseGraph + Loop Closure", self.var_pose_refine),
            ("Exportar 3D Gaussian Splatting", self.var_with_3dgs),
        ]
        for idx, (label, var) in enumerate(toggles):
            ttk.Checkbutton(c_opt, text=label, variable=var, style="Ink.TCheckbutton").grid(
                row=idx // 3, column=idx % 3, sticky="w", padx=4, pady=5
            )

        num_bar = tk.Frame(c_opt, bg=BG_CARD)
        num_bar.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 2))
        specs = [
            ("Voxel TSDF (m):", self.var_tsdf_voxel, 8),
            ("Upsample (1-3):", self.var_upsample, 4),
            ("Max Depth (m):", self.var_max_depth, 6),
            ("Voxel Nuvem (m):", self.var_voxel_size, 6),
            ("Confiança Mín.:", self.var_min_conf, 4),
        ]
        for lbl, var, w in specs:
            tk.Label(num_bar, text=lbl, bg=BG_CARD, fg=TEXT_SILVER, font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
            ttk.Entry(num_bar, textvariable=var, width=w, style="Ink.TEntry").pack(side="left", padx=(0, 14))

        act_bar = tk.Frame(parent, bg=BG_OBSIDIAN)
        act_bar.pack(fill="x", padx=2, pady=(4, 0))
        self._make_primary_button(act_bar, "▶  INICIAR PROCESSAMENTO", self._run_process).pack(side="left", padx=(0, 10))
        self._make_outline_button(act_bar, "■  INTERROMPER TAREFA", self._cancel_active).pack(side="left")

    def _build_tab_view(self, parent: ttk.Frame) -> None:
        c_tgt = self._make_card(parent, "ALVO DO VISUALIZADOR 3D")
        c_tgt.columnconfigure(1, weight=1)
        ttk.Label(c_tgt, text="Arquivo / Pasta:", style="CardText.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(c_tgt, textvariable=self.var_view_target, style="Ink.TEntry").grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        self._make_outline_button(c_tgt, "Arquivo...", self._pick_single_file).grid(row=0, column=2, pady=4)

        opt_row = tk.Frame(c_tgt, bg=BG_CARD)
        opt_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 2))
        ttk.Checkbutton(opt_row, text="Abrir sem cor (--no-color / Tecla C alterna)", variable=self.var_no_color, style="Ink.TCheckbutton").pack(side="left", padx=(0, 16))
        ttk.Checkbutton(opt_row, text="Mostrar Eixos XYZ", variable=self.var_show_axes, style="Ink.TCheckbutton").pack(side="left", padx=(0, 16))
        ttk.Checkbutton(opt_row, text="TSDF se pacote bruto", variable=self.var_view_tsdf, style="Ink.TCheckbutton").pack(side="left", padx=(0, 16))
        tk.Label(opt_row, text="Tamanho Ponto:", bg=BG_CARD, fg=TEXT_SILVER, font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        ttk.Entry(opt_row, textvariable=self.var_point_size, width=5, style="Ink.TEntry").pack(side="left")

        act_row = tk.Frame(c_tgt, bg=BG_CARD)
        act_row.grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 2))
        self._make_primary_button(act_row, "ABRIR VISOR 3D INTERATIVO", lambda: self._launch_view(measure=False)).pack(side="left", padx=(0, 10))
        self._make_outline_button(act_row, "📏  ABRIR RÉGUA 3D (1-Clique)", lambda: self._launch_view(measure=True)).pack(side="left")

        c_tbl = self._make_card(parent, "MEDIÇÕES DA RÉGUA 3D (ATUALIZA AO FECHAR COM 'Q')")
        cols = ("segment", "dist_m", "dist_cm", "horiz_m", "dv_m", "slope_pct")
        self.measure_tree = ttk.Treeview(c_tbl, columns=cols, show="headings", style="Ink.Treeview", height=8)
        headers = [
            ("segment", "Trecho", 80),
            ("dist_m", "Distância 3D (m)", 140),
            ("dist_cm", "Distância (cm)", 130),
            ("horiz_m", "Horizontal (m)", 140),
            ("dv_m", "Desnível dV (m)", 140),
            ("slope_pct", "Inclinação (%)", 130),
        ]
        for key, title, width in headers:
            self.measure_tree.heading(key, text=title)
            self.measure_tree.column(key, width=width, anchor="center")
        self.measure_tree.pack(fill="x", pady=(4, 4))
        tk.Label(
            c_tbl,
            text="Controles 3D: Clique curto = marca ponto P1->P2 | Arrastar = gira/olha | WASD + E/X = anda/sobe | Setas = olha | Modo Orbitar/Voo + Presets na barra superior | Z desfaz, Q conclui.",
            bg=BG_CARD,
            fg=TEXT_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(2, 0))

    def _build_tab_engineering(self, parent: ttk.Frame) -> None:
        c_room = self._make_card(parent, "ENGENHARIA DE AMBIENTE, PLANTA 2D (SVG / DXF) & PLANEZA DO PISO")
        c_room.columnconfigure(1, weight=1)

        ttk.Label(c_room, text="Nuvem / Pasta:", style="CardText.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(c_room, textvariable=self.var_room_target, style="Ink.TEntry").grid(row=0, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(c_room, text="Eixo UP:", style="CardText.TLabel").grid(row=0, column=2, padx=(6, 4))
        ttk.Combobox(c_room, textvariable=self.var_up_axis, values=["auto", "y", "z"], width=6, state="readonly", style="Ink.TCombobox").grid(row=0, column=3, pady=4)

        ttk.Label(c_room, text="Saída Planta:", style="CardText.TLabel").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(c_room, textvariable=self.var_room_out, style="Ink.TEntry").grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        room_btns = tk.Frame(c_room, bg=BG_CARD)
        room_btns.grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 10))
        self._make_primary_button(room_btns, "ANALISAR CÔMODO / GALPÃO", self._run_analyze_room).pack(side="left", padx=(0, 8))
        self._make_outline_button(room_btns, "Abrir Planta SVG", self._open_room_svg).pack(side="left", padx=(0, 8))
        self._make_outline_button(room_btns, "Abrir Pasta / DXF", self._open_room_folder).pack(side="left", padx=(0, 8))
        self._make_outline_button(room_btns, "Ver Mapa Planeza (.ply)", self._open_flatness_ply).pack(side="left")

        kpi_grid = tk.Frame(c_room, bg=BG_CARD)
        kpi_grid.grid(row=3, column=0, columnspan=4, sticky="ew")
        for c in range(3):
            kpi_grid.columnconfigure(c, weight=1)

        kpi_defs = [
            ("DIMENSÕES (m)", "dims"),
            ("ÁREA DO PISO", "area"),
            ("PÉ-DIREITO", "ceiling"),
            ("ÁREA DE PAREDES", "walls"),
            ("PLANEZA PISO (RMS/P95)", "flat_rms"),
            ("PRUMO MÁX PAREDES", "plumb"),
        ]
        for idx, (label, key) in enumerate(kpi_defs):
            box = tk.Frame(kpi_grid, bg=BG_INPUT, highlightbackground=BORDER_INK, highlightthickness=1, padx=12, pady=8)
            box.grid(row=idx // 3, column=idx % 3, sticky="ew", padx=4, pady=4)
            tk.Label(box, text=label, bg=BG_INPUT, fg=TEXT_MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w")
            tk.Label(box, textvariable=self.kpi_labels[key], bg=BG_INPUT, fg=TEXT_WHITE, font=("Consolas", 13, "bold")).pack(anchor="w", pady=(2, 0))

        # Card Terraplenagem (`level`)
        c_lvl = self._make_card(parent, "SIMULAÇÃO DE TERRAPLENAGEM & CORTE / ATERRO (MDT)")
        c_lvl.columnconfigure(1, weight=1)
        ttk.Label(c_lvl, text="Arquivo MDT (.tif):", style="CardText.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(c_lvl, textvariable=self.var_lvl_mdt, style="Ink.TEntry").grid(row=0, column=1, columnspan=5, sticky="ew", padx=8, pady=4)

        coords_row = tk.Frame(c_lvl, bg=BG_CARD)
        coords_row.grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 4))
        for lbl, var in [
            ("Min X:", self.var_lvl_min_x),
            ("Min Y:", self.var_lvl_min_y),
            ("Max X:", self.var_lvl_max_x),
            ("Max Y:", self.var_lvl_max_y),
            ("Cota Alvo (opc.):", self.var_lvl_target_z),
        ]:
            tk.Label(coords_row, text=lbl, bg=BG_CARD, fg=TEXT_SILVER, font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
            ttk.Entry(coords_row, textvariable=var, width=7, style="Ink.TEntry").pack(side="left", padx=(0, 10))
        ttk.Checkbutton(coords_row, text="Local", variable=self.var_lvl_local, style="Ink.TCheckbutton").pack(side="left")

        lvl_act = tk.Frame(c_lvl, bg=BG_CARD)
        lvl_act.grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 2))
        self._make_outline_button(lvl_act, "Simular Corte / Aterro", self._run_level).pack(side="left", padx=(0, 14))
        tk.Label(lvl_act, textvariable=self.kpi_labels["earthwork"], bg=BG_CARD, fg=TEXT_WHITE, font=("Consolas", 10, "bold")).pack(side="left")

    def _build_tab_simulations(self, parent: ttk.Frame) -> None:
        # Card 1: Inspeção 4D (Antes x Depois)
        c_diff = self._make_card(parent, "INSPEÇÃO 4D — COMPARAÇÃO ANTES × DEPOIS (MAPA DE CALOR MM)")
        c_diff.columnconfigure(1, weight=1)
        ttk.Label(c_diff, text="Antes (Ref.):", style="CardText.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(c_diff, textvariable=self.var_diff_before, style="Ink.TEntry").grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        self._make_outline_button(c_diff, "Arquivo...", lambda: self._pick_into(self.var_diff_before)).grid(row=0, column=2, pady=3)

        ttk.Label(c_diff, text="Depois:", style="CardText.TLabel").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(c_diff, textvariable=self.var_diff_after, style="Ink.TEntry").grid(row=1, column=1, sticky="ew", padx=8, pady=3)
        self._make_outline_button(c_diff, "Arquivo...", lambda: self._pick_into(self.var_diff_after)).grid(row=1, column=2, pady=3)

        diff_opts = tk.Frame(c_diff, bg=BG_CARD)
        diff_opts.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 4))
        tk.Label(diff_opts, text="Tolerância (mm):", bg=BG_CARD, fg=TEXT_SILVER, font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        ttk.Entry(diff_opts, textvariable=self.var_diff_tol, width=6, style="Ink.TEntry").pack(side="left", padx=(0, 14))
        ttk.Checkbutton(diff_opts, text="Alinhar automaticamente via ICP", variable=self.var_diff_align, style="Ink.TCheckbutton").pack(side="left", padx=(0, 14))
        self._make_primary_button(diff_opts, "COMPARAR 4D", self._run_diff).pack(side="left", padx=(0, 8))
        self._make_outline_button(diff_opts, "Abrir Heatmap 3D", self._open_diff_ply).pack(side="left")

        tk.Label(c_diff, textvariable=self.kpi_labels["diff_summary"], bg=BG_CARD, fg=TEXT_WHITE, font=("Consolas", 9, "bold")).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # Card 2: Simulação Hidrológica / Empoçamento (Priority-Flood Rust)
        c_drain = self._make_card(parent, "SIMULAÇÃO HIDROLÓGICA — EMPOÇAMENTO & DRENAGEM (RUST)")
        c_drain.columnconfigure(1, weight=1)
        ttk.Label(c_drain, text="Nuvem / MDT:", style="CardText.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(c_drain, textvariable=self.var_drain_target, style="Ink.TEntry").grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        tk.Label(c_drain, text="Célula (m):", bg=BG_CARD, fg=TEXT_SILVER, font=("Segoe UI", 9)).grid(row=0, column=2, padx=(6, 4))
        ttk.Entry(c_drain, textvariable=self.var_drain_cell, width=6, style="Ink.TEntry").grid(row=0, column=3, pady=3)

        drain_btns = tk.Frame(c_drain, bg=BG_CARD)
        drain_btns.grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 4))
        self._make_primary_button(drain_btns, "SIMULAR EMPOÇAMENTO", self._run_drainage).pack(side="left", padx=(0, 8))
        self._make_outline_button(drain_btns, "Ver Poças 3D", self._open_drain_ply).pack(side="left", padx=(0, 8))
        self._make_outline_button(drain_btns, "Abrir Planta SVG", self._open_drain_svg).pack(side="left")

        tk.Label(c_drain, textvariable=self.kpi_labels["drain_summary"], bg=BG_CARD, fg=TEXT_WHITE, font=("Consolas", 9, "bold")).grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # Card 3: Logística ("Cabe / Passa?") & Pontos Cegos CFTV
        c_clr = self._make_card(parent, "SIMULADOR LOGÍSTICO ('CABE / PASSA?') & PONTOS CEGOS CFTV (RUST)")
        c_clr.columnconfigure(1, weight=1)
        ttk.Label(c_clr, text="Nuvem / Sala:", style="CardText.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(c_clr, textvariable=self.var_clear_target, style="Ink.TEntry").grid(row=0, column=1, sticky="ew", padx=8, pady=3)

        box_row = tk.Frame(c_clr, bg=BG_CARD)
        box_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 4))
        for lbl, var in [("Largura L (m):", self.var_clear_w), ("Compr. C (m):", self.var_clear_d), ("Altura A (m):", self.var_clear_h)]:
            tk.Label(box_row, text=lbl, bg=BG_CARD, fg=TEXT_SILVER, font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
            ttk.Entry(box_row, textvariable=var, width=6, style="Ink.TEntry").pack(side="left", padx=(0, 12))

        clr_btns = tk.Frame(c_clr, bg=BG_CARD)
        clr_btns.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 4))
        self._make_primary_button(clr_btns, "TESTAR PASSAGEM & CFTV", self._run_clearance).pack(side="left", padx=(0, 8))
        self._make_outline_button(clr_btns, "Mapa Passagem 3D", lambda: self._open_clear_ply("clearance_map.ply")).pack(side="left", padx=(0, 8))
        self._make_outline_button(clr_btns, "Pontos Cegos CFTV 3D", lambda: self._open_clear_ply("viewshed_map.ply")).pack(side="left", padx=(0, 8))
        self._make_outline_button(clr_btns, "Planta SVG", self._open_clear_svg).pack(side="left")

        tk.Label(c_clr, textvariable=self.kpi_labels["clear_summary"], bg=BG_CARD, fg=TEXT_WHITE, font=("Consolas", 9, "bold")).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # Card 4: Cubagem (m³) & Vidros/Espelhos (confidence == 0)
        c_vol = self._make_card(parent, "CUBAGEM DE VOLUME (M³) & DETECTOR DE VIDROS / ESPELHOS")
        vol_btns = tk.Frame(c_vol, bg=BG_CARD)
        vol_btns.pack(fill="x", pady=(2, 4))
        self._make_primary_button(vol_btns, "CALCULAR VOLUME (M³)", self._run_volume).pack(side="left", padx=(0, 8))
        self._make_outline_button(vol_btns, "Ver Volume 3D", lambda: self._open_sim_ply("out_volume", "volume_heatmap.ply")).pack(side="left", padx=(0, 14))
        self._make_outline_button(vol_btns, "Detectar Vidros/Espelhos (Pacote Bruto)", self._run_reflective).pack(side="left", padx=(0, 8))
        self._make_outline_button(vol_btns, "Ver Reflexos 3D", lambda: self._open_sim_ply("out_reflective", "reflective_zones.ply")).pack(side="left")
        tk.Label(c_vol, textvariable=self.kpi_labels["vol_summary"], bg=BG_CARD, fg=TEXT_WHITE, font=("Consolas", 9, "bold")).pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------
    # Ações de descoberta e seleção
    # ------------------------------------------------------------------
    def _append_log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"

        def _write() -> None:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.root.after(0, _write)

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def refresh_workspace(self) -> None:
        base = Path(self.var_workspace.get().strip() or ".").resolve()
        self.workspace_dir = base
        self.discovery_data = discover_workspace_items(base)

        self.list_bundles.delete(0, "end")
        for b in self.discovery_data["bundles"]:
            self.list_bundles.insert("end", f"{b['name']}  ({b['frames']}f)")

        self.list_outputs.delete(0, "end")
        for o in self.discovery_data["outputs"]:
            tags: list[str] = []
            art = o["artifacts"]
            if art["obj_uv"]:
                tags.append("UV")
            elif art["mesh"]:
                tags.append("Mesh")
            elif art["cloud"]:
                tags.append("Nuvem")
            if art["svg"]:
                tags.append("2D")
            if art["mdt"]:
                tags.append("MDT")
            badge = ",".join(tags)
            self.list_outputs.insert("end", f"{o['name']}  [{badge}]" if badge else o["name"])

        if self.discovery_data["bundles"] and not self.var_bundle_path.get():
            self.list_bundles.selection_set(0)
            self._on_select_bundle()
        if self.discovery_data["outputs"] and not self.var_view_target.get():
            best_idx = 0
            for idx, item in enumerate(self.discovery_data["outputs"]):
                art = item["artifacts"]
                if art["obj_uv"] or art["mesh"] or art["cloud"]:
                    best_idx = idx
                    break
            self.list_outputs.selection_set(best_idx)
            self._on_select_output()

        self._append_log(
            f"Workspace atualizado: {len(self.discovery_data['bundles'])} captura(s) e {len(self.discovery_data['outputs'])} saída(s)."
        )

    def _on_select_bundle(self, _evt: Any = None) -> None:
        sel = self.list_bundles.curselection()
        if not sel:
            return
        item = self.discovery_data["bundles"][sel[0]]
        self.var_bundle_path.set(item["path"])
        short_id = item["name"].split("_")[1] if "_" in item["name"] else item["name"]
        self.var_out_dir.set(str(self.workspace_dir / f"out_{short_id}"))
        if not self.var_view_target.get():
            self.var_view_target.set(item["path"])

    def _on_select_output(self, _evt: Any = None) -> None:
        sel = self.list_outputs.curselection()
        if not sel:
            return
        item = self.discovery_data["outputs"][sel[0]]
        out_p = Path(item["path"])
        art = item["artifacts"]

        if art["obj_uv"]:
            self.var_view_target.set(str(out_p / "object_textured.obj"))
        elif art["mesh"]:
            self.var_view_target.set(str(out_p / "object_mesh_local.ply"))
        elif art["cloud"]:
            c_local = out_p / "cloud_local.ply"
            self.var_view_target.set(str(c_local if c_local.exists() else (out_p / "cloud_georef.ply")))
        else:
            fallback_ply = next(
                (
                    out_p / name
                    for name in (
                        "diff_heatmap.ply",
                        "ponding_map.ply",
                        "clearance_map.ply",
                        "viewshed_map.ply",
                        "volume_heatmap.ply",
                        "floor_flatness.ply",
                        "reflective_zones.ply",
                    )
                    if (out_p / name).exists()
                ),
                out_p,
            )
            self.var_view_target.set(str(fallback_ply))

        cloud_pref = str(out_p / "cloud_local.ply" if (out_p / "cloud_local.ply").exists() else out_p)
        self.var_room_target.set(cloud_pref)
        self.var_room_out.set(str(out_p if art["room_json"] else (out_p.parent / f"{out_p.name}_room")))
        self.var_drain_target.set(cloud_pref)
        self.var_drain_out.set(str(out_p.parent / f"{out_p.name}_drainage"))
        self.var_clear_target.set(cloud_pref)
        self.var_clear_out.set(str(out_p.parent / f"{out_p.name}_clearance"))
        if not self.var_diff_before.get():
            self.var_diff_before.set(cloud_pref)
        else:
            self.var_diff_after.set(cloud_pref)
        if not self.var_diff_out.get():
            self.var_diff_out.set(str(out_p.parent / f"{out_p.name}_diff"))
        if art["mdt"]:
            self.var_lvl_mdt.set(str(out_p / "mdt.tif"))

        room_json = out_p / "room_report.json"
        if room_json.exists():
            self._load_room_kpis(room_json)

    def _load_room_kpis(self, report_path: Path) -> None:
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.kpi_labels["dims"].set(f"{data['width_m']:.2f} × {data['length_m']:.2f} m")
            self.kpi_labels["area"].set(f"{data['floor_area_m2']:.2f} m²")
            self.kpi_labels["ceiling"].set(f"{data['ceiling_height_m']:.2f} m ({data.get('ceiling_source', '')})")
            self.kpi_labels["walls"].set(f"{data['wall_area_m2']:.2f} m²")
            self.kpi_labels["flat_rms"].set(f"{data['floor_flatness_rms_mm']:.1f} / {data['floor_flatness_p95_mm']:.1f} mm")
            self.kpi_labels["plumb"].set(f"{data.get('max_wall_plumb_mm_per_m', 0.0):.1f} mm/m")
        except Exception:
            pass

    def _pick_workspace(self) -> None:
        chosen = filedialog.askdirectory(initialdir=str(self.workspace_dir))
        if chosen:
            self.var_workspace.set(chosen)
            self.refresh_workspace()

    def _pick_bundle_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=str(self.workspace_dir))
        if chosen:
            self.var_bundle_path.set(chosen)

    def _pick_out_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=str(self.workspace_dir))
        if chosen:
            self.var_out_dir.set(chosen)

    def _pick_single_file(self) -> None:
        chosen = filedialog.askopenfilename(
            initialdir=str(self.workspace_dir),
            filetypes=[("Modelos 3D / Nuvens", "*.obj *.ply *.laz *.las"), ("Todos", "*.*")],
        )
        if chosen:
            self.var_view_target.set(chosen)
            self.var_room_target.set(chosen)

    def _apply_preset(self, preset: str) -> None:
        if preset == "object":
            self.var_tsdf.set(True)
            self.var_uv.set(True)
            self.var_mvs.set(True)
            self.var_photo_opt.set(True)
            self.var_hires_color.set(True)
            self.var_pose_refine.set(True)
            self.var_upsample.set("2")
            self.var_tsdf_voxel.set("0.008")
            self.var_max_depth.set("3.5")
        elif preset == "room":
            self.var_tsdf.set(True)
            self.var_uv.set(True)
            self.var_mvs.set(False)
            self.var_photo_opt.set(False)
            self.var_hires_color.set(True)
            self.var_pose_refine.set(True)
            self.var_upsample.set("2")
            self.var_tsdf_voxel.set("0.012")
            self.var_max_depth.set("4.5")
        elif preset == "terrain":
            self.var_tsdf.set(False)
            self.var_uv.set(False)
            self.var_mvs.set(False)
            self.var_photo_opt.set(False)
            self.var_hires_color.set(True)
            self.var_pose_refine.set(False)
            self.var_upsample.set("1")
            self.var_voxel_size.set("0.04")
            self.var_max_depth.set("5.0")
        elif preset == "fast":
            self.var_tsdf.set(False)
            self.var_uv.set(False)
            self.var_mvs.set(False)
            self.var_photo_opt.set(False)
            self.var_hires_color.set(False)
            self.var_pose_refine.set(False)
            self.var_upsample.set("1")
            self.var_voxel_size.set("0.03")
            self.var_max_depth.set("4.5")
        self._append_log(f"Preset aplicado: {preset.upper()}")

    # ------------------------------------------------------------------
    # Execução sem CMD (pythonw.exe + CREATE_NO_WINDOW)
    # ------------------------------------------------------------------
    def _spawn_cli_async(
        self,
        cli_args: list[str],
        *,
        on_done: Any | None = None,
        exclusive: bool = True,
    ) -> None:
        if exclusive and self.active_proc is not None and self.active_proc.poll() is None:
            self._append_log("Já existe uma tarefa pesada em execução. Aguarde ou clique em INTERROMPER.")
            return

        cmd = [_preferred_pythonw(), "-u", "-m", "lidup.cli", *cli_args]
        self._append_log(f"Executando: lidup {' '.join(cli_args)}")
        if exclusive:
            self.root.after(0, lambda: self.var_status.set("EXECUTANDO..."))

        def _runner() -> None:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            collected: list[str] = []
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.workspace_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
                env=env,
            )
            if exclusive:
                self.active_proc = proc

            assert proc.stdout is not None
            for line in proc.stdout:
                txt = line.rstrip()
                if txt:
                    collected.append(txt)
                    self._append_log(txt)

            rc = proc.wait()
            if exclusive:
                self.active_proc = None
                self.root.after(0, lambda: self.var_status.set("PRONTO" if rc == 0 else f"ERRO ({rc})"))
            if on_done is not None:
                self.root.after(0, lambda: on_done(rc, "\n".join(collected)))

        self.worker_thread = threading.Thread(target=_runner, daemon=True)
        self.worker_thread.start()

    def _cancel_active(self) -> None:
        if self.active_proc is not None and self.active_proc.poll() is None:
            self.active_proc.terminate()
            self._append_log("Tarefa interrompida pelo usuário.")
            self.var_status.set("INTERROMPIDO")
        else:
            self._append_log("Nenhuma tarefa em andamento.")

    def _run_process(self) -> None:
        bundle_p = self.var_bundle_path.get().strip()
        out_p = self.var_out_dir.get().strip()
        if not bundle_p or not out_p:
            self._append_log("Selecione um pacote bruto e a pasta de saída antes de iniciar.")
            return

        args = [
            "process",
            bundle_p,
            "--out",
            out_p,
            "--min-confidence",
            self.var_min_conf.get().strip() or "2",
            "--max-depth",
            self.var_max_depth.get().strip() or "4.0",
            "--voxel-size",
            self.var_voxel_size.get().strip() or "0.03",
        ]
        if self.var_tsdf.get():
            args.extend(["--tsdf", "--tsdf-voxel", self.var_tsdf_voxel.get().strip() or "0.008"])
        if self.var_uv.get():
            args.append("--uv-texture")
        if self.var_mvs.get():
            args.append("--mvs")
        if self.var_photo_opt.get():
            args.append("--photometric-opt")
        if self.var_hires_color.get():
            args.append("--hires-color")
        if self.var_pose_refine.get():
            args.append("--refine-poses")
        if self.var_with_3dgs.get():
            args.append("--with-3dgs")
        up = self.var_upsample.get().strip()
        if up in {"1", "2", "3"}:
            args.extend(["--depth-upsample", up])

        def _done(rc: int, _out: str) -> None:
            if rc == 0:
                self.refresh_workspace()
                out_dir = Path(out_p)
                if (out_dir / "object_textured.obj").exists():
                    self.var_view_target.set(str(out_dir / "object_textured.obj"))
                elif (out_dir / "object_mesh_local.ply").exists():
                    self.var_view_target.set(str(out_dir / "object_mesh_local.ply"))
                elif (out_dir / "cloud_local.ply").exists():
                    self.var_view_target.set(str(out_dir / "cloud_local.ply"))

        self._spawn_cli_async(args, on_done=_done, exclusive=True)

    def _launch_view(self, *, measure: bool) -> None:
        target = self.var_view_target.get().strip()
        if not target:
            self._append_log("Escolha um arquivo ou pasta para visualizar.")
            return
        args = ["view", target, "--point-size", self.var_point_size.get().strip() or "3.5"]
        if self.var_no_color.get():
            args.append("--no-color")
        if self.var_show_axes.get():
            args.append("--show-axes")
        if self.var_view_tsdf.get():
            args.append("--tsdf")
        if measure:
            args.append("--measure")

        def _done(rc: int, out_txt: str) -> None:
            rows = parse_measure_output(out_txt)
            if rows or (measure and rc == 0):
                for item in self.measure_tree.get_children():
                    self.measure_tree.delete(item)
                for r in rows:
                    self.measure_tree.insert(
                        "",
                        "end",
                        values=(r["segment"], r["dist_m"], r["dist_cm"], r["horiz_m"], r["dv_m"], r["slope_pct"]),
                    )

        self._spawn_cli_async(args, on_done=_done, exclusive=False)

    def _run_analyze_room(self) -> None:
        target = self.var_room_target.get().strip() or self.var_view_target.get().strip()
        out_p = self.var_room_out.get().strip()
        if not target:
            self._append_log("Selecione uma nuvem/malha para análise arquitetônica.")
            return
        if not out_p:
            p = Path(target)
            base_dir = p.parent if p.is_file() else p
            out_p = str(base_dir.parent / f"{base_dir.name}_room")
            self.var_room_out.set(out_p)

        args = ["analyze-room", target, "--out", out_p, "--up-axis", self.var_up_axis.get().strip() or "auto"]

        def _done(rc: int, _txt: str) -> None:
            if rc == 0:
                self._load_room_kpis(Path(out_p) / "room_report.json")
                self.refresh_workspace()

        self._spawn_cli_async(args, on_done=_done, exclusive=True)

    def _open_room_svg(self) -> None:
        out_p = Path(self.var_room_out.get().strip() or ".")
        svg = out_p / "floorplan.svg"
        if svg.exists():
            webbrowser.open(svg.resolve().as_uri())
        else:
            self._append_log(f"Planta SVG não encontrada em {out_p}")

    def _open_room_folder(self) -> None:
        out_p = Path(self.var_room_out.get().strip() or self.workspace_dir)
        if out_p.exists() and sys.platform.startswith("win"):
            os.startfile(str(out_p))

    def _open_flatness_ply(self) -> None:
        out_p = Path(self.var_room_out.get().strip() or ".")
        ply = out_p / "floor_flatness.ply"
        if ply.exists():
            self._spawn_cli_async(["view", str(ply), "--point-size", "4.0"], exclusive=False)
        else:
            self._append_log(f"Mapa de planeza não encontrado em {out_p}")

    def _run_level(self) -> None:
        mdt = self.var_lvl_mdt.get().strip()
        if not mdt or not Path(mdt).exists():
            self._append_log("Selecione um arquivo mdt.tif válido.")
            return
        out_dir = Path(mdt).parent / "terraplenagem"
        args = [
            "level",
            mdt,
            "--out",
            str(out_dir),
            "--min-x",
            self.var_lvl_min_x.get().strip() or "-2.0",
            "--min-y",
            self.var_lvl_min_y.get().strip() or "-2.0",
            "--max-x",
            self.var_lvl_max_x.get().strip() or "2.0",
            "--max-y",
            self.var_lvl_max_y.get().strip() or "2.0",
        ]
        if self.var_lvl_local.get():
            args.append("--local")
        tz = self.var_lvl_target_z.get().strip()
        if tz:
            args.extend(["--target-z", tz])

        def _done(rc: int, out_txt: str) -> None:
            if rc != 0:
                return
            cut_m = _LEVEL_CUT_RE.search(out_txt)
            fill_m = _LEVEL_FILL_RE.search(out_txt)
            z_m = _LEVEL_Z_RE.search(out_txt)
            parts: list[str] = []
            if z_m:
                parts.append(f"Cota: {z_m.group(1)} m")
            if cut_m:
                parts.append(f"Corte: {cut_m.group(1)} m³")
            if fill_m:
                parts.append(f"Aterro: {fill_m.group(1)} m³")
            self.kpi_labels["earthwork"].set(" | ".join(parts) if parts else "Simulação concluída")
            self.refresh_workspace()

        self._spawn_cli_async(args, on_done=_done, exclusive=True)

    def _pick_into(self, target_var: tk.StringVar) -> None:
        chosen = filedialog.askopenfilename(
            initialdir=str(self.workspace_dir),
            filetypes=[("Modelos 3D / Nuvens", "*.ply *.obj *.laz *.las"), ("Todos", "*.*")],
        )
        if chosen:
            target_var.set(chosen)

    def _run_diff(self) -> None:
        b = self.var_diff_before.get().strip()
        a = self.var_diff_after.get().strip()
        out_p = self.var_diff_out.get().strip() or str(self.workspace_dir / "out_diff")
        if not b or not a:
            self._append_log("Selecione os modelos 'Antes' e 'Depois' para comparação 4D.")
            return
        self.var_diff_out.set(out_p)
        args = ["diff", b, a, "--out", out_p, "--tol-mm", self.var_diff_tol.get().strip() or "15.0"]
        if not self.var_diff_align.get():
            args.append("--no-align")

        def _done(rc: int, _txt: str) -> None:
            if rc != 0:
                return
            rep_p = Path(out_p) / "diff_report.json"
            if rep_p.exists():
                d = json.loads(rep_p.read_text(encoding="utf-8"))
                self.kpi_labels["diff_summary"].set(
                    f"RMS: {d['rms_mm']:.1f} mm | P95: {d['p95_mm']:.1f} mm | Alterado: {d['changed_pct']:.1f}% | Vol: {d['net_volume_m3']:+.3f} m³"
                )
                self.refresh_workspace()

        self._spawn_cli_async(args, on_done=_done, exclusive=True)

    def _open_diff_ply(self) -> None:
        p = Path(self.var_diff_out.get().strip() or ".") / "diff_heatmap.ply"
        if p.exists():
            self._spawn_cli_async(["view", str(p), "--point-size", "4.0"], exclusive=False)
        else:
            self._append_log("Execute COMPARAR 4D primeiro.")

    def _run_drainage(self) -> None:
        tgt = self.var_drain_target.get().strip() or self.var_view_target.get().strip()
        out_p = self.var_drain_out.get().strip() or str(self.workspace_dir / "out_drainage")
        if not tgt:
            self._append_log("Selecione uma nuvem/MDT para simulação hidrológica.")
            return
        self.var_drain_out.set(out_p)
        args = ["drainage", tgt, "--out", out_p, "--cell-m", self.var_drain_cell.get().strip() or "0.04"]

        def _done(rc: int, _txt: str) -> None:
            if rc != 0:
                return
            rep_p = Path(out_p) / "drainage_report.json"
            if rep_p.exists():
                d = json.loads(rep_p.read_text(encoding="utf-8"))
                self.kpi_labels["drain_summary"].set(
                    f"Empoçado: {d['ponding_area_m2']:.2f} m² ({d['ponding_pct']:.1f}%) | Lâmina máx: {d['max_depth_mm']:.1f} mm | Volume: {d['water_volume_liters']:.1f} L"
                )
                self.refresh_workspace()

        self._spawn_cli_async(args, on_done=_done, exclusive=True)

    def _open_drain_ply(self) -> None:
        p = Path(self.var_drain_out.get().strip() or ".") / "ponding_map.ply"
        if p.exists():
            self._spawn_cli_async(["view", str(p), "--point-size", "5.0"], exclusive=False)

    def _open_drain_svg(self) -> None:
        p = Path(self.var_drain_out.get().strip() or ".") / "ponding_map.svg"
        if p.exists():
            webbrowser.open(p.resolve().as_uri())

    def _run_clearance(self) -> None:
        tgt = self.var_clear_target.get().strip() or self.var_view_target.get().strip()
        out_p = self.var_clear_out.get().strip() or str(self.workspace_dir / "out_clearance")
        if not tgt:
            self._append_log("Selecione uma nuvem/sala para análise logística e CFTV.")
            return
        self.var_clear_out.set(out_p)
        args = [
            "clearance",
            tgt,
            "--out",
            out_p,
            "--width",
            self.var_clear_w.get().strip() or "0.80",
            "--depth",
            self.var_clear_d.get().strip() or "1.20",
            "--height",
            self.var_clear_h.get().strip() or "1.80",
        ]

        def _done(rc: int, _txt: str) -> None:
            if rc != 0:
                return
            rep_p = Path(out_p) / "clearance_report.json"
            if rep_p.exists():
                d = json.loads(rep_p.read_text(encoding="utf-8"))
                self.kpi_labels["clear_summary"].set(
                    f"Passa livre: {d['passable_area_m2']:.2f} m² ({d['passable_pct']:.1f}%) | CFTV Visível: {d['viewshed_pct']:.1f}% | Ponto cego: {d['blindspot_area_m2']:.2f} m²"
                )
                self.refresh_workspace()

        self._spawn_cli_async(args, on_done=_done, exclusive=True)

    def _open_clear_ply(self, filename: str) -> None:
        p = Path(self.var_clear_out.get().strip() or ".") / filename
        if p.exists():
            self._spawn_cli_async(["view", str(p), "--point-size", "5.0"], exclusive=False)

    def _open_clear_svg(self) -> None:
        p = Path(self.var_clear_out.get().strip() or ".") / "clearance_map.svg"
        if p.exists():
            webbrowser.open(p.resolve().as_uri())

    def _run_volume(self) -> None:
        tgt = self.var_clear_target.get().strip() or self.var_view_target.get().strip()
        if not tgt:
            self._append_log("Selecione uma nuvem/malha para cubagem de volume.")
            return
        out_p = str(self.workspace_dir / "out_volume")

        def _done(rc: int, _txt: str) -> None:
            if rc != 0:
                return
            rep_p = Path(out_p) / "volume_report.json"
            if rep_p.exists():
                d = json.loads(rep_p.read_text(encoding="utf-8"))
                self.kpi_labels["vol_summary"].set(
                    f"Volume: {d['volume_m3']:.3f} m³ | Base: {d['footprint_m2']:.2f} m² | Altura máx: {d['max_height_m']:.2f} m"
                )
                self.refresh_workspace()

        self._spawn_cli_async(["volume", tgt, "--out", out_p], on_done=_done, exclusive=True)

    def _run_reflective(self) -> None:
        b = self.var_bundle_path.get().strip()
        if not b:
            self._append_log("Selecione uma captura bruta do iPhone na barra lateral para detectar vidros/espelhos.")
            return
        out_p = str(self.workspace_dir / "out_reflective")

        def _done(rc: int, _txt: str) -> None:
            if rc != 0:
                return
            rep_p = Path(out_p) / "reflective_report.json"
            if rep_p.exists():
                d = json.loads(rep_p.read_text(encoding="utf-8"))
                self.kpi_labels["vol_summary"].set(
                    f"Vidros/Espelhos (conf=0): {d['reflective_point_count']} pts ({d['reflective_ratio_pct']:.1f}%) em {d['frames_analyzed']} frames"
                )
                self.refresh_workspace()

        self._spawn_cli_async(["reflective", b, "--out", out_p], on_done=_done, exclusive=True)

    def _open_sim_ply(self, folder: str, filename: str) -> None:
        p = self.workspace_dir / folder / filename
        if p.exists():
            self._spawn_cli_async(["view", str(p), "--point-size", "4.0"], exclusive=False)
        else:
            self._append_log(f"Arquivo ainda não gerado: {p}")


def launch_studio(workspace_dir: str | Path | None = None) -> None:
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    ws = Path(workspace_dir).resolve() if workspace_dir else Path.cwd()
    LidUpStudioApp(root, workspace_dir=ws)
    root.mainloop()


def main() -> None:
    ws = sys.argv[1] if len(sys.argv) > 1 else None
    launch_studio(workspace_dir=ws)


if __name__ == "__main__":
    main()
