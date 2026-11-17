"""Interface Gráfica Desktop Nativa (Tkinter Dark Mode) para o Razuk Bot.

# GOLDCODE EXEMPTION: arquivo de UI Tkinter com um único componente coeso (AppWindow).
# A classe detém todos os widgets, estado compartilhado e API pública em um único lugar;
# dividir romperia o encapsulamento e criaria acoplamento ingerenciável entre módulos.

ARQUITETURA CORRIGIDA (Linux/X11):
- AppWindow.__init__() constrói a janela na THREAD PRINCIPAL
- run_mainloop() bloqueia a thread principal com root.mainloop()
- O asyncio (BotOrchestrator) roda em uma THREAD DE BACKGROUND
- Comunicação thread-safe protegida por threading.Lock com snapshots rápidos

Painéis:
- Header: Status de conexão, relógio e tipo de conta
- Gestão de Risco: Banca, Saldo, PnL e Online Learning
- Varredura EMA Wave 34: Pares analisados em tempo real
- Ranking: Melhores pares por assertividade e PnL
- Feed: Histórico recente de trades WIN/LOSS
"""

from datetime import datetime
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, List


class AppWindow:
    """Janela gráfica desktop construída com Tkinter Dark Mode.

    DEVE ser instanciada na thread principal ANTES de iniciar o asyncio.
    O loop asyncio deve rodar em uma thread de background separada.
    """

    def __init__(self, pares: List[str]) -> None:
        """Constrói todos os widgets Tkinter na thread chamadora (principal)."""
        self.pares = pares

        # Dados compartilhados (escritos pelo asyncio, lidos pelo Tkinter via GIL)
        self.banca_inicial: float = 0.0
        self.banca_atual: float = 0.0
        self.lucro_diario: float = 0.0
        self.status_conexao: str = "CONECTANDO..."
        self.conta_tipo: str = "PRACTICE"
        self.par_em_analise: str = "Aguardando..."
        # Online Learning state
        self.learning_estado: str = "ESTÁVEL"   # 'APRENDENDO' | 'ESTÁVEL' | 'SEM APRENDIZADO'
        self.learning_pct: float = 0.0           # 0.0 – 1.0
        self.pares_status: Dict[str, Dict] = {
            p: {"sinal": "NEUTRO", "prob": "-", "relogio": "3h"} for p in pares
        }
        self.pares_stats: Dict[str, Dict] = {
            p: {"wins": 0, "losses": 0, "pnl": 0.0} for p in pares
        }
        self.start_time: float = time.time()
        self.feed_logs: List[Dict[str, str]] = []
        self._lock = threading.Lock()

        # Constrói a janela e todos os widgets na thread principal
        self.root = tk.Tk()
        self.root.title("⚡ RAZUK BOT QUANT — Terminal Executivo")
        self.root.geometry("860x640")
        self.root.configure(bg="#12131C")
        self.root.resizable(True, True)

        self._apply_dark_theme()
        self._build_widgets()

    # ------------------------------------------------------------------
    # Tema e construção de layout
    # ------------------------------------------------------------------

    def _apply_dark_theme(self) -> None:
        """Configura paleta dark mode nos componentes ttk."""
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background="#12131C", foreground="#E0E0E0",
                        font=("Segoe UI", 9))
        style.configure("Treeview", background="#1E1F2E", foreground="#FFFFFF",
                        fieldbackground="#1E1F2E", rowheight=26)
        style.map("Treeview", background=[("selected", "#2A2D4A")])
        style.configure("Treeview.Heading", background="#26283B",
                        foreground="#00E5FF", font=("Segoe UI", 9, "bold"))

    def _build_widgets(self) -> None:
        """Constrói a estrutura de painéis e tabelas estáticas."""
        # ---- Header ----
        hdr = tk.Frame(self.root, bg="#1E1F2E", height=50)
        hdr.pack(fill=tk.X, padx=12, pady=(10, 6))
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⚡ RAZUK BOT QUANT", font=("Segoe UI", 13, "bold"),
                 fg="#00E5FF", bg="#1E1F2E").pack(side=tk.LEFT, padx=12, pady=10)

        self.lbl_status = tk.Label(hdr, text="● CONECTANDO...",
                                   font=("Segoe UI", 10, "bold"),
                                   fg="#FFD600", bg="#1E1F2E")
        self.lbl_status.pack(side=tk.LEFT, padx=16, pady=10)

        self.lbl_clock = tk.Label(hdr, text="🕒 --:--:--",
                                  font=("Segoe UI", 10), fg="#B0BEC5", bg="#1E1F2E")
        self.lbl_clock.pack(side=tk.RIGHT, padx=12, pady=10)

        self.lbl_conta = tk.Label(hdr, text="CONTA: PRACTICE",
                                  font=("Segoe UI", 9), fg="#78909C", bg="#1E1F2E")
        self.lbl_conta.pack(side=tk.RIGHT, padx=6, pady=10)

        # ---- Gestão de Risco ----
        risk_box = tk.LabelFrame(self.root, text=" 💼 Gestão de Risco & Capital ",
                                 font=("Segoe UI", 10, "bold"), fg="#E040FB",
                                 bg="#12131C", padx=10, pady=8)
        risk_box.pack(fill=tk.X, padx=12, pady=4)

        self.lbl_banca_base = tk.Label(risk_box, text="Base: $0.00",
                                       font=("Segoe UI", 10), fg="#FFFFFF", bg="#12131C")
        self.lbl_banca_base.pack(side=tk.LEFT, expand=True)

        self.lbl_tempo_sessao = tk.Label(
            risk_box, text="Tempo de Sessão: 00:00:00",
            font=("Segoe UI", 11, "bold"), fg="#00E5FF", bg="#12131C"
        )
        self.lbl_tempo_sessao.pack(side=tk.LEFT, expand=True)

        self.lbl_pnl = tk.Label(risk_box, text="PnL Hoje: $0.00",
                                 font=("Segoe UI", 11, "bold"), fg="#00E676", bg="#12131C")
        self.lbl_pnl.pack(side=tk.LEFT, expand=True)

        # ---- Indicador Online Learning ----
        learn_frame = tk.Frame(risk_box, bg="#12131C")
        learn_frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 4))

        self.lbl_learn_title = tk.Label(
            learn_frame, text="🧠 Online Learning",
            font=("Segoe UI", 8, "bold"), fg="#B0BEC5", bg="#12131C"
        )
        self.lbl_learn_title.pack(anchor="w")

        bar_frame = tk.Frame(learn_frame, bg="#2A2D4A", height=10, width=150)
        bar_frame.pack(fill=tk.X, pady=(2, 0))
        bar_frame.pack_propagate(False)

        self._learn_bar_bg = bar_frame
        self._learn_bar_fill = tk.Frame(bar_frame, bg="#78909C", height=10)
        self._learn_bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        self.lbl_learn_status = tk.Label(
            learn_frame, text="ESTÁVEL  0%",
            font=("Segoe UI", 8), fg="#78909C", bg="#12131C"
        )
        self.lbl_learn_status.pack(anchor="w")

        self.lbl_stake = tk.Label(risk_box, text="Stake: $2.00 (fixo)",
                                  font=("Segoe UI", 10), fg="#B0BEC5", bg="#12131C")
        self.lbl_stake.pack(side=tk.LEFT, expand=True)

        # ---- Container Central: Pares + Ranking ----
        mid = tk.Frame(self.root, bg="#12131C")
        mid.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        pairs_box = tk.LabelFrame(mid, text=" 🌊 Varredura EMA Wave 34 ",
                                  font=("Segoe UI", 10, "bold"), fg="#00E5FF", bg="#12131C")
        pairs_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self.tree_pairs = ttk.Treeview(pairs_box,
                                       columns=("par", "relogio", "sinal", "prob"),
                                       show="headings", height=6)
        self.tree_pairs.heading("par", text="Ativo")
        self.tree_pairs.heading("relogio", text="Relógio Raghee")
        self.tree_pairs.heading("sinal", text="Sinal")
        self.tree_pairs.heading("prob", text="P(Win)")
        self.tree_pairs.column("par", width=115, anchor="center")
        self.tree_pairs.column("relogio", width=115, anchor="center")
        self.tree_pairs.column("sinal", width=70, anchor="center")
        self.tree_pairs.column("prob", width=65, anchor="center")
        self.tree_pairs.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        rank_box = tk.LabelFrame(mid, text=" 🏆 Melhores Pares ",
                                 font=("Segoe UI", 10, "bold"), fg="#00E676", bg="#12131C")
        rank_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self.tree_rank = ttk.Treeview(rank_box,
                                      columns=("top", "par", "assert", "pnl"),
                                      show="headings", height=6)
        self.tree_rank.heading("top", text="#")
        self.tree_rank.heading("par", text="Ativo")
        self.tree_rank.heading("assert", text="Assertividade")
        self.tree_rank.heading("pnl", text="PnL")
        self.tree_rank.column("top", width=40, anchor="center")
        self.tree_rank.column("par", width=105, anchor="center")
        self.tree_rank.column("assert", width=105, anchor="center")
        self.tree_rank.column("pnl", width=70, anchor="center")
        self.tree_rank.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ---- Feed de Operações ----
        feed_box = tk.LabelFrame(self.root, text=" 📊 Feed de Operações ",
                                 font=("Segoe UI", 10, "bold"), fg="#FFD600", bg="#12131C")
        feed_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 10))

        self.tree_feed = ttk.Treeview(feed_box,
                                      columns=("hora", "par", "dir", "lote", "pnl", "res"),
                                      show="headings", height=4)
        headers = {"hora": "HORA", "par": "ATIVO", "dir": "DIR",
                   "lote": "STAKE", "pnl": "PnL", "res": "RES"}
        for col, heading in headers.items():
            self.tree_feed.heading(col, text=heading)
            self.tree_feed.column(col, width=90, anchor="center")
        self.tree_feed.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Loop de atualização periódica (sempre na thread principal Tkinter)
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Atualiza UI e reagenda para 1 segundo depois com resiliência a falhas."""
        try:
            self._update_ui_elements()
        except Exception:
            pass
        finally:
            self.root.after(1000, self._tick)

    def _update_ui_elements(self) -> None:
        """Injeta dados frescos nos widgets sem recriar nenhum componente."""
        hora = datetime.now().strftime("%H:%M:%S")
        self.lbl_clock.config(text=f"🕒 {hora}")
        self.lbl_conta.config(text=f"CONTA: {self.conta_tipo}")

        # Cor dinâmica do status
        cor_st = "#00E676" if "ONLINE" in self.status_conexao else (
            "#FF1744" if any(w in self.status_conexao for w in ("ERRO", "DESCONECT")) else "#FFD600"
        )
        self.lbl_status.config(
            text=f"● {self.status_conexao}  |  🔍 {self.par_em_analise}", fg=cor_st
        )

        # Métricas de capital e tempo
        cor_pnl = "#00E676" if self.lucro_diario >= 0 else "#FF1744"
        sinal = "+" if self.lucro_diario >= 0 else ""
        self.lbl_banca_base.config(text=f"Base: ${self.banca_inicial:.2f}")

        decorrido = int(time.time() - self.start_time)
        horas = decorrido // 3600
        minutos = (decorrido % 3600) // 60
        segundos = decorrido % 60
        self.lbl_tempo_sessao.config(text=f"Tempo de Sessão: {horas:02d}:{minutos:02d}:{segundos:02d}")

        self.lbl_pnl.config(text=f"PnL: {sinal}${self.lucro_diario:.2f}", fg=cor_pnl)

        # ---- Barra Online Learning ----
        _COR_APRENDENDO = "#00E676"     # verde
        _COR_ESTAVEL = "#78909C"        # cinza
        _COR_SEM = "#FF9100"            # laranja
        if self.learning_estado == "APRENDENDO":
            cor_learn = _COR_APRENDENDO
            icone = "⚡"
        elif self.learning_estado == "SEM APRENDIZADO":
            cor_learn = _COR_SEM
            icone = "⚠"
        else:
            cor_learn = _COR_ESTAVEL
            icone = "●"
        pct_txt = f"{self.learning_pct * 100:.0f}%"
        self.lbl_learn_status.config(
            text=f"{icone} {self.learning_estado}  {pct_txt}", fg=cor_learn
        )
        self._learn_bar_fill.config(bg=cor_learn)
        self._learn_bar_fill.place(x=0, y=0, relheight=1.0, relwidth=max(0.02, self.learning_pct))

        # Snapshot seguro sob lock para evitar colisão com escrita do asyncio
        with self._lock:
            status_items = list(self.pares_status.items())
            stats_items = [(p, dict(s)) for p, s in self.pares_stats.items()]
            feed_items = list(self.feed_logs)

        # Tabela de pares com distinção explícita entre Normal e OTC
        self.tree_pairs.delete(*self.tree_pairs.get_children())
        for par, info in status_items:
            mercado = "OTC" if "-OTC" in par.upper() else "NORMAL"
            par_base = par.replace("-OTC", "").replace("-otc", "")
            rotulo_par = f"{par_base} ({mercado})"
            self.tree_pairs.insert("", "end", values=(
                rotulo_par, info["relogio"], info["sinal"], info["prob"]
            ))

        # Ranking com distinção explícita entre Normal e OTC
        self.tree_rank.delete(*self.tree_rank.get_children())
        ranking = sorted(
            [
                (p, s["wins"] + s["losses"],
                 s["wins"], s["losses"],
                 s["wins"] / (s["wins"] + s["losses"]) * 100 if (s["wins"] + s["losses"]) > 0 else 0.0,
                 s["pnl"])
                for p, s in stats_items
            ],
            key=lambda x: (x[4], x[5]), reverse=True
        )
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for i, (p, tot, w, l, taxa, pnl) in enumerate(ranking[:4], 1):
            med = medals.get(i, f"#{i}")
            mercado = "OTC" if "-OTC" in p.upper() else "NORM"
            par_base = p.replace("-OTC", "").replace("-otc", "")
            rotulo_par = f"{par_base} ({mercado})"
            taxa_txt = f"{taxa:.1f}% ({int(w)}W-{int(l)}L)" if tot > 0 else "Sem ordens"
            pnl_txt = f"{'+'if pnl>=0 else ''}${pnl:.2f}"
            self.tree_rank.insert("", "end", values=(med, rotulo_par, taxa_txt, pnl_txt))

        # Feed de operações
        self.tree_feed.delete(*self.tree_feed.get_children())
        for entry in reversed(feed_items):
            self.tree_feed.insert("", "end", values=(
                entry["hora"], entry["par"], entry["dir"],
                entry["lote"], entry["res"], entry["tag"]
            ))

    def run_mainloop(self) -> None:
        """Inicia o ticker e bloqueia a thread principal com root.mainloop().

        Deve ser chamado APÓS lançar o asyncio em thread de background.
        """
        self._tick()
        self.root.mainloop()

    # ------------------------------------------------------------------
    # API pública — chamada pelo asyncio (thread de background)
    # ------------------------------------------------------------------

    def update_risk(self, banca_inicial: float, banca_atual: float,
                    lucro: float, dd_pct: float = 0.0, max_dd: float = 0.05) -> None:
        """Atualiza métricas de risco (seguro via GIL do Python)."""
        self.banca_inicial = banca_inicial
        self.banca_atual = banca_atual
        self.lucro_diario = lucro

    def update_learning(self, estado: str, pct: float) -> None:
        """Atualiza estado do Online Learning. Chamado pelo asyncio após cada ciclo."""
        self.learning_estado = estado
        self.learning_pct = max(0.0, min(1.0, pct))

    def set_current_analyzing_pair(self, par: str) -> None:
        self.par_em_analise = par

    def sync_active_pairs(self, active_pairs: List[str]) -> None:
        """Sincroniza os pares monitorados exibidos dinamicamente na grade."""
        with self._lock:
            self.pares = list(active_pairs)
            for p in list(self.pares_status.keys()):
                if p not in active_pairs:
                    self.pares_status.pop(p, None)

    def update_pair(self, par: str, sinal: str, prob: float, relogio: str) -> None:
        with self._lock:
            self.pares_status[par] = {
                "sinal": sinal,
                "prob": f"{prob * 100:.1f}%" if prob > 0 else "-",
                "relogio": relogio,
            }

    def add_trade_log(self, par: str, direcao: str, lote: float,
                      pnl: float, was_win: bool) -> None:
        hora = datetime.now().strftime("%H:%M:%S")
        res_str = f"+${pnl:.2f}" if was_win else f"-${abs(pnl):.2f}"
        tag = "WIN ✅" if was_win else "LOSS ❌"
        with self._lock:
            self.feed_logs.append({
                "hora": hora, "par": par.replace("-OTC", ""), "dir": direcao,
                "lote": f"${lote:.2f}", "res": res_str, "tag": tag,
            })
            if len(self.feed_logs) > 6:
                self.feed_logs.pop(0)
            if par not in self.pares_stats:
                self.pares_stats[par] = {"wins": 0, "losses": 0, "pnl": 0.0}
            if was_win:
                self.pares_stats[par]["wins"] += 1
                self.pares_stats[par]["pnl"] += pnl
            else:
                self.pares_stats[par]["losses"] += 1
                self.pares_stats[par]["pnl"] -= abs(pnl)

    def refresh(self) -> None:
        """Compatibilidade: atualização automática via _tick(). Sem ação."""
        pass
