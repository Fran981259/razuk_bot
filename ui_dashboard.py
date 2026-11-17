"""Motor de interface visual de terminal (TUI) via Rich.

Apresenta dashboard responsivo em tempo real com auto-adaptação à largura da tela:
- Header: Conta, Conexão, Relógio e Par sendo analisado no momento
- Painel de Capital: Banca base, Saldo atual, Lucro acumulado e Drawdown
- Tabela de Pares: Ativo, Indicador de Análise Ativa, Relógio Raghee Horner, Sinal e P(Win)
- Painel de Desempenho dos Pares: Ranking de assertividade real (Win Rate % e PnL)
- Feed de Operações: Últimos disparos com badge WIN/LOSS e PnL
"""

from datetime import datetime
from typing import Dict, List, Optional

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TerminalUI:
    """Gerenciador de exibição responsiva no terminal usando Rich."""

    def __init__(self, pares: List[str]) -> None:
        """Inicializa a interface de terminal."""
        self.console = Console()
        self.pares = pares
        self.banca_inicial: float = 0.0
        self.banca_atual: float = 0.0
        self.lucro_diario: float = 0.0
        self.drawdown_pct: float = 0.0
        self.max_drawdown_pct: float = 0.05
        self.status_conexao: str = "CONECTANDO..."
        self.conta_tipo: str = "PRACTICE"
        self.par_em_analise: str = "Aguardando vela..."
        self.feed_logs: List[Dict[str, str]] = []
        self.pares_status: Dict[str, Dict[str, str]] = {
            p: {"sinal": "NEUTRO", "prob": "-", "relogio": "3h"} for p in pares
        }
        # Estatísticas acumuladas de desempenho por par
        self.pares_stats: Dict[str, Dict[str, float]] = {
            p: {"wins": 0, "losses": 0, "pnl": 0.0} for p in pares
        }
        self.learning_estado: str = "Pronto"
        self.learning_pct: float = 0.0
        # Contador de trades reais para meta de coleta de dados
        self.total_trades_real: int = 0
        self.trades_target: int = 1500
        # Status do cooldown
        self.cooldown_status: str = "Inativo"
        self.cooldown_remaining: int = 0

    def update_risk(
        self, banca_inicial: float, banca_atual: float, lucro: float, dd_pct: float, max_dd: float
    ) -> None:
        """Atualiza indicadores financeiros de risco."""
        self.banca_inicial = banca_inicial
        self.banca_atual = banca_atual
        self.lucro_diario = lucro
        self.drawdown_pct = dd_pct
        self.max_drawdown_pct = max_dd

    def update_learning(self, estado: str, pct: float) -> None:
        """Atualiza indicadores de estado do retreino contínuo."""
        self.learning_estado = estado
        self.learning_pct = pct

    def update_trade_counter(self, total: int, target: int = 1500) -> None:
        """Atualiza o contador de trades reais e a meta de coleta."""
        self.total_trades_real = total
        self.trades_target = target

    def update_cooldown_status(self, status: str, remaining_minutes: int = 0) -> None:
        """Atualiza o status do cooldown de proteção."""
        self.cooldown_status = status
        self.cooldown_remaining = remaining_minutes

    def sync_active_pairs(self, active_pairs: List[str]) -> None:
        """Sincroniza os pares monitorados exibidos dinamicamente na tabela."""
        self.pares = list(active_pairs)
        for p in list(self.pares_status.keys()):
            if p not in active_pairs:
                self.pares_status.pop(p, None)
        for p in active_pairs:
            if p not in self.pares_status:
                self.pares_status[p] = {"sinal": "NEUTRO", "prob": "-", "relogio": "3h"}
            if p not in self.pares_stats:
                self.pares_stats[p] = {"wins": 0, "losses": 0, "pnl": 0.0}

    def set_current_analyzing_pair(self, par: str) -> None:
        """Define o par que está sendo analisado nesse exato milissegundo."""
        self.par_em_analise = par

    def update_pair(self, par: str, sinal: str, prob: float, relogio: str) -> None:
        """Atualiza o estado de um par específico."""
        self.pares_status[par] = {
            "sinal": sinal,
            "prob": f"{prob * 100:.1f}%" if prob > 0 else "-",
            "relogio": relogio,
        }

    def add_trade_log(self, par: str, direcao: str, lote: float, pnl: float, was_win: bool) -> None:
        """Adiciona resultado de trade e atualiza estatísticas de desempenho do par."""
        hora = datetime.now().strftime("%H:%M:%S")
        res_str = f"+${pnl:.2f}" if was_win else f"-${abs(pnl):.2f}"
        tag = "[bold green]WIN[/bold green]" if was_win else "[bold red]LOSS[/bold red]"
        self.feed_logs.append({
            "hora": hora,
            "par": par.replace("-OTC", ""),
            "dir": direcao,
            "lote": f"${lote:.2f}",
            "res": res_str,
            "tag": tag,
        })
        if len(self.feed_logs) > 6:
            self.feed_logs.pop(0)

        # Atualiza o ranking histórico do par
        if par not in self.pares_stats:
            self.pares_stats[par] = {"wins": 0, "losses": 0, "pnl": 0.0}
        if was_win:
            self.pares_stats[par]["wins"] += 1
            self.pares_stats[par]["pnl"] += pnl
        else:
            self.pares_stats[par]["losses"] += 1
            self.pares_stats[par]["pnl"] -= abs(pnl)

    def _render_header(self) -> Panel:
        """Renderiza barra de topo com o par sendo analisado em destaque."""
        hora_agora = datetime.now().strftime("%H:%M:%S")
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)

        par_analise_txt = f"🔍 [bold yellow]{self.par_em_analise}[/bold yellow]" if self.par_em_analise else "⏳ Aguardando..."

        grid.add_row(
            Text.from_markup(f"⚡ [bold cyan]RAZUK BOT QUANT[/bold cyan] | [bold yellow]{self.conta_tipo}[/bold yellow]"),
            Text.from_markup(f"Analisando Agora: {par_analise_txt}"),
            Text.from_markup(f"🕒 [cyan]{hora_agora}[/cyan]"),
        )
        return Panel(grid, box=ROUNDED, border_style="cyan")

    def _render_risk_table(self) -> Table:
        """Renderiza tabela de capital, limites de risco e status do cooldown."""
        pnl_color = "green" if self.lucro_diario >= 0 else "red"
        pnl_signal = "+" if self.lucro_diario >= 0 else ""
        t = Table(box=ROUNDED, expand=True, title="💼 Gestão de Risco & Capital", title_style="bold magenta")
        t.add_column("Banca Base", justify="center")
        t.add_column("Saldo Atual", justify="center")
        t.add_column("PnL Hoje", justify="center")
        t.add_column("Drawdown", justify="center")
        t.add_column("Limite Stop", justify="center")
        t.add_column("Cooldown", justify="center")

        # Status do cooldown com cor
        if self.cooldown_remaining > 0:
            cooldown_fmt = f"[bold red]⏳ {self.cooldown_remaining}min[/bold red]"
        else:
            cooldown_fmt = "[dim]Inativo[/dim]"

        t.add_row(
            f"${self.banca_inicial:.2f}" if self.banca_inicial > 0 else "Aguardando...",
            f"[bold cyan]${self.banca_atual:.2f}[/bold cyan]" if self.banca_atual > 0 else "Conectando...",
            f"[bold {pnl_color}]{pnl_signal}${self.lucro_diario:.2f}[/bold {pnl_color}]",
            f"[yellow]{self.drawdown_pct:.2%}[/yellow]",
            f"[bold red]{self.max_drawdown_pct:.1%}[/bold red]",
            cooldown_fmt,
        )
        return t

    def _render_pairs_table(self) -> Table:
        """Renderiza tabela de ativos com indicador visual de análise ativa."""
        t = Table(box=ROUNDED, expand=True, title="🌊 Varredura EMA Wave 34 (Raghee Horner)", title_style="bold cyan")
        t.add_column("Status", justify="center")
        t.add_column("Par / Ativo", justify="left")
        t.add_column("Relógio", justify="center")
        t.add_column("Sinal", justify="center")
        t.add_column("P(Win) ML", justify="center")

        for par, info in self.pares_status.items():
            par_short = par.replace("-OTC", " [dim](OTC)[/dim]")
            is_active = (par == self.par_em_analise)
            status_icon = "▶ [bold yellow]ANALISANDO[/bold yellow]" if is_active else "[dim]OK[/dim]"

            sinal = info["sinal"]
            relogio = info["relogio"]
            prob = info["prob"]

            sinal_fmt = "[bold green]▲ CALL[/bold green]" if "CALL" in sinal else (
                "[bold red]▼ PUT[/bold red]" if "PUT" in sinal else "[dim]NEUTRO[/dim]"
            )

            if "12h" in relogio:
                relogio_fmt = "[green]🕐 12h-2h (Alta)[/green]"
            elif "5h" in relogio or "3h30" in relogio:
                relogio_fmt = "[red]🕔 3h30-5h (Baixa)[/red]"
            else:
                relogio_fmt = "[dim]🕒 3h (Lateral)[/dim]"

            t.add_row(status_icon, par_short, relogio_fmt, sinal_fmt, prob)
        return t

    def _render_performance_table(self) -> Table:
        """Renderiza ranking dos pares com melhor desempenho e assertividade."""
        t = Table(box=ROUNDED, expand=True, title="🏆 Ranking de Desempenho dos Pares", title_style="bold green")
        t.add_column("Top", justify="center")
        t.add_column("Ativo", justify="left")
        t.add_column("Trades", justify="center")
        t.add_column("Assertividade", justify="center")
        t.add_column("PnL Acumulado", justify="center")

        # Ordena pares por assertividade (%) e PnL
        ranking = []
        for p, s in self.pares_stats.items():
            tot = s["wins"] + s["losses"]
            taxa = (s["wins"] / tot * 100) if tot > 0 else 0.0
            ranking.append((p, tot, s["wins"], s["losses"], taxa, s["pnl"]))

        # Ordena prioritariamente por maior taxa e depois maior PnL
        ranking.sort(key=lambda x: (x[4], x[5]), reverse=True)

        for i, (p, tot, w, l, taxa, pnl) in enumerate(ranking[:4], start=1):
            cor_taxa = "bold green" if taxa >= 55.0 else ("yellow" if taxa >= 50.0 else "red")
            cor_pnl = "green" if pnl >= 0 else "red"
            sinal_pnl = "+" if pnl >= 0 else ""
            medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"#{i}"))
            p_short = p.replace("-OTC", "")

            taxa_str = f"[{cor_taxa}]{taxa:.1f}%[/] ({int(w)}W-{int(l)}L)" if tot > 0 else "[dim]Sem ordens[/dim]"
            pnl_str = f"[{cor_pnl}]{sinal_pnl}${pnl:.2f}[/]"

            t.add_row(medal, p_short, str(int(tot)), taxa_str, pnl_str)
        return t

    def _render_data_collection_panel(self) -> Panel:
        """Renderiza painel de coleta de dados com barra de progresso até a meta."""
        pct = min(self.total_trades_real / max(self.trades_target, 1), 1.0)
        bar_width = 20
        filled = int(bar_width * pct)
        bar_str = "█" * filled + "░" * (bar_width - filled)
        color = "bold green" if pct >= 1.0 else ("green" if pct >= 0.5 else "yellow")
        progress_text = Text.from_markup(
            f"📊 Coleta: [bold cyan]{self.total_trades_real}[/]/{self.trades_target} trades "
            f"[{color}]{bar_str}[/] {pct:.0%}"
        )
        return Panel(progress_text, box=ROUNDED, border_style="cyan")

    def _render_trades_feed(self) -> Table:
        """Renderiza feed de ordens recentes."""
        t = Table(box=ROUNDED, expand=True, title="📊 Feed de Operações Recentes", title_style="bold yellow")
        t.add_column("Hora", justify="center")
        t.add_column("Ativo", justify="center")
        t.add_column("Dir", justify="center")
        t.add_column("Lote", justify="center")
        t.add_column("PnL", justify="center")
        t.add_column("Status", justify="center")

        if not self.feed_logs:
            t.add_row("-", "-", "-", "-", "-", "[dim]Aguardando ordens...[/dim]")
        else:
            for l in reversed(self.feed_logs):
                t.add_row(l["hora"], l["par"], l["dir"], l["lote"], l["res"], l["tag"])
        return t

    def refresh(self) -> None:
        """Limpa e renderiza o dashboard responsivo com as novas seções."""
        self.console.clear()
        self.console.print(self._render_header())
        self.console.print(self._render_risk_table())

        largura_terminal = self.console.width
        if largura_terminal >= 105:
            # Lado a lado: Varredura + Desempenho
            top_cols = Table.grid(expand=True)
            top_cols.add_column(ratio=1)
            top_cols.add_column(ratio=1)
            top_cols.add_row(self._render_pairs_table(), self._render_performance_table())
            self.console.print(top_cols)
            self.console.print(self._render_data_collection_panel())
            self.console.print(self._render_trades_feed())
        else:
            # Empilhado verticalmente para telas menores
            self.console.print(self._render_pairs_table())
            self.console.print(self._render_performance_table())
            self.console.print(self._render_data_collection_panel())
            self.console.print(self._render_trades_feed())
