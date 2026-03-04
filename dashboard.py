import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from backtest import BacktestResult, run_backtest
from genetic_optimizer import sharpe

log = logging.getLogger(__name__)

DARK  = "#0d1117"
PANEL = "#161b22"
GREEN = "#2ea043"
RED   = "#f85149"
BLUE  = "#58a6ff"
GREY  = "#8b949e"
WHITE = "#e6edf3"


def _ax(fig, gs, row, col, colspan=1):
    ax = fig.add_subplot(gs[row, col:col+colspan])
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.tick_params(colors=GREY, labelsize=8)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.title.set_color(WHITE)
    return ax


def plot(result: BacktestResult, save_path="dashboard.png") -> str:
    fig = plt.figure(figsize=(16, 10), facecolor=DARK)
    fig.suptitle(
        f"SENSE-ACT  |  {result.ticker}  |  {result.period}",
        color=WHITE, fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    eq = np.array(result.equity_curve)

    ax1 = _ax(fig, gs, 0, 0, colspan=3)
    c = GREEN if eq[-1] >= 0 else RED
    ax1.plot(eq, color=c, linewidth=1.4)
    ax1.fill_between(range(len(eq)), eq, alpha=0.12, color=c)
    ax1.axhline(0, color=GREY, linewidth=0.5, linestyle="--")
    ax1.set_title("Equity Curve", fontsize=11)
    ax1.set_ylabel("cumulative PnL")

    ax2 = _ax(fig, gs, 1, 0, colspan=2)
    dp = np.array(result.daily_pnl)
    if len(dp):
        colors = [GREEN if v >= 0 else RED for v in dp]
        ax2.bar(range(len(dp)), dp, color=colors, width=0.8)
        ax2.axhline(0, color=GREY, linewidth=0.5)
    ax2.set_title("Daily PnL", fontsize=11)
    ax2.set_ylabel("PnL")

    ax3 = _ax(fig, gs, 1, 2)
    if result.n_trades:
        wedges, _, autotexts = ax3.pie(
            [result.n_wins, result.n_losses],
            labels=["Wins", "Losses"],
            colors=[GREEN, RED],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": WHITE, "fontsize": 8},
        )
        for at in autotexts:
            at.set_color(DARK)
    ax3.set_title(f"Win Rate  {result.win_rate:.1f}%", fontsize=11)

    ax4 = _ax(fig, gs, 2, 0)
    ax4.axis("off")
    rows = [
        ("Total PnL",    f"{result.total_pnl:+.4f}", GREEN if result.total_pnl >= 0 else RED),
        ("Sharpe",       f"{result.sharpe_ratio:.4f}", WHITE),
        ("Max DD",       f"{result.max_drawdown:.2%}", RED if result.max_drawdown < -0.10 else WHITE),
        ("Signals",      str(result.n_signals), WHITE),
        ("Trades",       str(result.n_trades), WHITE),
    ]
    y = 0.92
    for label, val, color in rows:
        ax4.text(0.05, y, label, color=GREY, fontsize=9, transform=ax4.transAxes)
        ax4.text(0.62, y, val, color=color, fontsize=9, fontweight="bold", transform=ax4.transAxes)
        y -= 0.18

    ax5 = _ax(fig, gs, 2, 1)
    bars = ax5.bar(["avg win", "avg loss"], [result.avg_win, result.avg_loss],
                   color=[GREEN, RED], width=0.5)
    ax5.axhline(0, color=GREY, linewidth=0.5)
    for bar, val in zip(bars, [result.avg_win, result.avg_loss]):
        ax5.text(bar.get_x() + bar.get_width() / 2, val,
                 f"{val:+.4f}", ha="center",
                 va="bottom" if val >= 0 else "top",
                 color=WHITE, fontsize=8)
    ax5.set_title("Avg Win vs Avg Loss", fontsize=11)

    ax6 = _ax(fig, gs, 2, 2)
    if len(eq) > 32:
        rets   = np.diff(eq)
        window = min(30, len(rets) - 1)
        roll   = []
        for i in range(window, len(rets)):
            w  = rets[i - window:i]
            sd = w.std()
            roll.append(w.mean() / sd * np.sqrt(252) if sd > 1e-10 else 0.0)
        ax6.plot(roll, color=BLUE, linewidth=1)
        ax6.axhline(0, color=GREY, linewidth=0.5, linestyle="--")
        ax6.axhline(1, color=GREEN, linewidth=0.5, linestyle="--", alpha=0.5)
    ax6.set_title("Rolling Sharpe (30)", fontsize=11)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    log.info("saved → %s", save_path)
    print(f"dashboard → {save_path}")
    return save_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    r = run_backtest("XOM", "2y")
    plot(r)
