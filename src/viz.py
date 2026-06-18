"""Estilo visual unificado del proyecto.

Principios: títulos que afirman el hallazgo (no describen el eje), ejes
limpios sin bordes superiores/derechos, paleta propia consistente, números
legibles y anotaciones sobre el gráfico cuando ayudan a leerlo.
Toda figura del proyecto debe construirse con estas utilidades.
"""
import matplotlib.pyplot as plt
from matplotlib import ticker

PALETA = {
    "azul": "#2563eb",
    "naranja": "#f59e0b",
    "verde": "#16a34a",
    "rojo": "#dc2626",
    "morado": "#7c3aed",
    "gris": "#9ca3af",
    "gris_oscuro": "#4b5563",
}
CICLO = [PALETA[k] for k in ("azul", "naranja", "verde", "rojo", "morado", "gris")]

# Colores fijos por mercado (consistentes en TODO el proyecto)
MERCADOS = {
    "Philadelphia": PALETA["azul"],
    "Tampa": PALETA["naranja"],
    "New Orleans": PALETA["verde"],
}
COLOR_RESTO = "#d1d5db"


def aplicar_estilo() -> None:
    """Activa el estilo del proyecto (llamar una vez por notebook)."""
    plt.rcParams.update({
        "figure.figsize": (11, 4.5),
        "figure.dpi": 110,
        "axes.prop_cycle": plt.cycler(color=CICLO),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "font.size": 10.5,
        "legend.frameon": False,
        "figure.autolayout": True,
    })


def fmt_miles(ax, eje: str = "y") -> None:
    """Separador de miles en el eje indicado."""
    a = ax.yaxis if eje == "y" else ax.xaxis
    a.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))


def barras_h(ax, etiquetas, valores, resaltar=(), fmt="{:,.0f}", color=None):
    """Barras horizontales con el valor al final; `resaltar` colorea, el resto gris."""
    colores = [
        (MERCADOS.get(e, PALETA["azul"]) if (not resaltar or e in resaltar) else COLOR_RESTO)
        for e in etiquetas
    ] if color is None else color
    bars = ax.barh(list(etiquetas), list(valores), color=colores)
    for b, v in zip(bars, valores):
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2,
                fmt.format(v), va="center", fontsize=9, color=PALETA["gris_oscuro"])
    ax.margins(x=0.12)
    return bars


def anotar(ax, texto, xy, xytext, color=PALETA["gris_oscuro"]):
    """Flecha discreta con texto, para señalar el punto que importa."""
    ax.annotate(texto, xy=xy, xytext=xytext, fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1))


def ccdf(valores):
    """Complementary CDF: P(X >= x). Útil para colas largas en escala log-log."""
    import numpy as np

    v = np.sort(np.asarray(valores))
    p = 1.0 - np.arange(len(v)) / len(v)
    return v, p


def guardar(fig, nombre: str) -> str:
    """Exporta a ``docs/figs/`` y devuelve una ruta relativa portable."""
    from .config import ROOT

    figs = ROOT / "docs" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    ruta = figs / f"{nombre}.png"
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    return str(ruta.relative_to(ROOT))
