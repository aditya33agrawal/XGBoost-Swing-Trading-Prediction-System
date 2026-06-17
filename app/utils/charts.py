"""Shared Plotly chart helpers for the Streamlit dashboard."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

_GREEN  = "#22c55e"
_RED    = "#ef4444"
_AMBER  = "#f59e0b"
_BLUE   = "#3b82f6"
_GRAY   = "#6b7280"
_BG     = "#0f172a"
_GRID   = "#1e293b"


def _base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_GRID,
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=50, r=20, t=40, b=40),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------
def equity_curve_chart(trades_df: pd.DataFrame, initial_capital: float = 1_000_000) -> go.Figure:
    """Cumulative portfolio value from closed trades."""
    closed = trades_df[trades_df["status"] == "closed"].copy()
    if closed.empty:
        fig = go.Figure()
        fig.update_layout(title="Equity Curve (no closed trades yet)", **_base_layout())
        return fig

    closed["exit_date"] = pd.to_datetime(closed["exit_date"])
    closed = closed.sort_values("exit_date")
    closed["cum_pnl"]   = closed["pnl"].fillna(0).cumsum()
    closed["equity"]    = initial_capital + closed["cum_pnl"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=closed["exit_date"], y=closed["equity"],
        mode="lines+markers",
        line=dict(color=_GREEN, width=2),
        marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.08)",
        name="Portfolio Value",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Value: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=initial_capital, line_dash="dash", line_color=_GRAY, opacity=0.6,
                  annotation_text="Initial Capital")
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date", yaxis_title="Value (₹)",
        **_base_layout(),
    )
    return fig


# ---------------------------------------------------------------------------
# IC over time
# ---------------------------------------------------------------------------
def ic_trend_chart(ic_df: pd.DataFrame) -> go.Figure:
    """IC by week with warning zones."""
    if ic_df.empty:
        fig = go.Figure()
        fig.update_layout(title="IC Trend (no data yet)", **_base_layout())
        return fig

    ic_df = ic_df.copy()
    ic_df["ic"] = pd.to_numeric(ic_df["ic"], errors="coerce")
    colors = [
        _GREEN if (v or 0) >= 0.01 else (_AMBER if (v or 0) >= 0.0 else _RED)
        for v in ic_df["ic"]
    ]

    fig = go.Figure()
    # Warning zone
    fig.add_hrect(y0=-1, y1=0, fillcolor="rgba(239,68,68,0.05)", line_width=0)
    fig.add_hrect(y0=0, y1=0.01, fillcolor="rgba(245,158,11,0.05)", line_width=0)
    # IC bars
    fig.add_trace(go.Bar(
        x=ic_df["week_start"], y=ic_df["ic"],
        marker_color=colors,
        name="Weekly IC",
        hovertemplate="<b>%{x}</b><br>IC: %{y:.4f}<br>N: %{customdata}<extra></extra>",
        customdata=ic_df["n_obs"],
    ))
    # Reference lines
    fig.add_hline(y=0.01, line_dash="dash", line_color=_GREEN,  opacity=0.5, annotation_text="Target (0.01)")
    fig.add_hline(y=0.00, line_dash="solid", line_color=_GRAY, opacity=0.3)

    fig.update_layout(
        title="Information Coefficient by Week",
        xaxis_title="Week", yaxis_title="Spearman IC",
        **_base_layout(),
    )
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def feature_importance_chart(fi_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    if fi_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Feature Importance (no data)", **_base_layout())
        return fig

    score_col = "importance_score" if "importance_score" in fi_df.columns else "score"
    name_col  = "feature_name"     if "feature_name"     in fi_df.columns else "feature"
    fi_df = fi_df.nlargest(top_n, score_col).sort_values(score_col)

    fig = go.Figure(go.Bar(
        x=fi_df[score_col],
        y=fi_df[name_col],
        orientation="h",
        marker_color=_BLUE,
        hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance Score",
        height=max(400, top_n * 22),
        **_base_layout(),
    )
    return fig


# ---------------------------------------------------------------------------
# Win/loss breakdown
# ---------------------------------------------------------------------------
def exit_reason_chart(trades_df: pd.DataFrame) -> go.Figure:
    closed = trades_df[trades_df["status"] == "closed"]
    if closed.empty:
        fig = go.Figure()
        fig.update_layout(title="Exit Reasons (no closed trades)", **_base_layout())
        return fig

    counts = closed["exit_reason"].value_counts()
    colors_map = {"target": _GREEN, "stop": _RED, "expired": _AMBER, "manual": _GRAY}
    colors = [colors_map.get(r, _BLUE) for r in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=colors),
        hole=0.4,
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(title="Exit Reason Breakdown", **_base_layout())
    return fig


# ---------------------------------------------------------------------------
# Directional accuracy over time
# ---------------------------------------------------------------------------
def dir_accuracy_chart(ic_df: pd.DataFrame) -> go.Figure:
    if ic_df.empty or "dir_accuracy" not in ic_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Directional Accuracy (no data)", **_base_layout())
        return fig

    ic_df = ic_df.copy()
    ic_df["dir_accuracy"] = pd.to_numeric(ic_df["dir_accuracy"], errors="coerce") * 100

    fig = go.Figure(go.Bar(
        x=ic_df["week_start"],
        y=ic_df["dir_accuracy"],
        marker_color=[_GREEN if (v or 0) >= 53 else _AMBER for v in ic_df["dir_accuracy"]],
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1f}%%<extra></extra>",
    ))
    fig.add_hline(y=53, line_dash="dash", line_color=_GREEN, opacity=0.5,
                  annotation_text="Target (53%)")
    fig.add_hline(y=50, line_dash="solid", line_color=_GRAY, opacity=0.3)
    fig.update_layout(
        title="Directional Accuracy by Week",
        xaxis_title="Week", yaxis_title="Accuracy (%)",
        yaxis_range=[40, 70],
        **_base_layout(),
    )
    return fig


# ---------------------------------------------------------------------------
# P&L distribution
# ---------------------------------------------------------------------------
def pnl_distribution_chart(trades_df: pd.DataFrame) -> go.Figure:
    closed = trades_df[trades_df["status"] == "closed"].dropna(subset=["pnl_pct"])
    if closed.empty:
        fig = go.Figure()
        fig.update_layout(title="P&L Distribution (no closed trades)", **_base_layout())
        return fig

    fig = go.Figure(go.Histogram(
        x=closed["pnl_pct"],
        nbinsx=20,
        marker_color=_BLUE,
        hovertemplate="P&L bucket: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=_GRAY, line_dash="solid")
    fig.update_layout(
        title="Closed Trade P&L Distribution",
        xaxis_title="P&L (%)", yaxis_title="Count",
        **_base_layout(),
    )
    return fig
