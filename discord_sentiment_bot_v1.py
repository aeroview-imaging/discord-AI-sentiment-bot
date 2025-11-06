"""
Discord AI Sentiment Bot v1 — Hybrid, Short-Form Embed

Features
- Slash command: /sentiment <ticker>
- Live data via yfinance: price, RSI(14), ATR(14), 20‑EMA
- Options snapshot: nearest expiry (≥7d), PCR (OI/Vol), max OI strike/type
- GPT analysis: structured sections (Market, Technical, Options, Outlook) with confidence
- ATR-based entry/stop/TP; AI used as a **filter/sizer** (not to set numeric stops)
- Auto‑truncate AI text to fit Discord field limits (no more 1024‑char errors)
- Clean embed with bold headers and emojis

Setup
1) pip install -U discord.py python-dotenv yfinance pandas numpy openai
2) .env (same folder):
   DISCORD_TOKEN=your_discord_bot_token
   OPENAI_API_KEY=your_openai_api_key
3) Invite bot with scopes: bot, applications.commands; perms: Send Messages, Embed Links, Read Message History
4) Run: python discord_sentiment_bot_v1.py
"""

import os
import math
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

import discord
from discord import app_commands

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# ---------------------------
# Config & Logging
# ---------------------------
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("sentiment-bot")

client_oa = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        client_oa = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.warning(f"OpenAI init failed: {e}")

# ---------------------------
# Discord Bot Setup
# ---------------------------
intents = discord.Intents.default()
intents.guilds = True
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

# ---------------------------
# Indicator Utilities
# ---------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    s = series.dropna()
    if len(s) < period + 1:
        return float("nan")
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else float("nan")


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    if df.empty or len(df) < period + 1:
        return float("nan")
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    val = atr.iloc[-1]
    return float(val) if pd.notna(val) else float("nan")


def compute_ema(series: pd.Series, span: int = 20) -> float:
    if series.dropna().empty:
        return float("nan")
    ema = series.ewm(span=span, adjust=False).mean()
    val = ema.iloc[-1]
    return float(val) if pd.notna(val) else float("nan")

# ---------------------------
# Data Fetchers
# ---------------------------
async def fetch_price_history(ticker: str, lookback_days: int = 120) -> Tuple[pd.DataFrame, float]:
    tk = yf.Ticker(ticker)
    hist = tk.history(period=f"{lookback_days}d")
    last_price = float(hist['Close'].iloc[-1]) if not hist.empty else float("nan")
    return hist, last_price


async def fetch_options_summary(ticker: str) -> Dict[str, Optional[float]]:
    result = {"expiry": None, "pcr_oi": None, "pcr_vol": None, "max_oi_strike": None, "max_oi_type": None}
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options or []
        if not expiries:
            return result
        # pick nearest expiry ≥7 days out when possible
        picked = None
        from datetime import datetime as dt
        today = dt.utcnow().date()
        for e in expiries:
            try:
                d = dt.strptime(e, "%Y-%m-%d").date()
            except ValueError:
                continue
            if (d - today).days >= 7:
                picked = e
                break
        if picked is None:
            picked = expiries[0]
        chain = tk.option_chain(picked)
        calls, puts = chain.calls, chain.puts
        # sums
        p_oi = float(puts.get('openInterest', pd.Series(dtype=float)).fillna(0).sum())
        c_oi = float(calls.get('openInterest', pd.Series(dtype=float)).fillna(0).sum())
        p_vol = float(puts.get('volume', pd.Series(dtype=float)).fillna(0).sum())
        c_vol = float(calls.get('volume', pd.Series(dtype=float)).fillna(0).sum())
        pcr_oi = (p_oi / c_oi) if c_oi > 0 else None
        pcr_vol = (p_vol / c_vol) if c_vol > 0 else None
        # max OI node
        max_call_row = calls.loc[calls['openInterest'].idxmax()] if not calls.empty else None
        max_put_row = puts.loc[puts['openInterest'].idxmax()] if not puts.empty else None
        if max_call_row is not None and (max_put_row is None or max_call_row['openInterest'] >= max_put_row['openInterest']):
            max_oi_strike, max_oi_type = float(max_call_row['strike']), "CALL"
        elif max_put_row is not None:
            max_oi_strike, max_oi_type = float(max_put_row['strike']), "PUT"
        else:
            max_oi_strike, max_oi_type = None, None
        result.update({
            "expiry": picked,
            "pcr_oi": round(pcr_oi, 3) if pcr_oi is not None else None,
            "pcr_vol": round(pcr_vol, 3) if pcr_vol is not None else None,
            "max_oi_strike": max_oi_strike,
            "max_oi_type": max_oi_type,
        })
    except Exception as e:
        logger.warning(f"Options fetch failed for {ticker}: {e}")
    return result


async def fetch_headlines(ticker: str, max_items: int = 6) -> List[str]:
    out: List[str] = []
    try:
        news = yf.Ticker(ticker).news or []
        for item in news[:max_items]:
            title = item.get("title")
            if title:
                out.append(title.strip())
    except Exception as e:
        logger.warning(f"Headlines fetch failed for {ticker}: {e}")
    return out

# ---------------------------
# AI Sentiment (LLM)
# ---------------------------
async def ai_sentiment_summary(
    ticker: str,
    headlines: List[str],
    rsi: float,
    atr: float,
    price: float,
    options_info: Dict[str, Optional[float]],
) -> Tuple[str, float, str]:
    """Return (formatted_text, confidence_0_100, sentiment_label). Fallback to heuristic if no OpenAI."""
    # Heuristic fallback
    if client_oa is None:
        bias = 0.0
        if pd.notna(rsi):
            if rsi < 35: bias += 0.25
            elif rsi > 65: bias -= 0.25
        pcr = options_info.get("pcr_oi")
        if pcr is not None:
            if pcr < 0.7: bias += 0.15
            elif pcr > 1.2: bias -= 0.15
        conf = max(5, min(95, 50 + 40 * bias))
        label = "Slightly Bullish" if bias > 0.05 else ("Slightly Bearish" if bias < -0.05 else "Neutral")
        text = (
            f"📈 **Market Overview:** heuristic only; no strong catalyst from headlines.\n"
            f"🧭 **Technical Picture:** RSI {rsi:.1f}, ATR {atr:.3f}.\n"
            f"🧩 **Options Insight:** PCR(OI) {pcr if pcr is not None else 'n/a'}.\n"
            f"🔮 **AI Summary & Outlook:** {label} bias at price ${price:.2f}."
        )
        return text, float(round(conf, 1)), label

    # With OpenAI
    headlines_text = "\n".join(f"- {h}" for h in headlines) or "(no recent headlines)"
    sys = (
        "You are a professional equity analyst. Produce a concise, structured report with 4 sections: \n"
        "Market Overview, Technical Picture, Options Insight, AI Summary & Outlook. \n"
        "Return JSON: {\"sections\": {\"market\": str, \"technical\": str, \"options\": str, \"outlook\": str}, \"sentiment_label\": one of [Bullish, Slightly Bullish, Neutral, Slightly Bearish, Bearish], \"confidence\": number 0-100}.\n"
        "Keep each section punchy (1-3 sentences)."
    )
    user = (
        f"Ticker: {ticker}\n"
        f"Price: {price:.2f}\nRSI14: {rsi:.1f}\nATR14: {atr:.3f}\n"
        f"Options: {options_info}\n"
        f"Headlines:\n{headlines_text}"
    )
    try:
        resp = client_oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.3,
            max_tokens=520,
        )
        raw = resp.choices[0].message.content.strip()
        import json
        data = json.loads(raw)
        sec = data.get("sections", {})
        label = str(data.get("sentiment_label", "Neutral"))
        conf = float(data.get("confidence", 50))
        # Build formatted text with emojis and bold headers
        text = (
            f"🌐 **Market Overview:** {sec.get('market', '')}\n"
            f"🧭 **Technical Picture:** {sec.get('technical', '')}\n"
            f"🧩 **Options Insight:** {sec.get('options', '')}\n"
            f"🔮 **AI Summary & Outlook:** {sec.get('outlook', '')}"
        ).strip()
        return text, max(0.0, min(100.0, conf)), label
    except Exception as e:
        logger.warning(f"OpenAI parse failed: {e}")
        # graceful fallback
        fallback = (
            f"🌐 **Market Overview:** (AI unavailable)\n"
            f"🧭 **Technical Picture:** RSI {rsi:.1f}, ATR {atr:.3f}.\n"
            f"🧩 **Options Insight:** PCR(OI) {options_info.get('pcr_oi')}\n"
            f"🔮 **AI Summary & Outlook:** Neutral"
        )
        return fallback, 50.0, "Neutral"

# ---------------------------
# Trade Levels & Helpers
# ---------------------------

def propose_levels(last_price: float, atr: float, stop_mult: float = 1.5, tp_mult: float = 2.2) -> Dict[str, float]:
    if not (last_price and last_price > 0):
        return {"entry": float("nan"), "stop": float("nan"), "take_profit": float("nan"), "rr": float("nan")}
    if not (atr and atr > 0):
        stop = round(last_price * 0.95, 4)
        tp = round(last_price * 1.08, 4)
    else:
        stop = max(0.01, last_price - stop_mult * atr)
        tp = last_price + tp_mult * atr
    rr = (tp - last_price) / (last_price - stop) if (last_price - stop) > 0 else float("nan")
    return {"entry": round(last_price, 4), "stop": round(stop, 4), "take_profit": round(tp, 4), "rr": round(rr, 2)}


def clamp_field(text: str, limit: int = 1024, reserve: int = 0) -> str:
    """Ensure a Discord embed field stays under limit. Reserve lets us keep space for bold/labels."""
    max_len = max(0, limit - reserve)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."

# ---------------------------
# Embed Builder
# ---------------------------

def build_embed(
    ticker: str,
    ai_text: str,
    ai_conf: float,
    label: str,
    last_price: float,
    rsi: float,
    atr: float,
    ema20: float,
    options_info: Dict[str, Optional[float]],
    levels: Dict[str, float],
    eligible: bool,
) -> discord.Embed:
    emb = discord.Embed(title=f"{ticker.upper()} — AI Sentiment Snapshot", color=discord.Color.dark_blue(), timestamp=datetime.now(timezone.utc))
    # Main AI report (shortened)
    main_text = clamp_field(ai_text, limit=1024)
    emb.add_field(name="🧠 AI Analysis", value=main_text, inline=False)
    emb.add_field(name="🤖 AI Confidence", value=f"{ai_conf:.1f}/100\nLabel: **{label}**", inline=True)

    tech = (
        f"**Last:** ${last_price:.4f}\n"
        f"**RSI(14):** {('%.1f' % rsi) if pd.notna(rsi) else 'n/a'}\n"
        f"**ATR(14):** {('%.4f' % atr) if pd.notna(atr) else 'n/a'}\n"
        f"**EMA(20):** {('%.4f' % ema20) if pd.notna(ema20) else 'n/a'}"
    )
    emb.add_field(name="📊 Technicals", value=clamp_field(tech), inline=True)

    opt = (
        f"**Expiry:** {options_info.get('expiry') or 'n/a'}\n"
        f"**PCR(OI):** {options_info.get('pcr_oi') if options_info.get('pcr_oi') is not None else 'n/a'}\n"
        f"**PCR(Vol):** {options_info.get('pcr_vol') if options_info.get('pcr_vol') is not None else 'n/a'}\n"
        f"**Max OI:** {options_info.get('max_oi_type') or 'n/a'} @ {options_info.get('max_oi_strike') or '—'}"
    )
    emb.add_field(name="🧩 Options", value=clamp_field(opt), inline=True)

    plan = (
        f"**Entry:** ${levels['entry']:.4f}\n"
        f"**Stop:** ${levels['stop']:.4f}\n"
        f"**Target:** ${levels['take_profit']:.4f}\n"
        f"**R:R:** {levels['rr']}\n"
        f"**Trade Check:** {'✅ Eligible' if eligible else '⚠️ Filtered'} (AI≥60 & Price≥EMA20)"
    )
    emb.add_field(name="🎯 Plan (ATR-based)", value=clamp_field(plan), inline=False)

    emb.set_footer(text="Not financial advice. Manage risk.")
    return emb

# ---------------------------
# Slash Command
# ---------------------------
@tree.command(name="sentiment", description="Get AI sentiment, options skew, and ATR-based levels for a ticker")
@app_commands.describe(ticker="e.g., AAPL")
async def sentiment_slash(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer(thinking=True, ephemeral=False)
    t0 = time.time()
    t = ticker.strip().upper()
    try:
        hist, last_price = await fetch_price_history(t)
        if hist.empty or not last_price or math.isnan(last_price):
            await interaction.followup.send(f"Couldn't fetch price data for `{t}`. Check the symbol and try again.")
            return
        rsi = compute_rsi(hist['Close'], 14)
        atr = compute_atr(hist, 14)
        ema20 = compute_ema(hist['Close'], 20)
        options_info = await fetch_options_summary(t)
        headlines = await fetch_headlines(t)

        ai_text, ai_conf, label = await ai_sentiment_summary(t, headlines, rsi, atr, last_price, options_info)
        levels = propose_levels(last_price, atr)

        # Hybrid trade eligibility (simple filter)
        eligible = (ai_conf >= 60) and (pd.notna(ema20) and last_price >= ema20)

        emb = build_embed(t, ai_text, ai_conf, label, last_price, rsi, atr, ema20, options_info, levels, eligible)
        took = time.time() - t0
        emb.add_field(name="⏱ Latency", value=f"{took:.1f}s", inline=True)
        await interaction.followup.send(embed=emb)
    except Exception as e:
        logger.exception("/sentiment failed")
        await interaction.followup.send(f"Something went wrong: `{e}`")

# ---------------------------
# Ready & Sync
# ---------------------------
@bot.event
async def on_ready():
    try:
        synced = await tree.sync()
        logger.info(f"Synced {len(synced)} command(s). Logged in as {bot.user}.")
    except Exception as e:
        logger.error(f"Command sync failed: {e}")

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise SystemExit("Missing DISCORD_TOKEN in environment.")
    bot.run(DISCORD_TOKEN)
