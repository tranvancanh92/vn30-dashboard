"""
VN30 Real-time Dashboard — Vùng Mua/Bán Sớm
============================================
- Vùng XANH = điểm mua sớm (EMA20 + MACD↑ + RSI<70)
- Vùng ĐỎ   = điểm bán sớm (dưới EMA20 + MACD↓ + RSI>30)
- Ngày đáo hạn phái sinh (countdown)
- Danh sách quét tín hiệu MUA/BÁN toàn VN30
- Lưu giá 2 ngày liên tiếp
- Filter khung thời gian: 1H / 4H / 1D
Chạy: streamlit run Dashboard.py
"""

import time
import logging
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from vnstock3 import Quote

# ── Logging ──
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("VN30Dash")

# ── Hằng số ──
VN30_LIST = [
    "ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG",
    "MBB","MSN","MWG","PLX","POW","SAB","SHB","SSB","SSI","STB",
    "TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE",
]

INTERVAL_MAP = {"1W": "1W", "1D": "1D", "1H": "1H"}
DAYS_MAP     = {"1W": 1000, "1D": 300,  "1H": 10}

# ══════════════════════════════════════════
# 1. Ngày đáo hạn phái sinh
# ══════════════════════════════════════════
def get_expiry():
    today = date.today()
    def third_thursday(y, m):
        d = date(y, m, 1)
        d += timedelta(days=(3 - d.weekday() + 7) % 7)
        return d + timedelta(weeks=2)

    exp = third_thursday(today.year, today.month)
    if today > exp:
        nm = today.month % 12 + 1
        ny = today.year + (1 if today.month == 12 else 0)
        exp = third_thursday(ny, nm)
    return exp, (exp - today).days


# ══════════════════════════════════════════
# 2. Tải dữ liệu — cache 60s (bớt lâu hơn)
# ══════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def load_data(symbol: str, interval: str = "1D") -> pd.DataFrame:
    try:
        days  = DAYS_MAP.get(interval, 300)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end   = datetime.now().strftime("%Y-%m-%d")
        q     = Quote(symbol=symbol, source="VCI")
        df    = q.history(start=start, end=end, interval=interval)
        if df is None or df.empty:
            logger.warning(f"{symbol} returned empty data")
            return pd.DataFrame()
        df.columns = [c.capitalize() for c in df.columns]
        dc = "Date" if "Date" in df.columns else "Time"
        df["Date"] = pd.to_datetime(df[dc])
        return df.set_index("Date").sort_index()
    except Exception as e:
        logger.warning("%s %s: %s", symbol, interval, str(e)[:100])
        return pd.DataFrame()


# ══════════════════════════════════════════
# 3. Tính chỉ báo — cache theo symbol+interval
# ══════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def calc_indicators_cached(symbol: str, interval: str) -> pd.DataFrame:
    df = load_data(symbol, interval)
    return _calc_indicators(df)

def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 30:
        return df
    try:
        df = df.copy()
        # EMA + BB
        df["EMA20"]    = df["Close"].ewm(span=20, adjust=False).mean()
        df["Std"]      = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["EMA20"] + df["Std"] * 2
        df["BB_Lower"] = df["EMA20"] - df["Std"] * 2
        # BB Squeeze: BandWidth thấp = đang tích lũy
        df["BW"]       = (df["BB_Upper"] - df["BB_Lower"]) / df["EMA20"]
        bw_min20       = df["BW"].rolling(20).min()
        df["Squeeze"]  = df["BW"] <= bw_min20 * 1.05   # True = đang nén
        # MACD
        e1 = df["Close"].ewm(span=12, adjust=False).mean()
        e2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"]   = e1 - e2
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["Hist"]   = df["MACD"] - df["Signal"]
        # RSI
        delta        = df["Close"].diff()
        gain         = delta.where(delta > 0, 0).rolling(14).mean()
        loss         = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["RSI"]    = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        # Stochastic %K %D (14,3)
        lo14         = df["Low"].rolling(14).min()
        hi14         = df["High"].rolling(14).max()
        df["Stoch_K"] = 100 * (df["Close"] - lo14) / (hi14 - lo14).replace(0, np.nan)
        df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
        # Volume spike: so với trung bình 20 phiên
        df["Vol_MA20"]   = df["Volume"].rolling(20).mean() if "Volume" in df.columns else 1
        df["Vol_Ratio"]  = df["Volume"] / df["Vol_MA20"].replace(0, np.nan) if "Volume" in df.columns else 1
        
        # Phân kỳ MACD (Divergence)
        bull_divs = np.zeros(len(df), dtype=bool)
        bear_divs = np.zeros(len(df), dtype=bool)
        p = df["Close"].values
        h = df["Hist"].values
        for k in range(5, len(df)):
            start_idx = max(0, k-15)
            window_p = p[start_idx:k+1]
            window_h = h[start_idx:k+1]
            
            troughs = [idx for idx in range(1, len(window_p)-1) if window_p[idx] < window_p[idx-1] and window_p[idx] < window_p[idx+1]]
            peaks   = [idx for idx in range(1, len(window_p)-1) if window_p[idx] > window_p[idx-1] and window_p[idx] > window_p[idx+1]]
            
            if len(troughs) >= 2:
                t1, t2 = troughs[-2], troughs[-1]
                if window_p[t2] < window_p[t1] and window_h[t2] > window_h[t1]:
                    if t2 >= len(window_p) - 3: 
                        bull_divs[k] = True
                        
            if len(peaks) >= 2:
                p1, p2 = peaks[-2], peaks[-1]
                if window_p[p2] > window_p[p1] and window_h[p2] < window_h[p1]:
                    if p2 >= len(window_p) - 3:
                        bear_divs[k] = True
                        
        df["Bull_Div"] = bull_divs
        df["Bear_Div"] = bear_divs
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return _calc_indicators(df)


# ══════════════════════════════════════════
# 4a. MACD Divergence
# ══════════════════════════════════════════
def detect_divergence(df: pd.DataFrame, lookback: int = 10) -> str:
    """Trả về BULL/BEAR/NONE tại nến cuối cùng."""
    if "Bull_Div" not in df.columns or len(df) < 2:
        return "NONE"
    if df["Bull_Div"].iloc[-1] or df["Bull_Div"].iloc[-2]:
        return "BULL"
    if df["Bear_Div"].iloc[-1] or df["Bear_Div"].iloc[-2]:
        return "BEAR"
    return "NONE"


# ══════════════════════════════════════════
# 4b. Phát hiện vùng tín hiệu (multi-factor)
# ══════════════════════════════════════════
def score_signal(df: pd.DataFrame, i: int, direction: str) -> int:
    """Tính điểm tín hiệu Đảo chiều (Reversal) 0-4 tại vị trí i."""
    score = 0
    close     = df["Close"].iloc[i]
    low       = df["Low"].iloc[i]
    high      = df["High"].iloc[i]
    open_p    = df["Open"].iloc[i]
    hist      = df["Hist"].iloc[i]
    hist_prev = df["Hist"].iloc[i - 1]
    rsi       = df["RSI"].iloc[i]
    stk       = df["Stoch_K"].iloc[i] if "Stoch_K" in df.columns else 50
    stk_prev  = df["Stoch_K"].iloc[i-1] if "Stoch_K" in df.columns else 50
    vol_ratio = df["Vol_Ratio"].iloc[i] if "Vol_Ratio" in df.columns else 1
    bull_div  = df["Bull_Div"].iloc[i] if "Bull_Div" in df.columns else False
    bear_div  = df["Bear_Div"].iloc[i] if "Bear_Div" in df.columns else False

    is_green = close > open_p
    is_red   = close < open_p
    body     = abs(close - open_p)
    upper_wick = high - max(close, open_p)
    lower_wick = min(close, open_p) - low

    if direction == "BUY":
        # 0. Bộ lọc Trend giảm mạnh
        if is_red and lower_wick <= body * 0.5 and close <= df["BB_Lower"].iloc[i] * 1.01:
            return 0 
            
        # 1. Chạm/thủng BB dưới hoặc RSI quá bán
        if low <= df["BB_Lower"].iloc[i] * 1.005 or rsi < 35:
            score += 1
            
        # Cạn Cung (No Supply) - Giá sát hỗ trợ mà thanh khoản teo tóp
        if low <= df["BB_Lower"].iloc[i] * 1.01 and vol_ratio <= 0.6:
            score += 1
            
        # Phân kỳ Dương (+2 điểm)
        if bull_div:
            score += 2
            
        # 2. MACD Hist tạo đáy móc lên
        if hist < 0 and hist > hist_prev:
            score += 1
        # 3. Stochastic móc lên từ vùng thấp
        if stk < 30 and stk > stk_prev:
            score += 1
        # 4. Xác nhận từ Nến đảo chiều (Xanh hoặc Rút chân)
        is_pinbar = lower_wick > body * 1.5
        if (is_green or is_pinbar):
            score += 1
            if vol_ratio >= 1.2: score += 1
    else:
        # 0. Bộ lọc Trend tăng mạnh
        if is_green and upper_wick <= body * 0.5 and close >= df["BB_Upper"].iloc[i] * 0.99:
            return 0
            
        # 1. Chạm/vượt BB trên hoặc RSI quá mua
        if high >= df["BB_Upper"].iloc[i] * 0.995 or rsi > 65:
            score += 1
            
        # Thiếu Cầu Kéo Giá (No Demand)
        if high >= df["BB_Upper"].iloc[i] * 0.99 and vol_ratio <= 0.6:
            score += 1
            
        # Phân kỳ Âm (+2 điểm)
        if bear_div:
            score += 2
            
        # 2. MACD Hist tạo đỉnh cắm xuống
        if hist > 0 and hist < hist_prev:
            score += 1
        # 3. Stochastic cắm xuống từ vùng cao
        if stk > 70 and stk < stk_prev:
            score += 1
        # 4. Xác nhận từ Nến đảo chiều (Đỏ hoặc Cụt đầu)
        is_pinbar_down = upper_wick > body * 1.5
        if (is_red or is_pinbar_down):
            score += 1
            if vol_ratio >= 1.2: score += 1
            
    return score


def detect_zones(df: pd.DataFrame) -> list[dict]:
    if len(df) < 30 or "EMA20" not in df.columns:
        return []
    zones = []
    i = 7
    last_type = None

    while i < len(df):
        sc_buy = score_signal(df, i, "BUY")
        sc_sell = score_signal(df, i, "SELL")

        # Bắt buộc xen kẽ: MUA -> BÁN -> MUA -> BÁN (để thể hiện hết chu kỳ)
        can_buy = (last_type != "BUY") and (sc_buy >= 2)
        can_sell = (last_type != "SELL") and (sc_sell >= 2)

        if can_buy or can_sell:
            is_buy = can_buy and (sc_buy >= sc_sell if can_sell else True)
            direction = "BUY" if is_buy else "SELL"
            sc = sc_buy if is_buy else sc_sell

            alpha = min(0.10 + sc * 0.06, 0.30)
            if is_buy:
                fill = f"rgba(0,220,100,{alpha:.2f})"
                line = "#00DC64"
            else:
                fill = f"rgba(255,60,60,{alpha:.2f})"
                line = "#FF3C3C"
            
            start_i = i
            j = i
            while j < len(df) - 1:
                j += 1
                
                # Kéo dài vùng cho đến khi có tín hiệu đảo chiều ngược lại (Full Cycle)
                if is_buy:
                    if score_signal(df, j, "SELL") >= 2: break
                else:
                    if score_signal(df, j, "BUY") >= 2: break
                    
            zones.append({
                "x0": df.index[start_i], "x1": df.index[j],
                "y0": df["Low"].iloc[start_i:j+1].min()  * 0.995,
                "y1": df["High"].iloc[start_i:j+1].max() * 1.005,
                "fill": fill, "line": line, "type": direction,
                "score": sc, "start_idx": start_i, "end_idx": j,
            })
            
            last_type = direction
            i = j  # Bắt đầu dò tín hiệu tiếp theo ngay tại điểm đảo chiều
            continue
        i += 1
    return zones


def current_signal(df: pd.DataFrame, zones: list[dict]) -> tuple[str, int]:
    """Trả về (tín hiệu, score): BUY/SELL/NEUTRAL, 0-4"""
    if not zones:
        return "NEUTRAL", 0
    last = zones[-1]
    # Lọc điểm mua sớm TRONG PHIÊN: chỉ báo tín hiệu mới bắt đầu ở nến cuối hoặc áp chót
    if last.get("start_idx", -1) >= len(df) - 2:
        return last["type"], last.get("score", 0)
    return "NEUTRAL", 0


# ══════════════════════════════════════════
# 5. PriceStore — lưu giá 2 ngày
# ══════════════════════════════════════════
def update_price_store(symbol: str, df: pd.DataFrame):
    if df.empty or len(df) < 2:
        return
    if "price_store" not in st.session_state:
        st.session_state.price_store = {}
    st.session_state.price_store[symbol] = {
        "today":           round(df["Close"].iloc[-1], 2),
        "yesterday":       round(df["Close"].iloc[-2], 2),
        "today_date":      df.index[-1].strftime("%d/%m"),
        "yesterday_date":  df.index[-2].strftime("%d/%m"),
        "change_pct":      round((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100, 2),
        "open":            round(df["Open"].iloc[-1], 2),
        "volume":          int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0,
    }


# ══════════════════════════════════════════
# 6. Scanner toàn VN30
# ══════════════════════════════════════════
def run_scan(interval: str = "1D"):
    buys, sells = [], []
    prog = st.sidebar.progress(0, text="Đang quét VN30...")
    for n, sym in enumerate(VN30_LIST):
        try:
            # Dùng cache — không fetch lại nếu đã có
            df = calc_indicators_cached(sym, interval)
            if not df.empty:
                update_price_store(sym, df)
                zones  = detect_zones(df)
                sig, sc = current_signal(df, zones)
                div   = detect_divergence(df)
                
                # Check Multi-timeframe 1H
                mtf_buy = False
                if interval in ["1D", "1W"] and (df["RSI"].iloc[-1] < 45 or df["Close"].iloc[-1] <= df["BB_Lower"].iloc[-1]*1.03):
                    df_1h = calc_indicators_cached(sym, "1H")
                    if not df_1h.empty:
                        z_1h = detect_zones(df_1h)
                        s_1h, sc_1h = current_signal(df_1h, z_1h)
                        if s_1h == "BUY" and sc_1h >= 2:
                            mtf_buy = True
                
                store = st.session_state.get("price_store", {}).get(sym, {})
                info  = {
                    "symbol":  sym,
                    "price":   store.get("today", 0),
                    "chg":     store.get("change_pct", 0),
                    "rsi":     round(df["RSI"].iloc[-1], 1) if "RSI" in df.columns else 0,
                    "score":   sc,
                    "div":     div,
                    "squeeze": bool(df["Squeeze"].iloc[-1]) if "Squeeze" in df.columns else False,
                    "mtf_buy": mtf_buy,
                }
                if sig == "BUY" or mtf_buy:
                    buys.append(info)
                elif sig == "SELL": 
                    sells.append(info)
        except Exception as e:
            logger.warning(f"Scan error {sym}: {e}")
        
        prog.progress((n + 1) / len(VN30_LIST), text=f"Đang quét: {sym}")
    prog.empty()
    buys.sort(key=lambda x: (x["score"], x["chg"]), reverse=True)
    sells.sort(key=lambda x: (x["score"], -x["chg"]), reverse=True)
    st.session_state.buy_list      = buys
    st.session_state.sell_list     = sells
    st.session_state.last_scan     = time.time()
    st.session_state.scan_interval = interval


# ══════════════════════════════════════════
# 7. Vẽ biểu đồ
# ══════════════════════════════════════════
def build_chart(df: pd.DataFrame, symbol: str, zones: list[dict], interval: str) -> go.Figure:
    has_stoch = "Stoch_K" in df.columns
    rows      = 4 if has_stoch else 3
    heights   = [0.55, 0.18, 0.14, 0.13] if has_stoch else [0.65, 0.2, 0.15]
    titles    = (f"{symbol} [{interval}]", "MACD", "RSI", "Stochastic %K/%D") if has_stoch else (f"{symbol} [{interval}]", "MACD", "RSI")
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=heights,
        subplot_titles=titles,
    )

    # Vùng xu hướng nền (khoanh vùng mua/bán sớm)
    for z in zones:
        # Khoanh vùng bằng hình chữ nhật bọc sát giá (từ đáy thấp nhất đến đỉnh cao nhất của vùng)
        fig.add_shape(
            type="rect", 
            x0=z["x0"], x1=z["x1"],
            y0=z["y0"], y1=z["y1"],
            fillcolor=z["fill"], 
            line=dict(color=z["line"], width=2),
            layer="below",
            row=1, col=1,
        )
        
        # Đánh dấu chính xác vị trí "Bắt đầu" tín hiệu
        arrow_y = z["y0"] if z["type"] == "BUY" else z["y1"]
        ay_offset = 35 if z["type"] == "BUY" else -35
        txt = "🟢 MUA SỚM" if z["type"] == "BUY" else "🔴 BÁN SỚM"
        
        fig.add_annotation(
            x=z["x0"], y=arrow_y,
            text=f"<b>{txt}</b>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor=z["line"],
            ax=0, ay=ay_offset,
            font=dict(color=z["line"], size=11),
            bgcolor="rgba(0,0,0,0.7)", bordercolor=z["line"], borderwidth=1, borderpad=3,
            row=1, col=1
        )

    # BB bands
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"],
        line=dict(color="rgba(100,180,255,0.4)", width=1), name="BB Up"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"],
        line=dict(color="rgba(100,180,255,0.4)", width=1),
        fill="tonexty", fillcolor="rgba(100,180,255,0.05)", name="BB Lo"), row=1, col=1)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name=symbol,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # EMA20
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"],
        line=dict(color="#FFD700", width=1.8), name="EMA20"), row=1, col=1)

    # Nhãn giá hôm nay / hôm qua
    p_now  = df["Close"].iloc[-1]
    p_prev = df["Close"].iloc[-2] if len(df) >= 2 else p_now
    d_now  = df.index[-1].strftime("%d/%m")
    d_prev = df.index[-2].strftime("%d/%m") if len(df) >= 2 else d_now
    chg    = (p_now / p_prev - 1) * 100 if p_prev else 0
    clr    = "#00E676" if chg >= 0 else "#FF5252"
    arrow  = "▲" if chg >= 0 else "▼"
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.97,
        text=f"<b>Hôm nay ({d_now}): {p_now:,.1f} {arrow}{abs(chg):.1f}%</b>",
        showarrow=False, font=dict(size=13, color=clr),
        bgcolor="rgba(0,0,0,0.6)", bordercolor=clr, borderwidth=1,
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.91,
        text=f"Hôm qua ({d_prev}): {p_prev:,.1f}",
        showarrow=False, font=dict(size=11, color="#FF9800"),
        bgcolor="rgba(0,0,0,0.5)",
    )

    # MACD Histogram
    hist_clr = np.where(df["Hist"] >= 0, "#00E676", "#FF5252")
    fig.add_trace(go.Bar(x=df.index, y=df["Hist"],
        marker_color=hist_clr, name="Hist"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
        line=dict(color="#2196F3", width=1.2), name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"],
        line=dict(color="#FF9800", width=1.2), name="Signal"), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
        line=dict(color="#CE93D8", width=1.5), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, row=3, col=1, line=dict(color="red",   width=0.8, dash="dot"))
    fig.add_hline(y=30, row=3, col=1, line=dict(color="green", width=0.8, dash="dot"))
    fig.add_hline(y=50, row=3, col=1, line=dict(color="gray",  width=0.5, dash="dot"))

    # Stochastic %K/%D
    if has_stoch:
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"],
            line=dict(color="#00BCD4", width=1.4), name="%K"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_D"],
            line=dict(color="#FF9800", width=1.2, dash="dot"), name="%D"), row=4, col=1)
        fig.add_hline(y=80, row=4, col=1, line=dict(color="red",   width=0.8, dash="dot"))
        fig.add_hline(y=20, row=4, col=1, line=dict(color="green", width=0.8, dash="dot"))

    # BB Squeeze markers trên kênh giá
    if "Squeeze" in df.columns:
        sq_x = df.index[df["Squeeze"]]
        sq_y = df["BB_Lower"][df["Squeeze"]] * 0.998
        if len(sq_x):
            fig.add_trace(go.Scatter(
                x=sq_x, y=sq_y, mode="markers",
                marker=dict(symbol="circle", size=5, color="#FFD700"),
                name="BB Squeeze", hovertemplate="BB Squeeze<extra></extra>",
            ), row=1, col=1)

    # Volume spike highlight
    if "Vol_Ratio" in df.columns:
        spike_x = df.index[df["Vol_Ratio"] >= 2.0]
        spike_y = df["High"][df["Vol_Ratio"] >= 2.0] * 1.003
        if len(spike_x):
            fig.add_trace(go.Scatter(
                x=spike_x, y=spike_y, mode="markers",
                marker=dict(symbol="triangle-up", size=8, color="#FF6D00"),
                name="Vol Spike", hovertemplate="Volume x%{customdata:.1f}<extra></extra>",
                customdata=df["Vol_Ratio"][df["Vol_Ratio"] >= 2.0],
            ), row=1, col=1)

    fig.update_layout(
        template="plotly_dark", height=920 if has_stoch else 820,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=10, r=80, t=60, b=10),
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
    )
    if interval == "1D":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


# ══════════════════════════════════════════
# 8. Render bảng giá 2 ngày — FIX applymap()
# ══════════════════════════════════════════
def render_price_table(buy_syms: set, sell_syms: set):
    store = st.session_state.get("price_store", {})
    if not store:
        st.info("Chưa có dữ liệu bảng giá. Chờ quét xong.")
        return

    rows = []
    for sym in VN30_LIST:
        if sym not in store:
            continue
        d = store[sym]
        sig = "🟢 MUA" if sym in buy_syms else ("🔴 BÁN" if sym in sell_syms else "⚪")
        rows.append({
            "Mã":         sym,
            "Tín hiệu":  sig,
            f"Ngày trước ({d['yesterday_date']})": d["yesterday"],
            f"Hôm nay ({d['today_date']})":        d["today"],
            "% Thay đổi": d["change_pct"],
            "Volume":     f"{d['volume']:,}",
        })

    df_tbl = pd.DataFrame(rows)
    df_tbl = df_tbl.sort_values("% Thay đổi", ascending=False)

    def color_chg(val):
        if val > 0:   return "color: #00E676; font-weight:bold; font-size:15px"
        elif val < 0: return "color: #FF5252; font-weight:bold; font-size:15px"
        return "color: #E0E0E0; font-size:15px"

    def color_sig(val):
        if "MUA" in str(val): return "background-color: rgba(0,200,100,0.2); font-size:15px"
        if "BÁN" in str(val): return "background-color: rgba(255,60,60,0.2); font-size:15px"
        return "font-size:15px"

    try:
        # FIX: map() thay cho applymap() (pandas 2.1+)
        styled = (
            df_tbl.style
            .map(color_chg, subset=["% Thay đổi"])
            .map(color_sig, subset=["Tín hiệu"])
            .format({"% Thay đổi": "{:+.2f}%"})
        )
    except AttributeError:
        # Fallback để compatible với pandas cũ
        styled = (
            df_tbl.style
            .applymap(color_chg, subset=["% Thay đổi"])
            .applymap(color_sig, subset=["Tín hiệu"])
            .format({"% Thay đổi": "{:+.2f}%"})
        )
    
    st.dataframe(styled, use_container_width=True, height=350)


# ══════════════════════════════════════════
# 9. MAIN APP
# ══════════════════════════════════════════
def main():
    st.set_page_config(
        layout="wide",
        page_title="VN30 Dashboard — Vùng Mua/Bán Sớm",
        page_icon="📊",
    )

    # Custom CSS
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #1C2333;
        color: #F8F9FA;
    }
    .stMarkdown, .stText, p, label {
        font-size: 16px !important;
        color: #F8F9FA !important;
    }
    [data-testid="stSidebar"]  { background: #252D3D; }
    [data-testid="stHeader"]   { background: rgba(28,35,51,0.95); }
    .block-container { padding-top: 4rem; padding-bottom: 2rem; }
    .metric-card {
        background: #252D3D; border-radius: 10px;
        padding: 12px 16px; border: 1px solid #3D4F6B;
        text-align: center;
    }
    .signal-buy  { background: rgba(0,200,100,0.12); border-left: 3px solid #00C864;
                   border-radius:8px; padding:8px 12px; margin:4px 0; }
    .signal-sell { background: rgba(255,60,60,0.12);  border-left: 3px solid #FF3C3C;
                   border-radius:8px; padding:8px 12px; margin:4px 0; }
    .expiry-banner { border-radius: 10px; padding: 14px 20px;
                     text-align:center; font-size:18px; font-weight:bold; margin-bottom:12px; }
    .buy-panel { background: rgba(0,180,80,0.08); border: 1px solid rgba(0,200,100,0.3);
                 border-radius: 10px; padding: 10px 16px; margin-bottom: 10px; }
    .sell-panel { background: rgba(255,60,60,0.08); border: 1px solid rgba(255,80,80,0.3);
                  border-radius: 10px; padding: 10px 16px; margin-bottom: 10px; }
    /* Metric cards */
    [data-testid="stMetric"] { background: #252D3D; border-radius:10px; padding:10px 16px;
                               border:1px solid #3D4F6B; }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 32px !important; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #E0E0E0 !important; font-size: 16px !important; }
    div[data-testid="stSelectbox"] > div { background: #2A3449; color: #FFFFFF; font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Session state init ──
    for key, val in [
        ("buy_list", []), ("sell_list", []),
        ("last_scan", 0), ("scan_interval", "1D"),
        ("price_store", {}), ("selected_stock", "VHM"),
        ("force_scan", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

    # ════════════════════════════
    # SIDEBAR
    # ════════════════════════════
    with st.sidebar:
        st.markdown("## 📊 VN30 Dashboard")
        st.divider()

        # Chọn mã — ưu tiên selected_stock từ session (click từ panel)
        sel_idx = VN30_LIST.index(st.session_state.selected_stock) \
                  if st.session_state.selected_stock in VN30_LIST else VN30_LIST.index("VHM")
        selected = st.selectbox("🔍 Chọn mã", VN30_LIST, index=sel_idx,
                                key="sidebar_select")
        if selected != st.session_state.selected_stock:
            st.session_state.selected_stock = selected
            st.rerun()

        # Filter khung thời gian
        interval = st.radio("⏱ Khung thời gian", ["1W", "1D", "1H"], index=1, horizontal=True)

        # Tần suất refresh
        refresh  = st.slider("🔄 Auto-refresh (giây)", 5, 60, 10)

        st.divider()

        # Nút quét thủ công
        if st.button("🔍 Quét VN30 ngay", use_container_width=True, type="primary"):
            st.session_state.force_scan = True
            st.rerun()

        # Thời gian quét cuối
        if st.session_state.last_scan:
            elapsed = int(time.time() - st.session_state.last_scan)
            st.caption(f"Quét lần cuối: {elapsed}s trước | [{st.session_state.scan_interval}]")
            next_scan = max(0, 600 - elapsed)
            st.caption(f"Quét tiếp theo: ~{next_scan//60}p{next_scan%60}s")

        st.divider()

        # ── Danh sách MUA ──
        buy_list  = st.session_state.buy_list
        sell_list = st.session_state.sell_list

        st.markdown(f"### 🟢 Điểm MUA sớm ({len(buy_list)})")
        if buy_list:
            for m in buy_list:
                clr   = "#00E676" if m["chg"] >= 0 else "#FF5252"
                arr   = "▲" if m["chg"] >= 0 else "▼"
                stars = "★" * m.get("score", 0) + "☆" * (4 - m.get("score", 0))
                div_b = " 🔺<b style='color:#00E676'>BULL DIV</b>" if m.get("div") == "BULL" else ""
                sq_b  = " 🟡<b style='color:#FFD700'>SQZ</b>" if m.get("squeeze") else ""
                mtf_b = " ⚡<b style='color:#00DC64'>MTF 1H</b>" if m.get("mtf_buy") else ""
                st.markdown(
                    f'<div class="signal-buy">'
                    f'<b style="font-size:16px">{m["symbol"]}</b>'
                    f'<span style="float:right;color:{clr};font-size:16px;font-weight:bold">{arr}{abs(m["chg"]):.1f}%</span><br>'
                    f'<span style="color:#FFD700;font-size:14px">{stars}</span>{div_b}{sq_b}{mtf_b}<br>'
                    f'<span style="color:#E0E0E0;font-size:14px">Giá: {m["price"]:,.1f} | RSI: {m["rsi"]}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("ℹ️ Chưa có tín hiệu mua — nhấn Quét VN30")

        st.divider()

        # ── Danh sách BÁN ──
        st.markdown(f"### 🔴 Điểm BÁN sớm ({len(sell_list)})")
        if sell_list:
            for m in sell_list:
                clr   = "#FF5252" if m["chg"] <= 0 else "#00E676"
                arr   = "▼" if m["chg"] <= 0 else "▲"
                stars = "★" * m.get("score", 0) + "☆" * (4 - m.get("score", 0))
                div_b = " 🔻<b style='color:#FF5252'>BEAR DIV</b>" if m.get("div") == "BEAR" else ""
                sq_b  = " 🟡<b style='color:#FFD700'>SQZ</b>" if m.get("squeeze") else ""
                st.markdown(
                    f'<div class="signal-sell">'
                    f'<b style="font-size:16px">{m["symbol"]}</b>'
                    f'<span style="float:right;color:{clr};font-size:16px;font-weight:bold">{arr}{abs(m["chg"]):.1f}%</span><br>'
                    f'<span style="color:#FF9800;font-size:14px">{stars}</span>{div_b}{sq_b}<br>'
                    f'<span style="color:#E0E0E0;font-size:14px">Giá: {m["price"]:,.1f} | RSI: {m["rsi"]}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("ℹ️ Chưa có tín hiệu bán — nhấn Quét VN30")

    # ════════════════════════════
    # HEADER — Đồng hồ + Đáo hạn
    # ════════════════════════════
    expiry_date, days_left = get_expiry()
    now_str = datetime.now().strftime("%H:%M:%S  %d/%m/%Y")

    if days_left <= 3:
        banner_bg, banner_border = "#3D0000", "#FF1744"
        banner_icon = "🚨"
    elif days_left <= 7:
        banner_bg, banner_border = "#3D2B00", "#FF9800"
        banner_icon = "⚠️"
    else:
        banner_bg, banner_border = "#003322", "#00E676"
        banner_icon = "📅"

    col_clock, col_expiry = st.columns([1, 2])
    with col_clock:
        st.markdown(
            f'<div style="background:#161B22;border-radius:10px;padding:14px 20px;'
            f'border:1px solid #30363D;text-align:center">'
            f'<span style="font-size:26px;font-weight:bold;color:#FFFFFF;letter-spacing:2px">'
            f'🕐 {now_str}</span></div>',
            unsafe_allow_html=True,
        )
    with col_expiry:
        st.markdown(
            f'<div class="expiry-banner" style="background:{banner_bg};border:2px solid {banner_border}">'
            f'{banner_icon} Đáo hạn phái sinh VN30F: '
            f'<span style="color:{banner_border}">{expiry_date.strftime("%d/%m/%Y")} '
            f'— Còn {days_left} ngày</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ════════════════════════════
    # CHART CHÍNH (Đưa lên đầu)
    # ════════════════════════════
    selected = st.session_state.selected_stock

    with st.spinner(f"Đang tải {selected} [{interval}]..."):
        df_main = load_data(selected, interval)

    if df_main.empty:
        st.error(f"⚠️ Không tải được dữ liệu {selected} [{interval}]. Kiểm tra API hoặc khoá chứng chỉ.")
    else:
        df_main  = calc_indicators(df_main)
        update_price_store(selected, df_main)
        zones    = detect_zones(df_main)
        sig, sc  = current_signal(df_main, zones)
        div_now  = detect_divergence(df_main)

        sig_text = (
            f"🟢 MUA SỚM ({'★'*sc})"
            if sig == "BUY" else
            f"🔴 BÁN SỚM ({'★'*sc})"
            if sig == "SELL" else "⚪ TRUNG TÍNH"
        )
        if div_now == "BULL": sig_text += " 🔺BULL DIV"
        if div_now == "BEAR": sig_text += " 🔻BEAR DIV"

        c1, c2, c3, c4 = st.columns(4)
        ps = st.session_state.price_store.get(selected, {})
        c1.metric("📌 Tín hiệu", sig_text)
        c2.metric(f"💰 Giá hôm nay ({ps.get('today_date','--')})",
                  f"{ps.get('today', 0):,.1f}",
                  delta=f"{ps.get('change_pct', 0):+.2f}%")
        c3.metric(f"📆 Giá hôm qua ({ps.get('yesterday_date','--')})",
                  f"{ps.get('yesterday', 0):,.1f}")
        c4.metric("📊 RSI hiện tại",
                  f"{df_main['RSI'].iloc[-1]:.1f}" if "RSI" in df_main.columns else "--")

        fig = build_chart(df_main, selected, zones, interval)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected}_{interval}_{int(time.time())}")

    st.divider()

    # ════════════════════════════
    # PANEL MÃ MUA SỚM — click để chuyển chart
    # ════════════════════════════
    buy_list  = st.session_state.buy_list
    sell_list = st.session_state.sell_list

    if buy_list or sell_list:
        col_bp, col_sp = st.columns(2)
        with col_bp:
            st.markdown('<div class="buy-panel">', unsafe_allow_html=True)
            st.markdown(f"🟢 **ĐIỂM MUA SỚM — {len(buy_list)} mã** *(click để xem chart)*")
            if buy_list:
                # Hiển thị dạng nút bấm theo cột
                n_cols = min(len(buy_list), 5)
                btn_cols = st.columns(n_cols)
                for idx, m in enumerate(buy_list):
                    clr  = "#00E676" if m["chg"] >= 0 else "#FF5252"
                    arr  = "▲" if m["chg"] >= 0 else "▼"
                    with btn_cols[idx % n_cols]:
                        if st.button(
                            f"🟢 {m['symbol']}\n{arr}{abs(m['chg']):.1f}%",
                            key=f"buy_btn_{m['symbol']}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_stock = m["symbol"]
                            st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_sp:
            st.markdown('<div class="sell-panel">', unsafe_allow_html=True)
            st.markdown(f"🔴 **ĐIỂM BÁN SỚM — {len(sell_list)} mã** *(click để xem chart)*")
            if sell_list:
                n_cols = min(len(sell_list), 5)
                btn_cols = st.columns(n_cols)
                for idx, m in enumerate(sell_list):
                    arr  = "▼" if m["chg"] <= 0 else "▲"
                    with btn_cols[idx % n_cols]:
                        if st.button(
                            f"🔴 {m['symbol']}\n{arr}{abs(m['chg']):.1f}%",
                            key=f"sell_btn_{m['symbol']}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_stock = m["symbol"]
                            st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("🔍 Chưa có dữ liệu quét — nhấn **'Quét VN30 ngay'** ở sidebar để xãc định tín hiệu.")

    st.divider()

    # ════════════════════════════
    # BẢNG GIÁ 2 NGÀY VN30
    # ════════════════════════════
    st.markdown("### 📋 Bảng giá 2 ngày — Toàn bộ VN30")
    buy_syms  = {m["symbol"] for m in st.session_state.buy_list}
    sell_syms = {m["symbol"] for m in st.session_state.sell_list}
    render_price_table(buy_syms, sell_syms)

    # ════════════════════════════
    # AUTO SCAN — chạy SAU khi chart đã render
    # ════════════════════════════
    interval_changed = st.session_state.scan_interval != interval
    need_scan = (
        st.session_state.last_scan == 0          # lần đầu load
        or st.session_state.get("force_scan")    # nhấn nút quét
        or interval_changed                       # đổi khung thời gian
    )
    if need_scan:
        st.session_state.force_scan = False
        run_scan(interval)
        st.rerun()

    # ════════════════════════════
    # AUTO REFRESH — JS non-blocking (không block Python thread)
    # ════════════════════════════
    import streamlit.components.v1 as components
    components.html(
        f"<script>setTimeout(function(){{window.location.reload();}},{refresh*1000});</script>",
        height=0,
    )
    st.caption(f"🔄 Tự động làm mới sau {refresh}s | Cập nhật: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
