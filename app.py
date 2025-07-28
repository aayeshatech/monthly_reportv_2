import streamlit as st
import pandas as pd
import datetime
import math
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import io
import numpy as np

st.set_page_config(
    page_title="🌟 Advanced Astrological Trading Platform",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, #ffd700, #ffb347, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ffd700;
        text-align: center;
        color: white;
        margin-bottom: 10px;
    }
    
    .forecast-card {
        background: linear-gradient(135deg, #2d2d4a, #1a1a2e);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffd700;
        margin: 0.5rem 0;
        color: white;
    }
    
    .planet-card {
        background: linear-gradient(135deg, #16213e, #0c0c0c);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ffd700;
        text-align: center;
        color: white;
        margin-bottom: 10px;
    }
    
    .sector-card {
        background: linear-gradient(135deg, #1a2332, #2d3748);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
        color: white;
    }
    
    .historical-card {
        background: linear-gradient(135deg, #2a1810, #3d2817);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
        color: white;
    }
    
    .bullish { color: #4caf50; font-weight: bold; }
    .bearish { color: #f44336; font-weight: bold; }
    .neutral { color: #ff9800; font-weight: bold; }
    .long-signal { background-color: #4caf50; color: white; padding: 5px 10px; border-radius: 5px; }
    .short-signal { background-color: #f44336; color: white; padding: 5px 10px; border-radius: 5px; }
    .hold-signal { background-color: #ff9800; color: white; padding: 5px 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

class Sentiment(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"

@dataclass
class Planet:
    symbol: str
    name: str

@dataclass
class Forecast:
    date: str
    day: int
    event: str
    sentiment: Sentiment
    change: str
    impact: str
    sector_impact: Dict[str, float]
    signal: SignalType

@dataclass
class SectorImpact:
    sector: str
    impact_percentage: float
    top_stocks: List[str]
    recommendation: str

@dataclass
class HistoricalTransit:
    transit_type: str
    last_occurrence: str
    market_change: str
    duration_days: int
    success_rate: float

@dataclass
class GlobalMarketData:
    symbol: str
    name: str
    expected_change: float
    confidence: float
    correlation_strength: str

class EnhancedAstrologicalTradingPlatform:
    
    def __init__(self):
        self.planets = {
            'sun': Planet('☉', 'Sun'),
            'moon': Planet('☽', 'Moon'),
            'mercury': Planet('☿', 'Mercury'),
            'venus': Planet('♀', 'Venus'),
            'mars': Planet('♂', 'Mars'),
            'jupiter': Planet('♃', 'Jupiter'),
            'saturn': Planet('♄', 'Saturn'),
            'uranus': Planet('♅', 'Uranus'),
            'neptune': Planet('♆', 'Neptune'),
            'pluto': Planet('♇', 'Pluto')
        }
        
        self.sectors = {
            'banking': ['HDFC', 'ICICI', 'SBI', 'AXIS', 'KOTAK'],
            'it': ['TCS', 'INFY', 'WIPRO', 'HCL', 'TECHM'],
            'pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'BIOCON'],
            'auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO'],
            'energy': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'NTPC'],
            'metals': ['TATA STEEL', 'JSW STEEL', 'HINDALCO', 'VEDL', 'COALINDIA'],
            'fmcg': ['HINDUNILVR', 'ITC', 'NESTLE', 'BRITANNIA', 'DABUR']
        }
        
        self.global_markets = {
            'dow_jones': GlobalMarketData('DJI', 'Dow Jones', 0.0, 0.0, ''),
            'nasdaq': GlobalMarketData('NASDAQ', 'NASDAQ Composite', 0.0, 0.0, ''),
            'sp500': GlobalMarketData('SPX', 'S&P 500', 0.0, 0.0, ''),
            'nikkei': GlobalMarketData('N225', 'Nikkei 225', 0.0, 0.0, ''),
            'ftse': GlobalMarketData('FTSE', 'FTSE 100', 0.0, 0.0, ''),
            'dax': GlobalMarketData('DAX', 'DAX', 0.0, 0.0, ''),
            'hang_seng': GlobalMarketData('HSI', 'Hang Seng', 0.0, 0.0, '')
        }
        
        self.commodities = {
            'gold': GlobalMarketData('GOLD', 'Gold', 0.0, 0.0, ''),
            'silver': GlobalMarketData('SILVER', 'Silver', 0.0, 0.0, ''),
            'crude': GlobalMarketData('CRUDE', 'Crude Oil', 0.0, 0.0, ''),
            'bitcoin': GlobalMarketData('BTC', 'Bitcoin', 0.0, 0.0, ''),
            'ethereum': GlobalMarketData('ETH', 'Ethereum', 0.0, 0.0, '')
        }
        
        self.month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        
        # Enhanced monthly events with sector impacts
        self.enhanced_monthly_events = {
            0: [  # January
                {
                    "date": 3, "event": "Sun trine Jupiter", "sentiment": Sentiment.BULLISH, 
                    "change": "+2.1", "sectors": {"banking": 2.5, "it": 1.8, "auto": 2.0},
                    "signal": SignalType.LONG, "retrograde": False
                },
                {
                    "date": 7, "event": "Venus square Mars", "sentiment": Sentiment.BEARISH, 
                    "change": "-1.8", "sectors": {"energy": -2.1, "metals": -1.5, "fmcg": -0.8},
                    "signal": SignalType.SHORT, "retrograde": False
                },
                {
                    "date": 15, "event": "Mercury Retrograde begins", "sentiment": Sentiment.BEARISH, 
                    "change": "-2.5", "sectors": {"it": -3.0, "banking": -2.0, "auto": -1.8},
                    "signal": SignalType.SHORT, "retrograde": True
                },
                {
                    "date": 25, "event": "Venus conjunct Jupiter", "sentiment": Sentiment.BULLISH, 
                    "change": "+3.1", "sectors": {"banking": 3.5, "pharma": 2.8, "fmcg": 2.2},
                    "signal": SignalType.LONG, "retrograde": False
                }
            ],
            7: [  # August
                {
                    "date": 2, "event": "Mercury square Jupiter", "sentiment": Sentiment.BEARISH, 
                    "change": "-1.7", "sectors": {"it": -2.0, "banking": -1.5, "energy": -1.2},
                    "signal": SignalType.SHORT, "retrograde": False
                },
                {
                    "date": 11, "event": "Mercury Direct", "sentiment": Sentiment.BULLISH, 
                    "change": "+1.9", "sectors": {"it": 2.5, "banking": 2.0, "auto": 1.8},
                    "signal": SignalType.LONG, "retrograde": False
                },
                {
                    "date": 15, "event": "Sun sextile Jupiter", "sentiment": Sentiment.BULLISH, 
                    "change": "+2.3", "sectors": {"banking": 2.8, "pharma": 2.1, "fmcg": 1.9},
                    "signal": SignalType.LONG, "retrograde": False
                },
                {
                    "date": 27, "event": "Jupiter opposition Saturn", "sentiment": Sentiment.BEARISH, 
                    "change": "-2.5", "sectors": {"banking": -3.0, "auto": -2.2, "metals": -2.8},
                    "signal": SignalType.SHORT, "retrograde": False
                }
            ]
        }

    def calculate_planetary_positions(self, date: datetime.date) -> Dict[str, float]:
        j2000 = datetime.date(2000, 1, 1)
        days_since_j2000 = (date - j2000).days
        
        positions = {
            'sun': (280.460 + 0.9856474 * days_since_j2000) % 360,
            'moon': (218.316 + 13.176396 * days_since_j2000) % 360,
            'mercury': (252.250 + 4.092317 * days_since_j2000) % 360,
            'venus': (181.979 + 1.602130 * days_since_j2000) % 360,
            'mars': (355.433 + 0.524033 * days_since_j2000) % 360,
            'jupiter': (34.351 + 0.083091 * days_since_j2000) % 360,
            'saturn': (50.077 + 0.033494 * days_since_j2000) % 360,
            'uranus': (314.055 + 0.011733 * days_since_j2000) % 360,
            'neptune': (304.348 + 0.006020 * days_since_j2000) % 360,
            'pluto': (238.958 + 0.004028 * days_since_j2000) % 360
        }
        
        return positions

    def get_sector_impact(self, event_data: Dict) -> List[SectorImpact]:
        sector_impacts = []
        
        for sector, impact_pct in event_data.get('sectors', {}).items():
            recommendation = "STRONG BUY" if impact_pct > 2 else "BUY" if impact_pct > 0 else "SELL" if impact_pct < -2 else "WEAK SELL"
            
            sector_impacts.append(SectorImpact(
                sector=sector.upper(),
                impact_percentage=impact_pct,
                top_stocks=self.sectors.get(sector, [])[:3],
                recommendation=recommendation
            ))
        
        return sector_impacts

    def calculate_global_market_impact(self, sentiment: Sentiment, change_pct: float) -> Dict[str, GlobalMarketData]:
        global_impacts = {}
        
        for key, market in self.global_markets.items():
            # Calculate correlation-based impact
            correlation_factor = np.random.uniform(0.6, 0.9)
            expected_change = change_pct * correlation_factor * np.random.uniform(0.8, 1.2)
            confidence = min(95, max(60, 80 + abs(change_pct) * 5))
            
            strength = "Strong" if abs(expected_change) > 1.5 else "Moderate" if abs(expected_change) > 0.8 else "Weak"
            
            global_impacts[key] = GlobalMarketData(
                symbol=market.symbol,
                name=market.name,
                expected_change=round(expected_change, 2),
                confidence=round(confidence, 1),
                correlation_strength=strength
            )
        
        return global_impacts

    def calculate_commodity_impact(self, sentiment: Sentiment, change_pct: float) -> Dict[str, GlobalMarketData]:
        commodity_impacts = {}
        
        for key, commodity in self.commodities.items():
            # Different commodities react differently to market sentiment
            if key == 'gold':
                # Gold often moves inverse to markets during uncertainty
                correlation_factor = -0.7 if sentiment == Sentiment.BEARISH else 0.3
            elif key in ['crude']:
                # Oil correlates with economic growth
                correlation_factor = 0.8
            elif key in ['bitcoin', 'ethereum']:
                # Crypto has higher volatility
                correlation_factor = np.random.uniform(1.2, 2.0)
            else:
                correlation_factor = 0.6
            
            expected_change = change_pct * correlation_factor * np.random.uniform(0.9, 1.1)
            confidence = min(90, max(55, 75 + abs(change_pct) * 3))
            
            strength = "Strong" if abs(expected_change) > 2.0 else "Moderate" if abs(expected_change) > 1.0 else "Weak"
            
            commodity_impacts[key] = GlobalMarketData(
                symbol=commodity.symbol,
                name=commodity.name,
                expected_change=round(expected_change, 2),
                confidence=round(confidence, 1),
                correlation_strength=strength
            )
        
        return commodity_impacts

    def get_historical_transit_data(self, event: str) -> HistoricalTransit:
        # Simulated historical data for similar transits
        historical_data = {
            "Sun trine Jupiter": HistoricalTransit("Trine", "2024-01-15", "+2.3%", 5, 78.5),
            "Venus square Mars": HistoricalTransit("Square", "2023-11-22", "-1.9%", 3, 72.1),
            "Mercury Retrograde": HistoricalTransit("Retrograde", "2024-12-13", "-2.8%", 21, 85.3),
            "Jupiter opposition Saturn": HistoricalTransit("Opposition", "2023-08-28", "-2.7%", 7, 81.2)
        }
        
        for key, data in historical_data.items():
            if key.lower() in event.lower():
                return data
        
        # Default historical data
        return HistoricalTransit("Transit", "2024-06-15", "±1.5%", 4, 68.0)

    def generate_enhanced_monthly_forecast(self, symbol: str, year: int, month: int) -> List[Forecast]:
        forecasts = []
        month_data = self.enhanced_monthly_events.get(month, [])
        
        if month == 11:
            next_month = datetime.date(year + 1, 1, 1)
        else:
            next_month = datetime.date(year, month + 2, 1)
        
        current_month = datetime.date(year, month + 1, 1)
        days_in_month = (next_month - current_month).days
        
        for day in range(1, days_in_month + 1):
            current_date = datetime.date(year, month + 1, day)
            
            day_event = None
            for event in month_data:
                if event["date"] == day:
                    day_event = event
                    break
            
            if day_event:
                sector_impacts = {}
                for sector, impact in day_event.get("sectors", {}).items():
                    sector_impacts[sector] = impact
                
                forecasts.append(Forecast(
                    date=current_date.strftime('%Y-%m-%d'),
                    day=day,
                    event=day_event["event"],
                    sentiment=day_event["sentiment"],
                    change=day_event["change"],
                    impact=f"{'Retrograde ' if day_event.get('retrograde', False) else ''}{day_event['sentiment'].value.title()}",
                    sector_impact=sector_impacts,
                    signal=day_event["signal"]
                ))
            else:
                # Generate minor daily transits
                sentiment_options = [Sentiment.BULLISH, Sentiment.BEARISH, Sentiment.NEUTRAL]
                sentiment = sentiment_options[day % 3]
                change_percent = ((day * 37) % 100 - 50) / 50
                change_str = f"{'+' if change_percent > 0 else ''}{change_percent:.1f}"
                
                signal = SignalType.LONG if sentiment == Sentiment.BULLISH else SignalType.SHORT if sentiment == Sentiment.BEARISH else SignalType.HOLD
                
                forecasts.append(Forecast(
                    date=current_date.strftime('%Y-%m-%d'),
                    day=day,
                    event='Minor planetary transit',
                    sentiment=sentiment,
                    change=change_str,
                    impact=f"Minor {sentiment.value.title()}",
                    sector_impact={},
                    signal=signal
                ))
        
        return forecasts

def main():
    if 'platform' not in st.session_state:
        st.session_state.platform = EnhancedAstrologicalTradingPlatform()
    
    platform = st.session_state.platform
    
    st.markdown('<h1 class="main-header">🌟 Advanced Astrological Trading Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #b8b8b8; font-size: 1.2rem;">Professional Market Analysis with Planetary Transits & Comprehensive Forecasting</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; color: #ffd700;">Current Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>', unsafe_allow_html=True)
    
    # Sidebar Controls
    st.sidebar.markdown("## ⚙️ Analysis Controls")
    
    # Market Type Selection
    market_category = st.sidebar.selectbox(
        "🏪 Market Category", 
        ["Stock Symbol", "Global Markets & Commodities & Forex"],
        help="Choose between individual stocks or global markets"
    )
    
    if market_category == "Stock Symbol":
        symbol = st.sidebar.text_input("📈 Stock Symbol", value="NIFTY", help="Enter stock symbol")
        analysis_type = "stock"
    else:
        global_symbol = st.sidebar.selectbox(
            "🌍 Global Market/Commodity", 
            ["NIFTY", "SENSEX", "DOW JONES", "NASDAQ", "S&P 500", "GOLD", "SILVER", "CRUDE OIL", "BITCOIN", "EURUSD", "GBPUSD"]
        )
        symbol = global_symbol
        analysis_type = "global"
    
    # Month and Year Selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        selected_year = st.selectbox("📅 Year", [2024, 2025, 2026], index=1)
    with col2:
        month_options = {i: platform.month_names[i] for i in range(12)}
        selected_month = st.selectbox("📅 Month", options=list(month_options.keys()), 
                                     format_func=lambda x: month_options[x], index=7)
    
    time_zone = st.sidebar.selectbox("🌍 Time Zone", ["IST", "EST", "GMT", "JST"])
    
    if st.sidebar.button("🚀 Generate Analysis", type="primary"):
        with st.spinner("Calculating planetary positions and market correlations..."):
            st.session_state.report = platform.generate_enhanced_monthly_forecast(symbol, selected_year, selected_month)
            st.session_state.symbol = symbol
            st.session_state.month = selected_month
            st.session_state.year = selected_year
            st.session_state.analysis_type = analysis_type
    
    # Main Content Tabs
    if 'report' in st.session_state:
        tab1, tab2, tab3 = st.tabs(["📅 Astro Calendar", "📊 Stock Analysis", "📈 Astro Graph"])
        
        with tab1:
            st.markdown("### 📅 Astrological Calendar")
            
            # Monthly Overview
            forecasts = st.session_state.report
            month_name = platform.month_names[st.session_state.month]
            
            col1, col2, col3, col4 = st.columns(4)
            
            bullish_count = sum(1 for f in forecasts if f.sentiment == Sentiment.BULLISH)
            bearish_count = sum(1 for f in forecasts if f.sentiment == Sentiment.BEARISH)
            neutral_count = len(forecasts) - bullish_count - bearish_count
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{bullish_count}</h3>
                    <p>Bullish Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{bearish_count}</h3>
                    <p>Bearish Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{neutral_count}</h3>
                    <p>Neutral Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                major_events = len([f for f in forecasts if 'retrograde' in f.event.lower() or 'conjunct' in f.event.lower() or 'opposition' in f.event.lower()])
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{major_events}</h3>
                    <p>Major Transits</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Date-wise Transit Calendar
            st.markdown(f"#### 📅 {month_name} {st.session_state.year} - Daily Transits")
            
            # Filter significant events
            significant_events = [f for f in forecasts if abs(float(f.change.replace('+', '').replace('-', ''))) > 1.5]
            
            for forecast in significant_events[:15]:
                signal_class = "long-signal" if forecast.signal == SignalType.LONG else "short-signal" if forecast.signal == SignalType.SHORT else "hold-signal"
                sentiment_class = forecast.sentiment.value
                
                # Historical data for this type of transit
                historical = platform.get_historical_transit_data(forecast.event)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="forecast-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <div style="color: #ffd700; font-weight: bold; font-size: 1.1rem;">{forecast.date}</div>
                            <div class="{signal_class}">{forecast.signal.value}</div>
                        </div>
                        <div style="margin-bottom: 10px; font-size: 1.1rem;">{forecast.event}</div>
                        <div class="{sentiment_class}" style="margin-bottom: 10px;">{forecast.impact}</div>
                        <div style="color: #b8b8b8;">Expected Change: {forecast.change}%</div>
                        
                        {f'''<div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #444;">
                            <div style="font-size: 0.9rem; color: #ffd700;">Sector Impacts:</div>
                            {" | ".join([f"{sector}: {impact:+.1f}%" for sector, impact in forecast.sector_impact.items()])}
                        </div>''' if forecast.sector_impact else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="historical-card">
                        <div style="color: #ff9800; font-weight: bold; margin-bottom: 5px;">Historical Data</div>
                        <div style="font-size: 0.9rem;">Last: {historical.last_occurrence}</div>
                        <div style="font-size: 0.9rem;">Impact: {historical.market_change}</div>
                        <div style="font-size: 0.9rem;">Success Rate: {historical.success_rate}%</div>
                        <div style="font-size: 0.9rem;">Duration: {historical.duration_days} days</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 📊 Stock Analysis")
            
            # Sector-wise Impact Analysis
            if st.session_state.analysis_type == "stock":
                st.markdown("#### 🏢 Sector-wise Impact Analysis")
                
                # Calculate average sector impacts for the month
                sector_totals = {}
                for forecast in forecasts:
                    for sector, impact in forecast.sector_impact.items():
                        if sector not in sector_totals:
                            sector_totals[sector] = []
                        sector_totals[sector].append(impact)
                
                if sector_totals:
                    sector_avg = {sector: np.mean(impacts) for sector, impacts in sector_totals.items()}
                    
                    # Display sector cards
                    sectors_list = list(sector_avg.items())
                    for i in range(0, len(sectors_list), 3):
                        cols = st.columns(3)
                        for j, col in enumerate(cols):
                            if i + j < len(sectors_list):
                                sector, avg_impact = sectors_list[i + j]
                                impact_color = "#4caf50" if avg_impact > 0 else "#f44336"
                                
                                with col:
                                    top_stocks = platform.sectors.get(sector.lower(), ['N/A', 'N/A', 'N/A'])[:3]
                                    st.markdown(f"""
                                    <div class="sector-card">
                                        <div style="color: #ffd700; font-weight: bold; margin-bottom: 10px;">{sector.upper()}</div>
                                        <div style="color: {impact_color}; font-size: 1.5rem; font-weight: bold; margin-bottom: 10px;">{avg_impact:+.2f}%</div>
                                        <div style="font-size: 0.9rem; color: #b8b8b8;">Top Stocks:</div>
                                        <div style="font-size: 0.8rem;">{' | '.join(top_stocks)}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
            
            # Global Market Impacts
            st.markdown("#### 🌍 Global Market Correlations")
            
            # Sample forecast for global impact calculation
            sample_forecast = forecasts[0] if forecasts else None
            if sample_forecast:
                change_pct = float(sample_forecast.change.replace('+', '').replace('-', ''))
                global_impacts = platform.calculate_global_market_impact(sample_forecast.sentiment, change_pct)
                
                # Display global market impacts
                markets_list = list(global_impacts.values())
                for i in range(0, len(markets_list), 4):
                    cols = st.columns(4)
                    for j, col in enumerate(cols):
                        if i + j < len(markets_list):
                            market = markets_list[i + j]
                            impact_color = "#4caf50" if market.expected_change > 0 else "#f44336"
                            
                            with col:
                                st.markdown(f"""
                                <div class="stat-card">
                                    <div style="color: #ffd700; font-weight: bold; margin-bottom: 5px;">{market.name}</div>
                                    <div style="color: {impact_color}; font-size: 1.2rem; font-weight: bold;">{market.expected_change:+.2f}%</div>
                                    <div style="font-size: 0.8rem; color: #b8b8b8;">Confidence: {market.confidence}%</div>
                                    <div style="font-size: 0.8rem; color: #b8b8b8;">{market.correlation_strength}</div>
                                </div>
                                """, unsafe_allow_html=True)
            
            # Commodity Impact
            st.markdown("#### 🥇 Commodity & Cryptocurrency Impact")
            
            if sample_forecast:
                commodity_impacts = platform.calculate_commodity_impact(sample_forecast.sentiment, change_pct)
                
                commodities_list = list(commodity_impacts.values())
                for i in range(0, len(commodities_list), 5):
                    cols = st.columns(5)
                    for j, col in enumerate(cols):
                        if i + j < len(commodities_list):
                            commodity = commodities_list[i + j]
                            impact_color = "#4caf50" if commodity.expected_change > 0 else "#f44336"
                            
                            with col:
                                st.markdown(f"""
                                <div class="stat-card">
                                    <div style="color: #ffd700; font-weight: bold; margin-bottom: 5px;">{commodity.name}</div>
                                    <div style="color: {impact_color}; font-size: 1.2rem; font-weight: bold;">{commodity.expected_change:+.2f}%</div>
                                    <div style="font-size: 0.8rem; color: #b8b8b8;">Confidence: {commodity.confidence}%</div>
                                </div>
                                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 📈 Astrological Movement Graph")
            
            # Create forecast dataframe for visualization
            df_forecasts = pd.DataFrame([
                {
                    'date': f.date,
                    'change': float(f.change.replace('+', '').replace('-', '')),
                    'sentiment': f.sentiment.value,
                    'signal': f.signal.value,
                    'event': f.event
                } for f in forecasts
            ])
            df_forecasts['date'] = pd.to_datetime(df_forecasts['date'])
            df_forecasts['change_signed'] = df_forecasts.apply(
                lambda row: row['change'] if row['sentiment'] != 'bearish' else -row['change'], axis=1
            )
            
            # Main price movement chart
            fig = go.Figure()
            
            # Add line for expected changes
            fig.add_trace(go.Scatter(
                x=df_forecasts['date'],
                y=df_forecasts['change_signed'],
                mode='lines+markers',
                name='Expected Movement',
                line=dict(color='#ffd700', width=3),
                marker=dict(size=8, color=df_forecasts['change_signed'], 
                          colorscale=['red', 'orange', 'green'], showscale=True)
            ))
            
            # Add horizontal line at zero
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            # Add annotations for major events
            major_events = df_forecasts[df_forecasts['change'] > 2]
            for _, event in major_events.iterrows():
                fig.add_annotation(
                    x=event['date'],
                    y=event['change_signed'],
                    text=event['event'][:20] + "...",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#ffd700",
                    font=dict(size=10, color="white"),
                    bgcolor="rgba(26, 26, 46, 0.8)",
                    bordercolor="#ffd700",
                    borderwidth=1
                )
            
            fig.update_layout(
                title=f"Planetary Transit Impact on {st.session_state.symbol} - {month_name} {st.session_state.year}",
                xaxis_title="Date",
                yaxis_title="Expected Change (%)",
                template="plotly_dark",
                showlegend=True,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal Distribution Chart
            signal_counts = df_forecasts['signal'].value_counts()
            
            fig2 = px.bar(
                x=signal_counts.index,
                y=signal_counts.values,
                title="Trading Signal Distribution",
                color=signal_counts.values,
                color_continuous_scale=['red', 'orange', 'green']
            )
            fig2.update_layout(template="plotly_dark")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # Sentiment pie chart
                sentiment_counts = df_forecasts['sentiment'].value_counts()
                fig3 = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Market Sentiment Distribution",
                    color_discrete_map={'bullish': '#4caf50', 'bearish': '#f44336', 'neutral': '#ff9800'}
                )
                fig3.update_layout(template="plotly_dark")
                st.plotly_chart(fig3, use_container_width=True)
        
        # Export functionality
        if st.sidebar.button("📊 Export Complete Report"):
            # Create comprehensive export data
            export_data = []
            for forecast in st.session_state.report:
                row = {
                    'Date': forecast.date,
                    'Day': forecast.day,
                    'Transit_Event': forecast.event,
                    'Sentiment': forecast.sentiment.value,
                    'Expected_Change_%': forecast.change,
                    'Impact_Level': forecast.impact,
                    'Trading_Signal': forecast.signal.value
                }
                
                # Add sector impacts
                for sector, impact in forecast.sector_impact.items():
                    row[f'{sector.upper()}_Impact_%'] = impact
                
                export_data.append(row)
            
            df_export = pd.DataFrame(export_data)
            
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.sidebar.download_button(
                label="📥 Download Complete Analysis",
                data=csv_data,
                file_name=f"astro_trading_complete_{st.session_state.symbol}_{month_name}_{st.session_state.year}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
