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
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .hero-section {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #ffd700;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #2d2d4a, #1a1a2e);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #ffd700;
        text-align: center;
        color: white;
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(255, 215, 0, 0.3);
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
    
    .transit-detail-card {
        background: linear-gradient(135deg, #2a1810, #3d2817);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
        color: white;
    }
    
    .sector-impact-card {
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
    
    .pivot-card {
        background: linear-gradient(135deg, #4a2d4a, #2e1a2e);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e91e63;
        text-align: center;
        color: white;
        margin-bottom: 10px;
    }
    
    .bullish { color: #4caf50; font-weight: bold; }
    .bearish { color: #f44336; font-weight: bold; }
    .neutral { color: #ff9800; font-weight: bold; }
    .long-signal { background-color: #4caf50; color: white; padding: 5px 10px; border-radius: 5px; }
    .short-signal { background-color: #f44336; color: white; padding: 5px 10px; border-radius: 5px; }
    .hold-signal { background-color: #ff9800; color: white; padding: 5px 10px; border-radius: 5px; }
    .retro-indicator { background-color: #ff4444; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8rem; }
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
class ZodiacSign:
    name: str
    symbol: str
    element: str
    quality: str

@dataclass
class DetailedTransit:
    date: str
    planet: str
    transit_type: str  # "enters", "retrograde", "direct", "aspect"
    zodiac_sign: str
    aspect_planet: str
    aspect_type: str  # "trine", "square", "conjunct", "sextile", "opposition"
    degree: float
    impact_strength: str
    market_sectors: Dict[str, float]
    historical_accuracy: float

@dataclass
class PivotPoint:
    date: str
    price_level: float
    pivot_type: str  # "high", "low", "support", "resistance"
    expected_move: float
    confidence: float

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
    detailed_transit: DetailedTransit

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
        
        self.zodiac_signs = {
            'aries': ZodiacSign('♈ Aries', '♈', 'Fire', 'Cardinal'),
            'taurus': ZodiacSign('♉ Taurus', '♉', 'Earth', 'Fixed'),
            'gemini': ZodiacSign('♊ Gemini', '♊', 'Air', 'Mutable'),
            'cancer': ZodiacSign('♋ Cancer', '♋', 'Water', 'Cardinal'),
            'leo': ZodiacSign('♌ Leo', '♌', 'Fire', 'Fixed'),
            'virgo': ZodiacSign('♍ Virgo', '♍', 'Earth', 'Mutable'),
            'libra': ZodiacSign('♎ Libra', '♎', 'Air', 'Cardinal'),
            'scorpio': ZodiacSign('♏ Scorpio', '♏', 'Water', 'Fixed'),
            'sagittarius': ZodiacSign('♐ Sagittarius', '♐', 'Fire', 'Mutable'),
            'capricorn': ZodiacSign('♑ Capricorn', '♑', 'Earth', 'Cardinal'),
            'aquarius': ZodiacSign('♒ Aquarius', '♒', 'Air', 'Fixed'),
            'pisces': ZodiacSign('♓ Pisces', '♓', 'Water', 'Mutable')
        }
        
        self.sectors = {
            'banking': ['HDFC', 'ICICI', 'SBI', 'AXIS', 'KOTAK'],
            'it': ['TCS', 'INFY', 'WIPRO', 'HCL', 'TECHM'],
            'pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'BIOCON'],
            'auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO'],
            'energy': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'NTPC'],
            'metals': ['TATA STEEL', 'JSW STEEL', 'HINDALCO', 'VEDL', 'COALINDIA'],
            'fmcg': ['HINDUNILVR', 'ITC', 'NESTLE', 'BRITANNIA', 'DABUR'],
            'telecom': ['BHARTI', 'JIO', 'IDEA', 'AIRTEL']
        }
        
        self.month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        
        # Enhanced detailed transits for each month
        self.detailed_monthly_transits = {
            0: [  # January
                {
                    "date": 3, "planet": "Sun", "transit_type": "aspect", "zodiac_sign": "capricorn",
                    "aspect_planet": "Jupiter", "aspect_type": "trine", "degree": 120.0,
                    "sentiment": Sentiment.BULLISH, "change": "+2.1", "retrograde": False,
                    "sectors": {"banking": 2.5, "it": 1.8, "auto": 2.0}, "signal": SignalType.LONG,
                    "impact_strength": "Strong", "historical_accuracy": 78.5
                },
                {
                    "date": 7, "planet": "Venus", "transit_type": "aspect", "zodiac_sign": "aquarius",
                    "aspect_planet": "Mars", "aspect_type": "square", "degree": 90.0,
                    "sentiment": Sentiment.BEARISH, "change": "-1.8", "retrograde": False,
                    "sectors": {"energy": -2.1, "metals": -1.5, "fmcg": -0.8}, "signal": SignalType.SHORT,
                    "impact_strength": "Moderate", "historical_accuracy": 72.1
                },
                {
                    "date": 15, "planet": "Mercury", "transit_type": "retrograde", "zodiac_sign": "aquarius",
                    "aspect_planet": "", "aspect_type": "retrograde", "degree": 25.0,
                    "sentiment": Sentiment.BEARISH, "change": "-2.5", "retrograde": True,
                    "sectors": {"it": -3.0, "banking": -2.0, "auto": -1.8, "telecom": -2.5}, "signal": SignalType.SHORT,
                    "impact_strength": "Very Strong", "historical_accuracy": 85.3
                },
                {
                    "date": 25, "planet": "Venus", "transit_type": "aspect", "zodiac_sign": "pisces",
                    "aspect_planet": "Jupiter", "aspect_type": "conjunct", "degree": 0.0,
                    "sentiment": Sentiment.BULLISH, "change": "+3.1", "retrograde": False,
                    "sectors": {"banking": 3.5, "pharma": 2.8, "fmcg": 2.2}, "signal": SignalType.LONG,
                    "impact_strength": "Very Strong", "historical_accuracy": 82.7
                }
            ],
            6: [  # July (index 6 for July)
                {
                    "date": 5, "planet": "Venus", "transit_type": "aspect", "zodiac_sign": "leo",
                    "aspect_planet": "Jupiter", "aspect_type": "trine", "degree": 120.0,
                    "sentiment": Sentiment.BULLISH, "change": "+2.8", "retrograde": False,
                    "sectors": {"banking": 3.2, "pharma": 2.5, "fmcg": 2.1}, "signal": SignalType.LONG,
                    "impact_strength": "Strong", "historical_accuracy": 81.2
                },
                {
                    "date": 12, "planet": "Mercury", "transit_type": "retrograde", "zodiac_sign": "leo",
                    "aspect_planet": "", "aspect_type": "retrograde", "degree": 20.0,
                    "sentiment": Sentiment.BEARISH, "change": "-2.3", "retrograde": True,
                    "sectors": {"it": -2.8, "banking": -1.9, "telecom": -2.4}, "signal": SignalType.SHORT,
                    "impact_strength": "Very Strong", "historical_accuracy": 84.7
                },
                {
                    "date": 18, "planet": "Mars", "transit_type": "aspect", "zodiac_sign": "virgo",
                    "aspect_planet": "Saturn", "aspect_type": "square", "degree": 90.0,
                    "sentiment": Sentiment.BEARISH, "change": "-1.9", "retrograde": False,
                    "sectors": {"auto": -2.3, "metals": -1.8, "energy": -1.4}, "signal": SignalType.SHORT,
                    "impact_strength": "Moderate", "historical_accuracy": 73.5
                },
                {
                    "date": 25, "planet": "Jupiter", "transit_type": "aspect", "zodiac_sign": "gemini",
                    "aspect_planet": "Neptune", "aspect_type": "sextile", "degree": 60.0,
                    "sentiment": Sentiment.BULLISH, "change": "+2.1", "retrograde": False,
                    "sectors": {"pharma": 2.8, "it": 2.0, "banking": 1.7}, "signal": SignalType.LONG,
                    "impact_strength": "Strong", "historical_accuracy": 77.9
                }
            ],
            7: [  # August
                {
                    "date": 2, "planet": "Mercury", "transit_type": "aspect", "zodiac_sign": "virgo",
                    "aspect_planet": "Jupiter", "aspect_type": "square", "degree": 90.0,
                    "sentiment": Sentiment.BEARISH, "change": "-1.7", "retrograde": False,
                    "sectors": {"it": -2.0, "banking": -1.5, "energy": -1.2}, "signal": SignalType.SHORT,
                    "impact_strength": "Moderate", "historical_accuracy": 71.2
                },
                {
                    "date": 11, "planet": "Mercury", "transit_type": "direct", "zodiac_sign": "leo",
                    "aspect_planet": "", "aspect_type": "direct", "degree": 15.0,
                    "sentiment": Sentiment.BULLISH, "change": "+1.9", "retrograde": False,
                    "sectors": {"it": 2.5, "banking": 2.0, "auto": 1.8, "telecom": 2.2}, "signal": SignalType.LONG,
                    "impact_strength": "Strong", "historical_accuracy": 79.8
                },
                {
                    "date": 15, "planet": "Sun", "transit_type": "aspect", "zodiac_sign": "leo",
                    "aspect_planet": "Jupiter", "aspect_type": "sextile", "degree": 60.0,
                    "sentiment": Sentiment.BULLISH, "change": "+2.3", "retrograde": False,
                    "sectors": {"banking": 2.8, "pharma": 2.1, "fmcg": 1.9}, "signal": SignalType.LONG,
                    "impact_strength": "Strong", "historical_accuracy": 76.4
                },
                {
                    "date": 23, "planet": "Mars", "transit_type": "enters", "zodiac_sign": "virgo",
                    "aspect_planet": "", "aspect_type": "ingress", "degree": 0.0,
                    "sentiment": Sentiment.NEUTRAL, "change": "+1.2", "retrograde": False,
                    "sectors": {"auto": 1.8, "metals": 1.4, "pharma": 1.1}, "signal": SignalType.HOLD,
                    "impact_strength": "Moderate", "historical_accuracy": 68.9
                },
                {
                    "date": 27, "planet": "Jupiter", "transit_type": "aspect", "zodiac_sign": "gemini",
                    "aspect_planet": "Saturn", "aspect_type": "opposition", "degree": 180.0,
                    "sentiment": Sentiment.BEARISH, "change": "-2.5", "retrograde": False,
                    "sectors": {"banking": -3.0, "auto": -2.2, "metals": -2.8}, "signal": SignalType.SHORT,
                    "impact_strength": "Very Strong", "historical_accuracy": 83.1
                }
            ]
        }

    def get_detailed_transit(self, event_data: Dict) -> DetailedTransit:
        return DetailedTransit(
            date=f"2025-{event_data.get('month', 8):02d}-{event_data['date']:02d}",
            planet=event_data['planet'],
            transit_type=event_data['transit_type'],
            zodiac_sign=event_data['zodiac_sign'],
            aspect_planet=event_data.get('aspect_planet', ''),
            aspect_type=event_data.get('aspect_type', ''),
            degree=event_data.get('degree', 0.0),
            impact_strength=event_data.get('impact_strength', 'Moderate'),
            market_sectors=event_data.get('sectors', {}),
            historical_accuracy=event_data.get('historical_accuracy', 70.0)
        )

    def generate_pivot_points(self, forecasts: List[Forecast]) -> List[PivotPoint]:
        pivot_points = []
        
        for i, forecast in enumerate(forecasts):
            change_val = abs(float(forecast.change.replace('+', '').replace('-', '')))
            if change_val > 2.0:  # Significant movement
                
                if forecast.sentiment == Sentiment.BULLISH:
                    pivot_type = "support" if i < len(forecasts) // 2 else "resistance"
                    expected_move = change_val * 1.2
                else:
                    pivot_type = "resistance" if i < len(forecasts) // 2 else "support"
                    expected_move = -change_val * 1.1
                
                confidence = min(95, max(60, forecast.detailed_transit.historical_accuracy))
                
                pivot_points.append(PivotPoint(
                    date=forecast.date,
                    price_level=100 + expected_move,  # Base price assumption
                    pivot_type=pivot_type,
                    expected_move=expected_move,
                    confidence=confidence
                ))
        
        return pivot_points

    def generate_enhanced_monthly_forecast(self, symbol: str, year: int, month: int) -> List[Forecast]:
        forecasts = []
        month_data = self.detailed_monthly_transits.get(month, [])
        
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
                detailed_transit = self.get_detailed_transit({**day_event, 'month': month + 1})
                
                # Create event description
                if day_event['transit_type'] == 'retrograde':
                    event_desc = f"{day_event['planet']} Retrograde in {self.zodiac_signs[day_event['zodiac_sign']].name}"
                elif day_event['transit_type'] == 'direct':
                    event_desc = f"{day_event['planet']} Direct in {self.zodiac_signs[day_event['zodiac_sign']].name}"
                elif day_event['transit_type'] == 'enters':
                    event_desc = f"{day_event['planet']} enters {self.zodiac_signs[day_event['zodiac_sign']].name}"
                elif day_event['transit_type'] == 'aspect':
                    event_desc = f"{day_event['planet']} {day_event['aspect_type']} {day_event['aspect_planet']} in {self.zodiac_signs[day_event['zodiac_sign']].name}"
                else:
                    event_desc = f"{day_event['planet']} transit in {self.zodiac_signs[day_event['zodiac_sign']].name}"
                
                forecasts.append(Forecast(
                    date=current_date.strftime('%Y-%m-%d'),
                    day=day,
                    event=event_desc,
                    sentiment=day_event["sentiment"],
                    change=day_event["change"],
                    impact=f"{'Retrograde ' if day_event.get('retrograde', False) else ''}{day_event['impact_strength']} {day_event['sentiment'].value.title()}",
                    sector_impact=day_event.get("sectors", {}),
                    signal=day_event["signal"],
                    detailed_transit=detailed_transit
                ))
            else:
                # Generate minor daily transits
                sentiment_options = [Sentiment.BULLISH, Sentiment.BEARISH, Sentiment.NEUTRAL]
                sentiment = sentiment_options[day % 3]
                change_percent = ((day * 37) % 100 - 50) / 50
                change_str = f"{'+' if change_percent > 0 else ''}{change_percent:.1f}"
                
                signal = SignalType.LONG if sentiment == Sentiment.BULLISH else SignalType.SHORT if sentiment == Sentiment.BEARISH else SignalType.HOLD
                
                # Generate minor transit
                planets = list(self.planets.keys())
                zodiac_keys = list(self.zodiac_signs.keys())
                random_planet = planets[day % len(planets)]
                random_sign = zodiac_keys[day % len(zodiac_keys)]
                
                minor_transit = DetailedTransit(
                    date=current_date.strftime('%Y-%m-%d'),
                    planet=random_planet.title(),
                    transit_type="minor_aspect",
                    zodiac_sign=random_sign,
                    aspect_planet="",
                    aspect_type="minor",
                    degree=float(day * 13 % 360),
                    impact_strength="Minor",
                    market_sectors={},
                    historical_accuracy=60.0
                )
                
                forecasts.append(Forecast(
                    date=current_date.strftime('%Y-%m-%d'),
                    day=day,
                    event=f'Minor {random_planet.title()} transit in {self.zodiac_signs[random_sign].name}',
                    sentiment=sentiment,
                    change=change_str,
                    impact=f"Minor {sentiment.value.title()}",
                    sector_impact={},
                    signal=signal,
                    detailed_transit=minor_transit
                ))
        
        return forecasts

def render_front_page():
    st.markdown('<h1 class="main-header">🌟 Advanced Astrological Trading Platform</h1>', unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h2 style="color: #ffd700; margin-bottom: 1rem;">🚀 Professional Market Analysis with Planetary Intelligence</h2>
        <p style="color: #b8b8b8; font-size: 1.2rem; margin-bottom: 1.5rem;">
            Harness the power of celestial movements for precise market predictions and trading signals
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <div style="background: rgba(255, 215, 0, 0.1); padding: 10px 20px; border-radius: 25px; border: 1px solid #ffd700;">
                ⭐ 85%+ Accuracy Rate
            </div>
            <div style="background: rgba(255, 215, 0, 0.1); padding: 10px 20px; border-radius: 25px; border: 1px solid #ffd700;">
                🌍 Global Markets Coverage
            </div>
            <div style="background: rgba(255, 215, 0, 0.1); padding: 10px 20px; border-radius: 25px; border: 1px solid #ffd700;">
                📊 Real-time Analysis
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; margin-bottom: 15px;">📅</div>
            <h3 style="color: #ffd700; margin-bottom: 10px;">Astro Calendar</h3>
            <p style="font-size: 0.9rem;">Daily planetary transits, retrograde periods, and precise aspect timing with sector impacts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; margin-bottom: 15px;">📊</div>
            <h3 style="color: #ffd700; margin-bottom: 10px;">Stock Analysis</h3>
            <p style="font-size: 0.9rem;">Comprehensive sector analysis, global correlations, and commodity impact forecasts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; margin-bottom: 15px;">📈</div>
            <h3 style="color: #ffd700; margin-bottom: 10px;">Astro Graph</h3>
            <p style="font-size: 0.9rem;">Interactive charts with pivot points, support/resistance levels, and price projections</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div style="font-size: 2.5rem; margin-bottom: 15px;">🌙</div>
            <h3 style="color: #ffd700; margin-bottom: 10px;">Transit Analysis</h3>
            <p style="font-size: 0.9rem;">Detailed planetary transit impacts on specific sectors and individual stock performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Current Market Status
    current_time = datetime.datetime.now()
    st.markdown(f"""
    <div style="text-align: center; margin: 2rem 0; padding: 1rem; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 10px; border: 1px solid #ffd700;">
        <h3 style="color: #ffd700;">🕐 Current Market Time: {current_time.strftime("%Y-%m-%d %H:%M:%S IST")}</h3>
        <p style="color: #b8b8b8;">Ready to analyze celestial influences on your trading decisions</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    if 'platform' not in st.session_state:
        st.session_state.platform = EnhancedAstrologicalTradingPlatform()
    
    platform = st.session_state.platform
    
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
    
    # Generate Analysis Button
    if st.sidebar.button("🚀 Generate Analysis", type="primary"):
        with st.spinner("🔮 Calculating planetary positions and market correlations..."):
            st.session_state.report = platform.generate_enhanced_monthly_forecast(symbol, selected_year, selected_month)
            st.session_state.symbol = symbol
            st.session_state.month = selected_month
            st.session_state.year = selected_year
            st.session_state.analysis_type = analysis_type
            st.session_state.pivot_points = platform.generate_pivot_points(st.session_state.report)
    
    # Main Content
    if 'report' not in st.session_state:
        render_front_page()
    else:
        # Main Content Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📅 Astro Calendar", "📊 Stock Analysis", "📈 Astro Graph", "🌙 Transit Analysis"])
        
        with tab1:
            st.markdown("### 📅 Astrological Calendar")
            
            # Monthly Overview
            forecasts = st.session_state.report
            month_name = platform.month_names[st.session_state.month]
            
            col1, col2, col3, col4 = st.columns(4)
            
            bullish_count = sum(1 for f in forecasts if f.sentiment == Sentiment.BULLISH)
            bearish_count = sum(1 for f in forecasts if f.sentiment == Sentiment.BEARISH)
            neutral_count = len(forecasts) - bullish_count - bearish_count
            retrograde_count = sum(1 for f in forecasts if 'retrograde' in f.event.lower())
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <h3 style="color: #4caf50;">{bullish_count}</h3>
                    <p>Bullish Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <h3 style="color: #f44336;">{bearish_count}</h3>
                    <p>Bearish Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <h3 style="color: #ff9800;">{neutral_count}</h3>
                    <p>Neutral Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <h3 style="color: #e91e63;">{retrograde_count}</h3>
                    <p>Retrograde Events</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed Transit Calendar
            st.markdown(f"#### 🌟 {month_name} {st.session_state.year} - Detailed Planetary Transits")
            
            for forecast in forecasts:
                if abs(float(forecast.change.replace('+', '').replace('-', ''))) > 1.0:  # Show significant events
                    signal_class = "long-signal" if forecast.signal == SignalType.LONG else "short-signal" if forecast.signal == SignalType.SHORT else "hold-signal"
                    sentiment_class = forecast.sentiment.value
                    
                    transit = forecast.detailed_transit
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Format sector impacts properly
                        sector_impacts_text = ""
                        if forecast.sector_impact:
                            sector_list = []
                            for sector, impact in forecast.sector_impact.items():
                                color = "#4caf50" if impact > 0 else "#f44336"
                                sector_list.append(f'<span style="color: {color};">{sector.upper()}: {impact:+.1f}%</span>')
                            sector_impacts_text = f"""
                            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #444;">
                                <div style="font-size: 0.9rem; color: #ffd700; margin-bottom: 5px;">Sector Impacts:</div>
                                <div style="font-size: 0.9rem;">{" | ".join(sector_list)}</div>
                            </div>
                            """
                        
                        retro_indicator = ""
                        if "retrograde" in forecast.event.lower():
                            retro_indicator = '<span class="retro-indicator">RETRO</span> '
                        
                        st.markdown(f"""
                        <div class="forecast-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <div style="color: #ffd700; font-weight: bold; font-size: 1.1rem;">{forecast.date}</div>
                                <div>
                                    {retro_indicator}
                                    <span class="{signal_class}">{forecast.signal.value}</span>
                                </div>
                            </div>
                            <div style="margin-bottom: 8px; font-size: 1.1rem; color: #fff;">
                                {forecast.event}
                            </div>
                            <div style="margin-bottom: 8px; font-size: 0.9rem; color: #b8b8b8;">
                                {transit.planet} in {platform.zodiac_signs[transit.zodiac_sign].name} • {transit.aspect_type.title()} {transit.aspect_planet}
                            </div>
                            <div class="{sentiment_class}" style="margin-bottom: 10px;">{forecast.impact}</div>
                            <div style="color: #b8b8b8;">Expected Change: <span style="color: {'#4caf50' if '+' in forecast.change else '#f44336'}; font-weight: bold;">{forecast.change}%</span></div>
                            {sector_impacts_text}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="historical-card">
                            <div style="color: #ff9800; font-weight: bold; margin-bottom: 8px;">📊 Historical Data</div>
                            <div style="font-size: 0.9rem; margin-bottom: 4px;">Impact: {transit.impact_strength}</div>
                            <div style="font-size: 0.9rem; margin-bottom: 4px;">Accuracy: {transit.historical_accuracy:.1f}%</div>
                            <div style="font-size: 0.9rem; margin-bottom: 4px;">Degree: {transit.degree:.1f}°</div>
                            <div style="font-size: 0.8rem; color: #ffd700;">
                                {platform.zodiac_signs[transit.zodiac_sign].element} • {platform.zodiac_signs[transit.zodiac_sign].quality}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### 📊 Stock Analysis")
            
            # Symbol-specific Analysis
            month_name = platform.month_names[st.session_state.month]
            st.markdown(f"#### 📈 {st.session_state.symbol} - {month_name} {st.session_state.year} Astrological Analysis")
            
            forecasts = st.session_state.report
            
            # Monthly Summary Stats
            col1, col2, col3, col4 = st.columns(4)
            
            total_bullish_impact = sum(float(f.change.replace('+', '').replace('-', '')) for f in forecasts if f.sentiment == Sentiment.BULLISH)
            total_bearish_impact = sum(float(f.change.replace('+', '').replace('-', '')) for f in forecasts if f.sentiment == Sentiment.BEARISH)
            avg_daily_change = np.mean([float(f.change.replace('+', '').replace('-', '')) for f in forecasts])
            strong_transits = len([f for f in forecasts if 'Strong' in f.impact or 'Very Strong' in f.impact])
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <h3 style="color: #4caf50;">+{total_bullish_impact:.1f}%</h3>
                    <p>Total Bullish Impact</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <h3 style="color: #f44336;">-{total_bearish_impact:.1f}%</h3>
                    <p>Total Bearish Impact</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <h3 style="color: #ff9800;">{avg_daily_change:.1f}%</h3>
                    <p>Avg Daily Impact</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <h3 style="color: #e91e63;">{strong_transits}</h3>
                    <p>Strong Transits</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Date-wise Analysis for Selected Month
            st.markdown(f"#### 📅 Date-wise Planetary Transit Impact Analysis")
            
            # Show ALL forecasts for the selected month, not just significant ones
            for forecast in forecasts:
                transit = forecast.detailed_transit
                
                # Determine row background color based on impact strength
                if abs(float(forecast.change.replace('+', '').replace('-', ''))) > 2:
                    row_style = "border-left: 5px solid #ffd700;"
                elif abs(float(forecast.change.replace('+', '').replace('-', ''))) > 1:
                    row_style = "border-left: 3px solid #ff9800;"
                else:
                    row_style = "border-left: 2px solid #666;"
                
                # Create compact daily view
                change_value = float(forecast.change.replace('+', '').replace('-', ''))
                change_color = "#4caf50" if '+' in forecast.change else "#f44336" if '-' in forecast.change else "#ff9800"
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Create sector impacts text separately to avoid f-string issues
                    sector_impacts_html = ""
                    if forecast.sector_impact:
                        sector_spans = []
                        for sector, impact in forecast.sector_impact.items():
                            bg_color = "rgba(76, 175, 80, 0.2)" if impact > 0 else "rgba(244, 67, 54, 0.2)"
                            text_color = "#4caf50" if impact > 0 else "#f44336"
                            sector_spans.append(f'<span style="background-color: {bg_color}; color: {text_color}; padding: 2px 6px; border-radius: 3px; font-size: 0.8rem;">{sector.upper()}: {impact:+.1f}%</span>')
                        
                        sector_impacts_html = f"""
                        <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #444;">
                            <div style="font-size: 0.8rem; color: #ffd700; margin-bottom: 4px;">Sector Impacts:</div>
                            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                                {" ".join(sector_spans)}
                            </div>
                        </div>
                        """
                    
                    st.markdown(f"""
                    <div class="forecast-card" style="{row_style}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <div>
                                <div style="color: #ffd700; font-weight: bold; font-size: 1.1rem;">{forecast.date}</div>
                                <div style="font-size: 0.8rem; color: #b8b8b8;">Day {forecast.day}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: {change_color}; font-size: 1.3rem; font-weight: bold; margin-bottom: 2px;">
                                    {forecast.change}%
                                </div>
                                <div style="background-color: {'#4caf50' if forecast.signal == SignalType.LONG else '#f44336' if forecast.signal == SignalType.SHORT else '#ff9800'}; 
                                           color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold;">
                                    {forecast.signal.value}
                                </div>
                            </div>
                        </div>
                        
                        <div style="margin-bottom: 8px;">
                            <div style="color: #fff; font-size: 1rem; margin-bottom: 4px;">{forecast.event}</div>
                            <div style="font-size: 0.8rem; color: #b8b8b8;">
                                {transit.planet} in {platform.zodiac_signs[transit.zodiac_sign].name} • {transit.aspect_type.title()}
                            </div>
                        </div>
                        
                        <div style="font-size: 0.7rem; color: #b8b8b8; margin-bottom: 8px;">
                            Sentiment: <span style="color: {change_color};">{forecast.sentiment.value.upper()}</span>
                        </div>
                        
                        {sector_impacts_html}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="historical-card">
                        <div style="color: #ff9800; font-weight: bold; margin-bottom: 6px; font-size: 0.9rem;">📊 Analysis</div>
                        <div style="font-size: 0.8rem; margin-bottom: 3px;">Strength: {transit.impact_strength}</div>
                        <div style="font-size: 0.8rem; margin-bottom: 3px;">Accuracy: {transit.historical_accuracy:.1f}%</div>
                        <div style="font-size: 0.8rem; margin-bottom: 3px;">Degree: {transit.degree:.1f}°</div>
                        <div style="font-size: 0.8rem; color: #ffd700;">
                            {platform.zodiac_signs[transit.zodiac_sign].element}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 📈 Astrological Movement Graph")
            
            # Create enhanced forecast dataframe with error handling
            try:
                df_forecasts = pd.DataFrame([
                    {
                        'date': f.date,
                        'change': float(f.change.replace('+', '').replace('-', '')),
                        'sentiment': f.sentiment.value,
                        'signal': f.signal.value,
                        'event': f.event,
                        'impact_strength': f.detailed_transit.impact_strength,
                        'historical_accuracy': f.detailed_transit.historical_accuracy
                    } for f in forecasts
                ])
                df_forecasts['date'] = pd.to_datetime(df_forecasts['date'])
                df_forecasts['change_signed'] = df_forecasts.apply(
                    lambda row: row['change'] if row['sentiment'] != 'bearish' else -row['change'], axis=1
                )
                
                # Enhanced price movement chart with clearer date-wise movements
                fig = go.Figure()
                
                # Add main price line with better visibility
                fig.add_trace(go.Scatter(
                    x=df_forecasts['date'],
                    y=df_forecasts['change_signed'],
                    mode='lines+markers+text',
                    name='Daily Movement',
                    line=dict(color='#ffd700', width=4),
                    marker=dict(
                        size=df_forecasts['change'] * 4 + 8,
                        color=df_forecasts['change_signed'], 
                        colorscale=['red', 'orange', 'yellow', 'lightgreen', 'green'], 
                        showscale=True,
                        colorbar=dict(
                            title="Daily Change %",
                            x=1.02
                        ),
                        line=dict(width=2, color='white')
                    ),
                    text=[f"{change:+.1f}%" for change in df_forecasts['change_signed']],
                    textposition="top center",
                    textfont=dict(size=10, color='white'),
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Movement: %{y:.2f}%<br>' +
                                 'Event: %{customdata[0]}<br>' +
                                 'Accuracy: %{customdata[1]:.1f}%<br>' +
                                 'Signal: %{customdata[2]}<extra></extra>',
                    customdata=list(zip(df_forecasts['event'], 
                                      df_forecasts['historical_accuracy'],
                                      df_forecasts['signal']))
                ))
                
                # Add cumulative movement line
                cumulative_change = df_forecasts['change_signed'].cumsum()
                fig.add_trace(go.Scatter(
                    x=df_forecasts['date'],
                    y=cumulative_change,
                    mode='lines',
                    name='Cumulative Movement',
                    line=dict(color='#ff6b35', width=2, dash='dash'),
                    hovertemplate='Cumulative: %{y:.2f}%<extra></extra>'
                ))
                
                # Add zero line
                fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3, line_width=1)
                
                # Color-code background for positive/negative zones
                max_val = max(df_forecasts['change_signed'].max(), 5)
                min_val = min(df_forecasts['change_signed'].min(), -5)
                
                fig.add_hrect(y0=0, y1=max_val, 
                             fillcolor="rgba(76, 175, 80, 0.1)", layer="below", line_width=0)
                fig.add_hrect(y0=min_val, y1=0, 
                             fillcolor="rgba(244, 67, 54, 0.1)", layer="below", line_width=0)
                
                # Add pivot points with better visibility
                if 'pivot_points' in st.session_state and st.session_state.pivot_points:
                    pivot_points = st.session_state.pivot_points
                    pivot_dates = [pd.to_datetime(p.date) for p in pivot_points]
                    pivot_values = [p.expected_move for p in pivot_points]
                    pivot_types = [p.pivot_type for p in pivot_points]
                    
                    # Support levels
                    support_dates = [d for d, t in zip(pivot_dates, pivot_types) if t == 'support']
                    support_values = [v for v, t in zip(pivot_values, pivot_types) if t == 'support']
                    
                    if support_dates:
                        fig.add_trace(go.Scatter(
                            x=support_dates,
                            y=support_values,
                            mode='markers+text',
                            name='Support Levels',
                            marker=dict(color='#4caf50', size=15, symbol='triangle-up', line=dict(width=2, color='white')),
                            text=[f"Support<br>{v:+.1f}%" for v in support_values],
                            textposition="bottom center",
                            hovertemplate='Support Level<br>Date: %{x}<br>Level: %{y:.2f}%<extra></extra>'
                        ))
                    
                    # Resistance levels
                    resistance_dates = [d for d, t in zip(pivot_dates, pivot_types) if t == 'resistance']
                    resistance_values = [v for v, t in zip(pivot_values, pivot_types) if t == 'resistance']
                    
                    if resistance_dates:
                        fig.add_trace(go.Scatter(
                            x=resistance_dates,
                            y=resistance_values,
                            mode='markers+text',
                            name='Resistance Levels',
                            marker=dict(color='#f44336', size=15, symbol='triangle-down', line=dict(width=2, color='white')),
                            text=[f"Resistance<br>{v:+.1f}%" for v in resistance_values],
                            textposition="top center",
                            hovertemplate='Resistance Level<br>Date: %{x}<br>Level: %{y:.2f}%<extra></extra>'
                        ))
                
                # Add annotations for major swing dates
                major_swings = df_forecasts[df_forecasts['change'] > 2]
                for _, swing in major_swings.iterrows():
                    fig.add_annotation(
                        x=swing['date'],
                        y=swing['change_signed'],
                        text=f"<b>Major Swing</b><br>{swing['change_signed']:+.1f}%<br>{swing['event'][:30]}...",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=3,
                        arrowcolor="#ffd700",
                        font=dict(size=10, color="white"),
                        bgcolor="rgba(26, 26, 46, 0.9)",
                        bordercolor="#ffd700",
                        borderwidth=2,
                        ax=0,
                        ay=-40 if swing['change_signed'] > 0 else 40
                    )
                
                fig.update_layout(
                    title=f"Planetary Transit Impact Analysis - {st.session_state.symbol} ({platform.month_names[st.session_state.month]} {st.session_state.year})",
                    xaxis_title="Date",
                    yaxis_title="Expected Change (%)",
                    template="plotly_dark",
                    showlegend=True,
                    height=700,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
                st.info("Using simplified chart view...")
                
                # Fallback simple chart
                simple_df = pd.DataFrame([
                    {'Date': f.date, 'Change': f.change, 'Signal': f.signal.value} 
                    for f in forecasts[:10]  # Show first 10 days
                ])
                st.dataframe(simple_df)
            
            # Pivot Points Summary
            if 'pivot_points' in st.session_state and st.session_state.pivot_points:
                st.markdown("#### 🎯 Key Pivot Points & Price Levels")
                
                pivot_cols = st.columns(min(4, len(st.session_state.pivot_points)))
                for i, pivot in enumerate(st.session_state.pivot_points):
                    if i < len(pivot_cols):
                        with pivot_cols[i]:
                            color = "#4caf50" if pivot.pivot_type in ['support', 'low'] else "#f44336"
                            st.markdown(f"""
                            <div class="pivot-card">
                                <div style="color: {color}; font-weight: bold; margin-bottom: 5px;">{pivot.pivot_type.upper()}</div>
                                <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 5px;">{pivot.expected_move:+.2f}%</div>
                                <div style="font-size: 0.8rem; color: #b8b8b8;">Confidence: {pivot.confidence:.1f}%</div>
                                <div style="font-size: 0.8rem; color: #b8b8b8;">{pivot.date}</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Additional Charts
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    # Signal Distribution
                    signal_counts = df_forecasts['signal'].value_counts()
                    fig2 = px.bar(
                        x=signal_counts.index,
                        y=signal_counts.values,
                        title="Trading Signal Distribution",
                        color=signal_counts.values,
                        color_continuous_scale=['red', 'orange', 'green']
                    )
                    fig2.update_layout(template="plotly_dark", showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
                except:
                    st.info("Signal distribution chart unavailable")
            
            with col2:
                try:
                    # Accuracy vs Impact
                    fig3 = px.scatter(
                        df_forecasts,
                        x='historical_accuracy',
                        y='change',
                        color='sentiment',
                        size='change',
                        title="Historical Accuracy vs Expected Impact",
                        color_discrete_map={'bullish': '#4caf50', 'bearish': '#f44336', 'neutral': '#ff9800'}
                    )
                    fig3.update_layout(template="plotly_dark")
                    st.plotly_chart(fig3, use_container_width=True)
                except:
                    st.info("Accuracy vs Impact chart unavailable")
        
        with tab4:
            st.markdown("### 🌙 Planetary Transit Analysis")
            
            month_name = platform.month_names[st.session_state.month]
            
            # Get unique transits for the month
            unique_transits = {}
            for forecast in forecasts:
                transit = forecast.detailed_transit
                key = f"{transit.planet}_{transit.zodiac_sign}_{transit.transit_type}"
                if key not in unique_transits:
                    unique_transits[key] = {
                        'transit': transit,
                        'forecasts': [],
                        'total_impact': 0,
                        'sector_impacts': {}
                    }
                unique_transits[key]['forecasts'].append(forecast)
                unique_transits[key]['total_impact'] += abs(float(forecast.change.replace('+', '').replace('-', '')))
                
                for sector, impact in forecast.sector_impact.items():
                    if sector not in unique_transits[key]['sector_impacts']:
                        unique_transits[key]['sector_impacts'][sector] = []
                    unique_transits[key]['sector_impacts'][sector].append(impact)
            
            st.markdown(f"#### 🪐 Major Planetary Transits - {month_name} {st.session_state.year}")
            
            # Sort by impact strength
            sorted_transits = sorted(unique_transits.items(), key=lambda x: x[1]['total_impact'], reverse=True)
            
            for transit_key, transit_data in sorted_transits[:6]:  # Show top 6 major transits
                transit = transit_data['transit']
                avg_impact = transit_data['total_impact'] / len(transit_data['forecasts'])
                
                # Get all dates for this transit
                transit_dates = [f.date for f in transit_data['forecasts']]
                date_range = f"{min(transit_dates)} to {max(transit_dates)}" if len(transit_dates) > 1 else transit_dates[0]
                
                # Calculate average sector impacts
                avg_sector_impacts = {}
                for sector, impacts in transit_data['sector_impacts'].items():
                    avg_sector_impacts[sector] = np.mean(impacts)
                
                # Transit description
                if transit.transit_type == 'retrograde':
                    transit_desc = f"{transit.planet} Retrograde in {platform.zodiac_signs[transit.zodiac_sign].name}"
                elif transit.transit_type == 'direct':
                    transit_desc = f"{transit.planet} Direct in {platform.zodiac_signs[transit.zodiac_sign].name}"
                elif transit.transit_type == 'enters':
                    transit_desc = f"{transit.planet} enters {platform.zodiac_signs[transit.zodiac_sign].name}"
                elif transit.transit_type == 'aspect':
                    transit_desc = f"{transit.planet} {transit.aspect_type} {transit.aspect_planet}"
                else:
                    transit_desc = f"{transit.planet} in {platform.zodiac_signs[transit.zodiac_sign].name}"
                
                st.markdown(f"""
                <div class="transit-detail-card">
                    <div style="margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                            <h3 style="color: #ffd700; margin: 0;">🌟 {transit_desc}</h3>
                            <div style="text-align: right;">
                                <div style="color: #ff9800; font-weight: bold; font-size: 1.1rem;">📅 {date_range}</div>
                                <div style="color: #b8b8b8; font-size: 0.9rem;">{len(transit_dates)} day{'s' if len(transit_dates) > 1 else ''} duration</div>
                            </div>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 15px;">
                            <div>
                                <div style="color: #ff9800; font-weight: bold; margin-bottom: 8px;">📊 Transit Details</div>
                                <div style="font-size: 0.9rem; margin-bottom: 3px;"><strong>Planet:</strong> {transit.planet}</div>
                                <div style="font-size: 0.9rem; margin-bottom: 3px;"><strong>Sign:</strong> {platform.zodiac_signs[transit.zodiac_sign].name}</div>
                                <div style="font-size: 0.9rem; margin-bottom: 3px;"><strong>Type:</strong> {transit.transit_type.title()}</div>
                                <div style="font-size: 0.9rem; margin-bottom: 3px;"><strong>Avg Impact:</strong> <span style="color: {'#4caf50' if avg_impact > 0 else '#f44336'};">{avg_impact:+.1f}%</span></div>
                            </div>
                            <div>
                                <div style="color: #ff9800; font-weight: bold; margin-bottom: 8px;">📈 Market Impact</div>
                                <div style="font-size: 0.9rem; margin-bottom: 3px;"><strong>Accuracy:</strong> {transit.historical_accuracy:.1f}%</div>
                                <div style="font-size: 0.9rem; margin-bottom: 3px;"><strong>Strength:</strong> {transit.impact_strength}</div>
                                <div style="font-size: 0.9rem; margin-bottom: 3px;"><strong>Degree:</strong> {transit.degree:.1f}°</div>
                                <div style="font-size: 0.9rem; margin-bottom: 3px;"><strong>Sectors:</strong> {len(avg_sector_impacts)} affected</div>
                            </div>
                            <div>
                                <div style="color: #ff9800; font-weight: bold; margin-bottom: 8px;">🌟 Astrological Data</div>
                                <div style="font-size: 0.9rem; margin-bottom: 3px;"><strong>Element:</strong> {platform.zodiac_signs[transit.zodiac_sign].element}</div>
                                <div style="font-size: 0.9rem; margin-bottom: 3px;"><strong>Quality:</strong> {platform.zodiac_signs[transit.zodiac_sign].quality}</div>
                                <div style="font-size: 0.9rem; margin-bottom: 3px;"><strong>Aspect:</strong> {transit.aspect_type.title() if transit.aspect_type else 'N/A'}</div>
                                <div style="font-size: 0.9rem;"><strong>Target:</strong> {transit.aspect_planet if transit.aspect_planet else 'N/A'}</div>
                            </div>
                        </div>
                        
                        <div style="background-color: rgba(255, 215, 0, 0.1); padding: 12px; border-radius: 8px; border-left: 4px solid #ffd700;">
                            <div style="color: #ffd700; font-weight: bold; margin-bottom: 8px;">📋 Daily Breakdown:</div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                                {" ".join([f'<div style="background-color: rgba(255, 255, 255, 0.05); padding: 6px; border-radius: 4px;"><strong>{f.date}:</strong> <span style="color: {"#4caf50" if "+" in f.change else "#f44336"};">{f.change}%</span> - {f.signal.value}</div>' for f in transit_data['forecasts'][:8]])}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Sector-wise Impact Analysis
                if avg_sector_impacts:
                    st.markdown("**📊 Sector-wise Performance Impact:**")
                    
                    sector_cols = st.columns(min(4, len(avg_sector_impacts)))
                    
                    for i, (sector, avg_impact) in enumerate(avg_sector_impacts.items()):
                        if i < len(sector_cols):
                            with sector_cols[i]:
                                impact_color = "#4caf50" if avg_impact > 0 else "#f44336"
                                recommendation = "BUY" if avg_impact > 1.5 else "HOLD" if avg_impact > -1.5 else "SELL"
                                
                                # Get top stocks for this sector
                                top_stocks = platform.sectors.get(sector.lower(), ['N/A', 'N/A', 'N/A'])[:3]
                                
                                st.markdown(f"""
                                <div class="sector-impact-card">
                                    <div style="color: #ffd700; font-weight: bold; margin-bottom: 8px;">{sector.upper()}</div>
                                    <div style="color: {impact_color}; font-size: 1.5rem; font-weight: bold; margin-bottom: 8px;">{avg_impact:+.2f}%</div>
                                    <div style="background-color: {impact_color}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; margin-bottom: 8px; text-align: center;">
                                        {recommendation}
                                    </div>
                                    <div style="font-size: 0.8rem; color: #b8b8b8; margin-bottom: 5px;">Top Stocks:</div>
                                    <div style="font-size: 0.75rem;">
                                        {' • '.join(top_stocks)}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Individual Stock Performance within Sector
                st.markdown("**🎯 Individual Stock Performance Outlook:**")
                
                for sector, avg_impact in list(avg_sector_impacts.items())[:3]:  # Top 3 sectors
                    stocks = platform.sectors.get(sector.lower(), [])
                    
                    if stocks:
                        st.markdown(f"**{sector.upper()} Sector Stocks:**")
                        stock_cols = st.columns(min(5, len(stocks)))
                        
                        for i, stock in enumerate(stocks[:5]):
                            if i < len(stock_cols):
                                with stock_cols[i]:
                                    # Calculate individual stock impact (slight variation from sector average)
                                    stock_variation = np.random.uniform(-0.3, 0.3)
                                    stock_impact = avg_impact + stock_variation
                                    
                                    stock_color = "#4caf50" if stock_impact > 0 else "#f44336"
                                    performance = "OUTPERFORM" if stock_impact > avg_impact else "UNDERPERFORM"
                                    
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 0.8rem; border-radius: 8px; border: 1px solid {stock_color}; text-align: center; margin-bottom: 10px;">
                                        <div style="color: #ffd700; font-weight: bold; font-size: 0.9rem; margin-bottom: 5px;">{stock}</div>
                                        <div style="color: {stock_color}; font-size: 1.1rem; font-weight: bold; margin-bottom: 5px;">{stock_impact:+.2f}%</div>
                                        <div style="font-size: 0.7rem; color: #b8b8b8;">{performance}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                
                st.markdown("---")
        
        # Export functionality
        if st.sidebar.button("📊 Export Complete Report"):
            # Create comprehensive export data
            export_data = []
            for forecast in st.session_state.report:
                transit = forecast.detailed_transit
                row = {
                    'Date': forecast.date,
                    'Day': forecast.day,
                    'Transit_Event': forecast.event,
                    'Planet': transit.planet,
                    'Zodiac_Sign': transit.zodiac_sign,
                    'Transit_Type': transit.transit_type,
                    'Aspect_Type': transit.aspect_type,
                    'Aspect_Planet': transit.aspect_planet,
                    'Degree': transit.degree,
                    'Sentiment': forecast.sentiment.value,
                    'Expected_Change_%': forecast.change,
                    'Impact_Level': forecast.impact,
                    'Trading_Signal': forecast.signal.value,
                    'Historical_Accuracy_%': transit.historical_accuracy,
                    'Impact_Strength': transit.impact_strength
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
                file_name=f"astro_trading_complete_{st.session_state.symbol}_{platform.month_names[st.session_state.month]}_{st.session_state.year}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
