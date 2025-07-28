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
import random

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
    
    .astro-card {
        background: linear-gradient(135deg, #2d2d4a, #1a1a2e);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #ffd700;
        margin-bottom: 1rem;
        color: white;
    }
    
    .bullish-card {
        border-left-color: #4caf50 !important;
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(26, 26, 46, 0.9));
    }
    
    .bearish-card {
        border-left-color: #f44336 !important;
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.1), rgba(26, 26, 46, 0.9));
    }
    
    .neutral-card {
        border-left-color: #ff9800 !important;
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.1), rgba(26, 26, 46, 0.9));
    }
    
    .transit-forecast-card {
        background: linear-gradient(135deg, #1e1e3f, #2d2d4a);
        border: 2px solid #ffd700;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .transit-forecast-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ffd700, #ff6b35, #ffd700);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .symbol-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ffd700;
        text-align: center;
        margin-bottom: 1rem;
    }
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
    transit_type: str
    zodiac_sign: str
    aspect_planet: str
    aspect_type: str
    degree: float
    impact_strength: str
    market_sectors: Dict[str, float]
    historical_accuracy: float

@dataclass
class PivotPoint:
    date: str
    price_level: float
    pivot_type: str
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
        
        # Symbol-specific astrological influences
        self.symbol_planetary_rulers = {
            'NIFTY': ['jupiter', 'saturn'],
            'SENSEX': ['sun', 'jupiter'],
            'RELIANCE': ['jupiter', 'venus'],
            'TCS': ['mercury', 'uranus'],
            'HDFC': ['venus', 'jupiter'],
            'INFY': ['mercury', 'neptune'],
            'ICICI': ['moon', 'venus'],
            'SBI': ['saturn', 'mars'],
            'WIPRO': ['mercury', 'pluto'],
            'MARUTI': ['mars', 'mercury'],
            'TATAMOTORS': ['mars', 'saturn'],
            'SUNPHARMA': ['sun', 'jupiter'],
            'DRREDDY': ['mars', 'jupiter'],
            'CIPLA': ['mercury', 'venus'],
            'LUPIN': ['moon', 'neptune'],
            'BIOCON': ['pluto', 'mars'],
            'HINDUNILVR': ['venus', 'moon'],
            'ITC': ['saturn', 'venus'],
            'NESTLE': ['venus', 'jupiter'],
            'BRITANNIA': ['moon', 'venus'],
            'DABUR': ['venus', 'mercury'],
            'BHARTI': ['mercury', 'uranus'],
            'AIRTEL': ['mercury', 'uranus'],
            'JIO': ['uranus', 'mercury'],
            'IDEA': ['neptune', 'mercury'],
            'AXIS': ['mars', 'venus'],
            'KOTAK': ['mercury', 'venus'],
            'HCLTECH': ['mercury', 'saturn'],
            'TECHM': ['mercury', 'mars'],
            'GOLD': ['sun', 'venus'],
            'SILVER': ['moon', 'venus'],
            'CRUDE OIL': ['mars', 'pluto'],
            'BITCOIN': ['uranus', 'pluto'],
            'DOW JONES': ['jupiter', 'saturn'],
            'NASDAQ': ['mercury', 'uranus'],
            'S&P 500': ['jupiter', 'sun'],
            'EURUSD': ['venus', 'jupiter'],
            'GBPUSD': ['mercury', 'venus']
        }

    def get_symbol_specific_influence(self, symbol: str, planet: str) -> float:
        """Calculate symbol-specific planetary influence multiplier"""
        ruling_planets = self.symbol_planetary_rulers.get(symbol, ['jupiter'])
        
        if planet.lower() in [p.lower() for p in ruling_planets]:
            return 1.5  # Stronger influence for ruling planets
        elif planet.lower() in ['mercury', 'venus', 'mars']:  # Fast moving planets
            return 1.2
        elif planet.lower() in ['jupiter', 'saturn']:  # Slow moving planets
            return 1.1
        else:
            return 1.0

    def generate_symbol_specific_transits(self, symbol: str, year: int, month: int) -> List[Dict]:
        """Generate symbol-specific planetary transits"""
        transits = []
        ruling_planets = self.symbol_planetary_rulers.get(symbol, ['jupiter', 'saturn'])
        
        # Get month-specific data or generate based on symbol
        base_transits = [
            {
                "date": 5, "planet": ruling_planets[0], "transit_type": "aspect", "zodiac_sign": "leo",
                "aspect_planet": "Jupiter", "aspect_type": "trine", "degree": 120.0,
                "sentiment": Sentiment.BULLISH, "retrograde": False,
                "impact_strength": "Strong", "historical_accuracy": 78.5
            },
            {
                "date": 12, "planet": "Mercury", "transit_type": "retrograde", "zodiac_sign": "leo",
                "aspect_planet": "", "aspect_type": "retrograde", "degree": 20.0,
                "sentiment": Sentiment.BEARISH, "retrograde": True,
                "impact_strength": "Very Strong", "historical_accuracy": 84.7
            },
            {
                "date": 18, "planet": ruling_planets[-1], "transit_type": "aspect", "zodiac_sign": "virgo",
                "aspect_planet": "Saturn", "aspect_type": "square", "degree": 90.0,
                "sentiment": Sentiment.BEARISH, "retrograde": False,
                "impact_strength": "Strong", "historical_accuracy": 73.5
            },
            {
                "date": 25, "planet": "Jupiter", "transit_type": "aspect", "zodiac_sign": "gemini",
                "aspect_planet": "Neptune", "aspect_type": "sextile", "degree": 60.0,
                "sentiment": Sentiment.BULLISH, "retrograde": False,
                "impact_strength": "Strong", "historical_accuracy": 77.9
            }
        ]
        
        # Apply symbol-specific modifications
        for transit in base_transits:
            influence = self.get_symbol_specific_influence(symbol, transit["planet"])
            
            # Adjust change percentage based on symbol influence
            base_change = random.uniform(1.0, 3.0)
            if transit["sentiment"] == Sentiment.BEARISH:
                base_change = -base_change
            
            transit["change"] = f"{base_change * influence:+.1f}"
            
            # Calculate sector impacts based on symbol
            transit["sectors"] = self.get_symbol_sector_impact(symbol, transit["sentiment"], influence)
            
            # Determine signal
            if abs(base_change * influence) > 2.0:
                transit["signal"] = SignalType.LONG if transit["sentiment"] == Sentiment.BULLISH else SignalType.SHORT
            else:
                transit["signal"] = SignalType.HOLD
        
        return base_transits

    def get_symbol_sector_impact(self, symbol: str, sentiment: Sentiment, influence: float) -> Dict[str, float]:
        """Get sector-specific impacts for a symbol"""
        impacts = {}
        
        # Determine primary sector for symbol
        primary_sector = None
        for sector, stocks in self.sectors.items():
            if symbol in stocks:
                primary_sector = sector
                break
        
        if primary_sector:
            # Primary sector gets stronger impact
            base_impact = random.uniform(1.5, 3.0) * influence
            if sentiment == Sentiment.BEARISH:
                base_impact = -base_impact
            impacts[primary_sector] = base_impact
            
            # Related sectors get moderate impact
            related_sectors = ['banking', 'it'] if primary_sector in ['banking', 'it'] else ['auto', 'metals']
            for sector in related_sectors:
                if sector != primary_sector:
                    impacts[sector] = base_impact * 0.6
        else:
            # For indices and commodities, impact multiple sectors
            all_sectors = list(self.sectors.keys())
            for sector in random.sample(all_sectors, 3):
                base_impact = random.uniform(1.0, 2.5) * influence
                if sentiment == Sentiment.BEARISH:
                    base_impact = -base_impact
                impacts[sector] = base_impact
        
        return impacts

    def generate_enhanced_monthly_forecast(self, symbol: str, year: int, month: int) -> List[Forecast]:
        forecasts = []
        
        # Get symbol-specific transits
        symbol_transits = self.generate_symbol_specific_transits(symbol, year, month)
        
        if month == 11:
            next_month = datetime.date(year + 1, 1, 1)
        else:
            next_month = datetime.date(year, month + 2, 1)
        
        current_month = datetime.date(year, month + 1, 1)
        days_in_month = (next_month - current_month).days
        
        for day in range(1, days_in_month + 1):
            current_date = datetime.date(year, month + 1, day)
            
            # Check for major transits
            day_event = None
            for event in symbol_transits:
                if event["date"] == day:
                    day_event = event
                    break
            
            if day_event:
                detailed_transit = self.get_detailed_transit({**day_event, 'month': month + 1})
                
                # Create event description
                if day_event['transit_type'] == 'retrograde':
                    event_desc = f"{day_event['planet']} Retrograde in {self.zodiac_signs[day_event['zodiac_sign']].name if day_event['zodiac_sign'] in self.zodiac_signs else day_event['zodiac_sign'].title()}"
                elif day_event['transit_type'] == 'direct':
                    event_desc = f"{day_event['planet']} Direct in {self.zodiac_signs[day_event['zodiac_sign']].name if day_event['zodiac_sign'] in self.zodiac_signs else day_event['zodiac_sign'].title()}"
                elif day_event['transit_type'] == 'enters':
                    event_desc = f"{day_event['planet']} enters {self.zodiac_signs[day_event['zodiac_sign']].name if day_event['zodiac_sign'] in self.zodiac_signs else day_event['zodiac_sign'].title()}"
                elif day_event['transit_type'] == 'aspect':
                    event_desc = f"{day_event['planet']} {day_event['aspect_type']} {day_event['aspect_planet']}"
                else:
                    event_desc = f"{day_event['planet']} transit in {self.zodiac_signs[day_event['zodiac_sign']].name if day_event['zodiac_sign'] in self.zodiac_signs else day_event['zodiac_sign'].title()}"
                
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
                # Generate minor daily transits with symbol influence
                ruling_planets = self.symbol_planetary_rulers.get(symbol, ['jupiter'])
                sentiment_options = [Sentiment.BULLISH, Sentiment.BEARISH, Sentiment.NEUTRAL]
                sentiment = sentiment_options[day % 3]
                
                # Symbol-specific minor influence
                base_change = ((day * 37) % 100 - 50) / 100
                planet_influence = self.get_symbol_specific_influence(symbol, ruling_planets[0])
                change_percent = base_change * planet_influence
                change_str = f"{'+' if change_percent > 0 else ''}{change_percent:.1f}"
                
                signal = SignalType.LONG if sentiment == Sentiment.BULLISH else SignalType.SHORT if sentiment == Sentiment.BEARISH else SignalType.HOLD
                
                # Generate minor transit
                planets = list(self.planets.keys())
                zodiac_keys = list(self.zodiac_signs.keys())
                random_planet = ruling_planets[0] if day % 2 == 0 else planets[day % len(planets)]
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
                    historical_accuracy=60.0 + (planet_influence - 1) * 10
                )
                
                forecasts.append(Forecast(
                    date=current_date.strftime('%Y-%m-%d'),
                    day=day,
                    event=f'Minor {random_planet.title()} transit affecting {symbol}',
                    sentiment=sentiment,
                    change=change_str,
                    impact=f"Minor {sentiment.value.title()}",
                    sector_impact={},
                    signal=signal,
                    detailed_transit=minor_transit
                ))
        
        return forecasts

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
            if change_val > 1.5:  # Significant movement threshold
                
                if forecast.sentiment == Sentiment.BULLISH:
                    pivot_type = "support" if i < len(forecasts) // 2 else "resistance"
                    expected_move = change_val * 1.2
                else:
                    pivot_type = "resistance" if i < len(forecasts) // 2 else "support"
                    expected_move = -change_val * 1.1
                
                confidence = min(95, max(60, forecast.detailed_transit.historical_accuracy))
                
                pivot_points.append(PivotPoint(
                    date=forecast.date,
                    price_level=100 + expected_move,
                    pivot_type=pivot_type,
                    expected_move=expected_move,
                    confidence=confidence
                ))
        
        return pivot_points

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
    
    features = [
        ("📅", "Astro Calendar", "Daily planetary transits, retrograde periods, and precise aspect timing with sector impacts"),
        ("📊", "Stock Analysis", "Comprehensive symbol-specific analysis, global correlations, and sector impact forecasts"),
        ("📈", "Astro Graph", "Interactive charts with pivot points, support/resistance levels, and price projections"),
        ("🌙", "Transit Analysis", "Advanced planetary transit impacts on specific symbols and market performance")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="feature-card">
                <div style="font-size: 2.5rem; margin-bottom: 15px;">{icon}</div>
                <h3 style="color: #ffd700; margin-bottom: 10px;">{title}</h3>
                <p style="font-size: 0.9rem;">{desc}</p>
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

def render_astro_calendar_grid(forecasts: List[Forecast], month_name: str, year: int, platform):
    """Render astro calendar in the requested format"""
    st.markdown(f"""
    <div class="symbol-header">
        <h2 style="color: #ffd700; margin: 0;">📅 Monthly Planetary Transit Forecast</h2>
        <p style="color: #b8b8b8; margin: 0.5rem 0 0 0;">{month_name} {year}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter significant transits only
    significant_forecasts = [f for f in forecasts if abs(float(f.change.replace('+', '').replace('-', ''))) > 1.0][:9]
    
    # Create 3x3 grid
    for row in range(3):
        cols = st.columns(3)
        for col in range(3):
            index = row * 3 + col
            if index < len(significant_forecasts):
                forecast = significant_forecasts[index]
                transit = forecast.detailed_transit
                
                # Determine card class based on sentiment
                card_class = {
                    Sentiment.BULLISH: "bullish-card",
                    Sentiment.BEARISH: "bearish-card",
                    Sentiment.NEUTRAL: "neutral-card"
                }.get(forecast.sentiment, "neutral-card")
                
                # Determine impact label and color
                if forecast.sentiment == Sentiment.BULLISH:
                    impact_label = "MODERATE BULLISH" if "moderate" in forecast.impact.lower() else "STRONG BULLISH"
                    impact_color = "#4caf50"
                elif forecast.sentiment == Sentiment.BEARISH:
                    impact_label = "MODERATE BEARISH" if "moderate" in forecast.impact.lower() else "STRONG BEARISH"
                    impact_color = "#f44336"
                else:
                    impact_label = "MODERATE NEUTRAL"
                    impact_color = "#ff9800"
                
                with cols[col]:
                    st.markdown(f"""
                    <div class="transit-forecast-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <h3 style="color: #ffd700; margin: 0; font-size: 1.1rem;">{forecast.date.replace('2025-', '')}</h3>
                            <div style="background: {impact_color}; color: white; padding: 0.2rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: bold;">
                                {impact_label}
                            </div>
                        </div>
                        
                        <h4 style="color: white; margin: 0 0 0.5rem 0; font-size: 1rem; line-height: 1.3;">
                            {transit.planet} {transit.aspect_type} {transit.aspect_planet if transit.aspect_planet else ''}
                        </h4>
                        
                        <p style="color: #b8b8b8; margin: 0 0 1rem 0; font-size: 0.85rem;">
                            {transit.planet} in {platform.zodiac_signs[transit.zodiac_sign].name if transit.zodiac_sign in platform.zodiac_signs else transit.zodiac_sign.title()} • {transit.aspect_type.title()}
                        </p>
                        
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="color: #ffd700; font-weight: bold; font-size: 1.1rem;">Expected Change: {forecast.change}%</span>
                            </div>
                        </div>
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
        symbol = st.sidebar.text_input("📈 Stock Symbol", value="NIFTY", help="Enter stock symbol (e.g., NIFTY, RELIANCE, TCS)")
        if symbol:
            symbol = symbol.upper().strip()  # Normalize input
            # Show if symbol is recognized
            if symbol in platform.symbol_planetary_rulers:
                st.sidebar.success(f"✅ {symbol} - Recognized symbol with specific planetary analysis")
            else:
                st.sidebar.info(f"ℹ️ {symbol} - Custom symbol with general planetary analysis")
        analysis_type = "stock"
    else:
        global_symbols = ["DOW JONES", "NASDAQ", "S&P 500", "GOLD", "SILVER", "CRUDE OIL", "BITCOIN", "EURUSD", "GBPUSD"]
        symbol = st.sidebar.selectbox("🌍 Global Market/Commodity", global_symbols)
        analysis_type = "global"
    
    # Month and Year Selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        selected_year = st.selectbox("📅 Year", [2024, 2025, 2026], index=1)
    with col2:
        month_options = {i: platform.month_names[i] for i in range(12)}
        selected_month = st.selectbox("📅 Month", options=list(month_options.keys()), 
                                     format_func=lambda x: month_options[x], index=7)  # Default to August
    
    time_zone = st.sidebar.selectbox("🌍 Time Zone", ["IST", "EST", "GMT", "JST"])
    
    # Generate Analysis Button
    if st.sidebar.button("🚀 Generate Analysis", type="primary"):
        if symbol and symbol.strip():  # Ensure symbol is not empty
            with st.spinner(f"🔮 Calculating planetary positions for {symbol}..."):
                st.session_state.report = platform.generate_enhanced_monthly_forecast(symbol, selected_year, selected_month)
                st.session_state.symbol = symbol
                st.session_state.month = selected_month
                st.session_state.year = selected_year
                st.session_state.analysis_type = analysis_type
                st.session_state.pivot_points = platform.generate_pivot_points(st.session_state.report)
        else:
            st.sidebar.error("Please enter a valid stock symbol")
    
    # Main Content
    if 'report' not in st.session_state:
        render_front_page()
    else:
        # Main Content Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📅 Astro Calendar", "📊 Stock Analysis", "📈 Astro Graph", "🌙 Transit Analysis"])
        
        with tab1:
            forecasts = st.session_state.report
            month_name = platform.month_names[st.session_state.month]
            
            # Monthly Overview Stats
            col1, col2, col3, col4 = st.columns(4)
            
            bullish_count = sum(1 for f in forecasts if f.sentiment == Sentiment.BULLISH)
            bearish_count = sum(1 for f in forecasts if f.sentiment == Sentiment.BEARISH)
            neutral_count = len(forecasts) - bullish_count - bearish_count
            strong_transits = len([f for f in forecasts if 'Strong' in f.impact])
            
            stats = [
                (bullish_count, "Bullish Days", "#4caf50"),
                (bearish_count, "Bearish Days", "#f44336"),
                (neutral_count, "Neutral Days", "#ff9800"),
                (strong_transits, "Strong Transits", "#e91e63")
            ]
            
            for i, (count, label, color) in enumerate(stats):
                with [col1, col2, col3, col4][i]:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3 style="color: {color};">{count}</h3>
                        <p>{label}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Render Astro Calendar Grid
            render_astro_calendar_grid(forecasts, month_name, st.session_state.year, platform)
            
            # Additional daily details
            st.markdown("### 📋 Complete Daily Transit Details")
            for forecast in forecasts:
                if abs(float(forecast.change.replace('+', '').replace('-', ''))) > 0.5:
                    transit = forecast.detailed_transit
                    
                    sentiment_emoji = "📈" if forecast.sentiment == Sentiment.BULLISH else "📉" if forecast.sentiment == Sentiment.BEARISH else "➡️"
                    signal_emoji = "🟢" if forecast.signal == SignalType.LONG else "🔴" if forecast.signal == SignalType.SHORT else "🟡"
                    
                    st.markdown(f"""
                    <div class="astro-card {{'bullish-card' if forecast.sentiment == Sentiment.BULLISH else 'bearish-card' if forecast.sentiment == Sentiment.BEARISH else 'neutral-card'}}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <h4 style="margin: 0; color: #ffd700;">📅 {forecast.date} (Day {forecast.day})</h4>
                            <div style="display: flex; gap: 10px;">
                                <span>{signal_emoji} {forecast.signal.value}</span>
                                <span style="color: #ffd700; font-weight: bold;">{forecast.change}%</span>
                            </div>
                        </div>
                        
                        <h5 style="margin: 0.5rem 0; color: white;">{forecast.event}</h5>
                        <p style="margin: 0.5rem 0; color: #b8b8b8; font-style: italic;">
                            {transit.planet} in {platform.zodiac_signs[transit.zodiac_sign].name} • {transit.aspect_type.title()} • 
                            Accuracy: {transit.historical_accuracy:.1f}%
                        </p>
                        
                        {f'<p style="margin: 0; color: #ffd700;"><strong>Sector Impacts:</strong> {" | ".join([f"{k.upper()}: {v:+.1f}%" for k, v in forecast.sector_impact.items()])}</p>' if forecast.sector_impact else ''}
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown(f"""
            <div class="symbol-header">
                <h2 style="color: #ffd700; margin: 0;">📊 {st.session_state.symbol} Analysis</h2>
                <p style="color: #b8b8b8; margin: 0.5rem 0 0 0;">{platform.month_names[st.session_state.month]} {st.session_state.year} Astrological Analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            forecasts = st.session_state.report
            
            # Enhanced Monthly Summary Stats
            col1, col2, col3, col4 = st.columns(4)
            
            total_bullish_impact = sum(float(f.change.replace('+', '').replace('-', '')) for f in forecasts if f.sentiment == Sentiment.BULLISH)
            total_bearish_impact = sum(float(f.change.replace('+', '').replace('-', '')) for f in forecasts if f.sentiment == Sentiment.BEARISH)
            avg_daily_change = np.mean([float(f.change.replace('+', '').replace('-', '')) for f in forecasts])
            high_accuracy_transits = len([f for f in forecasts if f.detailed_transit.historical_accuracy > 75])
            
            stats = [
                (f"+{total_bullish_impact:.1f}%", "Total Bullish Impact", "#4caf50"),
                (f"-{total_bearish_impact:.1f}%", "Total Bearish Impact", "#f44336"),
                (f"{avg_daily_change:.1f}%", "Avg Daily Impact", "#ff9800"),
                (high_accuracy_transits, "High Accuracy Transits", "#e91e63")
            ]
            
            for i, (value, label, color) in enumerate(stats):
                with [col1, col2, col3, col4][i]:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h3 style="color: {color};">{value}</h3>
                        <p>{label}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Symbol-specific ruling planets info
            ruling_planets = platform.symbol_planetary_rulers.get(st.session_state.symbol, ['jupiter'])
            st.markdown(f"""
            <div class="astro-card">
                <h4 style="color: #ffd700; margin: 0 0 0.5rem 0;">🪐 Ruling Planetary Influences for {st.session_state.symbol}</h4>
                <p style="color: white; margin: 0;">
                    <strong>Primary Ruling Planets:</strong> {', '.join([p.title() for p in ruling_planets])}
                </p>
                <p style="color: #b8b8b8; margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    These planets have the strongest influence on {st.session_state.symbol}'s price movements according to astrological analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Monthly forecast breakdown
            st.markdown("### 📈 Symbol-Specific Monthly Forecast")
            
            # Create weekly breakdown
            weeks = [forecasts[i:i+7] for i in range(0, len(forecasts), 7)]
            
            for week_num, week_forecasts in enumerate(weeks, 1):
                if not week_forecasts:
                    continue
                    
                st.markdown(f"#### Week {week_num}")
                
                weekly_cols = st.columns(min(7, len(week_forecasts)))
                for i, forecast in enumerate(week_forecasts):
                    if i < len(weekly_cols):
                        with weekly_cols[i]:
                            sentiment_color = {
                                Sentiment.BULLISH: "#4caf50",
                                Sentiment.BEARISH: "#f44336",
                                Sentiment.NEUTRAL: "#ff9800"
                            }.get(forecast.sentiment, "#ff9800")
                            
                            signal_text = {
                                SignalType.LONG: "🟢 BUY",
                                SignalType.SHORT: "🔴 SELL", 
                                SignalType.HOLD: "🟡 HOLD"
                            }.get(forecast.signal, "🟡 HOLD")
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {sentiment_color}15, #1a1a2e); 
                                        border: 1px solid {sentiment_color}; border-radius: 8px; padding: 0.8rem; 
                                        text-align: center; margin-bottom: 0.5rem;">
                                <div style="color: #ffd700; font-weight: bold; margin-bottom: 0.3rem;">
                                    {forecast.date.split('-')[2]}/{forecast.date.split('-')[1]}
                                </div>
                                <div style="color: {sentiment_color}; font-weight: bold; font-size: 1.1rem; margin-bottom: 0.3rem;">
                                    {forecast.change}%
                                </div>
                                <div style="color: white; font-size: 0.8rem; margin-bottom: 0.3rem;">
                                    {signal_text}
                                </div>
                                <div style="color: #b8b8b8; font-size: 0.7rem;">
                                    {forecast.detailed_transit.planet} Transit
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown(f"""
            <div class="symbol-header">
                <h2 style="color: #ffd700; margin: 0;">📈 {st.session_state.symbol} Astrological Movement Graph</h2>
                <p style="color: #b8b8b8; margin: 0.5rem 0 0 0;">Dynamic Planetary Transit Impact Visualization</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create enhanced forecast dataframe
            try:
                df_forecasts = pd.DataFrame([
                    {
                        'date': f.date,
                        'change': float(f.change.replace('+', '').replace('-', '')),
                        'sentiment': f.sentiment.value,
                        'signal': f.signal.value,
                        'event': f.event,
                        'impact_strength': f.detailed_transit.impact_strength,
                        'historical_accuracy': f.detailed_transit.historical_accuracy,
                        'planet': f.detailed_transit.planet
                    } for f in forecasts
                ])
                df_forecasts['date'] = pd.to_datetime(df_forecasts['date'])
                df_forecasts['change_signed'] = df_forecasts.apply(
                    lambda row: row['change'] if row['sentiment'] != 'bearish' else -row['change'], axis=1
                )
                
                # Enhanced interactive chart
                fig = go.Figure()
                
                # Main price movement line with dynamic colors
                colors = ['#f44336' if s == 'bearish' else '#4caf50' if s == 'bullish' else '#ff9800' 
                         for s in df_forecasts['sentiment']]
                
                fig.add_trace(go.Scatter(
                    x=df_forecasts['date'],
                    y=df_forecasts['change_signed'],
                    mode='lines+markers+text',
                    name=f'{st.session_state.symbol} Movement',
                    line=dict(color='#ffd700', width=4),
                    marker=dict(
                        size=df_forecasts['change'] * 3 + 8,
                        color=colors,
                        line=dict(width=2, color='black'),  # Changed from white to black
                        opacity=0.8
                    ),
                    text=[f"{change:+.1f}%" for change in df_forecasts['change_signed']],
                    textposition="top center",
                    textfont=dict(size=10, color='black'),  # Changed from white to black
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Movement: %{y:.2f}%<br>' +
                                 'Event: %{customdata[0]}<br>' +
                                 'Planet: %{customdata[1]}<br>' +
                                 'Accuracy: %{customdata[2]:.1f}%<br>' +
                                 'Signal: %{customdata[3]}<extra></extra>',
                    customdata=list(zip(df_forecasts['event'], 
                                      df_forecasts['planet'],
                                      df_forecasts['historical_accuracy'],
                                      df_forecasts['signal']))
                ))
                
                # Add cumulative movement
                cumulative_change = df_forecasts['change_signed'].cumsum()
                fig.add_trace(go.Scatter(
                    x=df_forecasts['date'],
                    y=cumulative_change,
                    mode='lines',
                    name='Cumulative Movement',
                    line=dict(color='#ff6b35', width=3, dash='dash'),
                    hovertemplate='Cumulative: %{y:.2f}%<extra></extra>'
                ))
                
                # Add zero line
                fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3, line_width=1)
                
                # Enhanced pivot points with black text
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
                            marker=dict(color='#4caf50', size=18, symbol='triangle-up', 
                                       line=dict(width=3, color='black')),  # Changed from white to black
                            text=[f"Support<br>{v:+.1f}%" for v in support_values],
                            textposition="bottom center",
                            textfont=dict(size=10, color='black'),  # Changed from white to black
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
                            marker=dict(color='#f44336', size=18, symbol='triangle-down', 
                                       line=dict(width=3, color='black')),  # Changed from white to black
                            text=[f"Resistance<br>{v:+.1f}%" for v in resistance_values],
                            textposition="top center",
                            textfont=dict(size=10, color='black'),  # Changed from white to black
                            hovertemplate='Resistance Level<br>Date: %{x}<br>Level: %{y:.2f}%<extra></extra>'
                        ))
                
                # Enhanced annotations with black text
                major_swings = df_forecasts[df_forecasts['change'] > 2]
                for _, swing in major_swings.iterrows():
                    fig.add_annotation(
                        x=swing['date'],
                        y=swing['change_signed'],
                        text=f"<b>Major Transit</b><br>{swing['change_signed']:+.1f}%<br>{swing['planet']} Impact",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=3,
                        arrowcolor="#ffd700",
                        font=dict(size=11, color="black"),  # Changed from white to black
                        bgcolor="rgba(255, 215, 0, 0.9)",
                        bordercolor="black",  # Changed border to black
                        borderwidth=2,
                        ax=0,
                        ay=-50 if swing['change_signed'] > 0 else 50
                    )
                
                # Color zones
                max_val = max(df_forecasts['change_signed'].max(), 5)
                min_val = min(df_forecasts['change_signed'].min(), -5)
                
                fig.add_hrect(y0=0, y1=max_val, 
                             fillcolor="rgba(76, 175, 80, 0.1)", layer="below", line_width=0)
                fig.add_hrect(y0=min_val, y1=0, 
                             fillcolor="rgba(244, 67, 54, 0.1)", layer="below", line_width=0)
                
                fig.update_layout(
                    title=f"Planetary Transit Impact Analysis - {st.session_state.symbol} ({platform.month_names[st.session_state.month]} {st.session_state.year})",
                    xaxis_title="Date",
                    yaxis_title="Expected Change (%)",
                    template="plotly_dark",
                    showlegend=True,
                    height=700,
                    hovermode='x unified',
                    font=dict(color="white")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
            
            # Dynamic pivot points summary
            if 'pivot_points' in st.session_state and st.session_state.pivot_points:
                st.markdown("### 🎯 Dynamic Pivot Points & Price Levels")
                
                pivot_data = []
                for pivot in st.session_state.pivot_points:
                    pivot_data.append({
                        'Date': pivot.date,
                        'Type': pivot.pivot_type.upper(),
                        'Expected Move': f"{pivot.expected_move:+.2f}%",
                        'Confidence': f"{pivot.confidence:.1f}%",
                        'Price Level': f"{pivot.price_level:.2f}"
                    })
                
                df_pivots = pd.DataFrame(pivot_data)
                st.dataframe(df_pivots, use_container_width=True)
            
            # Additional analysis charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Planetary influence chart
                planet_impacts = df_forecasts.groupby('planet')['change'].mean().reset_index()
                fig_planets = px.bar(
                    planet_impacts,
                    x='planet',
                    y='change',
                    title=f"Average Planetary Impact on {st.session_state.symbol}",
                    color='change',
                    color_continuous_scale='RdYlGn'
                )
                fig_planets.update_layout(template="plotly_dark")
                st.plotly_chart(fig_planets, use_container_width=True)
            
            with col2:
                # Signal distribution
                signal_counts = df_forecasts['signal'].value_counts()
                fig_signals = px.pie(
                    values=signal_counts.values,
                    names=signal_counts.index,
                    title=f"Trading Signal Distribution - {st.session_state.symbol}",
                    color_discrete_map={'LONG': '#4caf50', 'SHORT': '#f44336', 'HOLD': '#ff9800'}
                )
                fig_signals.update_layout(template="plotly_dark")
                st.plotly_chart(fig_signals, use_container_width=True)
        
        with tab4:
            st.markdown(f"""
            <div class="symbol-header">
                <h2 style="color: #ffd700; margin: 0;">🌙 Advanced Planetary Transit Analysis</h2>
                <p style="color: #b8b8b8; margin: 0.5rem 0 0 0;">{st.session_state.symbol} - {platform.month_names[st.session_state.month]} {st.session_state.year}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Symbol-specific ruling planet analysis
            ruling_planets = platform.symbol_planetary_rulers.get(st.session_state.symbol, ['jupiter'])
            
            st.markdown(f"""
            <div class="astro-card bullish-card">
                <h4 style="color: #ffd700; margin: 0 0 0.5rem 0;">🪐 Symbol-Specific Planetary Rulership</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                    <div>
                        <strong style="color: white;">Primary Ruling Planets:</strong><br>
                        <span style="color: #4caf50;">{', '.join([p.title() for p in ruling_planets])}</span>
                    </div>
                    <div>
                        <strong style="color: white;">Symbol Type:</strong><br>
                        <span style="color: #ffd700;">{st.session_state.analysis_type.title()}</span>
                    </div>
                    <div>
                        <strong style="color: white;">Enhanced Sensitivity:</strong><br>
                        <span style="color: #ff9800;">50% stronger impact from ruling planets</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get unique transits for the month with enhanced analysis
            unique_transits = {}
            for forecast in forecasts:
                transit = forecast.detailed_transit
                key = f"{transit.planet}_{transit.zodiac_sign}_{transit.transit_type}"
                if key not in unique_transits:
                    unique_transits[key] = {
                        'transit': transit,
                        'forecasts': [],
                        'total_impact': 0,
                        'sector_impacts': {},
                        'is_ruling_planet': transit.planet.lower() in [p.lower() for p in ruling_planets]
                    }
                unique_transits[key]['forecasts'].append(forecast)
                unique_transits[key]['total_impact'] += abs(float(forecast.change.replace('+', '').replace('-', '')))
                
                for sector, impact in forecast.sector_impact.items():
                    if sector not in unique_transits[key]['sector_impacts']:
                        unique_transits[key]['sector_impacts'][sector] = []
                    unique_transits[key]['sector_impacts'][sector].append(impact)
            
            # Sort by impact strength and ruling planet priority
            sorted_transits = sorted(
                unique_transits.items(), 
                key=lambda x: (x[1]['is_ruling_planet'], x[1]['total_impact']), 
                reverse=True
            )
            
            st.markdown("### 🌟 Major Planetary Transits Affecting Your Symbol")
            
            for transit_key, transit_data in sorted_transits[:6]:
                transit = transit_data['transit']
                avg_impact = transit_data['total_impact'] / len(transit_data['forecasts'])
                is_ruling = transit_data['is_ruling_planet']
                
                # Enhanced card styling for ruling planets
                card_class = "bullish-card" if is_ruling else "astro-card"
                ruling_indicator = "👑 RULING PLANET" if is_ruling else ""
                
                # Get all dates for this transit
                transit_dates = [f.date for f in transit_data['forecasts']]
                date_range = f"{min(transit_dates)} to {max(transit_dates)}" if len(transit_dates) > 1 else transit_dates[0]
                
                # Calculate average sector impacts
                avg_sector_impacts = {}
                for sector, impacts in transit_data['sector_impacts'].items():
                    avg_sector_impacts[sector] = np.mean(impacts)
                
                # Enhanced transit description
                if transit.transit_type == 'retrograde':
                    transit_desc = f"{transit.planet} Retrograde in {platform.zodiac_signs[transit.zodiac_sign].name}"
                    impact_multiplier = 1.3  # Retrograde has stronger impact
                elif transit.transit_type == 'direct':
                    transit_desc = f"{transit.planet} Direct in {platform.zodiac_signs[transit.zodiac_sign].name}"
                    impact_multiplier = 1.2
                elif transit.transit_type == 'enters':
                    transit_desc = f"{transit.planet} enters {platform.zodiac_signs[transit.zodiac_sign].name}"
                    impact_multiplier = 1.1
                elif transit.transit_type == 'aspect':
                    transit_desc = f"{transit.planet} {transit.aspect_type} {transit.aspect_planet}"
                    impact_multiplier = 1.15
                else:
                    transit_desc = f"{transit.planet} in {platform.zodiac_signs[transit.zodiac_sign].name}"
                    impact_multiplier = 1.0
                
                if is_ruling:
                    impact_multiplier *= 1.5
                
                st.markdown(f"""
                <div class="{card_class}" style="margin-bottom: 2rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h3 style="color: #ffd700; margin: 0;">🌟 {transit_desc}</h3>
                        {f'<span style="background: linear-gradient(45deg, #ffd700, #ff6b35); padding: 0.3rem 0.8rem; border-radius: 15px; color: black; font-weight: bold; font-size: 0.8rem;">{ruling_indicator}</span>' if is_ruling else ''}
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                        <div>
                            <strong style="color: #ffd700;">Duration:</strong><br>
                            <span style="color: white;">{date_range}</span><br>
                            <span style="color: #b8b8b8;">({len(transit_dates)} day{'s' if len(transit_dates) > 1 else ''})</span>
                        </div>
                        <div>
                            <strong style="color: #ffd700;">Average Impact:</strong><br>
                            <span style="color: {'#4caf50' if avg_impact > 0 else '#f44336'}; font-size: 1.2rem; font-weight: bold;">
                                {avg_impact * impact_multiplier:+.1f}%
                            </span>
                        </div>
                        <div>
                            <strong style="color: #ffd700;">Accuracy:</strong><br>
                            <span style="color: white;">{transit.historical_accuracy:.1f}%</span><br>
                            <span style="color: #b8b8b8;">{transit.impact_strength} Impact</span>
                        </div>
                        <div>
                            <strong style="color: #ffd700;">Astrological Data:</strong><br>
                            <span style="color: white;">{platform.zodiac_signs[transit.zodiac_sign].element} • {platform.zodiac_signs[transit.zodiac_sign].quality}</span><br>
                            <span style="color: #b8b8b8;">{transit.degree:.1f}°</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Daily breakdown with enhanced visualization
                if len(transit_data['forecasts']) > 1:
                    st.markdown("**📋 Daily Progression:**")
                    
                    daily_cols = st.columns(min(7, len(transit_data['forecasts'])))
                    for i, f in enumerate(transit_data['forecasts'][:7]):
                        with daily_cols[i]:
                            change_val = float(f.change.replace('+', '').replace('-', ''))
                            if is_ruling:
                                change_val *= 1.5
                            
                            sentiment_color = {
                                Sentiment.BULLISH: "#4caf50",
                                Sentiment.BEARISH: "#f44336", 
                                Sentiment.NEUTRAL: "#ff9800"
                            }.get(f.sentiment, "#ff9800")
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {sentiment_color}20, #1a1a2e);
                                        border: 1px solid {sentiment_color}; border-radius: 8px; padding: 0.8rem;
                                        text-align: center; margin-bottom: 0.5rem;">
                                <div style="color: #ffd700; font-weight: bold; margin-bottom: 0.3rem;">
                                    {f.date.split('-')[2]}/{f.date.split('-')[1]}
                                </div>
                                <div style="color: {sentiment_color}; font-weight: bold; font-size: 1.1rem;">
                                    {change_val:+.1f}%
                                </div>
                                <div style="color: white; font-size: 0.8rem;">
                                    {f.signal.value}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Enhanced sector analysis for symbol
                if avg_sector_impacts:
                    st.markdown("**📊 Symbol-Specific Sector Impact Analysis:**")
                    
                    sector_analysis = []
                    for sector, avg_impact in avg_sector_impacts.items():
                        if is_ruling:
                            avg_impact *= 1.5
                        
                        # Get symbol's relationship to this sector
                        symbol_in_sector = st.session_state.symbol in platform.sectors.get(sector.lower(), [])
                        relevance = "PRIMARY" if symbol_in_sector else "SECONDARY"
                        
                        recommendation = "STRONG BUY" if avg_impact > 2.5 else "BUY" if avg_impact > 1.0 else "HOLD" if avg_impact > -1.0 else "SELL" if avg_impact > -2.5 else "STRONG SELL"
                        
                        sector_analysis.append({
                            'sector': sector,
                            'impact': avg_impact,
                            'relevance': relevance,
                            'recommendation': recommendation
                        })
                    
                    # Sort by impact and relevance
                    sector_analysis.sort(key=lambda x: (x['relevance'] == 'PRIMARY', abs(x['impact'])), reverse=True)
                    
                    sector_cols = st.columns(min(4, len(sector_analysis)))
                    for i, sector_data in enumerate(sector_analysis[:4]):
                        with sector_cols[i]:
                            impact_color = "#4caf50" if sector_data['impact'] > 0 else "#f44336"
                            relevance_color = "#ffd700" if sector_data['relevance'] == 'PRIMARY' else "#b8b8b8"
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, {impact_color}15, #1a1a2e);
                                        border: 2px solid {impact_color}; border-radius: 10px; padding: 1rem;
                                        text-align: center; margin-bottom: 1rem;">
                                <h4 style="color: {relevance_color}; margin: 0 0 0.5rem 0;">
                                    {sector_data['sector'].upper()}
                                </h4>
                                <div style="color: {impact_color}; font-weight: bold; font-size: 1.3rem; margin-bottom: 0.5rem;">
                                    {sector_data['impact']:+.2f}%
                                </div>
                                <div style="color: white; font-weight: bold; margin-bottom: 0.3rem;">
                                    {sector_data['recommendation']}
                                </div>
                                <div style="color: {relevance_color}; font-size: 0.8rem;">
                                    {sector_data['relevance']} RELEVANCE
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
        
        # Enhanced export functionality
        if st.sidebar.button("📊 Export Complete Report"):
            export_data = []
            for forecast in st.session_state.report:
                transit = forecast.detailed_transit
                ruling_planets = platform.symbol_planetary_rulers.get(st.session_state.symbol, ['jupiter'])
                is_ruling = transit.planet.lower() in [p.lower() for p in ruling_planets]
                
                row = {
                    'Symbol': st.session_state.symbol,
                    'Date': forecast.date,
                    'Day': forecast.day,
                    'Transit_Event': forecast.event,
                    'Planet': transit.planet,
                    'Is_Ruling_Planet': is_ruling,
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
                    'Impact_Strength': transit.impact_strength,
                    'Element': platform.zodiac_signs[transit.zodiac_sign].element,
                    'Quality': platform.zodiac_signs[transit.zodiac_sign].quality
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
                file_name=f"astro_trading_{st.session_state.symbol}_{platform.month_names[st.session_state.month]}_{st.session_state.year}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
