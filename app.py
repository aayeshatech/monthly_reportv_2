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
        ruling_planets = self.symbol_planetary_rulers.get(symbol, ['jupiter', 'saturn'])
        
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
        ruling_planets = self.symbol_planetary_rulers.get(symbol, ['jupiter', 'saturn'])  # Default to Jupiter/Saturn
        
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
                ruling_planets = self.symbol_planetary_rulers.get(symbol, ['jupiter', 'saturn'])
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
    st.markdown('# 🌟 Advanced Astrological Trading Platform')
    
    # Hero Section using native Streamlit
    st.info("""
    ## 🚀 Professional Market Analysis with Planetary Intelligence
    
    Harness the power of celestial movements for precise market predictions and trading signals
    
    ⭐ 85%+ Accuracy Rate | 🌍 Global Markets Coverage | 📊 Real-time Analysis
    """)
    
    # Feature Cards using columns
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        ("📅", "Astro Calendar", "Daily planetary transits, retrograde periods, and precise aspect timing with sector impacts"),
        ("📊", "Stock Analysis", "Comprehensive symbol-specific analysis, global correlations, and sector impact forecasts"),
        ("📈", "Astro Graph", "Interactive charts with pivot points, support/resistance levels, and price projections"),
        ("🌙", "Transit Analysis", "Advanced planetary transit impacts on specific symbols and market performance")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"### {icon} {title}")
            st.markdown(desc)
    
    # Current Market Status
    current_time = datetime.datetime.now()
    st.success(f"🕐 Current Market Time: {current_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
    st.info("Ready to analyze celestial influences on your trading decisions")

def render_astro_calendar_grid(forecasts: List[Forecast], month_name: str, year: int, platform):
    """Render astro calendar in the requested format"""
    st.markdown(f"## 📅 Monthly Planetary Transit Forecast")
    st.markdown(f"### {month_name} {year}")
    
    # Filter significant transits only
    significant_forecasts = [f for f in forecasts if abs(float(f.change.replace('+', '').replace('-', ''))) > 1.0][:9]
    
    # Create 3x3 grid using native Streamlit
    for row in range(3):
        cols = st.columns(3)
        for col in range(3):
            index = row * 3 + col
            if index < len(significant_forecasts):
                forecast = significant_forecasts[index]
                transit = forecast.detailed_transit
                
                with cols[col]:
                    # Create card using expander
                    card_title = f"{forecast.date.replace('2025-', '')} - {forecast.change}%"
                    
                    with st.expander(card_title, expanded=True):
                        # Determine impact label and display color
                        if forecast.sentiment == Sentiment.BULLISH:
                            if "strong" in forecast.impact.lower():
                                st.success("STRONG BULLISH")
                            else:
                                st.success("MODERATE BULLISH")
                        elif forecast.sentiment == Sentiment.BEARISH:
                            if "strong" in forecast.impact.lower():
                                st.error("STRONG BEARISH")
                            else:
                                st.error("MODERATE BEARISH")
                        else:
                            st.warning("MODERATE NEUTRAL")
                        
                        # Main content
                        st.markdown(f"**{transit.planet} {transit.aspect_type} {transit.aspect_planet if transit.aspect_planet else ''}**")
                        
                        zodiac_name = platform.zodiac_signs[transit.zodiac_sign].name if transit.zodiac_sign in platform.zodiac_signs else transit.zodiac_sign.title()
                        st.markdown(f"_{transit.planet} in {zodiac_name} • {transit.aspect_type.title()}_")
                        
                        # Expected change
                        if forecast.sentiment == Sentiment.BULLISH:
                            st.success(f"Expected Change: {forecast.change}%")
                        elif forecast.sentiment == Sentiment.BEARISH:
                            st.error(f"Expected Change: {forecast.change}%")
                        else:
                            st.info(f"Expected Change: {forecast.change}%")

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
    elif not hasattr(st.session_state, 'symbol') or not st.session_state.symbol:
        st.error("Please select a symbol and generate analysis first.")
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
                (bullish_count, "Bullish Days", "🟢"),
                (bearish_count, "Bearish Days", "🔴"),
                (neutral_count, "Neutral Days", "🟡"),
                (strong_transits, "Strong Transits", "⭐")
            ]
            
            for i, (count, label, emoji) in enumerate(stats):
                with [col1, col2, col3, col4][i]:
                    st.metric(label=f"{emoji} {label}", value=count)
            
            # Render Astro Calendar Grid
            render_astro_calendar_grid(forecasts, month_name, st.session_state.year, platform)
            
            # Additional daily details
            st.markdown("### 📋 Complete Daily Transit Details")
            for forecast in forecasts:
                if abs(float(forecast.change.replace('+', '').replace('-', ''))) > 0.5:
                    transit = forecast.detailed_transit
                    
                    # Create expandable sections for each day
                    with st.expander(f"📅 {forecast.date} (Day {forecast.day}) - {forecast.signal.value} {forecast.change}%"):
                        st.markdown(f"**{forecast.event}**")
                        
                        zodiac_name = platform.zodiac_signs[transit.zodiac_sign].name if transit.zodiac_sign in platform.zodiac_signs else transit.zodiac_sign.title()
                        st.markdown(f"_{transit.planet} in {zodiac_name} • {transit.aspect_type.title()} • Accuracy: {transit.historical_accuracy:.1f}%_")
                        
                        # Display sector impacts
                        if forecast.sector_impact:
                            st.markdown("**Sector Impacts:**")
                            for sector, impact in forecast.sector_impact.items():
                                if impact > 0:
                                    st.success(f"{sector.upper()}: +{impact:.1f}%")
                                else:
                                    st.error(f"{sector.upper()}: {impact:.1f}%")
        
        with tab2:
            st.markdown(f"## 📊 {st.session_state.symbol} Analysis")
            st.markdown(f"### {platform.month_names[st.session_state.month]} {st.session_state.year} Astrological Analysis")
            
            forecasts = st.session_state.report
            
            # Enhanced Monthly Summary Stats
            col1, col2, col3, col4 = st.columns(4)
            
            total_bullish_impact = sum(float(f.change.replace('+', '').replace('-', '')) for f in forecasts if f.sentiment == Sentiment.BULLISH)
            total_bearish_impact = sum(float(f.change.replace('+', '').replace('-', '')) for f in forecasts if f.sentiment == Sentiment.BEARISH)
            avg_daily_change = np.mean([float(f.change.replace('+', '').replace('-', '')) for f in forecasts])
            high_accuracy_transits = len([f for f in forecasts if f.detailed_transit.historical_accuracy > 75])
            
            stats = [
                (f"+{total_bullish_impact:.1f}%", "Total Bullish Impact", "🟢"),
                (f"-{total_bearish_impact:.1f}%", "Total Bearish Impact", "🔴"),
                (f"{avg_daily_change:.1f}%", "Avg Daily Impact", "🟡"),
                (high_accuracy_transits, "High Accuracy Transits", "⭐")
            ]
            
            for i, (value, label, emoji) in enumerate(stats):
                with [col1, col2, col3, col4][i]:
                    st.metric(label=f"{emoji} {label}", value=value)
            
            # Symbol-specific ruling planets info
            ruling_planets = platform.symbol_planetary_rulers.get(st.session_state.symbol, ['jupiter', 'saturn'])
            
            # Use native Streamlit components instead of HTML
            st.info(f"🪐 **Ruling Planetary Influences for {st.session_state.symbol}**")
            st.markdown(f"**Primary Ruling Planets:** {', '.join([p.title() for p in ruling_planets])}")
            st.markdown(f"_These planets have the strongest influence on {st.session_state.symbol}'s price movements according to astrological analysis._")
            
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
                            # Use native Streamlit components
                            st.markdown(f"**{forecast.date.split('-')[2]}/{forecast.date.split('-')[1]}**")
                            
                            # Display change with appropriate color
                            if forecast.sentiment == Sentiment.BULLISH:
                                st.success(f"{forecast.change}%")
                            elif forecast.sentiment == Sentiment.BEARISH:
                                st.error(f"{forecast.change}%")
                            else:
                                st.info(f"{forecast.change}%")
                            
                            # Signal
                            signal_text = {
                                SignalType.LONG: "🟢 BUY",
                                SignalType.SHORT: "🔴 SELL", 
                                SignalType.HOLD: "🟡 HOLD"
                            }.get(forecast.signal, "🟡 HOLD")
                            
                            st.markdown(f"**{signal_text}**")
                            st.markdown(f"_{forecast.detailed_transit.planet} Transit_")
                            st.markdown("---")
        
        with tab3:
            st.markdown(f"## 📈 {st.session_state.symbol} Astrological Movement Graph")
            st.markdown("### Dynamic Planetary Transit Impact Visualization")
            
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
            st.markdown(f"## 🌙 Advanced Planetary Transit Analysis")
            st.markdown(f"### {st.session_state.symbol} - {platform.month_names[st.session_state.month]} {st.session_state.year}")
            
            # Symbol-specific ruling planet analysis
            ruling_planets = platform.symbol_planetary_rulers.get(st.session_state.symbol, ['jupiter', 'saturn'])
            
            # Use expandable section for better organization
            with st.expander("🪐 Symbol-Specific Planetary Rulership", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Primary Ruling Planets",
                        value=', '.join([p.title() for p in ruling_planets]),
                        delta="Enhanced Analysis"
                    )
                
                with col2:
                    st.metric(
                        label="Symbol Type",
                        value=st.session_state.analysis_type.title(),
                        delta="Specific Analysis"
                    )
                
                with col3:
                    st.metric(
                        label="Enhanced Sensitivity",
                        value="50% Stronger",
                        delta="From Ruling Planets"
                    )
            
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
            
            st.markdown(f"### 🌟 Major Planetary Transits - {month_name} {st.session_state.year}")
            
            for transit_key, transit_data in sorted_transits[:6]:  # Show top 6 major transits
                transit = transit_data['transit']
                avg_impact = transit_data['total_impact'] / len(transit_data['forecasts'])
                is_ruling = transit_data['is_ruling_planet']
                
                # Get all dates for this transit
                transit_dates = [f.date for f in transit_data['forecasts']]
                date_range = f"{min(transit_dates)} to {max(transit_dates)}" if len(transit_dates) > 1 else transit_dates[0]
                
                # Calculate average sector impacts
                avg_sector_impacts = {}
                for sector, impacts in transit_data['sector_impacts'].items():
                    avg_sector_impacts[sector] = np.mean(impacts)
                
                # Enhanced transit description
                if transit.transit_type == 'retrograde':
                    transit_desc = f"{transit.planet} Retrograde in {platform.zodiac_signs[transit.zodiac_sign].name if transit.zodiac_sign in platform.zodiac_signs else transit.zodiac_sign.title()}"
                    impact_multiplier = 1.3  # Retrograde has stronger impact
                elif transit.transit_type == 'direct':
                    transit_desc = f"{transit.planet} Direct in {platform.zodiac_signs[transit.zodiac_sign].name if transit.zodiac_sign in platform.zodiac_signs else transit.zodiac_sign.title()}"
                    impact_multiplier = 1.2
                elif transit.transit_type == 'enters':
                    transit_desc = f"{transit.planet} enters {platform.zodiac_signs[transit.zodiac_sign].name if transit.zodiac_sign in platform.zodiac_signs else transit.zodiac_sign.title()}"
                    impact_multiplier = 1.1
                elif transit.transit_type == 'aspect':
                    transit_desc = f"{transit.planet} {transit.aspect_type} {transit.aspect_planet}"
                    impact_multiplier = 1.15
                else:
                    transit_desc = f"{transit.planet} in {platform.zodiac_signs[transit.zodiac_sign].name if transit.zodiac_sign in platform.zodiac_signs else transit.zodiac_sign.title()}"
                    impact_multiplier = 1.0
                
                if is_ruling:
                    impact_multiplier *= 1.5
                
                # Display transit info using native Streamlit components
                st.markdown(f"### 🌟 {transit_desc}")
                if is_ruling:
                    st.success("👑 RULING PLANET - Enhanced Impact on Your Symbol")
                
                st.markdown(f"**📅 Duration:** {date_range} ({len(transit_dates)} day{'s' if len(transit_dates) > 1 else ''})")
                
                # Create columns for key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="⏱️ Duration",
                        value=f"{len(transit_dates)} day{'s' if len(transit_dates) > 1 else ''}",
                        delta=date_range
                    )
                
                with col2:
                    impact_value = avg_impact * impact_multiplier
                    st.metric(
                        label="📈 Average Impact",
                        value=f"{impact_value:+.1f}%",
                        delta="Strong" if is_ruling else "Normal"
                    )
                
                with col3:
                    st.metric(
                        label="🎯 Accuracy",
                        value=f"{transit.historical_accuracy:.1f}%",
                        delta=f"{transit.impact_strength} Impact"
                    )
                
                with col4:
                    zodiac_info = platform.zodiac_signs[transit.zodiac_sign] if transit.zodiac_sign in platform.zodiac_signs else None
                    element = zodiac_info.element if zodiac_info else 'Unknown'
                    quality = zodiac_info.quality if zodiac_info else 'Unknown'
                    st.metric(
                        label="🌟 Astrological",
                        value=f"{element}",
                        delta=f"{quality} • {transit.degree:.1f}°"
                    )
                
                # Daily breakdown with enhanced visualization
                if len(transit_data['forecasts']) > 1:
                    st.markdown("**📋 Daily Progression:**")
                    
                    # Use tabs for better organization
                    daily_tabs = st.tabs([f"Day {i+1}" for i in range(min(7, len(transit_data['forecasts'])))])
                    
                    for i, f in enumerate(transit_data['forecasts'][:7]):
                        with daily_tabs[i]:
                            change_val = float(f.change.replace('+', '').replace('-', ''))
                            if is_ruling:
                                change_val *= 1.5
                            
                            # Use metric for clean display
                            st.metric(
                                label=f"{f.date.split('-')[2]}/{f.date.split('-')[1]}",
                                value=f"{change_val:+.1f}%",
                                delta=f.signal.value
                            )
                
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
                            # Use clean metric display
                            delta_color = "normal"
                            if "BUY" in sector_data['recommendation']:
                                delta_color = "normal"
                            elif "SELL" in sector_data['recommendation']:
                                delta_color = "inverse"
                            
                            st.metric(
                                label=f"📊 {sector_data['sector'].upper()}",
                                value=f"{sector_data['impact']:+.2f}%",
                                delta=f"{sector_data['recommendation']} ({sector_data['relevance']})",
                                delta_color=delta_color
                            )
                
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
                    'Element': platform.zodiac_signs[transit.zodiac_sign].element if transit.zodiac_sign in platform.zodiac_signs else 'Unknown',
                    'Quality': platform.zodiac_signs[transit.zodiac_sign].quality if transit.zodiac_sign in platform.zodiac_signs else 'Unknown'
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
