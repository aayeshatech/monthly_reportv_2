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
    }
    
    .bullish { color: #4caf50; font-weight: bold; }
    .bearish { color: #f44336; font-weight: bold; }
    .neutral { color: #ff9800; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class Sentiment(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

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

@dataclass
class PlanetaryPosition:
    planet: str
    symbol: str
    degree: float
    zodiac_sign: str
    sign_degree: float

class AstrologicalTradingPlatform:
    
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
        
        self.zodiac_signs = [
            {"name": "♈ Aries", "start": 0},
            {"name": "♉ Taurus", "start": 30},
            {"name": "♊ Gemini", "start": 60},
            {"name": "♋ Cancer", "start": 90},
            {"name": "♌ Leo", "start": 120},
            {"name": "♍ Virgo", "start": 150},
            {"name": "♎ Libra", "start": 180},
            {"name": "♏ Scorpio", "start": 210},
            {"name": "♐ Sagittarius", "start": 240},
            {"name": "♑ Capricorn", "start": 270},
            {"name": "♒ Aquarius", "start": 300},
            {"name": "♓ Pisces", "start": 330}
        ]
        
        self.month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        
        self.monthly_events_2025 = {
            0: [
                {"date": 1, "event": "New Year - Mercury sextile Venus", "sentiment": Sentiment.BULLISH, "change": "+1.2"},
                {"date": 3, "event": "Sun trine Jupiter", "sentiment": Sentiment.BULLISH, "change": "+2.1"},
                {"date": 7, "event": "Venus square Mars", "sentiment": Sentiment.BEARISH, "change": "-1.8"},
                {"date": 14, "event": "Mercury conjunct Saturn", "sentiment": Sentiment.BEARISH, "change": "-1.5"},
                {"date": 21, "event": "Mars trine Uranus", "sentiment": Sentiment.BULLISH, "change": "+2.3"},
                {"date": 25, "event": "Venus conjunct Jupiter", "sentiment": Sentiment.BULLISH, "change": "+3.1"},
                {"date": 28, "event": "Mercury square Pluto", "sentiment": Sentiment.BEARISH, "change": "-2.0"}
            ],
            7: [
                {"date": 2, "event": "Mercury square Jupiter", "sentiment": Sentiment.BEARISH, "change": "-1.7"},
                {"date": 6, "event": "Mars trine Neptune", "sentiment": Sentiment.NEUTRAL, "change": "+1.4"},
                {"date": 10, "event": "Venus opposition Uranus", "sentiment": Sentiment.BEARISH, "change": "-2.2"},
                {"date": 11, "event": "Mercury Direct", "sentiment": Sentiment.BULLISH, "change": "+1.9"},
                {"date": 15, "event": "Sun sextile Jupiter", "sentiment": Sentiment.BULLISH, "change": "+2.3"},
                {"date": 19, "event": "Mars square Pluto", "sentiment": Sentiment.BEARISH, "change": "-2.1"},
                {"date": 23, "event": "Venus trine Saturn", "sentiment": Sentiment.NEUTRAL, "change": "+1.2"},
                {"date": 27, "event": "Jupiter opposition Saturn", "sentiment": Sentiment.BEARISH, "change": "-2.5"},
                {"date": 31, "event": "Sun square Mars", "sentiment": Sentiment.BEARISH, "change": "-1.8"}
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

    def get_zodiac_sign(self, degree: float) -> Tuple[str, float]:
        normalized_degree = ((degree % 360) + 360) % 360
        
        for i in range(len(self.zodiac_signs) - 1, -1, -1):
            if normalized_degree >= self.zodiac_signs[i]["start"]:
                sign_degree = normalized_degree - self.zodiac_signs[i]["start"]
                return self.zodiac_signs[i]["name"], sign_degree
        
        return self.zodiac_signs[0]["name"], normalized_degree

    def get_impact_level(self, sentiment: Sentiment, change_str: str) -> str:
        try:
            change = abs(float(change_str.replace('+', '').replace('±', '').replace('-', '')))
        except (ValueError, AttributeError):
            change = 1.0
            
        if sentiment == Sentiment.BULLISH:
            if change > 2.5:
                return 'Very Strong Bullish'
            elif change > 1.5:
                return 'Strong Bullish'
            else:
                return 'Moderate Bullish'
        elif sentiment == Sentiment.BEARISH:
            if change > 2.5:
                return 'Very Strong Bearish'
            elif change > 1.5:
                return 'Strong Bearish'
            else:
                return 'Moderate Bearish'
        else:
            if change > 1.5:
                return 'Significant Neutral'
            else:
                return 'Moderate Neutral'

    def generate_monthly_forecast(self, symbol: str, year: int, month: int) -> List[Forecast]:
        forecasts = []
        month_data = self.monthly_events_2025.get(month, [])
        
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
                forecasts.append(Forecast(
                    date=current_date.strftime('%Y-%m-%d'),
                    day=day,
                    event=day_event["event"],
                    sentiment=day_event["sentiment"],
                    change=day_event["change"],
                    impact=self.get_impact_level(day_event["sentiment"], day_event["change"])
                ))
            else:
                sentiment_options = [Sentiment.BULLISH, Sentiment.BEARISH, Sentiment.NEUTRAL]
                sentiment = sentiment_options[day % 3]
                change_percent = ((day * 37) % 100 - 50) / 50
                
                change_str = f"{'+' if change_percent > 0 else ''}{change_percent:.1f}"
                
                forecasts.append(Forecast(
                    date=current_date.strftime('%Y-%m-%d'),
                    day=day,
                    event='Minor planetary transit',
                    sentiment=sentiment,
                    change=change_str,
                    impact=self.get_impact_level(sentiment, change_str)
                ))
        
        return forecasts

    def get_planetary_positions_display(self, date: datetime.date) -> List[PlanetaryPosition]:
        positions = self.calculate_planetary_positions(date)
        display_positions = []
        
        for planet_key, degree in positions.items():
            zodiac_sign, sign_degree = self.get_zodiac_sign(degree)
            display_positions.append(PlanetaryPosition(
                planet=self.planets[planet_key].name,
                symbol=self.planets[planet_key].symbol,
                degree=degree,
                zodiac_sign=zodiac_sign,
                sign_degree=sign_degree
            ))
        
        return display_positions

    def generate_report(self, symbol: str = 'NIFTY', year: int = 2025, month: int = 7) -> Dict:
        current_date = datetime.date.today()
        
        forecasts = self.generate_monthly_forecast(symbol, year, month)
        planetary_positions = self.get_planetary_positions_display(current_date)
        
        bullish_count = sum(1 for f in forecasts if f.sentiment == Sentiment.BULLISH)
        bearish_count = sum(1 for f in forecasts if f.sentiment == Sentiment.BEARISH)
        
        market_sentiment = 'BULLISH' if bullish_count > bearish_count else 'BEARISH' if bearish_count > bullish_count else 'NEUTRAL'
        risk_level = 'HIGH' if bearish_count > 15 else 'MEDIUM' if bearish_count > 10 else 'LOW'
        
        report = {
            'symbol': symbol,
            'month': self.month_names[month],
            'year': year,
            'generated_date': current_date.isoformat(),
            'market_sentiment': market_sentiment,
            'risk_level': risk_level,
            'active_transits': len(planetary_positions),
            'forecasts': forecasts,
            'planetary_positions': planetary_positions,
            'statistics': {
                'total_days': len(forecasts),
                'bullish_days': bullish_count,
                'bearish_days': bearish_count,
                'neutral_days': len(forecasts) - bullish_count - bearish_count
            }
        }
        
        return report

def main():
    if 'platform' not in st.session_state:
        st.session_state.platform = AstrologicalTradingPlatform()
    
    platform = st.session_state.platform
    
    st.markdown('<h1 class="main-header">🌟 Advanced Astrological Trading Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #b8b8b8; font-size: 1.2rem;">Professional Market Analysis with Planetary Transits & Forecasting</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; color: #ffd700;">Current Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>', unsafe_allow_html=True)
    
    st.sidebar.markdown("## ⚙️ Analysis Controls")
    
    symbol = st.sidebar.text_input("📈 Symbol/Index", value="NIFTY", help="Enter trading symbol")
    
    month_options = {i: f"{platform.month_names[i]} 2025" for i in range(12)}
    selected_month = st.sidebar.selectbox("📅 Select Month", options=list(month_options.keys()), 
                                         format_func=lambda x: month_options[x], index=7)
    
    market_type = st.sidebar.selectbox("🏪 Market Type", ["Equity", "Commodity", "Currency", "Cryptocurrency"])
    
    time_zone = st.sidebar.selectbox("🌍 Time Zone", ["IST", "EST", "GMT", "JST"])
    
    if st.sidebar.button("🚀 Generate Analysis", type="primary"):
        with st.spinner("Calculating planetary positions and market correlations..."):
            st.session_state.report = platform.generate_report(symbol, 2025, selected_month)
    
    if 'report' in st.session_state:
        report = st.session_state.report
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Forecast", "🌙 Planets", "📈 Signals"])
        
        with tab1:
            st.markdown("### 📊 Market Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{report['market_sentiment']}</h3>
                    <p>Market Sentiment</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{report['statistics']['bullish_days']}</h3>
                    <p>Bullish Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{report['statistics']['bearish_days']}</h3>
                    <p>Bearish Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <h3>{report['risk_level']}</h3>
                    <p>Risk Level</p>
                </div>
                """, unsafe_allow_html=True)
            
            sentiment_data = {
                'Sentiment': ['Bullish', 'Bearish', 'Neutral'],
                'Days': [
                    report['statistics']['bullish_days'],
                    report['statistics']['bearish_days'],
                    report['statistics']['neutral_days']
                ]
            }
            
            fig = px.pie(sentiment_data, values='Days', names='Sentiment', 
                        title=f"Market Sentiment Distribution for {report['month']} {report['year']}",
                        color_discrete_map={'Bullish': '#4caf50', 'Bearish': '#f44336', 'Neutral': '#ff9800'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 🔮 Monthly Forecast")
            
            forecasts = report['forecasts'][:12]
            
            for i in range(0, len(forecasts), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(forecasts):
                        forecast = forecasts[i + j]
                        sentiment_class = forecast.sentiment.value
                        
                        with col:
                            st.markdown(f"""
                            <div class="forecast-card">
                                <div style="color: #ffd700; font-weight: bold; margin-bottom: 10px;">{forecast.date}</div>
                                <div style="margin-bottom: 10px;">{forecast.event}</div>
                                <div class="{sentiment_class}" style="margin-bottom: 10px;">{forecast.impact}</div>
                                <div style="color: #b8b8b8;">Expected Change: {forecast.change}%</div>
                            </div>
                            """, unsafe_allow_html=True)
            
            df_forecasts = pd.DataFrame([asdict(f) for f in report['forecasts']])
            df_forecasts['change_numeric'] = df_forecasts['change'].str.replace('+', '').str.replace('±', '').astype(float)
            df_forecasts['date'] = pd.to_datetime(df_forecasts['date'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_forecasts['date'],
                y=df_forecasts['change_numeric'],
                mode='lines+markers',
                name='Expected Change %',
                line=dict(color='#ffd700', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f"Daily Market Change Forecast - {report['month']} {report['year']}",
                xaxis_title="Date",
                yaxis_title="Expected Change (%)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🌙 Planetary Positions")
            
            positions = report['planetary_positions']
            
            for i in range(0, len(positions), 5):
                cols = st.columns(5)
                for j, col in enumerate(cols):
                    if i + j < len(positions):
                        pos = positions[i + j]
                        
                        with col:
                            st.markdown(f"""
                            <div class="planet-card">
                                <div style="font-size: 2rem; margin-bottom: 10px;">{pos.symbol}</div>
                                <div style="color: #ffd700; font-weight: bold; margin-bottom: 5px;">{pos.degree:.1f}°</div>
                                <div style="color: #b8b8b8; font-size: 0.9rem;">{pos.zodiac_sign}</div>
                            </div>
                            """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### 📈 Trading Signals")
            
            significant_forecasts = [f for f in report['forecasts'] if abs(float(f.change.replace('+', '').replace('±', '').replace('-', ''))) > 2]
            
            for forecast in significant_forecasts[:8]:
                signal_color = '#4caf50' if forecast.sentiment == Sentiment.BULLISH else '#f44336' if forecast.sentiment == Sentiment.BEARISH else '#ff9800'
                signal_text = '📈 BUY' if forecast.sentiment == Sentiment.BULLISH else '📉 SELL' if forecast.sentiment == Sentiment.BEARISH else '➡️ HOLD'
                
                st.markdown(f"""
                <div class="forecast-card">
                    <div style="color: #ffd700; font-weight: bold; margin-bottom: 10px;">{forecast.date}</div>
                    <div style="color: {signal_color}; font-weight: bold; margin-bottom: 10px;">{signal_text}</div>
                    <div style="margin-bottom: 10px;">{forecast.event}</div>
                    <div style="color: #b8b8b8;">Expected: {forecast.change}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        if st.sidebar.button("📊 Export Report"):
            df_export = pd.DataFrame([
                {
                    'Date': f.date,
                    'Day': f.day,
                    'Event': f.event,
                    'Sentiment': f.sentiment.value,
                    'Change_%': f.change,
                    'Impact_Level': f.impact
                } for f in report['forecasts']
            ])
            
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.sidebar.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name=f"astro_trading_report_{report['symbol']}_{report['month']}_{report['year']}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
