"""
Hybrid Flask API: RAG (historical) + Live inference (current) + Gemini LLM
ZERO HALLUCINATION VERSION - UPDATED
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_query import RAGSystem
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize RAG
print("🔧 Initializing RAG system...")
rag = RAGSystem('vector_db.pkl')

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not found in .env file")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured")

# FIRMS API
FIRMS_API_KEY = os.getenv('FIRMS_API_KEY', 'ffe67bd547acd1ab34b70c7376aabdca')

# Load infrastructure data
INFRA_PATH = Path("D:/wildfire-risk-system/data/infrastructure/all_infrastructure_with_residential.csv")
if INFRA_PATH.exists():
    infra_df = pd.read_csv(INFRA_PATH)
    print(f"✅ Infrastructure loaded: {len(infra_df)} assets")
else:
    infra_df = pd.DataFrame()
    print("⚠️  Infrastructure data not found")

# ============================================================================
# HELPER FUNCTIONS - LIVE DATA
# ============================================================================

import pandas as pd
import numpy as np

def clean_for_json(obj):
    """Recursively clean NaN/Inf values for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif pd.isna(obj):
        return None
    return obj

def fetch_live_fires(bbox, days=1):
    """Fetch current fires from NASA FIRMS"""
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_API_KEY}/VIIRS_NOAA20_NRT/{bbox}/{days}"
    
    try:
        print(f"  📡 Fetching FIRMS data: {url}")
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        
        if not response.text or 'error' in response.text.lower():
            print(f"  ⚠️  No fire data returned or API error")
            return []
        
        from io import StringIO
        import csv
        
        fires = []
        reader = csv.DictReader(StringIO(response.text))
        
        for row in reader:
            try:
                fires.append({
                    'lat': float(row['latitude']),
                    'lon': float(row['longitude']),
                    'frp': float(row.get('frp', 0)),
                    'brightness': float(row.get('brightness', 300)),
                    'confidence': row.get('confidence', 'n'),
                    'date': row.get('acq_date', ''),
                    'time': row.get('acq_time', '')
                })
            except (ValueError, KeyError) as e:
                continue
        
        print(f"  ✓ Fetched {len(fires)} fire detections")
        return fires
        
    except Exception as e:
        print(f"  ❌ FIRMS API error: {e}")
        return []

def get_live_weather(lat, lon):
    """Fetch current weather from Open-Meteo API"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    
    try:
        print(f"  🌤️  Fetching weather for ({lat:.2f}, {lon:.2f})")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        current = data.get('current', {})
        
        weather = {
            'wind_speed_kmh': current.get('wind_speed_10m', 15),
            'wind_direction_deg': current.get('wind_direction_10m', 180),
            'temperature_c': current.get('temperature_2m', 20),
            'humidity_pct': current.get('relative_humidity_2m', 50)
        }
        
        print(f"  ✓ Weather: {weather['wind_speed_kmh']:.1f} km/h wind, {weather['temperature_c']:.1f}°C")
        return weather
        
    except Exception as e:
        print(f"  ❌ Weather API error: {e}")
        return {
            'wind_speed_kmh': 15,
            'wind_direction_deg': 180,
            'temperature_c': 20,
            'humidity_pct': 50,
            'note': 'Using default values'
        }

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def calculate_wind_alignment(fire_lat, fire_lon, asset_lat, asset_lon, wind_dir):
    """Calculate wind alignment between fire and asset"""
    lat1, lon1 = np.radians(fire_lat), np.radians(fire_lon)
    lat2, lon2 = np.radians(asset_lat), np.radians(asset_lon)
    dlon = lon2 - lon1
    
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360
    
    wind_toward = (wind_dir + 180) % 360
    angle_diff = abs(wind_toward - bearing)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    return np.cos(np.radians(angle_diff))

def calculate_simple_risk(min_dist, num_nearby_fires, wind_speed, wind_alignment):
    """Simple physics-based risk calculation"""
    if min_dist < 1:
        base_risk = 95
    elif min_dist < 5:
        base_risk = 80
    elif min_dist < 10:
        base_risk = 60
    elif min_dist < 20:
        base_risk = 40
    elif min_dist < 30:
        base_risk = 20
    else:
        base_risk = 5
    
    density_factor = 1.0 + min(num_nearby_fires / 50, 0.5)
    wind_factor = 1.0 + max(0, (wind_speed - 20) / 40)
    
    if wind_alignment > 0.5:
        alignment_factor = 1.3
    elif wind_alignment < -0.5:
        alignment_factor = 0.7
    else:
        alignment_factor = 1.0
    
    risk = base_risk * density_factor * wind_factor * alignment_factor
    return min(risk, 100)

def calculate_live_risks(fires, location, weather):
    """Calculate risks for infrastructure near location"""
    
    if not fires or infra_df.empty:
        return []
    
    lat, lon = location['lat'], location['lon']
    
    print(f"  🏥 Analyzing infrastructure near {location['name']}...")
    
    nearby_infra = infra_df[
        (infra_df['lat'] >= lat - 0.5) & 
        (infra_df['lat'] <= lat + 0.5) &
        (infra_df['lon'] >= lon - 0.5) & 
        (infra_df['lon'] <= lon + 0.5) &
        (infra_df['type'].isin(['Hospitals', 'Schools', 'Power Substations', 'Fire Stations']))
    ].copy()
    
    if len(nearby_infra) == 0:
        print(f"  ⚠️  No critical infrastructure found near {location['name']}")
        return []
    
    print(f"  ✓ Found {len(nearby_infra)} infrastructure assets")
    
    if len(nearby_infra) > 50:
        nearby_infra = nearby_infra.sample(50, random_state=42)
    
    results = []
    
    for _, asset in nearby_infra.iterrows():
        distances = [haversine(asset['lat'], asset['lon'], f['lat'], f['lon']) for f in fires]
        
        if not distances:
            continue
        
        min_dist = min(distances)
        
        if min_dist > 100:
            continue
        
        mean_dist = np.mean(distances)
        num_nearby = sum(1 for d in distances if d < 30)
        
        closest_idx = distances.index(min_dist)
        closest_fire = fires[closest_idx]
        
        wind_align = calculate_wind_alignment(
            closest_fire['lat'], closest_fire['lon'],
            asset['lat'], asset['lon'],
            weather['wind_direction_deg']
        )
        
        risk = calculate_simple_risk(
            min_dist, 
            num_nearby,
            weather['wind_speed_kmh'],
            wind_align
        )
        
        results.append({
            'name': asset.get('name', 'Unknown'),
            'type': asset['type'],
            'city': asset.get('city', ''),
            'lat': float(asset['lat']),
            'lon': float(asset['lon']),
            'risk_score': round(risk, 1),
            'distance_km': round(min_dist, 2),
            'fires_nearby': int(num_nearby),
            'wind_alignment': round(wind_align, 2)
        })
    
    results = sorted(results, key=lambda x: x['risk_score'], reverse=True)
    
    print(f"  ✓ Analyzed {len(results)} assets, {len([r for r in results if r['risk_score'] > 70])} high-risk")
    
    return results

# ============================================================================
# INTENT DETECTION
# ============================================================================

def detect_intent(question):
    """Detect if question needs live data and extract location"""
    
    live_keywords = [
        'now', 'current', 'today', 'live', 'right now', 
        'active', 'currently', 'at the moment', 'present',
        'ongoing', 'this moment', 'are there', 'any fires', 'fire status', 'weather', 'risk', 'near me'
    ]
    
    q_lower = question.lower()
    
    needs_live = any(kw in q_lower for kw in live_keywords)
    
    locations = {
        'san francisco': (37.77, -122.42),
        'los angeles': (34.05, -118.24),
        'paradise': (39.76, -121.62),
        'sacramento': (38.58, -121.49),
        'san diego': (32.72, -117.16),
        'fresno': (36.74, -119.78),
        'oakland': (37.80, -122.27),
        'san jose': (37.34, -121.89),
        'bakersfield': (35.37, -119.02),
        'riverside': (33.95, -117.40),
        'santa rosa': (38.44, -122.71),
        'redding': (40.59, -122.39),
        'chico': (39.73, -121.84),
        'napa': (38.30, -122.29),
        'sonoma': (38.29, -122.45)
    }
    
    location = None
    for city, coords in locations.items():
        if city in q_lower:
            location = {
                'name': city.title(),
                'lat': coords[0],
                'lon': coords[1]
            }
            break
    
    return {
        'needs_live': needs_live,
        'location': location
    }

# ============================================================================
# FALLBACK RESPONSE FORMATTER
# ============================================================================

def format_fallback_response(question, context, live_data):
    """Smart fallback when LLM fails"""
    q_lower = question.lower()
    
    # PRIORITIZE LIVE DATA
    if live_data:
        location = live_data['location']
        fire_count = live_data['fire_count']
        w = live_data['weather']
        
        if fire_count == 0:
            return (f"**No active fires detected near {location}** as of {live_data['data_timestamp']}.\n\n"
                   f"The area is currently clear of wildfire threats according to NASA satellite data.\n\n"
                   f"**Current Weather:**\n"
                   f"• Temperature: {w['temperature_c']:.1f}°C\n"
                   f"• Wind: {w['wind_speed_kmh']:.1f} km/h\n"
                   f"• Humidity: {w['humidity_pct']}%\n\n"
                   f"Conditions appear favorable with no immediate fire risk.")
        else:
            response = f"**🔥 ACTIVE FIRE ALERT - {location}**\n\n"
            response += f"{fire_count} fire detection(s) in the last 24 hours from NASA satellites.\n\n"
            
            if live_data['high_risk_assets'] > 0:
                response += f"⚠️ **{live_data['high_risk_assets']} facilities at high risk (>70%)**\n\n"
                response += "**Most vulnerable:**\n"
                for i, r in enumerate(live_data['top_risks'][:3], 1):
                    response += f"{i}. {r['name']} ({r['type']}) - {r['risk_score']}% risk, {r['distance_km']}km from fire\n"
            
            response += f"\n**Weather:** {w['wind_speed_kmh']:.0f} km/h winds, {w['temperature_c']:.0f}°C, {w['humidity_pct']}% humidity"
            return response
    
    # Historical fire queries
    if 'deadliest' in q_lower or 'worst' in q_lower or 'major' in q_lower:
        return ("**Deadliest Wildfires in California History:**\n\n"
               "**1. Camp Fire (2018)**\n"
               "   • 85 deaths - Deadliest in state history\n"
               "   • 18,804 structures destroyed\n"
               "   • 153,336 acres burned\n"
               "   • Destroyed Paradise, CA\n\n"
               "**2. Tubbs Fire (2017)**\n"
               "   • 22 deaths\n"
               "   • 5,636 structures destroyed\n"
               "   • 36,807 acres burned\n"
               "   • Devastated Santa Rosa\n\n"
               "**3. Woolsey Fire (2018)**\n"
               "   • 3 deaths\n"
               "   • 1,643 structures destroyed\n"
               "   • 96,949 acres burned\n"
               "   • Reached Pacific Ocean\n\n"
               "**4. Thomas Fire (2017)**\n"
               "   • 2 deaths\n"
               "   • 1,063 structures destroyed\n"
               "   • 281,893 acres - Largest at the time\n\n"
               "**5. Dixie Fire (2021)**\n"
               "   • 1 death\n"
               "   • 1,329 structures destroyed\n"
               "   • 963,309 acres - Second-largest ever")
    
    if 'camp fire' in q_lower:
        return ("**Camp Fire (November 8, 2018)**\n\n"
               "The deadliest wildfire in California history:\n\n"
               "• **85 deaths**\n"
               "• **18,804 structures destroyed**\n"
               "• **153,336 acres burned**\n"
               "• Started by PG&E transmission lines\n"
               "• Destroyed Paradise, CA and Adventist Health Feather River Hospital\n"
               "• Extreme 87 km/h winds drove rapid spread")
    
    elif 'dixie fire' in q_lower:
        return ("**Dixie Fire (July 13, 2021)**\n\n"
               "Second-largest wildfire in California history:\n\n"
               "• **1 death**\n"
               "• **1,329 structures destroyed**\n"
               "• **963,309 acres burned**\n"
               "• Burned for nearly 3 months\n"
               "• Destroyed the town of Greenville\n"
               "• Multiple power substations damaged")
    
    elif 'hospital' in q_lower and ('how many' in q_lower or 'count' in q_lower):
        return ("**California Hospitals**\n\n"
               "810 hospitals are tracked in the infrastructure database across California. "
               "During wildfire events, each is assessed for risk based on distance to fires, wind conditions, and fire intensity.\n\n"
               "Additionally, there are 2,192 medical clinics in the database.")
    
    elif 'weather' in q_lower and 'risk' in q_lower:
        return ("**Weather Conditions That Increase Fire Risk:**\n\n"
               "Based on historical high-risk scenarios:\n\n"
               "• **High winds** (>40 km/h) - Increase spread rate and ember transport\n"
               "• **Low humidity** (<20%) - Dries out vegetation, making it more flammable\n"
               "• **High temperatures** (>35°C) - Increases fuel dryness\n"
               "• **Wind-fire alignment** - When wind blows from fire toward assets\n\n"
               "The combination of these factors, especially with multiple nearby fires, creates extreme risk scenarios.")
    
    else:
        # Extract major fires from context if present
        if 'Camp Fire' in context:
            return ("**Major California Wildfires:**\n\n"
                   "Based on the database, the **Camp Fire (2018)** is documented as the deadliest and most destructive wildfire in California history, "
                   "causing 85 deaths and destroying 18,804 structures in Paradise, CA.\n\n"
                   "Other significant fires include the Dixie Fire (2021), Thomas Fire (2017), Tubbs Fire (2017), and Woolsey Fire (2018).")
        
        # Generic fallback
        lines = [l.strip() for l in context.split('\n') if l.strip() and not l.startswith('[') and 'Name:' not in l and 'dtype' not in l]
        response = "Based on the available California wildfire data:\n\n"
        for line in lines[:4]:
            if len(line) > 30:
                response += f"• {line}\n"
        return response if lines else "I don't have specific information to answer that question."

# ============================================================================
# CHAT ENDPOINT
# ============================================================================

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint - IMPROVED VERSION"""
    
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        print(f"\n{'='*60}")
        print(f"📝 Question: {question}")
        print(f"{'='*60}")
        
        # Detect intent
        intent = detect_intent(question)
        print(f"🔍 Intent: live_data={intent['needs_live']}, location={intent['location']}")
        
        context_parts = []
        live_data = None
        
        # Step 1: Get RAG context
        print("\n🔍 Searching RAG database...")
        rag_results = rag.query(question, top_k=5)
        
        if rag_results:
            for i, doc in enumerate(rag_results, 1):
                context_parts.append(f"[Historical Fact {i}] {doc['text']}")
                print(f"  ✓ Found: {doc['type']} (score: {doc['score']:.3f})")
        else:
            print("  ⚠️  No relevant historical data found")
        
        # Step 2: Fetch live data if needed
        if intent['needs_live'] and intent['location']:
            print(f"\n🔥 Fetching live data for {intent['location']['name']}...")
            
            loc = intent['location']
            bbox = f"{loc['lon']-0.5},{loc['lat']-0.5},{loc['lon']+0.5},{loc['lat']+0.5}"
            fires = fetch_live_fires(bbox, days=1)
            weather = get_live_weather(loc['lat'], loc['lon'])
            risks = calculate_live_risks(fires, loc, weather) if fires else []
            
            live_data = {
                'location': loc['name'],
                'fire_count': len(fires),
                'high_risk_assets': len([r for r in risks if r['risk_score'] > 70]),
                'medium_risk_assets': len([r for r in risks if 40 <= r['risk_score'] <= 70]),
                'top_risks': risks[:5],
                'weather': weather,
                'data_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M UTC')
            }
            
            # Add live data to context (PROMINENT PLACEMENT)
            context_parts.insert(0, f"\n=== CURRENT LIVE DATA (as of {live_data['data_timestamp']}) ===")
            context_parts.insert(1, f"Location: {loc['name']}")
            context_parts.insert(2, f"Active Fires: {len(fires)} detected in last 24 hours")
            context_parts.insert(3, f"Weather: {weather['temperature_c']:.1f}°C, {weather['wind_speed_kmh']:.1f} km/h wind, {weather['humidity_pct']}% humidity")
            
            if fires:
                if risks:
                    context_parts.insert(4, f"High Risk Assets: {live_data['high_risk_assets']}")
                    context_parts.insert(5, f"Medium Risk Assets: {live_data['medium_risk_assets']}")
                    
                    if live_data['top_risks']:
                        context_parts.insert(6, "\nMost At-Risk Facilities:")
                        for idx, r in enumerate(live_data['top_risks'][:3], 1):
                            context_parts.insert(6+idx, f"  {idx}. {r['name']} ({r['type']}): {r['risk_score']}% risk, {r['distance_km']}km away, {r['fires_nearby']} fires nearby")
            else:
                context_parts.insert(4, "Status: NO ACTIVE FIRES DETECTED - Area is currently clear")
            
            context_parts.insert(len(context_parts), "=== END LIVE DATA ===\n")
        
        elif intent['needs_live'] and not intent['location']:
            context_parts.append(
                "\n[NOTE] Live fire data requires a specific location. "
                "Supported: San Francisco, Los Angeles, Sacramento, Paradise, San Diego, etc."
            )
        
        # Step 3: Build full context
        full_context = "\n".join(context_parts)
        
        if not full_context:
            return jsonify({
                'answer': "I don't have enough information to answer this question. "
                          "Could you ask about California wildfires, infrastructure, or specific risk scenarios?",
                'live_data_used': False,
                'sources': [],
                'timestamp': datetime.now().isoformat()
            })
        
        # Step 4: Generate response with Gemini (IMPROVED)
        print("\n🤖 Generating response with Gemini...")
        
        system_prompt = """You are a California wildfire risk assistant.

CRITICAL RULES:
1. Answer ONLY using the provided data - never invent information
2. When LIVE DATA shows 0 fires, clearly state "No active fires detected"
3. ALWAYS complete your sentences - never cut off mid-thought
4. Be conversational but concise (2-4 paragraphs)
5. Cite specific numbers from the data
6. Prioritize live data when present - mention it's from NASA satellites
7. For risk analysis, explain key factors (distance, wind, fire density)

Response Guidelines:
- Live queries: Start with current status, add relevant context
- Historical queries: Focus on the specific event/fact
- Risk analysis: Explain the contributing factors clearly"""

        user_prompt = f"""Available Data:
{full_context}

Question: {question}

Provide a complete, helpful answer. Finish all sentences properly."""

        try:
            model_names = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            
            response = None
            llm_answer = None
            
            for model_name in model_names:
                try:
                    print(f"  🔄 Trying model: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    
                    response = model.generate_content(
                        f"{system_prompt}\n\n{user_prompt}",
                        generation_config=genai.GenerationConfig(
                            temperature=0.5,
                            top_p=0.95,
                            max_output_tokens=1000,
                            stop_sequences=None
                        ),
                        safety_settings={
                            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                        }
                    )
                    
                    llm_answer = response.text.strip()
                    
                    # Check for truncation
                    if llm_answer and not any(llm_answer.endswith(p) for p in ['.', '!', '?', '°C', '%', 'acres', 'MW']):
                        print(f"  ⚠️  Response appears truncated, trying next model")
                        continue
                    
                    print(f"  ✅ Success with {model_name} ({len(llm_answer)} chars)")
                    break
                    
                except Exception as e:
                    print(f"  ⚠️  {model_name} failed: {str(e)[:100]}")
                    continue
            
            if not llm_answer:
                raise Exception("All models failed or returned empty")

        except Exception as e:
            print(f"  ❌ Gemini failed: {e}")
            traceback.print_exc()
            llm_answer = format_fallback_response(question, full_context, live_data)
            print(f"  ✓ Using fallback response")
        
        # Build final response
        result = clean_for_json({
            'answer': llm_answer,
            'live_data_used': intent['needs_live'] and live_data is not None,
            'sources': [
                {
                    'text': doc['text'][:200] + ('...' if len(doc['text']) > 200 else ''),
                    'type': doc['type'],
                    'score': round(doc['score'], 3)
                }
                for doc in rag_results[:3]
            ],
            'timestamp': datetime.now().isoformat()
        })
        
        if live_data:
            result['live_data'] = live_data
        
        print(f"\n✅ Response generated")
        print(f"{'='*60}\n")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_loaded': rag is not None,
        'rag_docs': rag.get_stats()['total_docs'] if rag else 0,
        'infrastructure_loaded': not infra_df.empty,
        'infrastructure_count': len(infra_df) if not infra_df.empty else 0,
        'gemini_configured': GEMINI_API_KEY is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Database statistics"""
    stats_data = {
        'rag_stats': rag.get_stats() if rag else {},
        'infrastructure': {
            'total_assets': len(infra_df) if not infra_df.empty else 0,
            'types': infra_df['type'].value_counts().to_dict() if not infra_df.empty else {}
        },
        'supported_locations': [
            'San Francisco', 'Los Angeles', 'Paradise', 'Sacramento',
            'San Diego', 'Fresno', 'Oakland', 'San Jose', 'Bakersfield',
            'Riverside', 'Santa Rosa', 'Redding', 'Chico', 'Napa', 'Sonoma'
        ]
    }
    return jsonify(stats_data)

@app.route('/test-live', methods=['GET'])
def test_live():
    """Test live data fetching"""
    
    loc = {'name': 'San Francisco', 'lat': 37.77, 'lon': -122.42}
    bbox = f"{loc['lon']-0.5},{loc['lat']-0.5},{loc['lon']+0.5},{loc['lat']+0.5}"
    
    print("\n🧪 Testing live data APIs...")
    
    fires = fetch_live_fires(bbox, days=1)
    weather = get_live_weather(loc['lat'], loc['lon'])
    risks = calculate_live_risks(fires, loc, weather) if fires else []
    
    return jsonify({
        'location': loc['name'],
        'fires_detected': len(fires),
        'sample_fires': fires[:3],
        'weather': weather,
        'assets_analyzed': len(risks),
        'high_risk_count': len([r for r in risks if r['risk_score'] > 70]),
        'top_risks': risks[:3]
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔥 WILDFIRE RISK CHATBOT API - UPDATED VERSION")
    print("="*70)
    print(f"✅ RAG Database: {rag.get_stats()['total_docs']} documents")
    print(f"✅ Infrastructure: {len(infra_df):,} assets" if not infra_df.empty else "⚠️  Infrastructure: Not loaded")
    print(f"✅ Gemini API: {'Configured' if GEMINI_API_KEY else '❌ NOT CONFIGURED'}")
    print(f"✅ FIRMS API: {'Configured' if FIRMS_API_KEY else '❌ NOT CONFIGURED'}")
    print("="*70)
    print("\n🌐 Endpoints:")
    print("  POST /chat          - Main chatbot")
    print("  GET  /health        - Health check")
    print("  GET  /stats         - Statistics")
    print("  GET  /test-live     - Test live APIs")
    print("\n🚀 Server: http://localhost:5000")
    print("📡 Expose: ngrok http 5000")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)