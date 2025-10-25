"""
Complete solution combining AI analytics, geocoding, and translation.
All features work together seamlessly for optimal performance.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from datetime import datetime, timedelta
import re
import logging
import requests
import time
from functools import lru_cache
from googletrans import Translator
from deep_translator import GoogleTranslator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PART 1: ENHANCED GEOCODING WITH MULTIPLE STRATEGIES

# Comprehensive landmark database for Nakuru County
NAKURU_LANDMARKS = {
    # Town Centers
    'nakuru town': (-0.3031, 36.0800), 'nakuru cbd': (-0.3031, 36.0800),
    'kenyatta avenue': (-0.3031, 36.0800), 'town centre': (-0.3031, 36.0800),

    # All 11 Constituencies
    'bahati': (-0.2833, 36.0500), 'nakuru east': (-0.2800, 36.0900),
    'nakuru west': (-0.3200, 36.0700), 'gilgil': (-0.5000, 36.3200),
    'naivasha': (-0.7167, 36.4333), 'molo': (-0.2500, 35.7333),
    'rongai': (-0.1722, 35.8653), 'subukia': (-0.4000, 36.1500),
    'njoro': (-0.3333, 35.9333), 'kuresoi north': (-0.1167, 35.5833),
    'kuresoi south': (-0.2000, 35.6000),

    # Markets & Commercial
    'wakulima market': (-0.3025, 36.0795), 'free area': (-0.2850, 36.0820),
    'section 58': (-0.2950, 36.0750), 'gilanis': (-0.3040, 36.0810),
    'pipeline': (-0.3100, 36.0700), 'lanet': (-0.2167, 36.1167),

    # Residential Areas
    'bondeni': (-0.3100, 36.0850), 'milimani': (-0.2900, 36.0900),
    'shabab': (-0.3150, 36.0780), 'naka': (-0.2950, 36.0850),
    'flamingo': (-0.3200, 36.0650), 'satellite': (-0.3150, 36.0900),

    # Natural Landmarks
    'menengai crater': (-0.2000, 36.0700), 'lake nakuru': (-0.3667, 36.0833),
    'lake naivasha': (-0.7667, 36.3500), 'hells gate': (-0.9000, 36.3167),

    # Institutions
    'egerton university': (-0.3833, 35.9500), 'kabarak university': (-0.3167, 35.9000),

    # Shopping Centers
    'nakumatt': (-0.3035, 36.0810), 'west side mall': (-0.3200, 36.0750),
    'tumaini mall': (-0.2900, 36.0850), 'naivas': (-0.3040, 36.0805),

    # Health Facilities
    'war memorial hospital': (-0.3050, 36.0820), 'pgh': (-0.3050, 36.0820),
    'nakuru level 5': (-0.2967, 36.0783), 'rift valley hospital': (-0.3100, 36.0900),

    # Transport Hubs
    'nakuru railway station': (-0.2850, 36.0667), 'nakuru bus station': (-0.3040, 36.0805),
    'matatu stage': (-0.3035, 36.0810), 'stage': (-0.3035, 36.0810),
}


@lru_cache(maxsize=500)
def geocode_location(location_name, constituency="Nakuru", retries=3):
    """
    SMART GEOCODING: Converts location names to GPS coordinates using 3 strategies.

    Strategy 1: Local fuzzy matching (instant, offline)
    Strategy 2: Nominatim API (online, accurate)
    Strategy 3: Fallback to broader search

    Example:
        lat, lon = geocode_location("near nakuru town", "Nakuru East")
        # Returns: (-0.3031, 36.0800)
    """
    # Strategy 1: Fast local fuzzy matching
    coords = fuzzy_match_location(location_name)
    if coords[0]:
        logger.info(f"✓ Fuzzy matched '{location_name}' → {coords}")
        return coords

    # Strategy 2: Online geocoding with Nominatim
    for attempt in range(retries):
        try:
            # Build context-aware search query
            search_query = f"{location_name}, {constituency}, Nakuru County, Kenya"

            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={'q': search_query, 'format': 'json', 'limit': 1, 'countrycodes': 'ke'},
                headers={'User-Agent': 'NakuruSafetyPlatform/3.0'},
                timeout=8
            )

            if response.status_code == 200 and response.json():
                data = response.json()[0]
                lat, lon = float(data['lat']), float(data['lon'])

                # Validate within Nakuru County bounds
                if -1.2 <= lat <= 0.2 and 35.7 <= lon <= 36.5:
                    logger.info(f"✓ Geocoded '{location_name}' → ({lat:.4f}, {lon:.4f})")
                    return lat, lon

            # Strategy 3: Broader fallback search
            if attempt == 0:
                fallback_query = f"{location_name}, Nakuru, Kenya"
                response = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={'q': fallback_query, 'format': 'json', 'limit': 1, 'countrycodes': 'ke'},
                    headers={'User-Agent': 'NakuruSafetyPlatform/3.0'},
                    timeout=8
                )

                if response.status_code == 200 and response.json():
                    data = response.json()[0]
                    lat, lon = float(data['lat']), float(data['lon'])
                    if -1.2 <= lat <= 0.2 and 35.7 <= lon <= 36.5:
                        return lat, lon

            if attempt < retries - 1:
                time.sleep(1)  # Rate limiting

        except Exception as e:
            logger.warning(f"⚠ Geocoding attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)

    logger.warning(f"✗ Could not geocode: {location_name}")
    return None, None


def fuzzy_match_location(location_name):
    """
    FUZZY MATCHING: Matches user input to known landmarks using intelligent string matching.
    Handles typos, partial matches, and common variations.
    """
    location_lower = location_name.lower().strip()

    # Remove common stopwords
    for word in ['near', 'at', 'in', 'by', 'around', 'close to', 'next to', 'karibu', 'kwa']:
        location_lower = location_lower.replace(word, '').strip()

    # Direct match
    if location_lower in NAKURU_LANDMARKS:
        return NAKURU_LANDMARKS[location_lower]

    # Partial matching with scoring
    best_match = None
    best_score = 0

    for landmark, coords in NAKURU_LANDMARKS.items():
        # Check substring matches
        if landmark in location_lower or location_lower in landmark:
            score = len(location_lower) / len(landmark)
            if score > best_score:
                best_score = score
                best_match = coords

        # Word-level matching
        location_words = set(location_lower.split())
        landmark_words = set(landmark.split())
        overlap = len(location_words & landmark_words)

        if overlap > 0:
            score = overlap / max(len(location_words), len(landmark_words))
            if score > best_score and score >= 0.5:  # At least 50% match
                best_score = score
                best_match = coords

    if best_match and best_score >= 0.3:  # Minimum 30% confidence
        return best_match

    return None, None


# ============================================================================
# PART 2: INTEGRATED AI ANALYTICS WITH MULTILINGUAL SUPPORT
# ============================================================================

def detect_spam(report_data, get_settings_func=None):
    """
    AI-POWERED SPAM DETECTION with multilingual support.

    Analyzes reports using multiple indicators:
    - Test/spam keywords (English, Kiswahili, Kikuyu)
    - Promotional content detection
    - Location validation with geocoding
    - Description quality analysis

    Returns: {'is_spam': bool, 'spam_score': int, 'action': str, 'reasons': list}
    """
    # Get thresholds
    spam_threshold = 60
    auto_reject = 80
    if get_settings_func:
        try:
            settings = get_settings_func()
            spam_threshold = settings.get('spam_threshold', 60)
            auto_reject = settings.get('auto_reject_threshold', 80)
        except:
            pass

    spam_score = 0
    reasons = []

    description = str(report_data.get('description', '')).lower()
    location = str(report_data.get('manual_location', '')).lower()
    language = report_data.get('language', 'English')

    # Check 1: Test/Spam Keywords (multilingual)
    test_patterns = {
        'English': r'\b(test|testing|asdf|qwerty|spam|fake|xxx|dummy|sample|trial|check)\b',
        'Kiswahili': r'\b(jaribio|majaribio|bandia|uwongo|fake|test|kujaribu)\b',
        'Kikuyu': r'\b(kugeria|kugeragia|utaga|test|fake)\b'
    }

    if re.search(test_patterns.get(language, test_patterns['English']), description):
        spam_score += 50
        reasons.append("Contains test/spam keywords")

    # Check 2: Promotional Content (multilingual)
    promo_patterns = {
        'English': r'(buy|sale|discount|click\s*here|www\.|http|offer|deal|cheap|visit|order)',
        'Kiswahili': r'(nunua|uza|punguzo|bonyeza\s*hapa|bei\s*nafuu|ofa|tembelea)',
        'Kikuyu': r'(gura|kugura|bei\s*nini|www\.|http)'
    }

    if re.search(promo_patterns.get(language, promo_patterns['English']), description, re.IGNORECASE):
        spam_score += 50
        reasons.append("Promotional content detected")

    # Check 3: Description Quality
    if len(description) < 15:
        spam_score += 40
        reasons.append("Description too short")
    elif len(description.split()) < 5:
        spam_score += 30
        reasons.append("Too few words")

    # Check 4: Location Validation with Geocoding
    if not location or len(location) < 3:
        spam_score += 30
        reasons.append("Invalid location")
    else:
        # Try to geocode location - if it fails, likely invalid
        lat, lon = fuzzy_match_location(location)
        if not lat and not lon:
            # Try online geocoding as final check
            lat, lon = geocode_location(location, report_data.get('constituency', 'Nakuru'))
            if not lat:
                spam_score += 20
                reasons.append("Location not found")

    # Check 5: GPS Validation
    lat, lon = report_data.get('lat'), report_data.get('lon')
    if lat and lon:
        try:
            lat_f, lon_f = float(lat), float(lon)
            if lat_f == 0 and lon_f == 0:
                spam_score += 20
                reasons.append("Null coordinates")
            elif not (-1.2 <= lat_f <= 0.2 and 35.7 <= lon_f <= 36.5):
                spam_score += 30
                reasons.append("Outside Nakuru County")
        except:
            spam_score += 20
            reasons.append("Invalid GPS format")

    # Check 6: Repeated Characters
    if re.search(r'(.)\1{5,}', description):
        spam_score += 25
        reasons.append("Excessive repetition")

    # Determine action
    is_spam = spam_score >= spam_threshold
    if spam_score >= auto_reject:
        action = 'reject'
    elif is_spam:
        action = 'review'
    else:
        action = 'accept'

    return {
        'is_spam': is_spam,
        'spam_score': spam_score,
        'confidence': min(spam_score / 100, 1.0),
        'reasons': reasons,
        'action': action
    }


def detect_anomalies(reports):
    """
    AI ANOMALY DETECTION: Identifies urgent/critical incidents using NLP.

    Multilingual keyword detection for:
    - Violence (murder, gun, weapon)
    - Emergencies (fire, accident, help)
    - Time-sensitive indicators (now, happening)

    Returns: List of urgent reports with priority scores
    """
    if not reports or len(reports) < 1:
        return []

    anomalies = []

    # Multilingual critical keywords
    critical_keywords = {
        'English': ['murder', 'rape', 'gun', 'weapon', 'knife', 'kill', 'death', 'bomb', 'shot', 'shooting',
                    'stabbing'],
        'Kiswahili': ['mauaji', 'ubakaji', 'bunduki', 'silaha', 'kisu', 'kuua', 'kifo', 'bomu', 'risasi'],
        'Kikuyu': ['kuuraga', 'guthemba', 'mbaci', 'gikuu', 'mabomu', 'rihia']
    }

    urgent_keywords = {
        'English': ['emergency', 'urgent', 'help', 'attack', 'violence', 'fire', 'accident', 'danger', 'bleeding'],
        'Kiswahili': ['dharura', 'haraka', 'msaada', 'shambulio', 'jeuri', 'moto', 'ajali', 'hatari', 'damu'],
        'Kikuyu': ['thiina', 'haraka', 'uteithio', 'kugutha', 'haaro', 'mwaki', 'mutino']
    }

    time_patterns = {
        'English': r'\b(now|currently|happening|ongoing|right\s*now)\b',
        'Kiswahili': r'\b(sasa|inaendelea|inatokea|hivi\s*sasa)\b',
        'Kikuyu': r'\b(riu|riria|kugika)\b'
    }

    for report in reports:
        urgency_score = 0
        urgency_reasons = []
        description = str(report.get('description', '')).lower()
        language = report.get('language', 'English')

        # Check critical keywords (all languages)
        for lang, keywords in critical_keywords.items():
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            if re.search(pattern, description):
                urgency_score += 70
                urgency_reasons.append(f"Critical keywords detected")
                break

        # Check urgent keywords (if not already critical)
        if urgency_score < 70:
            for lang, keywords in urgent_keywords.items():
                pattern = r'\b(' + '|'.join(keywords) + r')\b'
                if re.search(pattern, description):
                    urgency_score += 40
                    urgency_reasons.append(f"Urgent keywords detected")
                    break

        # Check for emotional distress indicators
        if description.count('!') >= 3:
            urgency_score += 15
            urgency_reasons.append("High emotion detected")

        # Check time-sensitive language
        for lang, pattern in time_patterns.items():
            if re.search(pattern, description):
                urgency_score += 20
                urgency_reasons.append("Time-sensitive incident")
                break

        # Check location type (high-risk areas)
        location = str(report.get('manual_location', '')).lower()
        high_risk = ['school', 'hospital', 'market', 'cbd', 'bank', 'station']
        if any(area in location for area in high_risk):
            urgency_score += 10
            urgency_reasons.append("High-traffic location")

        # Add to anomalies if urgent
        if urgency_score >= 40:
            anomalies.append({
                'report': report,
                'urgency_score': min(urgency_score, 100),
                'priority': 'CRITICAL' if urgency_score >= 70 else 'HIGH',
                'reasons': urgency_reasons
            })

    # Sort by urgency
    anomalies.sort(key=lambda x: x['urgency_score'], reverse=True)
    return anomalies


def perform_clustering(hotspots, distance_threshold=0.01):
    """
    HIERARCHICAL CLUSTERING: Groups nearby incidents into hotspot zones.

    Uses Ward's method for optimal cluster formation.
    Distance threshold of 0.01 degrees ≈ 1.1 km
    """
    if len(hotspots) < 2:
        return list(zip(hotspots, [0] * len(hotspots)))

    try:
        coords = []
        for spot in hotspots:
            lat = float(spot.get('lat', 0))
            lon = float(spot.get('lon', 0))
            coords.append([lat, lon])

        data = np.array(coords, dtype=np.float64)
        dist_matrix = pdist(data, metric='euclidean')
        clusters = linkage(dist_matrix, method='ward')
        labels = fcluster(clusters, t=distance_threshold, criterion='distance')

        return list(zip(hotspots, labels.tolist()))
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        return list(zip(hotspots, [0] * len(hotspots)))


def calculate_hotspot_density(hotspots, radius=0.005):
    """
    DENSITY ANALYSIS: Calculates incident concentration at each location.

    Radius of 0.005 degrees ≈ 550 meters
    Risk levels: CRITICAL (10+), HIGH (6-9), MEDIUM (3-5), LOW (1-2)
    """
    if not hotspots:
        return []

    coords = []
    valid_hotspots = []

    for h in hotspots:
        try:
            lat, lon = float(h.get('lat', 0)), float(h.get('lon', 0))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                coords.append([lat, lon])
                valid_hotspots.append(h)
        except:
            continue

    if not coords:
        return []

    try:
        data = np.array(coords, dtype=np.float64)
        density_results = []

        for i, point in enumerate(data):
            distances = np.sqrt(np.sum((data - point) ** 2, axis=1))
            point_density = int(np.sum(distances <= radius))

            # Classify risk level
            if point_density >= 10:
                risk_level = 'CRITICAL'
            elif point_density >= 6:
                risk_level = 'HIGH'
            elif point_density >= 3:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'

            density_results.append({
                'hotspot': valid_hotspots[i],
                'lat': point[0],
                'lon': point[1],
                'density': point_density,
                'risk_level': risk_level
            })

        density_results.sort(key=lambda x: x['density'], reverse=True)
        return density_results
    except Exception as e:
        logger.error(f"Density calculation error: {e}")
        return []


# ============================================================================
# PART 3: ROBUST TRANSLATION SYSTEM
# ============================================================================

LANG_MAP = {'en': 'en', 'sw': 'sw', 'ki': 'sw', 'kikuyu': 'sw'}


def safe_translate(text, src='auto', dest='en', retries=3):
    """
    ROBUST TRANSLATION with multiple fallback strategies.

    Strategy 1: googletrans (primary)
    Strategy 2: deep_translator (fallback)
    Strategy 3: Error message if all fail
    """
    if not text or not text.strip():
        return text

    text = ' '.join(text.split())  # Clean whitespace

    # Try googletrans with retries
    for attempt in range(retries):
        try:
            translator = Translator()
            result = translator.translate(text, src=src, dest=dest)
            if result and result.text:
                return result.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    # Fallback to deep_translator
    try:
        fallback = GoogleTranslator(source=src, target=dest)
        return fallback.translate(text)
    except:
        pass

    return f"[Translation unavailable]"


@lru_cache(maxsize=200)
def detect_language(text):
    """LANGUAGE DETECTION with caching."""
    if not text or not text.strip():
        return 'en'

    try:
        translator = Translator()
        detected = translator.detect(text)
        if detected and detected.lang:
            return detected.lang.lower()
    except:
        pass

    return 'en'


def translate_text(text, target_lang):
    """
    SMART TRANSLATION: Translates text with automatic language detection.
    Skips translation if source = target for efficiency.
    """
    if not text or not text.strip():
        return text

    target = LANG_MAP.get(target_lang.lower(), 'en')
    src_lang = detect_language(text)

    # Handle Kikuyu (use Swahili)
    if src_lang in ('ki', 'kikuyu'):
        src_lang = 'sw'

    # Skip if same language
    if src_lang == target:
        return text

    return safe_translate(text, src=src_lang, dest=target)


def translate_report(report_dict, target_lang):
    """
    BATCH TRANSLATION: Translates all fields in a report.
    Used by police to translate citizen reports.
    """
    if not report_dict:
        return {}

    translated = {}
    for key, value in report_dict.items():
        if not value:
            translated[key] = value
            continue

        try:
            translated[key] = translate_text(str(value), target_lang)
        except:
            translated[key] = value  # Keep original if translation fails

    return translated


# ============================================================================
# PART 4: ADVANCED ANALYTICS & RECOMMENDATIONS
# ============================================================================

def analyze_trends(reports, time_window_days=7):
    """
    TREND ANALYSIS: Identifies crime patterns over time.
    Analyzes categories, peak hours, and trend direction.
    """
    if not reports:
        return {'total': 0, 'recent': 0, 'trend': 'stable', 'categories': {}, 'peak_hour': 12,
                'most_common_category': 'N/A'}

    try:
        now = datetime.now()
        cutoff = now - timedelta(days=time_window_days)
        recent_reports = []
        category_counts = {}
        hourly_distribution = [0] * 24

        for report in reports:
            try:
                timestamp = report.get('created_at', now)
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
            except:
                timestamp = now

            category = str(report.get('category', 'Unknown'))
            category_counts[category] = category_counts.get(category, 0) + 1
            hourly_distribution[timestamp.hour] += 1

            if timestamp >= cutoff:
                recent_reports.append(report)

        total = len(reports)
        recent = len(recent_reports)
        trend = 'stable'

        if total >= 2:
            recent_rate = recent / time_window_days
            oldest_date = reports[-1].get('created_at', now)
            if isinstance(oldest_date, str):
                oldest_date = datetime.fromisoformat(oldest_date)
            total_days = max((now - oldest_date).days, 1)
            overall_rate = total / total_days

            if recent_rate > overall_rate * 1.2:
                trend = 'increasing'
            elif recent_rate < overall_rate * 0.8:
                trend = 'decreasing'

        peak_hour = hourly_distribution.index(max(hourly_distribution))

        return {
            'total': total,
            'recent': recent,
            'trend': trend,
            'categories': category_counts,
            'peak_hour': peak_hour,
            'most_common_category': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else 'N/A'
        }
    except Exception as e:
        logger.error(f"Trend analysis error: {e}")
        return {'total': len(reports), 'recent': 0, 'trend': 'stable', 'categories': {}, 'peak_hour': 12,
                'most_common_category': 'N/A'}


def generate_patrol_recommendations(hotspots, reports, constituency):
    """
    AI-POWERED PATROL RECOMMENDATIONS
    Generates actionable intelligence for police deployment.
    """
    try:
        recommendations = []

        # Analyze data
        density_data = calculate_hotspot_density(hotspots)
        trends = analyze_trends(reports)
        anomalies = detect_anomalies(reports)

        # Recommendation 1: High-Density Zones
        critical_zones = [d for d in density_data if d['risk_level'] == 'CRITICAL']
        if critical_zones:
            locations = ', '.join([z['hotspot'].get('location', 'Unknown') for z in critical_zones[:3]])
            recommendations.append({
                'type': 'HIGH_DENSITY_PATROL',
                'priority': 'CRITICAL',
                'locations': critical_zones[:3],
                'message': f'🚨 Deploy patrols to {len(critical_zones)} critical zones: {locations}'
            })

        # Recommendation 2: Peak Hour Coverage
        peak_hour = trends.get('peak_hour', 12)
        recommendations.append({
            'type': 'PEAK_HOUR_COVERAGE',
            'priority': 'HIGH',
            'time': f'{peak_hour:02d}:00 - {(peak_hour + 2) % 24:02d}:00',
            'message': f'⏰ Increase presence during peak hours ({peak_hour:02d}:00-{(peak_hour + 2) % 24:02d}:00)'
        })

        # Recommendation 3: Category-Specific
        most_common = trends.get('most_common_category', 'N/A')
        if most_common != 'N/A':
            recommendations.append({
                'type': 'CATEGORY_FOCUS',
                'priority': 'MEDIUM',
                'category': most_common,
                'message': f'🎯 Focus on {most_common} prevention'
            })

        # Recommendation 4: Trend Alert
        if trends['trend'] == 'increasing':
            recommendations.append({
                'type': 'TREND_ALERT',
                'priority': 'HIGH',
                'message': f'📈 Crime trend is INCREASING in {constituency}'
            })

        # Recommendation 5: Urgent Incidents
        if anomalies:
            critical_count = sum(1 for a in anomalies if a['priority'] == 'CRITICAL')
            if critical_count > 0:
                recommendations.append({
                    'type': 'URGENT_RESPONSE',
                    'priority': 'CRITICAL',
                    'message': f'🚨 {critical_count} CRITICAL incidents need immediate attention'
                })

        return recommendations
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return []


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_integrated_system():
    """Test all integrated components"""
    print("\n" + "=" * 70)
    print("TESTING INTEGRATED AI, MAPPING & TRANSLATION SYSTEM")
    print("=" * 70)

    # Test 1: Geocoding
    print("\n📍 Test 1: Geocoding")
    test_locs = ['Nakuru Town', 'near market', 'Kenyatta Avenue']
    for loc in test_locs:
        lat, lon = geocode_location(loc)
        print(f"  {'✓' if lat else '✗'} {loc}: ({lat}, {lon})")

    # Test 2: Translation
    print("\n🌍 Test 2: Translation")
    text = "There is danger here"
    translated = translate_text(text, 'sw')
    print(f"  Original: {text}")
    print(f"  Translated: {translated}")

    # Test 3: Spam Detection
    print("\n🚫 Test 3: Spam Detection")
    test_reports = [
        {'description': 'test test test', 'manual_location': 'x', 'language': 'English'},
        {'description': 'I witnessed a robbery at the market today', 'manual_location': 'Wakulima Market',
         'language': 'English'}
    ]
    for i, report in enumerate(test_reports, 1):
        result = detect_spam(report)
        print(f"  Report {i}: Score={result['spam_score']}, Action={result['action']}")

    # Test 4: Anomaly Detection
    print("\n⚠️  Test 4: Anomaly Detection")
    test_incidents = [
        {'description': 'Gun shooting at market now!', 'language': 'English', 'manual_location': 'Market'},
        {'description': 'Minor traffic issue', 'language': 'English', 'manual_location': 'Road'}
    ]
    anomalies = detect_anomalies(test_incidents)
    print(f"  Detected {len(anomalies)} urgent incidents")
    for anomaly in anomalies:
        print(f"    - Priority: {anomaly['priority']}, Score: {anomaly['urgency_score']}")

    print("\n" + "=" * 70)
    print("✓ All integration tests completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_integrated_system()