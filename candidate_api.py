"""
Flask API for candidate scoring and retrieval
Provides endpoints to score candidates based on recruiter preferences and retrieve paginated results
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from typing import Dict, Optional
import os
import re

# Import scoring functions from score_candidates module
from score_candidates import process_candidates, calculate_composite_score, f_composite_score

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Default dataset path
DEFAULT_DATASET_PATH = 'candidates.csv'

# Simple city to coordinates mapping (can be extended or use geocoding service)
CITY_COORDINATES = {
    'new york': (40.7128, -74.0060),
    'san francisco': (37.7749, -122.4194),
    'los angeles': (34.0522, -118.2437),
    'chicago': (41.8781, -87.6298),
    'houston': (29.7604, -95.3698),
    'phoenix': (33.4484, -112.074),
    'philadelphia': (39.9526, -75.1652),
    'san antonio': (29.4241, -98.4936),
    'san diego': (32.7157, -117.1611),
    'dallas': (32.7767, -96.797),
    'austin': (30.2672, -97.7431),
    'jacksonville': (30.3322, -81.6557),
    'fort worth': (32.7555, -97.3308),
    'columbus': (39.9612, -82.9988),
    'charlotte': (35.2271, -80.8431),
    'san jose': (37.3382, -121.8863),
    'seattle': (47.6062, -122.3321),
    'denver': (39.7392, -104.9903),
    'washington': (38.9072, -77.0369),
    'boston': (42.3601, -71.0589),
    'el paso': (31.7619, -106.4850),
    'detroit': (42.3314, -83.0458),
    'nashville': (36.1627, -86.7816),
    'portland': (45.5152, -122.6784),
    'oklahoma city': (35.4676, -97.5164),
    'las vegas': (36.1699, -115.1398),
    'memphis': (35.1495, -90.0490),
    'louisville': (38.2527, -85.7585),
    'baltimore': (39.2904, -76.6122),
    'milwaukee': (43.0389, -87.9065),
    'atlanta': (33.749, -84.388),
    'miami': (25.7617, -80.1918),
}


def parse_location(location_str: str) -> tuple:
    """
    Parse location string to (lat, lon) tuple.
    Accepts formats:
    - "lat,lon" (e.g., "40.7128,-74.0060")
    - City name (e.g., "New York")
    - Empty string returns (None, None)
    """
    if not location_str or not isinstance(location_str, str) or location_str.strip() == '':
        return (None, None)
    
    location_str = location_str.strip()
    
    # Try to parse as "lat,lon"
    if ',' in location_str:
        try:
            parts = location_str.split(',')
            if len(parts) == 2:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                return (lat, lon)
        except (ValueError, IndexError):
            pass
    
    # Try to match city name (case-insensitive)
    city_key = location_str.lower()
    if city_key in CITY_COORDINATES:
        return CITY_COORDINATES[city_key]
    
    # If no match, return None
    return (None, None)


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights from 0-100 scale to 0-1 scale.
    Also maps user-friendly keys to internal keys.
    """
    # Map user-friendly keys to internal keys
    key_mapping = {
        'location': ['latitude', 'longitude'],
        'experience': 'experience_years',
        'skills': 'skills',
        'ctc': 'ctc_expectation_k',
        'role': 'role_applying',
        'degree': 'degree',  # Not used in current dataset, but handle gracefully
        'github': 'github_url',
    }
    
    normalized = {}
    total_weight = 0.0
    
    # First pass: collect all weights and calculate total
    for key, value in weights.items():
        if value is None or value == '':
            continue
        try:
            weight_value = float(value)
            if weight_value < 0:
                weight_value = 0.0
            total_weight += weight_value
        except (ValueError, TypeError):
            continue
    
    # If total is 0, return empty dict
    if total_weight == 0:
        return {}
    
    # Second pass: normalize and map keys
    for key, value in weights.items():
        if value is None or value == '':
            continue
        try:
            weight_value = float(value)
            if weight_value < 0:
                continue
            
            # Normalize to 0-1 scale
            normalized_weight = weight_value / total_weight if total_weight > 0 else 0.0
            
            # Map to internal keys
            if key in key_mapping:
                mapped_keys = key_mapping[key]
                if isinstance(mapped_keys, list):
                    # Split weight equally for location (latitude + longitude)
                    for mapped_key in mapped_keys:
                        normalized[mapped_key] = normalized_weight / len(mapped_keys)
                else:
                    normalized[mapped_keys] = normalized_weight
        except (ValueError, TypeError):
            continue
    
    return normalized


def paginate_dataframe(df: pd.DataFrame, page: int = 1, per_page: int = 20) -> Dict:
    """
    Paginate a DataFrame and return paginated results.
    
    Args:
        df: DataFrame to paginate
        page: Page number (1-indexed)
        per_page: Number of items per page
    
    Returns:
        Dictionary with paginated data and metadata
    """
    total_items = len(df)
    total_pages = (total_items + per_page - 1) // per_page if total_items > 0 else 0  # Ceiling division
    
    # Validate page number
    if page < 1:
        page = 1
    if total_pages > 0 and page > total_pages:
        page = total_pages
    
    # Calculate start and end indices
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    # Get paginated data
    paginated_df = df.iloc[start_idx:end_idx] if total_items > 0 else df
    
    # Convert to dictionary records
    records = paginated_df.to_dict(orient='records')
    
    return {
        'data': records,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_items': total_items,
            'total_pages': total_pages,
            'has_next': page < total_pages if total_pages > 0 else False,
            'has_prev': page > 1
        }
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Candidate API is running'}), 200


@app.route('/api/candidates/score', methods=['POST'])
def score_candidates():
    """
    Score candidates based on recruiter preferences and weights.
    
    Expected JSON body:
    {
        "location": "40.7128,-74.0060" or "New York",
        "ctc_range": { "min": "30", "max": "100" },
        "experience": { "min": "3", "max": "7" },
        "degree": "",  // Optional, not used in current dataset
        "role": "technology",
        "skills": ["Python", "SQL", "AWS"],
        "weights": {
            "location": 50,
            "experience": 50,
            "degree": 50,
            "role": 50,
            "skills": 50
        },
        "page": 1,  // Optional, default 1
        "per_page": 20  // Optional, default 20
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract and parse location
        location_str = data.get('location', '')
        target_lat, target_lon = parse_location(location_str)
        
        # Extract CTC range
        ctc_range = data.get('ctc_range', {})
        budget_min_k = ctc_range.get('min', '')
        budget_max_k = ctc_range.get('max', '')
        try:
            budget_min_k = float(budget_min_k) if budget_min_k else 0.0
            budget_max_k = float(budget_max_k) if budget_max_k else 1000.0
        except (ValueError, TypeError):
            budget_min_k = 0.0
            budget_max_k = 1000.0
        
        # Extract experience range
        experience = data.get('experience', {})
        min_exp = experience.get('min', '')
        max_exp = experience.get('max', '')
        try:
            min_exp = float(min_exp) if min_exp else 0.0
            max_exp = float(max_exp) if max_exp else 20.0
        except (ValueError, TypeError):
            min_exp = 0.0
            max_exp = 20.0
        
        # Extract role
        target_role = data.get('role', '')
        if not isinstance(target_role, str):
            target_role = str(target_role) if target_role else ''
        
        # Extract skills
        required_skills = data.get('skills', [])
        if not isinstance(required_skills, list):
            required_skills = []
        
        # Extract weights
        weights_raw = data.get('weights', {})
        if not weights_raw:
            return jsonify({'error': 'weights parameter is required'}), 400
        
        # Normalize weights from 0-100 to 0-1 scale
        weights = normalize_weights(weights_raw)
        
        if not weights:
            return jsonify({'error': 'No valid weights provided'}), 400
        
        # Build recruiter preferences
        recruiter_preferences = {
            'target_lat': target_lat,
            'target_lon': target_lon,
            'max_distance_km': 1000,  # Default max distance
            'min_exp': min_exp,
            'max_exp': max_exp,
            'required_skills': required_skills,
            'budget_min_k': budget_min_k,
            'budget_max_k': budget_max_k,
            'target_role': target_role
        }
        
        # Get pagination parameters
        page = data.get('page', 1)
        per_page = data.get('per_page', 20)
        
        # Validate pagination
        try:
            page = int(page)
            per_page = int(per_page)
            if page < 1:
                page = 1
            if per_page < 1:
                per_page = 20
        except (ValueError, TypeError):
            page = 1
            per_page = 20
        
        # Validate dataset exists
        dataset_path = data.get('dataset_path', DEFAULT_DATASET_PATH)
        if not os.path.exists(dataset_path):
            return jsonify({'error': f'Dataset file not found: {dataset_path}'}), 404
        
        # Process candidates
        results_df = process_candidates(
            dataset_path=dataset_path,
            weights=weights,
            recruiter_preferences=recruiter_preferences,
            output_path=None  # Don't save to file, just return results
        )
        
        # Paginate results
        paginated_results = paginate_dataframe(results_df, page=page, per_page=per_page)
        
        return jsonify({
            'success': True,
            'data': paginated_results['data'],
            'pagination': paginated_results['pagination']
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500


@app.route('/api/candidates', methods=['GET'])
def get_candidates():
    """
    Get candidates with optional pagination (without scoring).
    
    Query parameters:
    - page: Page number (default: 1)
    - per_page: Items per page (default: 20)
    - dataset_path: Path to dataset (default: candidates.csv)
    """
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        dataset_path = request.args.get('dataset_path', DEFAULT_DATASET_PATH)
        
        # Validate dataset exists
        if not os.path.exists(dataset_path):
            return jsonify({'error': f'Dataset file not found: {dataset_path}'}), 404
        
        # Read dataset
        df = pd.read_csv(dataset_path)
        
        # Paginate results
        paginated_results = paginate_dataframe(df, page=page, per_page=per_page)
        
        return jsonify({
            'success': True,
            'data': paginated_results['data'],
            'pagination': paginated_results['pagination']
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
