#!/usr/bin/env python3
"""
scored_candidates.py

Candidate Scoring System
Processes candidate dataset and calculates weighted composite scores based on recruiter preferences.
Implements subscores for location, experience, skills, ctc, role, and github presence.
"""

import pandas as pd
import numpy as np
import math
import re
from typing import Dict, Optional, List, Tuple


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth using the haversine formula.
    Inputs are in decimal degrees. Output is in kilometers.
    Returns NaN if any input is None/NaN.
    """
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (lat1, lon1, lat2, lon2)):
        return float('nan')

    # convert degrees to radians
    φ1, λ1, φ2, λ2 = map(math.radians, (lat1, lon1, lat2, lon2))

    # haversine formula
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    c = 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))  # numeric safety

    R = 6371.0  # mean Earth radius in kilometers
    return R * c


def subscore_location(candidate_lat: float, candidate_lon: float,
                      target_lat: float, target_lon: float,
                      max_distance_km: float = 1000) -> float:
    """
    Location subscore: returns value in [0,1]
    - 1.0 at target location (distance 0)
    - ~0.0 at or beyond max_distance_km
    - 0.5 returned for missing coordinates (neutral)
    Uses a smooth inverse-square decay scaled so that score ~0.05 at max_distance_km.
    """
    # Handle missing coords
    if candidate_lat is None or candidate_lon is None or pd.isna(candidate_lat) or pd.isna(candidate_lon):
        return 0.5
    if target_lat is None or target_lon is None or pd.isna(target_lat) or pd.isna(target_lon):
        return 0.5

    dist = calculate_distance(candidate_lat, candidate_lon, target_lat, target_lon)
    if math.isnan(dist):
        return 0.5

    # If max_distance_km is zero or negative, fallback to a binary match
    if max_distance_km <= 0:
        return 1.0 if dist == 0 else 0.0

    # scale parameter so that score decays smoothly; using inverse-square with small epsilon
    scale = max_distance_km / 6.0  # places most decay near max_distance_km
    score = 1.0 / (1.0 + (dist / (scale + 1e-9)) ** 2)

    # If distance >= max_distance_km, score should be near 0 (clamp)
    if dist >= max_distance_km:
        score = max(0.0, score * 0.05)  # push far-away to near-zero

    return float(max(0.0, min(1.0, score)))


def subscore_experience(experience_years: float, min_exp: float, max_exp: float) -> float:
    """
    Experience subscore in [0,1].
    - If experience inside [min_exp, max_exp] => closer to 1.0 (peak at mid-range).
    - Below min_exp => score decreases smoothly toward 0.
    - Above max_exp => small penalty (but not zero) to prefer range fits.
    - Missing experience returns 0.5 (neutral).
    Implementation: triangular / piecewise with smooth tails using logistic-ish mapping.
    """
    if experience_years is None or pd.isna(experience_years):
        return 0.5

    try:
        exp = float(experience_years)
    except Exception:
        return 0.5

    # Ensure sensible bounds
    if min_exp is None:
        min_exp = 0.0
    if max_exp is None or max_exp <= min_exp:
        max_exp = min_exp + 5.0

    # Preferred center is midpoint
    center = (min_exp + max_exp) / 2.0
    half_range = max(0.5, (max_exp - min_exp) / 2.0)

    # If within preferred window, use a smooth bell-ish peak
    if min_exp <= exp <= max_exp:
        # use a quadratic peak where center -> 1.0, edges -> 0.6
        norm = 1.0 - ((exp - center) / half_range) ** 2
        score = 0.6 + 0.4 * max(0.0, norm)  # range [0.6,1.0]
    elif exp < min_exp:
        # below range: scale from 0 -> 0.6 as exp goes from 0 to min_exp
        if min_exp <= 0.0:
            score = 0.0
        else:
            frac = max(0.0, exp / min_exp)
            score = 0.2 + 0.4 * frac  # range [0.2,0.6)
    else:  # exp > max_exp
        # slight penalty for overqualified but not too severe
        # scale down from 0.6 at max_exp to 0.3 at +3*range, then floor 0.2
        over = exp - max_exp
        scale = max(1.0, (max_exp - min_exp) if (max_exp - min_exp) > 0 else 1.0)
        frac = max(0.0, 1.0 - over / (3.0 * scale))
        score = 0.3 + 0.3 * frac  # range [0.3,0.6]
    return float(max(0.0, min(1.0, score)))

def process_candidates

def _split_skills(skills_field: Optional[str]) -> List[str]:
    """
    Normalize skills string/iterable into a cleaned list of lowercase tokens.
    Accepts comma/semicolon/pipe separated strings or lists.
    """
    if skills_field is None or (isinstance(skills_field, float) and math.isnan(skills_field)):
        return []
    if isinstance(skills_field, (list, tuple, set)):
        tokens = [str(s).strip().lower() for s in skills_field if str(s).strip()]
        return tokens
    s = str(skills_field)
    # common separators
    parts = re.split(r'[;,|/]+|\s{2,}', s)
    tokens = []
    for p in parts:
        for t in re.split(r'\s+', p.strip()):
            t = t.strip().lower()
            if t:
                tokens.append(t)
    return tokens


def subscore_skills(candidate_skills: str, required_skills: List[str]) -> float:
    """
    Skills subscore in [0,1].
    - Exact matches (case-insensitive token matches) earn full points.
    - Partial matches proportionate to count of required skills found.
    - If required_skills is empty => neutral 0.5.
    - Missing candidate_skills => 0.0
    """
    req = [r.strip().lower() for r in (required_skills or []) if str(r).strip()]
    if not req:
        return 0.5

    cand_tokens = _split_skills(candidate_skills)
    if not cand_tokens:
        return 0.0

    # Match by exact token or substring (to allow "python3" match "python")
    matches = 0
    for r in req:
        for t in cand_tokens:
            if r == t or r in t or t in r:
                matches += 1
                break

    frac = matches / len(req)
    # Map fraction to score with some non-linearity favoring full matches
    score = 0.2 + 0.8 * frac  # range [0.2,1.0]; if none => 0.2
    if matches == 0:
        score = 0.0
    return float(max(0.0, min(1.0, score)))


def subscore_ctc(ctc_expectation_k: float, budget_min_k: float, budget_max_k: float) -> float:
    """
    CTC (expected compensation) subscore in [0,1].
    - If expectation within budget range => high score (1.0 at mid-range)
    - If expectation below budget_min => good but capped (candidate cheaper than budget)
    - If expectation above budget_max => penalty decreasing to 0.
    - Missing expectation => 0.5
    Assumes inputs are in same units (k).
    """
    if ctc_expectation_k is None or pd.isna(ctc_expectation_k):
        return 0.5

    try:
        exp = float(ctc_expectation_k)
    except Exception:
        return 0.5

    # Ensure sensible budget
    if budget_min_k is None:
        budget_min_k = 0.0
    if budget_max_k is None or budget_max_k < budget_min_k:
        budget_max_k = budget_min_k + 50.0

    # If expectation within range
    if budget_min_k <= exp <= budget_max_k:
        mid = (budget_min_k + budget_max_k) / 2.0
        # closeness to mid gives better score
        dist = abs(exp - mid)
        half_range = max(1.0, (budget_max_k - budget_min_k) / 2.0)
        score = 0.7 + 0.3 * max(0.0, 1.0 - (dist / half_range))  # [0.7,1.0]
    elif exp < budget_min_k:
        # cheaper than min budget: good (cost-saving) but not perfect
        # more discount -> slightly lower score (we don't want suspiciously low)
        frac = max(0.0, exp / budget_min_k) if budget_min_k > 0 else 0.0
        score = 0.6 + 0.3 * frac  # [0.6,0.9)
    else:  # exp > budget_max_k
        # penalize proportionally; expectation far above budget approaches 0
        over = exp - budget_max_k
        span = max(1.0, budget_max_k - budget_min_k)
        frac = 1.0 / (1.0 + (over / span))
        score = 0.0 + 0.6 * frac  # up to 0.6 near the budget, down to 0 as over grows

    return float(max(0.0, min(1.0, score)))


def subscore_role(candidate_role: str, target_role: str) -> float:
    """
    Role subscore in [0,1].
    - Exact or strong substring match -> high score.
    - Related roles (e.g., 'backend' vs 'backend engineer') get partial credit.
    - Missing role -> 0.5
    """
    if candidate_role is None or pd.isna(candidate_role):
        return 0.5
    if target_role is None or pd.isna(target_role) or str(target_role).strip() == '':
        return 0.5

    cand = str(candidate_role).strip().lower()
    targ = str(target_role).strip().lower()

    # exact match
    if cand == targ:
        return 1.0

    # token overlap
    cand_tokens = set(re.split(r'[\s\-_/]+', cand))
    targ_tokens = set(re.split(r'[\s\-_/]+', targ))
    common = cand_tokens.intersection(targ_tokens)
    if common:
        frac = len(common) / len(targ_tokens)
        return float(max(0.0, min(1.0, 0.5 + 0.5 * frac)))  # gives 0.5-1.0 based on overlap

    # substring checks
    if targ in cand or cand in targ:
        return 0.8

    return 0.2


def subscore_github(github_url: str) -> float:
    """
    GitHub subscore: presence and basic heuristics.
    - No URL -> 0.0
    - Valid-looking URL -> 0.6
    - If repo/user looks active (heuristic: username/repo present) -> 1.0
    This function keeps it lightweight (no network calls).
    """
    if github_url is None or pd.isna(github_url) or str(github_url).strip() == '':
        return 0.0

    url = str(github_url).strip()
    # Normalize and check patterns
    # Accept formats like: github.com/username, https://github.com/username/repo
    m = re.search(r'github\.com/([A-Za-z0-9_\-\.]+)(/([A-Za-z0-9_\-\.]+))?', url, flags=re.IGNORECASE)
    if not m:
        return 0.4  # present but not a clear github profile/repo

    username = m.group(1)
    repo = m.group(3)
    # heuristics:
    if username and repo:
        return 1.0
    if username:
        # if username looks plausible length -> give good score
        return 0.8 if len(username) >= 3 else 0.6

    return 0.6


def calculate_composite_score(candidate: pd.Series, weights: Dict[str, float],
                              recruiter_preferences: Dict) -> Tuple[float, Dict[str, float]]:
    """
    Calculate weighted composite score for a candidate.

    Returns:
        (composite_score [0-1], subscores dict)
    """
    subscores = {}

    # Location subscore (using latitude/longitude)
    if 'latitude' in weights or 'longitude' in weights:
        loc_weight = weights.get('latitude', 0) + weights.get('longitude', 0)
        if loc_weight > 0:
            subscores['location'] = subscore_location(
                candidate.get('latitude'),
                candidate.get('longitude'),
                recruiter_preferences.get('target_lat'),
                recruiter_preferences.get('target_lon'),
                recruiter_preferences.get('max_distance_km', 1000)
            )

    # Experience subscore
    if 'experience_years' in weights and weights['experience_years'] > 0:
        subscores['experience_years'] = subscore_experience(
            candidate.get('experience_years'),
            recruiter_preferences.get('min_exp', 0),
            recruiter_preferences.get('max_exp', 20)
        )

    # Skills subscore
    if 'skills' in weights and weights['skills'] > 0:
        subscores['skills'] = subscore_skills(
            candidate.get('skills'),
            recruiter_preferences.get('required_skills', [])
        )

    # CTC subscore
    if 'ctc_expectation_k' in weights and weights['ctc_expectation_k'] > 0:
        subscores['ctc_expectation_k'] = subscore_ctc(
            candidate.get('ctc_expectation_k'),
            recruiter_preferences.get('budget_min_k', 0),
            recruiter_preferences.get('budget_max_k', 1000)
        )

    # Role subscore
    if 'role_applying' in weights and weights['role_applying'] > 0:
        subscores['role_applying'] = subscore_role(
            candidate.get('role_applying'),
            recruiter_preferences.get('target_role', '')
        )

    # GitHub subscore
    if 'github_url' in weights and weights['github_url'] > 0:
        subscores['github_url'] = subscore_github(
            candidate.get('github_url')
        )

    # Calculate weighted composite score starting from base 0.5
    base_score = 0.5
    composite_score = base_score
    total_weight = 0.0

    # Handle location weight (latitude + longitude combined)
    if 'latitude' in weights or 'longitude' in weights:
        loc_weight = weights.get('latitude', 0) + weights.get('longitude', 0)
        if loc_weight > 0 and 'location' in subscores:
            subscore = subscores['location']
            adjustment = (subscore - base_score) * loc_weight
            composite_score += adjustment
            total_weight += loc_weight

    # Handle other parameters
    for param in ['experience_years', 'skills', 'ctc_expectation_k', 'role_applying', 'github_url']:
        if param in weights and weights[param] > 0:
            if param in subscores:
                subscore = subscores[param]
                param_weight = weights[param]
                adjustment = (subscore - base_score) * param_weight
                composite_score += adjustment
                total_weight += param_weight

    # Normalize composite score to 0-1 based on possible range
    if total_weight > 0:
        max_adjustment = 0.5 * total_weight
        min_possible = base_score - max_adjustment
        max_possible = base_score + max_adjustment
        if max_possible > min_possible:
            composite_score = (composite_score - min_possible) / (max_possible - min_possible)
        else:
            composite_score = base_score
    else:
        composite_score = base_score

    composite_score = float(max(0.0, min(1.0, composite_score)))
    return composite_score, subscores


def f_composite_score(composite_score: float, base_score: float = 0.5) -> float:
    """
    Calculate f_composite_score using sigmoid function scaled 0-100.
    """
    deviation = (composite_score - base_score) * 100
    scale_factor = 20.0
    # numeric safety for large magnitudes
    x = deviation / scale_factor
    try:
        sigmoid_output = 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        sigmoid_output = 0.0 if x < 0 else 1.0
    return float(sigmoid_output * 100.0)


def process_candidates(dataset_path: str, weights: Dict[str, float],
                       recruiter_preferences: Dict, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Process candidate dataset and calculate scores.

    Args:
        dataset_path: Path to candidate dataset CSV file
        weights: Dictionary of parameter weights
        recruiter_preferences: Dictionary with recruiter preferences
        output_path: Optional path to save results CSV

    Returns:
        DataFrame with candidates and their scores
    """
    # Read dataset with permissive parsing
    df = pd.read_csv(dataset_path, dtype=str).replace({np.nan: None})

    # Try to coerce numeric columns commonly used
    numeric_cols = ['latitude', 'longitude', 'experience_years', 'ctc_expectation_k']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure skills and github_url and role_applying exist
    for col in ['skills', 'github_url', 'role_applying', 'name', 'id']:
        if col not in df.columns:
            df[col] = None

    base_score = 0.5
    results = []
    for _, candidate in df.iterrows():
        composite_score, subscores = calculate_composite_score(candidate, weights, recruiter_preferences)
        f_comp_score = f_composite_score(composite_score, base_score)

        result = candidate.to_dict()
        result['composite_score'] = composite_score
        result['f_composite_score'] = f_comp_score
        # attach subscores prefixed
        for k, v in subscores.items():
            result[f'subscore_{k}'] = v
        results.append(result)

    results_df = pd.DataFrame(results)

    # Fill missing id/name columns if absent
    if 'id' not in results_df.columns:
        results_df['id'] = results_df.index + 1
    if 'name' not in results_df.columns:
        results_df['name'] = results_df['id'].astype(str)

    # Sort by composite score descending, then f_comp_score
    results_df = results_df.sort_values(['composite_score', 'f_composite_score'], ascending=[False, False]).reset_index(drop=True)

    if output_path:
        results_df.to_csv(output_path, index=False)

    return results_df


if __name__ == "__main__":
    # Example usage: adjust paths as needed
    weights = {
        'latitude': 0.1,
        'longitude': 0.1,
        'experience_years': 0.2,
        'skills': 0.3,
        'ctc_expectation_k': 0.15,
        'role_applying': 0.1,
        'github_url': 0.05
    }

    recruiter_preferences = {
        'target_lat': 40.7128,
        'target_lon': -74.0060,
        'max_distance_km': 1000,
        'min_exp': 3,
        'max_exp': 7,
        'required_skills': ['Python', 'SQL', 'AWS'],
        'budget_max_k': 100,
        'budget_min_k': 30,
        'target_role': 'technology'
    }

    import argparse
    parser = argparse.ArgumentParser(description="Score candidates from CSV")
    parser.add_argument('input_csv', nargs='?', default='candidates.csv', help='Input candidates CSV')
    parser.add_argument('--output', '-o', default='scored_candidates.csv', help='Output CSV path')
    args = parser.parse_args()

    out_df = process_candidates(args.input_csv, weights, recruiter_preferences, args.output)
    # print top 10
    cols_to_show = [c for c in ['id', 'name', 'role_applying', 'composite_score', 'f_composite_score'] if c in out_df.columns]
    print(out_df[cols_to_show].head(10).to_string(index=False))
