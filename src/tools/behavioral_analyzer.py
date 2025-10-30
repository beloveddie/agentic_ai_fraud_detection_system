# tools/behavioral_analyzer.py
"""
Behavioral Pattern Analysis Tool
- Analyzes customer behavioral patterns for fraud detection
- Provides comprehensive behavioral insights and risk indicators
- Supports code-as-action paradigm with rich structured data
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from collections import Counter
import math

from typing_extensions import TypedDict
from agents import function_tool

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# TypedDict definitions
class SpendingPatterns(TypedDict):
    """Spending behavior analysis"""
    avg_amount_7d: float
    avg_amount_30d: float
    avg_amount_90d: float
    spending_acceleration_7d: float
    spending_acceleration_30d: float
    max_single_transaction: float
    spending_consistency_score: float
    unusual_spending_spike: bool
    spending_trend: str

class TemporalPatterns(TypedDict):
    """Time-based behavior analysis"""
    preferred_hours: List[int]
    preferred_days: List[int]
    time_consistency_score: float
    unusual_time_transaction: bool
    night_transaction_ratio: float
    weekend_transaction_ratio: float
    time_deviation_score: float

class GeographicPatterns(TypedDict):
    """Location-based behavior analysis"""
    primary_locations: List[str]
    location_diversity_score: float
    location_consistency_score: float
    unusual_location: bool
    location_change_rate: float
    international_transaction_ratio: float
    impossible_travel_detected: bool
    travel_distance_km: Optional[float]

class DeviceChannelPatterns(TypedDict):
    """Device and channel usage patterns"""
    preferred_devices: List[str]
    preferred_channels: List[str]
    device_consistency_score: float
    channel_consistency_score: float
    device_change_rate: float
    channel_change_rate: float
    new_device_detected: bool
    unusual_channel_usage: bool

class TransactionTypePatterns(TypedDict):
    """Transaction type and merchant patterns"""
    preferred_transaction_types: List[str]
    preferred_merchant_categories: List[str]
    transaction_type_diversity: float
    merchant_category_diversity: float
    unusual_transaction_type: bool
    unusual_merchant_category: bool
    merchant_loyalty_score: float

class FrequencyPatterns(TypedDict):
    """Transaction frequency analysis"""
    avg_transactions_per_day: float
    avg_transactions_per_week: float
    transaction_frequency_trend: str
    frequency_acceleration: float
    unusual_frequency_spike: bool
    frequency_consistency_score: float
    burst_activity_detected: bool

class RiskIndicators(TypedDict):
    """Behavioral risk indicators"""
    behavioral_risk_score: float
    risk_level: str
    top_risk_factors: List[Tuple[str, float]]
    anomaly_count: int
    pattern_deviation_score: float
    account_age_factor: float
    behavioral_confidence: float

class BehavioralInsights(TypedDict):
    """High-level behavioral insights"""
    customer_profile: str
    behavioral_stability: str
    fraud_likelihood: str
    recommended_actions: List[str]
    behavioral_summary: str

class BehavioralAnalysis(TypedDict):
    """Complete behavioral analysis result"""
    transaction_id: str
    customer_id: str
    analysis_timestamp: datetime
    analysis_window_days: int
    total_transactions_analyzed: int
    
    spending_patterns: Dict[str, Any]
    temporal_patterns: Dict[str, Any]
    geographic_patterns: Dict[str, Any]
    device_channel_patterns: Dict[str, Any]
    transaction_type_patterns: Dict[str, Any]
    frequency_patterns: Dict[str, Any]
    
    risk_indicators: Dict[str, Any]
    behavioral_insights: Dict[str, Any]
    
    analysis_quality: str
    data_sufficiency_score: float
    analysis_confidence: float
    
    raw_patterns: Dict[str, Any]
    pattern_metadata: Dict[str, Any]

class BehavioralAnalyzer:
    def __init__(self):
        """Initialize the behavioral analyzer"""
        self.analysis_cache = {}
        
    def analyze_behavioral_patterns(
        self, 
        transaction: Dict[str, Any], 
        customer_history: List[Dict[str, Any]],
        analysis_window_days: int = 30
    ) -> BehavioralAnalysis:
        """
        Analyze customer behavioral patterns for fraud detection.
        
        Args:
            transaction: Current transaction to analyze
            customer_history: Historical transactions for this customer
            analysis_window_days: Days to look back for pattern analysis
            
        Returns:
            Dict[str, Any] with comprehensive behavioral insights
        """
        try:
            # Filter history to analysis window
            cutoff_date = pd.to_datetime(transaction.get('time', datetime.now())) - timedelta(days=analysis_window_days)
            filtered_history = [
                tx for tx in customer_history 
                if pd.to_datetime(tx.get('time', datetime.now())) >= cutoff_date
            ]
            
            # Calculate data sufficiency
            data_sufficiency = self._calculate_data_sufficiency(filtered_history)
            
            # Analyze each behavioral domain
            spending_patterns = self._analyze_spending_patterns(transaction, filtered_history)
            temporal_patterns = self._analyze_temporal_patterns(transaction, filtered_history)
            geographic_patterns = self._analyze_geographic_patterns(transaction, filtered_history)
            device_channel_patterns = self._analyze_device_channel_patterns(transaction, filtered_history)
            transaction_type_patterns = self._analyze_transaction_type_patterns(transaction, filtered_history)
            frequency_patterns = self._analyze_frequency_patterns(transaction, filtered_history)
            
            # Generate risk indicators
            risk_indicators = self._generate_risk_indicators(
                transaction, filtered_history,
                spending_patterns, temporal_patterns, geographic_patterns,
                device_channel_patterns, transaction_type_patterns, frequency_patterns
            )
            
            # Generate behavioral insights
            behavioral_insights = self._generate_behavioral_insights(
                risk_indicators, spending_patterns, temporal_patterns, geographic_patterns,
                device_channel_patterns
            )
            
            # Calculate analysis confidence
            analysis_confidence = self._calculate_analysis_confidence(data_sufficiency, len(filtered_history))
            
            # Determine analysis quality
            analysis_quality = self._determine_analysis_quality(data_sufficiency, analysis_confidence)
            
            # Compile raw patterns for code-as-action
            raw_patterns = self._compile_raw_patterns(
                spending_patterns, temporal_patterns, geographic_patterns,
                device_channel_patterns, transaction_type_patterns, frequency_patterns
            )
            
            return {
                "transaction_id": transaction.get('transaction_id', 'unknown'),
                "customer_id": transaction.get('customer_id', 'unknown'),
                "analysis_timestamp": datetime.now(),
                "analysis_window_days": analysis_window_days,
                "total_transactions_analyzed": len(filtered_history),
                
                "spending_patterns": spending_patterns,
                "temporal_patterns": temporal_patterns,
                "geographic_patterns": geographic_patterns,
                "device_channel_patterns": device_channel_patterns,
                "transaction_type_patterns": transaction_type_patterns,
                "frequency_patterns": frequency_patterns,
                
                "risk_indicators": risk_indicators,
                "behavioral_insights": behavioral_insights,
                
                "analysis_quality": analysis_quality,
                "data_sufficiency_score": data_sufficiency,
                "analysis_confidence": analysis_confidence,
                
                "raw_patterns": raw_patterns,
                "pattern_metadata": {
                    'analysis_version': '1.0',
                    'algorithm_used': 'behavioral_pattern_analyzer',
                    'cache_key': f"{transaction.get('customer_id', 'unknown')}_{analysis_window_days}"
                }
            }
            
        except Exception as e:
            # Return minimal analysis on error
            return self._create_fallback_analysis(transaction, str(e))
    
    def _analyze_spending_patterns(self, transaction: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze spending behavior patterns"""
        amounts = [float(tx.get('amount', 0)) for tx in history if tx.get('amount') is not None]
        current_amount = float(transaction.get('amount', 0))
        
        if not amounts:
            return {
                "avg_amount_7d": 0.0, "avg_amount_30d": 0.0, "avg_amount_90d": 0.0,
                "spending_acceleration_7d": 1.0, "spending_acceleration_30d": 1.0,
                "max_single_transaction": current_amount, "spending_consistency_score": 0.5,
                "unusual_spending_spike": False, "spending_trend": "unknown"
            }
        
        # Calculate averages for different time windows
        now = pd.to_datetime(transaction.get('time', datetime.now()))
        amounts_7d = [float(tx.get('amount', 0)) for tx in history 
                      if (now - pd.to_datetime(tx.get('time', now))).days <= 7]
        amounts_30d = amounts  # Already filtered to 30 days
        
        avg_7d = np.mean(amounts_7d) if amounts_7d else 0.0
        avg_30d = np.mean(amounts_30d)
        avg_90d = avg_30d  # Approximate for now
        
        # Calculate accelerations
        accel_7d = avg_7d / avg_30d if avg_30d > 0 else 1.0
        accel_30d = avg_30d / avg_90d if avg_90d > 0 else 1.0
        
        # Calculate consistency score (inverse of coefficient of variation)
        if len(amounts) > 1:
            cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 1.0
            consistency_score = max(0, 1 - cv)
        else:
            consistency_score = 0.5
        
        # Detect unusual spending spike
        unusual_spike = current_amount > (avg_30d * 3) if avg_30d > 0 else False
        
        # Determine spending trend
        if len(amounts) >= 3:
            recent_avg = np.mean(amounts[-3:])
            older_avg = np.mean(amounts[:-3]) if len(amounts) > 3 else recent_avg
            if recent_avg > older_avg * 1.2:
                trend = "increasing"
            elif recent_avg < older_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        return {
            "avg_amount_7d": round(avg_7d, 2),
            "avg_amount_30d": round(avg_30d, 2),
            "avg_amount_90d": round(avg_90d, 2),
            "spending_acceleration_7d": round(accel_7d, 3),
            "spending_acceleration_30d": round(accel_30d, 3),
            "max_single_transaction": round(max(amounts + [current_amount]), 2),
            "spending_consistency_score": round(consistency_score, 3),
            "unusual_spending_spike": unusual_spike,
            "spending_trend": trend
        }
    
    def _analyze_temporal_patterns(self, transaction: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal behavior patterns"""
        if not history:
            current_time = pd.to_datetime(transaction.get('time', datetime.now()))
            return Dict[str, Any](
                preferred_hours=[current_time.hour], preferred_days=[current_time.weekday()],
                time_consistency_score=0.5, unusual_time_transaction=False,
                night_transaction_ratio=0.0, weekend_transaction_ratio=0.0,
                time_deviation_score=0.0
            )
        
        # Extract time patterns
        hours = [pd.to_datetime(tx.get('time', datetime.now())).hour for tx in history]
        days = [pd.to_datetime(tx.get('time', datetime.now())).weekday() for tx in history]
        
        # Find preferred hours and days
        hour_counts = Counter(hours)
        day_counts = Counter(days)
        preferred_hours = [h for h, count in hour_counts.most_common(3)]
        preferred_days = [d for d, count in day_counts.most_common(3)]
        
        # Calculate time consistency score
        hour_entropy = -sum((count/len(hours)) * math.log(count/len(hours)) for count in hour_counts.values() if count > 0)
        max_hour_entropy = math.log(24)
        time_consistency_score = 1 - (hour_entropy / max_hour_entropy) if max_hour_entropy > 0 else 0.5
        
        # Analyze current transaction timing
        current_time = pd.to_datetime(transaction.get('time', datetime.now()))
        current_hour = current_time.hour
        current_day = current_time.weekday()
        
        # Check if current time is unusual
        unusual_time = current_hour not in preferred_hours[:2] if len(preferred_hours) >= 2 else False
        
        # Calculate night and weekend ratios
        night_transactions = sum(1 for h in hours if h < 6 or h > 22)
        weekend_transactions = sum(1 for d in days if d >= 5)  # Saturday=5, Sunday=6
        
        night_ratio = night_transactions / len(hours) if hours else 0.0
        weekend_ratio = weekend_transactions / len(days) if days else 0.0
        
        # Calculate time deviation score
        if preferred_hours:
            avg_preferred_hour = np.mean(preferred_hours)
            time_deviation = abs(current_hour - avg_preferred_hour)
            time_deviation_score = min(time_deviation / 12, 1.0)  # Normalize to 0-1
        else:
            time_deviation_score = 0.0
        
        return Dict[str, Any](
            preferred_hours=preferred_hours,
            preferred_days=preferred_days,
            time_consistency_score=round(time_consistency_score, 3),
            unusual_time_transaction=unusual_time,
            night_transaction_ratio=round(night_ratio, 3),
            weekend_transaction_ratio=round(weekend_ratio, 3),
            time_deviation_score=round(time_deviation_score, 3)
        )
    
    def _analyze_geographic_patterns(self, transaction: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze geographic behavior patterns"""
        if not history:
            return Dict[str, Any](
                primary_locations=[transaction.get('location_city', 'unknown')],
                location_diversity_score=0.0, location_consistency_score=0.5,
                unusual_location=False, location_change_rate=0.0,
                international_transaction_ratio=0.0, impossible_travel_detected=False,
                travel_distance_km=None
            )
        
        # Extract location data
        cities = [tx.get('location_city', 'unknown') for tx in history]
        countries = [tx.get('location_country', 'unknown') for tx in history]
        
        # Find primary locations
        city_counts = Counter(cities)
        primary_locations = [city for city, count in city_counts.most_common(3)]
        
        # Calculate location diversity score
        unique_locations = len(set(cities))
        location_diversity_score = min(unique_locations / len(cities), 1.0) if cities else 0.0
        
        # Calculate location consistency score
        if len(cities) > 1:
            most_common_city_count = city_counts.most_common(1)[0][1]
            location_consistency_score = most_common_city_count / len(cities)
        else:
            location_consistency_score = 1.0
        
        # Check if current location is unusual
        current_city = transaction.get('location_city', 'unknown')
        unusual_location = current_city not in primary_locations[:2] if len(primary_locations) >= 2 else False
        
        # Calculate location change rate
        location_changes = sum(1 for i in range(1, len(cities)) if cities[i] != cities[i-1])
        location_change_rate = location_changes / len(cities) if len(cities) > 1 else 0.0
        
        # Calculate international transaction ratio
        current_country = transaction.get('location_country', 'unknown')
        international_count = sum(1 for country in countries if country != current_country)
        international_ratio = international_count / len(countries) if countries else 0.0
        
        # Detect impossible travel (simplified)
        impossible_travel = False
        travel_distance = None
        
        if len(history) >= 1:
            last_transaction = history[-1]
            last_city = last_transaction.get('location_city', 'unknown')
            last_time = pd.to_datetime(last_transaction.get('time', datetime.now()))
            current_time = pd.to_datetime(transaction.get('time', datetime.now()))
            
            if last_city != current_city:
                # Simple distance calculation (in practice, use proper geocoding)
                time_diff_hours = (current_time - last_time).total_seconds() / 3600
                # Assume max reasonable travel speed of 800 km/h (airplane)
                max_distance = time_diff_hours * 800
                
                # Simplified distance calculation (would use proper geocoding in production)
                if last_city != current_city:
                    travel_distance = 100.0  # Placeholder distance
                    impossible_travel = travel_distance > max_distance
        
        return Dict[str, Any](
            primary_locations=primary_locations,
            location_diversity_score=round(location_diversity_score, 3),
            location_consistency_score=round(location_consistency_score, 3),
            unusual_location=unusual_location,
            location_change_rate=round(location_change_rate, 3),
            international_transaction_ratio=round(international_ratio, 3),
            impossible_travel_detected=impossible_travel,
            travel_distance_km=travel_distance
        )
    
    def _analyze_device_channel_patterns(self, transaction: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze device and channel usage patterns"""
        if not history:
            return Dict[str, Any](
                preferred_devices=[transaction.get('device_id', 'unknown')],
                preferred_channels=[transaction.get('channel', 'unknown')],
                device_consistency_score=0.5, channel_consistency_score=0.5,
                device_change_rate=0.0, channel_change_rate=0.0,
                new_device_detected=False, unusual_channel_usage=False
            )
        
        # Extract device and channel data
        devices = [tx.get('device_id', 'unknown') for tx in history]
        channels = [tx.get('channel', 'unknown') for tx in history]
        
        # Find preferred devices and channels
        device_counts = Counter(devices)
        channel_counts = Counter(channels)
        preferred_devices = [device for device, count in device_counts.most_common(3)]
        preferred_channels = [channel for channel, count in channel_counts.most_common(3)]
        
        # Calculate consistency scores
        device_consistency_score = device_counts.most_common(1)[0][1] / len(devices) if devices else 0.5
        channel_consistency_score = channel_counts.most_common(1)[0][1] / len(channels) if channels else 0.5
        
        # Calculate change rates
        device_changes = sum(1 for i in range(1, len(devices)) if devices[i] != devices[i-1])
        channel_changes = sum(1 for i in range(1, len(channels)) if channels[i] != channels[i-1])
        
        device_change_rate = device_changes / len(devices) if len(devices) > 1 else 0.0
        channel_change_rate = channel_changes / len(channels) if len(channels) > 1 else 0.0
        
        # Check for new device and unusual channel usage
        current_device = transaction.get('device_id', 'unknown')
        current_channel = transaction.get('channel', 'unknown')
        
        new_device_detected = current_device not in devices
        unusual_channel_usage = current_channel not in preferred_channels[:2] if len(preferred_channels) >= 2 else False
        
        return Dict[str, Any](
            preferred_devices=preferred_devices,
            preferred_channels=preferred_channels,
            device_consistency_score=round(device_consistency_score, 3),
            channel_consistency_score=round(channel_consistency_score, 3),
            device_change_rate=round(device_change_rate, 3),
            channel_change_rate=round(channel_change_rate, 3),
            new_device_detected=new_device_detected,
            unusual_channel_usage=unusual_channel_usage
        )
    
    def _analyze_transaction_type_patterns(self, transaction: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transaction type and merchant patterns"""
        if not history:
            return Dict[str, Any](
                preferred_transaction_types=[transaction.get('transaction_type', 'unknown')],
                preferred_merchant_categories=[transaction.get('merchant_category', 'unknown')],
                transaction_type_diversity=0.0, merchant_category_diversity=0.0,
                unusual_transaction_type=False, unusual_merchant_category=False,
                merchant_loyalty_score=0.5
            )
        
        # Extract transaction type and merchant data
        transaction_types = [tx.get('transaction_type', 'unknown') for tx in history]
        merchant_categories = [tx.get('merchant_category', 'unknown') for tx in history]
        
        # Find preferred types and categories
        type_counts = Counter(transaction_types)
        category_counts = Counter(merchant_categories)
        preferred_types = [ttype for ttype, count in type_counts.most_common(3)]
        preferred_categories = [category for category, count in category_counts.most_common(3)]
        
        # Calculate diversity scores
        type_diversity = len(set(transaction_types)) / len(transaction_types) if transaction_types else 0.0
        category_diversity = len(set(merchant_categories)) / len(merchant_categories) if merchant_categories else 0.0
        
        # Check for unusual current transaction
        current_type = transaction.get('transaction_type', 'unknown')
        current_category = transaction.get('merchant_category', 'unknown')
        
        unusual_type = current_type not in preferred_types[:2] if len(preferred_types) >= 2 else False
        unusual_category = current_category not in preferred_categories[:2] if len(preferred_categories) >= 2 else False
        
        # Calculate merchant loyalty score
        merchant_loyalty_score = category_counts.most_common(1)[0][1] / len(merchant_categories) if merchant_categories else 0.5
        
        return Dict[str, Any](
            preferred_transaction_types=preferred_types,
            preferred_merchant_categories=preferred_categories,
            transaction_type_diversity=round(type_diversity, 3),
            merchant_category_diversity=round(category_diversity, 3),
            unusual_transaction_type=unusual_type,
            unusual_merchant_category=unusual_category,
            merchant_loyalty_score=round(merchant_loyalty_score, 3)
        )
    
    def _analyze_frequency_patterns(self, transaction: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transaction frequency patterns"""
        if not history:
            return Dict[str, Any](
                avg_transactions_per_day=0.0, avg_transactions_per_week=0.0,
                transaction_frequency_trend="unknown", frequency_acceleration=1.0,
                unusual_frequency_spike=False, frequency_consistency_score=0.5,
                burst_activity_detected=False
            )
        
        # Calculate frequency metrics
        total_days = (pd.to_datetime(transaction.get('time', datetime.now())) - 
                    pd.to_datetime(history[0].get('time', datetime.now()))).days + 1
        
        avg_per_day = len(history) / total_days if total_days > 0 else 0.0
        avg_per_week = avg_per_day * 7
        
        # Calculate frequency trend
        if len(history) >= 6:
            recent_freq = len([tx for tx in history[-3:]]) / 3  # Last 3 transactions
            older_freq = len([tx for tx in history[:-3]]) / max(len(history) - 3, 1)
            frequency_acceleration = recent_freq / older_freq if older_freq > 0 else 1.0
            
            if frequency_acceleration > 1.5:
                trend = "increasing"
            elif frequency_acceleration < 0.7:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"
            frequency_acceleration = 1.0
        
        # Detect unusual frequency spike
        current_time = pd.to_datetime(transaction.get('time', datetime.now()))
        recent_24h = [tx for tx in history 
                     if (current_time - pd.to_datetime(tx.get('time', current_time))).total_seconds() <= 86400]
        
        unusual_spike = len(recent_24h) > (avg_per_day * 3)
        
        # Calculate frequency consistency
        if len(history) > 1:
            daily_counts = {}
            for tx in history:
                day = pd.to_datetime(tx.get('time', datetime.now())).date()
                daily_counts[day] = daily_counts.get(day, 0) + 1
            
            if daily_counts:
                freq_values = list(daily_counts.values())
                freq_cv = np.std(freq_values) / np.mean(freq_values) if np.mean(freq_values) > 0 else 1.0
                frequency_consistency_score = max(0, 1 - freq_cv)
            else:
                frequency_consistency_score = 0.5
        else:
            frequency_consistency_score = 0.5
        
        # Detect burst activity
        burst_activity = len(recent_24h) > 10  # More than 10 transactions in 24h
        
        return Dict[str, Any](
            avg_transactions_per_day=round(avg_per_day, 3),
            avg_transactions_per_week=round(avg_per_week, 3),
            transaction_frequency_trend=trend,
            frequency_acceleration=round(frequency_acceleration, 3),
            unusual_frequency_spike=unusual_spike,
            frequency_consistency_score=round(frequency_consistency_score, 3),
            burst_activity_detected=burst_activity
        )
    
    def _generate_risk_indicators(
        self, transaction: Dict[str, Any], history: List[Dict[str, Any]],
        spending_patterns: Dict[str, Any], temporal_patterns: Dict[str, Any],
        geographic_patterns: Dict[str, Any], device_channel_patterns: Dict[str, Any],
        transaction_type_patterns: Dict[str, Any], frequency_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive risk indicators"""
        
        risk_factors = []
        
        # Spending risk factors
        if spending_patterns['unusual_spending_spike']:
            risk_factors.append(("unusual_spending_spike", 0.8))
        if spending_patterns['spending_acceleration_7d'] > 3.0:
            risk_factors.append(("high_spending_acceleration", 0.7))
        if spending_patterns['spending_consistency_score'] < 0.3:
            risk_factors.append(("inconsistent_spending", 0.6))
        
        # Temporal risk factors
        if temporal_patterns['unusual_time_transaction']:
            risk_factors.append(("unusual_timing", 0.5))
        if temporal_patterns['night_transaction_ratio'] > 0.3:
            risk_factors.append(("high_night_activity", 0.4))
        if temporal_patterns['time_deviation_score'] > 0.7:
            risk_factors.append(("high_time_deviation", 0.6))
        
        # Geographic risk factors
        if geographic_patterns['unusual_location']:
            risk_factors.append(("unusual_location", 0.7))
        if geographic_patterns['impossible_travel_detected']:
            risk_factors.append(("impossible_travel", 0.9))
        if geographic_patterns['location_change_rate'] > 0.5:
            risk_factors.append(("high_location_change", 0.6))
        
        # Device/Channel risk factors
        if device_channel_patterns['new_device_detected']:
            risk_factors.append(("new_device", 0.6))
        if device_channel_patterns['device_change_rate'] > 0.3:
            risk_factors.append(("high_device_change", 0.7))
        if device_channel_patterns['unusual_channel_usage']:
            risk_factors.append(("unusual_channel", 0.4))
        
        # Transaction type risk factors
        if transaction_type_patterns['unusual_transaction_type']:
            risk_factors.append(("unusual_transaction_type", 0.5))
        if transaction_type_patterns['unusual_merchant_category']:
            risk_factors.append(("unusual_merchant", 0.5))
        
        # Frequency risk factors
        if frequency_patterns['unusual_frequency_spike']:
            risk_factors.append(("frequency_spike", 0.7))
        if frequency_patterns['burst_activity_detected']:
            risk_factors.append(("burst_activity", 0.8))
        
        # Calculate overall behavioral risk score
        if risk_factors:
            behavioral_risk_score = min(sum(score for _, score in risk_factors) / len(risk_factors), 1.0)
        else:
            behavioral_risk_score = 0.1
        
        # Determine risk level
        if behavioral_risk_score >= 0.8:
            risk_level = "CRITICAL"
        elif behavioral_risk_score >= 0.6:
            risk_level = "HIGH"
        elif behavioral_risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate pattern deviation score
        deviation_scores = [
            1 - spending_patterns['spending_consistency_score'],
            1 - temporal_patterns['time_consistency_score'],
            1 - geographic_patterns['location_consistency_score'],
            1 - device_channel_patterns['device_consistency_score'],
            1 - frequency_patterns['frequency_consistency_score']
        ]
        pattern_deviation_score = np.mean(deviation_scores)
        
        # Account age factor
        account_age_days = transaction.get('account_age_days', 0)
        if account_age_days < 7:
            account_age_factor = 0.8  # New accounts are riskier
        elif account_age_days < 30:
            account_age_factor = 0.6
        elif account_age_days < 90:
            account_age_factor = 0.4
        else:
            account_age_factor = 0.2
        
        # Calculate behavioral confidence
        data_points = len(history)
        if data_points >= 20:
            behavioral_confidence = 0.9
        elif data_points >= 10:
            behavioral_confidence = 0.7
        elif data_points >= 5:
            behavioral_confidence = 0.5
        else:
            behavioral_confidence = 0.3
        
        return Dict[str, Any](
            behavioral_risk_score=round(behavioral_risk_score, 3),
            risk_level=risk_level,
            top_risk_factors=sorted(risk_factors, key=lambda x: x[1], reverse=True)[:5],
            anomaly_count=len(risk_factors),
            pattern_deviation_score=round(pattern_deviation_score, 3),
            account_age_factor=round(account_age_factor, 3),
            behavioral_confidence=round(behavioral_confidence, 3)
        )
    
    def _generate_behavioral_insights(
        self, risk_indicators: Dict[str, Any], spending_patterns: Dict[str, Any],
        temporal_patterns: Dict[str, Any], geographic_patterns: Dict[str, Any],
        device_channel_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate high-level behavioral insights"""
        
        # Determine customer profile
        if spending_patterns['spending_consistency_score'] > 0.7 and temporal_patterns['time_consistency_score'] > 0.7:
            customer_profile = "conservative"
        elif spending_patterns['spending_acceleration_7d'] > 2.0:
            customer_profile = "aggressive"
        elif risk_indicators['anomaly_count'] > 3:
            customer_profile = "unusual"
        else:
            customer_profile = "moderate"
        
        # Determine behavioral stability
        if risk_indicators['pattern_deviation_score'] < 0.3:
            behavioral_stability = "stable"
        elif risk_indicators['pattern_deviation_score'] < 0.6:
            behavioral_stability = "evolving"
        else:
            behavioral_stability = "volatile"
        
        # Determine fraud likelihood
        if risk_indicators['behavioral_risk_score'] >= 0.7:
            fraud_likelihood = "high"
        elif risk_indicators['behavioral_risk_score'] >= 0.4:
            fraud_likelihood = "medium"
        else:
            fraud_likelihood = "low"
        
        # Generate recommendations
        recommendations = []
        if risk_indicators['risk_level'] in ["HIGH", "CRITICAL"]:
            recommendations.append("Enhanced verification required")
            recommendations.append("Manual review recommended")
        if geographic_patterns['impossible_travel_detected']:
            recommendations.append("Verify customer location")
        if device_channel_patterns['new_device_detected']:
            recommendations.append("Device verification needed")
        if spending_patterns['unusual_spending_spike']:
            recommendations.append("Amount verification required")
        
        # Generate behavioral summary
        summary_parts = [
            f"Customer shows {customer_profile} spending behavior",
            f"with {behavioral_stability} patterns",
            f"and {fraud_likelihood} fraud likelihood"
        ]
        behavioral_summary = " ".join(summary_parts)
        
        return Dict[str, Any](
            customer_profile=customer_profile,
            behavioral_stability=behavioral_stability,
            fraud_likelihood=fraud_likelihood,
            recommended_actions=recommendations,
            behavioral_summary=behavioral_summary
        )
    
    def _calculate_data_sufficiency(self, history: List[Dict[str, Any]]) -> float:
        """Calculate data sufficiency score"""
        if not history:
            return 0.0
        
        # Factors affecting data sufficiency
        transaction_count = len(history)
        time_span_days = 0
        
        if len(history) > 1:
            start_time = pd.to_datetime(history[0].get('time', datetime.now()))
            end_time = pd.to_datetime(history[-1].get('time', datetime.now()))
            time_span_days = (end_time - start_time).days + 1
        
        # Calculate sufficiency score
        count_score = min(transaction_count / 20, 1.0)  # Optimal at 20+ transactions
        time_score = min(time_span_days / 30, 1.0)  # Optimal at 30+ days
        
        return (count_score + time_score) / 2
    
    def _calculate_analysis_confidence(self, data_sufficiency: float, transaction_count: int) -> float:
        """Calculate analysis confidence"""
        base_confidence = data_sufficiency
        
        # Adjust based on transaction count
        if transaction_count >= 20:
            count_factor = 1.0
        elif transaction_count >= 10:
            count_factor = 0.8
        elif transaction_count >= 5:
            count_factor = 0.6
        else:
            count_factor = 0.4
        
        return min(base_confidence * count_factor, 1.0)
    
    def _determine_analysis_quality(self, data_sufficiency: float, analysis_confidence: float) -> str:
        """Determine analysis quality"""
        avg_score = (data_sufficiency + analysis_confidence) / 2
        
        if avg_score >= 0.8:
            return "high"
        elif avg_score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _compile_raw_patterns(self, *pattern_dicts) -> Dict[str, Any]:
        """Compile raw patterns for code-as-action"""
        raw_patterns = {}
        
        for pattern_dict in pattern_dicts:
            for key, value in pattern_dict.items():
                raw_patterns[key] = value
        
        return raw_patterns
    
    def _create_fallback_analysis(self, transaction: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Create fallback analysis on error"""
        return Dict[str, Any](
            transaction_id=transaction.get('transaction_id', 'unknown'),
            customer_id=transaction.get('customer_id', 'unknown'),
            analysis_timestamp=datetime.now(),
            analysis_window_days=30,
            total_transactions_analyzed=0,
            
            spending_patterns=Dict[str, Any](
                avg_amount_7d=0.0, avg_amount_30d=0.0, avg_amount_90d=0.0,
                spending_acceleration_7d=1.0, spending_acceleration_30d=1.0,
                max_single_transaction=float(transaction.get('amount', 0)),
                spending_consistency_score=0.5, unusual_spending_spike=False,
                spending_trend="unknown"
            ),
            
            temporal_patterns=Dict[str, Any](
                preferred_hours=[], preferred_days=[],
                time_consistency_score=0.5, unusual_time_transaction=False,
                night_transaction_ratio=0.0, weekend_transaction_ratio=0.0,
                time_deviation_score=0.0
            ),
            
            geographic_patterns=Dict[str, Any](
                primary_locations=[], location_diversity_score=0.0,
                location_consistency_score=0.5, unusual_location=False,
                location_change_rate=0.0, international_transaction_ratio=0.0,
                impossible_travel_detected=False, travel_distance_km=None
            ),
            
            device_channel_patterns=Dict[str, Any](
                preferred_devices=[], preferred_channels=[],
                device_consistency_score=0.5, channel_consistency_score=0.5,
                device_change_rate=0.0, channel_change_rate=0.0,
                new_device_detected=False, unusual_channel_usage=False
            ),
            
            transaction_type_patterns=Dict[str, Any](
                preferred_transaction_types=[], preferred_merchant_categories=[],
                transaction_type_diversity=0.0, merchant_category_diversity=0.0,
                unusual_transaction_type=False, unusual_merchant_category=False,
                merchant_loyalty_score=0.5
            ),
            
            frequency_patterns=Dict[str, Any](
                avg_transactions_per_day=0.0, avg_transactions_per_week=0.0,
                transaction_frequency_trend="unknown", frequency_acceleration=1.0,
                unusual_frequency_spike=False, frequency_consistency_score=0.5,
                burst_activity_detected=False
            ),
            
            risk_indicators=Dict[str, Any](
                behavioral_risk_score=0.5, risk_level="UNKNOWN",
                top_risk_factors=[], anomaly_count=0,
                pattern_deviation_score=0.5, account_age_factor=0.5,
                behavioral_confidence=0.0
            ),
            
            behavioral_insights=Dict[str, Any](
                customer_profile="unknown", behavioral_stability="unknown",
                fraud_likelihood="unknown", recommended_actions=["Manual review required"],
                behavioral_summary="Analysis failed - insufficient data"
            ),
            
            analysis_quality="low", data_sufficiency_score=0.0, analysis_confidence=0.0,
            raw_patterns={"error": error_msg}, pattern_metadata={"error": True}
        )


# Convenience factory
_analyzer: BehavioralAnalyzer = None

def get_default_analyzer() -> BehavioralAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = BehavioralAnalyzer()
    return _analyzer

# OpenAI SDK format function tool
@function_tool(strict_mode=False)
async def analyze_behavioral_patterns(
    transaction: dict, 
    customer_history: list,
    analysis_window_days: int = 30
) -> dict:
    """
    Analyze customer behavioral patterns for fraud detection.
    
    Args:
        transaction: Current transaction to analyze
        customer_history: Historical transactions for this customer
        analysis_window_days: Days to look back for pattern analysis
        
    Returns:
        BehavioralAnalysis with comprehensive behavioral insights and risk indicators
    """
    analyzer = get_default_analyzer()
    return analyzer.analyze_behavioral_patterns(transaction, customer_history, analysis_window_days)

# CLI test
if __name__ == "__main__":
    import asyncio
    
    # Sample test data
    sample_transaction = {
        "transaction_id": "TXN_test_001",
        "customer_id": "CUST_test_001",
        "amount": 15000.0,
        "currency": "NGN",
        "transaction_type": "transfer",
        "channel": "web",
        "merchant_category": "electronics",
        "device_id": "DEV_test_001",
        "ip_address": "192.168.1.100",
        "location_country": "Nigeria",
        "location_city": "Lagos",
        "time": "2025-01-22 14:30:00",
        "account_age_days": 45
    }
    
    sample_history = [
        {
            "transaction_id": "TXN_hist_001",
            "customer_id": "CUST_test_001",
            "amount": 5000.0,
            "currency": "NGN",
            "transaction_type": "purchase",
            "channel": "mobile",
            "merchant_category": "groceries",
            "device_id": "DEV_test_001",
            "location_country": "Nigeria",
            "location_city": "Lagos",
            "time": "2025-01-20 10:15:00"
        },
        {
            "transaction_id": "TXN_hist_002",
            "customer_id": "CUST_test_001",
            "amount": 8000.0,
            "currency": "NGN",
            "transaction_type": "transfer",
            "channel": "web",
            "merchant_category": "utilities",
            "device_id": "DEV_test_001",
            "location_country": "Nigeria",
            "location_city": "Lagos",
            "time": "2025-01-18 16:45:00"
        }
    ]
    
    async def test_behavioral_analyzer():
        print("🧠 Testing Behavioral Pattern Analyzer")
        print("=" * 50)
        
        result = await analyze_behavioral_patterns(sample_transaction, sample_history)
        
        print(f"📊 Analysis Results:")
        print(f"   Transaction ID: {result['transaction_id']}")
        print(f"   Customer ID: {result['customer_id']}")
        print(f"   Analysis Quality: {result['analysis_quality']}")
        print(f"   Data Sufficiency: {result['data_sufficiency_score']:.3f}")
        print(f"   Analysis Confidence: {result['analysis_confidence']:.3f}")
        
        print(f"\n💰 Spending Patterns:")
        spending = result['spending_patterns']
        print(f"   Avg Amount (7d): {spending['avg_amount_7d']}")
        print(f"   Spending Acceleration: {spending['spending_acceleration_7d']}")
        print(f"   Consistency Score: {spending['spending_consistency_score']}")
        print(f"   Unusual Spike: {spending['unusual_spending_spike']}")
        
        print(f"\n⏰ Temporal Patterns:")
        temporal = result['temporal_patterns']
        print(f"   Preferred Hours: {temporal['preferred_hours']}")
        print(f"   Time Consistency: {temporal['time_consistency_score']}")
        print(f"   Unusual Time: {temporal['unusual_time_transaction']}")
        
        print(f"\n🌍 Geographic Patterns:")
        geo = result['geographic_patterns']
        print(f"   Primary Locations: {geo['primary_locations']}")
        print(f"   Location Consistency: {geo['location_consistency_score']}")
        print(f"   Unusual Location: {geo['unusual_location']}")
        
        print(f"\n🎯 Risk Indicators:")
        risk = result['risk_indicators']
        print(f"   Behavioral Risk Score: {risk['behavioral_risk_score']}")
        print(f"   Risk Level: {risk['risk_level']}")
        print(f"   Anomaly Count: {risk['anomaly_count']}")
        print(f"   Top Risk Factors: {risk['top_risk_factors']}")
        
        print(f"\n💡 Behavioral Insights:")
        insights = result['behavioral_insights']
        print(f"   Customer Profile: {insights['customer_profile']}")
        print(f"   Behavioral Stability: {insights['behavioral_stability']}")
        print(f"   Fraud Likelihood: {insights['fraud_likelihood']}")
        print(f"   Recommendations: {insights['recommended_actions']}")
        print(f"   Summary: {insights['behavioral_summary']}")
        
        print(f"\n🔧 Raw Patterns (for code-as-action):")
        print(f"   Available patterns: {list(result['raw_patterns'].keys())}")
    
    asyncio.run(test_behavioral_analyzer())
