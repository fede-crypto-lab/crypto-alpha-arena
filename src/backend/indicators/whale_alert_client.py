"""
Whale Alert Client - Monitor large cryptocurrency transactions
Uses public whale-alert.io JSON feed (FREE, no API key required)
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class WhaleAlertClient:
    """
    Client for monitoring large cryptocurrency transactions.
    
    Trading signals based on whale movements:
    - Large inflows TO exchanges = potential SELL pressure (bearish)
    - Large outflows FROM exchanges = accumulation (bullish)
    - Stablecoin flows TO exchanges = dry powder ready to buy (bullish)
    """
    
    # Public JSON feed URL (no API key needed)
    PUBLIC_URL = "https://whale-alert.io/alerts.json"
    
    # Known exchange wallets (partial list - exchanges have many addresses)
    EXCHANGE_KEYWORDS = [
        'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi', 'okex', 'kucoin',
        'bybit', 'ftx', 'gemini', 'bitstamp', 'bittrex', 'poloniex', 'gate.io',
        'crypto.com', 'exchange', 'hot wallet'
    ]
    
    # Minimum USD value to consider "whale" transaction
    MIN_WHALE_USD = 10_000_000  # $10M
    
    def __init__(self, cache_ttl: int = 300):
        """
        Initialize whale alert client.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.cache_ttl = cache_ttl
        self._cache: Optional[List[Dict]] = None
        self._cache_time: Optional[float] = None
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self._cache_time is None:
            return False
        return (time.time() - self._cache_time) < self.cache_ttl
    
    def _is_exchange(self, wallet_name: str) -> bool:
        """Check if wallet name indicates an exchange."""
        if not wallet_name:
            return False
        wallet_lower = wallet_name.lower()
        return any(ex in wallet_lower for ex in self.EXCHANGE_KEYWORDS)
    
    def get_recent_alerts(self, hours: int = 24) -> Optional[List[Dict]]:
        """
        Fetch recent whale alerts from public feed.
        
        Args:
            hours: How many hours of data to fetch
            
        Returns:
            List of whale alert dicts or None on error
        """
        # Check cache first
        if self._is_cache_valid() and self._cache:
            logger.debug("Using cached whale alerts")
            return self._cache
        
        try:
            # The public feed provides last 30 days of alerts
            params = {"range": "last_30_days"}
            response = requests.get(self.PUBLIC_URL, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            alerts = data if isinstance(data, list) else data.get("alerts", [])
            
            if not alerts:
                logger.warning("No whale alerts returned")
                return []
            
            # Filter by time window
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_alerts = []
            
            for alert in alerts:
                try:
                    # Parse timestamp
                    timestamp = alert.get("timestamp")
                    if timestamp:
                        if isinstance(timestamp, (int, float)):
                            alert_time = datetime.fromtimestamp(timestamp)
                        else:
                            alert_time = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                        
                        if alert_time < cutoff_time:
                            continue
                    
                    # Parse alert data
                    parsed = self._parse_alert(alert)
                    if parsed and parsed.get("amount_usd", 0) >= self.MIN_WHALE_USD:
                        recent_alerts.append(parsed)
                        
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error parsing alert: {e}")
                    continue
            
            # Cache results
            self._cache = recent_alerts
            self._cache_time = time.time()
            
            logger.info(f"Fetched {len(recent_alerts)} whale alerts (last {hours}h)")
            return recent_alerts
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch whale alerts: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing whale alerts: {e}")
            return None
    
    def _parse_alert(self, alert: Dict) -> Optional[Dict]:
        """Parse a single whale alert into standardized format."""
        try:
            # Different feed formats exist, handle both
            symbol = alert.get("symbol") or alert.get("blockchain") or alert.get("currency", "").upper()
            amount = float(alert.get("amount", 0) or alert.get("amount_usd", 0) or 0)
            amount_usd = float(alert.get("amount_usd", 0) or alert.get("usd_value", 0) or amount)
            
            from_wallet = alert.get("from", {})
            to_wallet = alert.get("to", {})
            
            # Handle different data structures
            if isinstance(from_wallet, str):
                from_name = from_wallet
                from_is_exchange = self._is_exchange(from_wallet)
            else:
                from_name = from_wallet.get("owner", "") or from_wallet.get("name", "")
                from_is_exchange = from_wallet.get("owner_type") == "exchange" or self._is_exchange(from_name)
            
            if isinstance(to_wallet, str):
                to_name = to_wallet
                to_is_exchange = self._is_exchange(to_wallet)
            else:
                to_name = to_wallet.get("owner", "") or to_wallet.get("name", "")
                to_is_exchange = to_wallet.get("owner_type") == "exchange" or self._is_exchange(to_name)
            
            # Determine flow type
            if from_is_exchange and not to_is_exchange:
                flow_type = "exchange_outflow"  # Bullish - accumulation
            elif not from_is_exchange and to_is_exchange:
                flow_type = "exchange_inflow"  # Bearish - potential selling
            elif from_is_exchange and to_is_exchange:
                flow_type = "exchange_to_exchange"
            else:
                flow_type = "wallet_to_wallet"
            
            return {
                "symbol": symbol.upper() if symbol else "UNKNOWN",
                "amount": amount,
                "amount_usd": amount_usd,
                "from": from_name or "unknown",
                "to": to_name or "unknown",
                "from_is_exchange": from_is_exchange,
                "to_is_exchange": to_is_exchange,
                "flow_type": flow_type,
                "timestamp": alert.get("timestamp"),
                "tx_hash": alert.get("hash") or alert.get("tx_hash"),
            }
        except Exception as e:
            logger.debug(f"Error parsing alert: {e}")
            return None
    
    def analyze_whale_activity(self, asset: str = "BTC", hours: int = 24) -> Dict[str, Any]:
        """
        Analyze whale activity for trading signals.
        
        Args:
            asset: Asset to analyze (BTC, ETH, etc.)
            hours: Time window in hours
            
        Returns:
            Dict with analysis results and trading signal
        """
        alerts = self.get_recent_alerts(hours=hours)
        
        if alerts is None:
            return {
                "error": "Failed to fetch whale alerts",
                "signal": "NEUTRAL",
                "confidence": 0
            }
        
        # Filter for specific asset
        asset_upper = asset.upper()
        asset_alerts = [a for a in alerts if a["symbol"] == asset_upper]
        
        # Also check stablecoin flows (USDT, USDC) as they indicate buying power
        stablecoin_alerts = [a for a in alerts if a["symbol"] in ["USDT", "USDC", "BUSD", "DAI"]]
        
        # Aggregate flows
        exchange_inflows = sum(a["amount_usd"] for a in asset_alerts if a["flow_type"] == "exchange_inflow")
        exchange_outflows = sum(a["amount_usd"] for a in asset_alerts if a["flow_type"] == "exchange_outflow")
        
        stablecoin_to_exchange = sum(a["amount_usd"] for a in stablecoin_alerts if a["flow_type"] == "exchange_inflow")
        stablecoin_from_exchange = sum(a["amount_usd"] for a in stablecoin_alerts if a["flow_type"] == "exchange_outflow")
        
        # Calculate net flow (positive = more going TO exchanges = bearish)
        net_flow = exchange_inflows - exchange_outflows
        stablecoin_net = stablecoin_to_exchange - stablecoin_from_exchange
        
        # Determine signal
        signal, confidence = self._calculate_signal(net_flow, stablecoin_net, exchange_inflows, exchange_outflows)
        
        return {
            "asset": asset_upper,
            "period_hours": hours,
            "exchange_inflows_usd": exchange_inflows,
            "exchange_outflows_usd": exchange_outflows,
            "net_flow_usd": net_flow,
            "stablecoin_to_exchange_usd": stablecoin_to_exchange,
            "stablecoin_from_exchange_usd": stablecoin_from_exchange,
            "stablecoin_net_usd": stablecoin_net,
            "alert_count": len(asset_alerts),
            "signal": signal,
            "confidence": confidence,
            "interpretation": self._get_interpretation(signal, net_flow, stablecoin_net)
        }
    
    def _calculate_signal(self, net_flow: float, stablecoin_net: float, 
                          inflows: float, outflows: float) -> tuple:
        """
        Calculate trading signal from whale flows.
        
        Returns:
            (signal, confidence) tuple
        """
        # Thresholds (in USD)
        SIGNIFICANT_FLOW = 50_000_000  # $50M
        MAJOR_FLOW = 200_000_000  # $200M
        
        # Score from -100 (very bearish) to +100 (very bullish)
        score = 0
        
        # Net flow score (negative flow = bullish, positive flow = bearish)
        if abs(net_flow) > MAJOR_FLOW:
            score -= 40 if net_flow > 0 else -40
        elif abs(net_flow) > SIGNIFICANT_FLOW:
            score -= 20 if net_flow > 0 else -20
        
        # Stablecoin score (stables TO exchange = bullish buying power)
        if stablecoin_net > MAJOR_FLOW:
            score += 30
        elif stablecoin_net > SIGNIFICANT_FLOW:
            score += 15
        elif stablecoin_net < -SIGNIFICANT_FLOW:
            score -= 10  # Stables leaving = less buying power
        
        # Convert score to signal
        if score >= 40:
            return "BULLISH", min(90, 50 + score)
        elif score >= 20:
            return "LEAN_BULLISH", min(70, 50 + score // 2)
        elif score <= -40:
            return "BEARISH", min(90, 50 - score)
        elif score <= -20:
            return "LEAN_BEARISH", min(70, 50 - score // 2)
        else:
            return "NEUTRAL", 50
    
    def _get_interpretation(self, signal: str, net_flow: float, stablecoin_net: float) -> str:
        """Get human-readable interpretation of whale activity."""
        parts = []
        
        if net_flow > 50_000_000:
            parts.append(f"${net_flow/1e6:.0f}M net inflow to exchanges (selling pressure)")
        elif net_flow < -50_000_000:
            parts.append(f"${abs(net_flow)/1e6:.0f}M net outflow from exchanges (accumulation)")
        
        if stablecoin_net > 50_000_000:
            parts.append(f"${stablecoin_net/1e6:.0f}M stablecoins moved to exchanges (buying power ready)")
        
        if not parts:
            parts.append("No significant whale movements detected")
        
        return "; ".join(parts)
    
    def get_whale_context(self, asset: str = "BTC") -> str:
        """
        Get formatted whale activity context for LLM prompt.
        
        Args:
            asset: Asset to analyze
            
        Returns:
            Formatted string describing whale activity
        """
        analysis = self.analyze_whale_activity(asset=asset, hours=24)
        
        if analysis.get("error"):
            return f"Whale Activity: {analysis['error']}"
        
        context = f"""
## Whale Activity ({asset}) - Last 24h
- Exchange Inflows: ${analysis['exchange_inflows_usd']/1e6:.1f}M (potential sell pressure)
- Exchange Outflows: ${analysis['exchange_outflows_usd']/1e6:.1f}M (accumulation)
- Net Flow: ${analysis['net_flow_usd']/1e6:+.1f}M
- Stablecoin to Exchanges: ${analysis['stablecoin_to_exchange_usd']/1e6:.1f}M (buying power)
- Signal: {analysis['signal']} (confidence: {analysis['confidence']}%)
- Interpretation: {analysis['interpretation']}
"""
        return context.strip()


# Convenience function
def get_whale_analysis(asset: str = "BTC") -> Dict[str, Any]:
    """Quick access to whale activity analysis."""
    client = WhaleAlertClient()
    return client.analyze_whale_activity(asset=asset)
