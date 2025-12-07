"""
Trading Bot Engine - Core trading logic separated from UI
Refactored from ai-trading-agent/src/main.py

PATCH v4: 
- CRITICAL: Startup sync with exchange before any trading
- CRITICAL: Block trading if sync fails (don't continue with stale data)
- CRITICAL: Validate TP/SL sanity in code
- CRITICAL: Handle existing positions on startup
- Improved rate limit handling with proper backoff
- NEW v4: TP/SL response validation - check if orders were actually accepted
- NEW v4: Log actual error messages when TP/SL orders are rejected
- NEW v4: Handle new structured response format from hyperliquid_api v4
"""

import asyncio
import json
import logging
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

from src.backend.indicators import EnhancedMarketContext
from src.backend.agent.decision_maker import TradingAgent
from src.backend.config_loader import CONFIG
from src.backend.indicators.taapi_client import TAAPIClient
from src.backend.models.trade_proposal import TradeProposal
from src.backend.trading.hyperliquid_api import HyperliquidAPI
from src.backend.utils.prompt_utils import json_default


@dataclass
class BotState:
    """Bot state for UI updates"""
    is_running: bool = False
    balance: float = 0.0
    total_value: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    positions: List[Dict] = field(default_factory=list)
    active_trades: List[Dict] = field(default_factory=list)
    open_orders: List[Dict] = field(default_factory=list)
    recent_fills: List[Dict] = field(default_factory=list)
    market_data: List[Dict] = field(default_factory=list)
    pending_proposals: List[Dict] = field(default_factory=list)
    last_reasoning: Dict = field(default_factory=dict)
    last_update: str = ""
    error: Optional[str] = None
    invocation_count: int = 0


class TradingBotEngine:
    """
    Core trading bot engine independent of UI.
    Communicates with GUI via callback system.
    """

    def __init__(
        self,
        assets: List[str],
        interval: str,
        on_state_update: Optional[Callable[[BotState], None]] = None,
        on_trade_executed: Optional[Callable[[Dict], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.assets = assets
        self.interval = interval
        self.on_state_update = on_state_update
        self.on_trade_executed = on_trade_executed
        self.on_error = on_error

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("bot.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.taapi = TAAPIClient()
        self.hyperliquid = HyperliquidAPI()
        self.agent = TradingAgent()
        self.enhanced_context = EnhancedMarketContext(
            sentiment_cache_ttl=3600,
            whale_cache_ttl=300
        )
        self.logger.info("EnhancedMarketContext initialized")

        self.state = BotState()
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

        self.start_time: Optional[datetime] = None
        self.invocation_count = 0
        self.trade_log: List[float] = []
        self.active_trades: List[Dict] = []
        self.recent_events: deque = deque(maxlen=200)
        self.initial_account_value: Optional[float] = None
        self.price_history: Dict[str, deque] = {asset: deque(maxlen=60) for asset in assets}
        
        self.trading_mode = CONFIG.get("trading_mode", "auto").lower()
        self.pending_proposals: List[TradeProposal] = []
        self.logger.info(f"Trading mode: {self.trading_mode.upper()}")

        self.diary_path = Path("data/diary.jsonl")
        self.diary_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ===== NEW: Rate limit tracking =====
        self._last_hl_call = 0.0
        self._hl_call_delay = 0.5  # Minimum delay between Hyperliquid API calls
        self._consecutive_429s = 0
        self._max_consecutive_429s = 5  # Skip iteration after this many 429s

    async def _rate_limited_call(self, coro):
        """
        Execute a coroutine with rate limiting for Hyperliquid API.
        Adds delay between calls and handles 429 errors gracefully.
        """
        import time
        
        # Ensure minimum delay between calls
        now = time.time()
        elapsed = now - self._last_hl_call
        if elapsed < self._hl_call_delay:
            await asyncio.sleep(self._hl_call_delay - elapsed)
        
        try:
            result = await coro
            self._last_hl_call = time.time()
            self._consecutive_429s = 0  # Reset on success
            return result
        except Exception as e:
            if '429' in str(e):
                self._consecutive_429s += 1
                # Exponential backoff
                wait_time = min(30, 2 ** self._consecutive_429s)
                self.logger.warning(f"⏳ Rate limited (429). Waiting {wait_time}s... (consecutive: {self._consecutive_429s})")
                await asyncio.sleep(wait_time)
            raise

    async def start(self):
        """Start the trading bot"""
        if self.is_running:
            self.logger.warning("Bot already running")
            return

        self.is_running = True
        self.state.is_running = True
        self.start_time = datetime.now(UTC)
        self.invocation_count = 0

        # ===== CRITICAL: STARTUP SYNC =====
        # Must sync with exchange BEFORE any trading to detect existing positions
        self.logger.info("🚀 STARTUP: Syncing with exchange...")
        try:
            startup_success = await self._startup_sync()
            if not startup_success:
                self.logger.error("❌ STARTUP SYNC FAILED - Bot will not trade until sync succeeds")
                self.state.error = "Startup sync failed - waiting for exchange connection"
        except Exception as e:
            self.logger.error(f"❌ STARTUP SYNC ERROR: {e}")
            self.state.error = f"Startup sync error: {e}"

        # ===== DELAY AFTER STARTUP before main loop =====
        self.logger.info("⏳ Waiting 10s after startup before main loop (rate limit protection)...")
        await asyncio.sleep(10)

        self._task = asyncio.create_task(self._main_loop())
        self.logger.info(f"Bot started - Assets: {self.assets}, Interval: {self.interval}")
        self._notify_state_update()

    async def _startup_sync(self) -> bool:
        """
        CRITICAL: Sync with exchange on startup.
        Detects existing positions and initializes active_trades tracking.
        
        Returns:
            True if sync successful, False otherwise
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"🔄 STARTUP SYNC attempt {attempt + 1}/{max_retries}")
                
                # Get account state with retry
                await asyncio.sleep(2)  # Initial delay to avoid burst
                state = await self.hyperliquid.get_user_state()
                
                # ===== DIAGNOSTIC: Log positions received =====
                self.logger.info(f"📊 STARTUP: Received {len(state.get('positions', []))} positions from exchange")
                
                self.initial_account_value = state.get('total_value', 0.0)
                if self.initial_account_value == 0.0:
                    self.initial_account_value = state.get('balance', 10000.0)
                
                # Get open orders
                await asyncio.sleep(2)  # Delay between calls
                open_orders = await self.hyperliquid.get_open_orders()
                
                # ===== DIAGNOSTIC: Log what we received =====
                self.logger.info(f"📋 STARTUP: Received {len(open_orders)} open orders")
                for idx, order in enumerate(open_orders):
                    order_type = order.get('orderType', {})
                    coin = order.get('coin')
                    oid = order.get('oid')
                    side = order.get('side')
                    sz = order.get('sz')
                    
                    # Parse trigger info
                    trigger_info = "N/A"
                    if isinstance(order_type, dict) and 'trigger' in order_type:
                        trigger = order_type.get('trigger', {})
                        tpsl = trigger.get('tpsl', 'unknown')
                        trigger_px = trigger.get('triggerPx', 'unknown')
                        trigger_info = f"{tpsl}@{trigger_px}"
                    
                    self.logger.info(f"  📋 Order[{idx}]: {coin} oid={oid} side={side} sz={sz} trigger={trigger_info}")
                    self.logger.debug(f"  📋 Order[{idx}] RAW: {order}")
                
                # ===== CRITICAL: Import existing positions into active_trades =====
                existing_positions = []
                for pos in state['positions']:
                    asset = pos.get('coin')
                    size = float(pos.get('szi', 0) or 0)
                    
                    if abs(size) > 0.00001:
                        entry_price = float(pos.get('entryPx', 0) or 0)
                        is_long = size > 0
                        
                        self.logger.debug(f"🔍 Looking for TP/SL orders for {asset} in {len(open_orders)} orders...")
                        
                        # Find associated TP/SL orders WITH PRICES
                        tp_oid = None
                        sl_oid = None
                        tp_price = None
                        sl_price = None
                        
                        for order in open_orders:
                            order_coin = order.get('coin')
                            if order_coin == asset:
                                self.logger.debug(f"🔍 Found order for {asset}: {order}")
                                order_type = order.get('orderType', {})
                                if isinstance(order_type, dict) and 'trigger' in order_type:
                                    trigger = order_type.get('trigger', {})
                                    tpsl = trigger.get('tpsl')
                                    trigger_px = trigger.get('triggerPx')
                                    
                                    if tpsl == 'tp':
                                        tp_oid = order.get('oid')
                                        tp_price = float(trigger_px) if trigger_px else None
                                    elif tpsl == 'sl':
                                        sl_oid = order.get('oid')
                                        sl_price = float(trigger_px) if trigger_px else None
                        
                        trade_entry = {
                            'asset': asset,
                            'is_long': is_long,
                            'amount': abs(size),
                            'entry_price': entry_price,
                            'tp_oid': tp_oid,
                            'sl_oid': sl_oid,
                            'tp_price': tp_price,  # NEW: Store the actual price
                            'sl_price': sl_price,  # NEW: Store the actual price
                            'exit_plan': 'Imported from existing position on startup',
                            'opened_at': datetime.now(UTC).isoformat(),
                            'imported_on_startup': True
                        }
                        
                        self.active_trades.append(trade_entry)
                        
                        # Build detailed position info for logging
                        tp_info = f"TP@${tp_price:,.2f}" if tp_price else "no TP"
                        sl_info = f"SL@${sl_price:,.2f}" if sl_price else "no SL"
                        existing_positions.append(f"{asset} {'LONG' if is_long else 'SHORT'} {abs(size):.6f}")
                        
                        self.logger.info(f"📥 IMPORTED: {asset} {'LONG' if is_long else 'SHORT'} {abs(size):.6f} @ ${entry_price:,.2f} | {tp_info} | {sl_info}")
                
                if existing_positions:
                    self.logger.warning(f"⚠️ STARTUP: Found {len(existing_positions)} existing positions: {existing_positions}")
                    self._write_diary_entry({
                        'timestamp': datetime.now(UTC).isoformat(),
                        'action': 'startup_import',
                        'imported_positions': existing_positions,
                        'note': 'Positions imported from exchange on bot startup'
                    })
                else:
                    self.logger.info("✅ STARTUP: No existing positions found")
                
                # Clean up orphan orders (orders without positions)
                await self._cleanup_orphan_orders(state['positions'], open_orders)
                
                self.logger.info(f"✅ STARTUP SYNC COMPLETE: Balance=${state['balance']:,.2f}, Positions={len(existing_positions)}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ STARTUP SYNC attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    self.logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
        
        return False

    async def _cleanup_orphan_orders(self, positions: List[Dict], open_orders: List[Dict]):
        """Cancel orders for assets that don't have positions (orphan TP/SL)"""
        position_assets = {pos.get('coin') for pos in positions if abs(float(pos.get('szi', 0) or 0)) > 0.00001}
        
        orphan_count = 0
        for order in open_orders:
            asset = order.get('coin')
            if asset and asset not in position_assets:
                try:
                    oid = order.get('oid')
                    if oid:
                        await asyncio.sleep(0.3)  # Rate limit protection
                        await self.hyperliquid.cancel_order(asset, oid)
                        orphan_count += 1
                        self.logger.info(f"🧹 Cancelled orphan order: {asset} oid={oid}")
                except Exception as e:
                    self.logger.error(f"Failed to cancel orphan order: {e}")
        
        if orphan_count > 0:
            self.logger.info(f"🧹 Cleaned up {orphan_count} orphan orders")

    async def stop(self):
        """Stop the trading bot"""
        if not self.is_running:
            return

        self.is_running = False
        self.state.is_running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self.logger.info("Bot stopped")
        self._notify_state_update()

    # ===== NEW: TP/SL VALIDATION =====
    def _validate_tp_sl(self, action: str, current_price: float, tp_price: Optional[float], sl_price: Optional[float]) -> Dict[str, Any]:
        """
        Validate and optionally fix TP/SL prices.
        
        Rules:
        - BUY (LONG): TP > current_price, SL < current_price
        - SELL (SHORT): TP < current_price, SL > current_price
        
        Returns:
            Dict with 'tp_price', 'sl_price', 'warnings' list
        """
        warnings = []
        validated_tp = tp_price
        validated_sl = sl_price
        
        if action == 'buy':
            # LONG position
            if tp_price is not None and tp_price <= current_price:
                warnings.append(f"TP {tp_price} <= entry {current_price} for LONG - INVALID, setting to None")
                validated_tp = None
            
            if sl_price is not None and sl_price >= current_price:
                warnings.append(f"SL {sl_price} >= entry {current_price} for LONG - INVALID, setting to None")
                validated_sl = None
                
            # Sanity check: TP shouldn't be too far (>50% gain)
            if validated_tp and validated_tp > current_price * 1.5:
                warnings.append(f"TP {validated_tp} is >50% above entry - unusually high")
                
            # Sanity check: SL shouldn't be too far (>30% loss)
            if validated_sl and validated_sl < current_price * 0.7:
                warnings.append(f"SL {validated_sl} is >30% below entry - consider tighter stop")
                
        elif action == 'sell':
            # SHORT position
            if tp_price is not None and tp_price >= current_price:
                warnings.append(f"TP {tp_price} >= entry {current_price} for SHORT - INVALID, setting to None")
                validated_tp = None
            
            if sl_price is not None and sl_price <= current_price:
                warnings.append(f"SL {sl_price} <= entry {current_price} for SHORT - INVALID, setting to None")
                validated_sl = None
                
            # Sanity check: TP shouldn't be too far (>50% gain)
            if validated_tp and validated_tp < current_price * 0.5:
                warnings.append(f"TP {validated_tp} is >50% below entry - unusually low")
                
            # Sanity check: SL shouldn't be too far (>30% loss)
            if validated_sl and validated_sl > current_price * 1.3:
                warnings.append(f"SL {validated_sl} is >30% above entry - consider tighter stop")
        
        return {
            'tp_price': validated_tp,
            'sl_price': validated_sl,
            'warnings': warnings
        }

    async def _sync_exchange_state(self) -> Optional[Dict[str, Any]]:
        """
        Sync with exchange to get fresh position state.
        
        IMPROVED v3: Returns None on failure instead of raising.
        Caller should handle None appropriately (skip trading).
        """
        self.logger.info("🔄 PRE-DECISION SYNC: Fetching fresh exchange state...")
        
        try:
            await asyncio.sleep(2)  # Rate limit protection - increased for testnet
            state = await self.hyperliquid.get_user_state()
            
            await asyncio.sleep(1)
            open_orders_raw = await self.hyperliquid.get_open_orders()
            
            position_assets = {pos.get('coin') for pos in state['positions'] 
                            if abs(float(pos.get('szi', 0) or 0)) > 0.00001}
            
            # Clean orphan orders
            orphan_orders = [o for o in open_orders_raw if o.get('coin') not in position_assets]
            if orphan_orders:
                self.logger.warning(f"🧹 Found {len(orphan_orders)} orphan orders to cancel")
                for order in orphan_orders:
                    try:
                        oid = order.get('oid')
                        asset = order.get('coin')
                        if oid and asset:
                            await asyncio.sleep(0.3)
                            await self.hyperliquid.cancel_order(asset, oid)
                            self.logger.info(f"🧹 Cancelled orphan order: {asset} oid={oid}")
                    except Exception as e:
                        self.logger.error(f"Failed to cancel orphan order: {e}")
                
                await asyncio.sleep(0.5)
                open_orders_raw = await self.hyperliquid.get_open_orders()
            
            # Enrich positions
            enriched_positions = []
            for pos in state['positions']:
                symbol = pos.get('coin')
                size = float(pos.get('szi', 0) or 0)
                if abs(size) < 0.00001:
                    continue
                    
                try:
                    await asyncio.sleep(0.2)
                    current_price = await self.hyperliquid.get_current_price(symbol)
                    enriched_positions.append({
                        'symbol': symbol,
                        'quantity': size,
                        'entry_price': float(pos.get('entryPx', 0) or 0),
                        'current_price': current_price,
                        'liquidation_price': float(pos.get('liquidationPx', 0) or 0),
                        'unrealized_pnl': pos.get('pnl', 0.0),
                        'leverage': pos.get('leverage', {}).get('value', 1) if isinstance(pos.get('leverage'), dict) else pos.get('leverage', 1)
                    })
                except Exception as e:
                    self.logger.error(f"Error enriching position for {symbol}: {e}")
            
            # Parse open orders
            open_orders = []
            for o in open_orders_raw:
                order_type_obj = o.get('orderType', {})
                trigger_price = None
                order_type_str = 'limit'

                if isinstance(order_type_obj, dict) and 'trigger' in order_type_obj:
                    order_type_str = 'trigger'
                    trigger_data = order_type_obj.get('trigger', {})
                    if 'triggerPx' in trigger_data:
                        trigger_price = float(trigger_data['triggerPx'])

                open_orders.append({
                    'coin': o.get('coin'),
                    'oid': o.get('oid'),
                    'is_buy': o.get('side') == 'B',
                    'size': float(o.get('sz', 0)),
                    'price': float(o.get('limitPx', 0)),
                    'trigger_price': trigger_price,
                    'order_type': order_type_str
                })
            
            self.logger.info(f"🔄 SYNC COMPLETE: {len(enriched_positions)} positions, {len(open_orders)} orders")
            self._consecutive_429s = 0  # Reset on success
            
            return {
                'positions': enriched_positions,
                'positions_raw': state['positions'],
                'open_orders': open_orders,
                'open_orders_raw': open_orders_raw,
                'balance': state['balance'],
                'total_value': state['total_value']
            }
            
        except Exception as e:
            error_str = str(e)
            if '429' in error_str:
                self._consecutive_429s += 1
                self.logger.error(f"❌ SYNC RATE LIMITED (429): {self._consecutive_429s} consecutive failures")
            else:
                self.logger.error(f"❌ SYNC FAILED: {e}")
            return None  # Return None instead of raising

    async def _get_real_position(self, asset: str) -> Optional[Dict]:
        """Get the REAL current position for an asset from the exchange."""
        try:
            await asyncio.sleep(0.3)
            state = await self.hyperliquid.get_user_state()
            for pos in state['positions']:
                if pos.get('coin') == asset:
                    size = float(pos.get('szi', 0) or 0)
                    if abs(size) > 0.00001:
                        return {
                            'side': 'long' if size > 0 else 'short',
                            'size': abs(size),
                            'entry_price': float(pos.get('entryPx', 0) or 0)
                        }
            return None
        except Exception as e:
            self.logger.error(f"Failed to get real position for {asset}: {e}")
            return None

    async def _reconcile_pre_decision(self, positions_raw: List[Dict], open_orders_raw: List[Dict]) -> List[str]:
        """Reconcile active trades with fresh exchange state BEFORE making LLM decision."""
        real_positions = {}
        for pos in positions_raw:
            asset = pos.get('coin')
            size = float(pos.get('szi', 0) or 0)
            if asset and abs(size) > 0.00001:
                real_positions[asset] = {
                    'side': 'long' if size > 0 else 'short',
                    'size': abs(size),
                    'entry_price': float(pos.get('entryPx', 0) or 0)
                }
        
        removed = []
        updated = []
        
        for trade in self.active_trades[:]:
            asset = trade['asset']
            
            if asset not in real_positions:
                self.active_trades.remove(trade)
                removed.append(asset)
            else:
                real = real_positions[asset]
                expected_side = 'long' if trade.get('is_long') else 'short'
                
                if real['side'] != expected_side:
                    self.logger.error(f"⚠️ SIDE MISMATCH {asset}: expected {expected_side}, got {real['side']}")
                    trade['is_long'] = (real['side'] == 'long')
                    trade['amount'] = real['size']
                    trade['entry_price'] = real['entry_price']
                    updated.append(asset)
                elif abs(trade.get('amount', 0) - real['size']) > 0.0001:
                    self.logger.warning(f"⚠️ SIZE MISMATCH {asset}: tracked {trade.get('amount')}, actual {real['size']}")
                    trade['amount'] = real['size']
                    updated.append(asset)

        # ===== NEW: Add positions that exist on exchange but not tracked =====
        for asset, real in real_positions.items():
            tracked = next((t for t in self.active_trades if t['asset'] == asset), None)
            if not tracked:
                # Position exists on exchange but we're not tracking it!
                # Try to find associated TP/SL orders
                tp_oid = None
                sl_oid = None
                tp_price = None
                sl_price = None
                
                for order in open_orders_raw:
                    if order.get('coin') == asset:
                        order_type = order.get('orderType', {})
                        if isinstance(order_type, dict) and 'trigger' in order_type:
                            trigger = order_type.get('trigger', {})
                            tpsl = trigger.get('tpsl')
                            trigger_px = trigger.get('triggerPx')
                            
                            if tpsl == 'tp':
                                tp_oid = order.get('oid')
                                tp_price = float(trigger_px) if trigger_px else None
                            elif tpsl == 'sl':
                                sl_oid = order.get('oid')
                                sl_price = float(trigger_px) if trigger_px else None
                
                tp_info = f"TP@${tp_price:,.2f}" if tp_price else "no TP"
                sl_info = f"SL@${sl_price:,.2f}" if sl_price else "no SL"
                
                self.logger.warning(f"⚠️ UNTRACKED POSITION FOUND: {asset} {real['side']} {real['size']} | {tp_info} | {sl_info}")
                self.active_trades.append({
                    'asset': asset,
                    'is_long': (real['side'] == 'long'),
                    'amount': real['size'],
                    'entry_price': real['entry_price'],
                    'tp_oid': tp_oid,
                    'sl_oid': sl_oid,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'exit_plan': 'Discovered untracked position during reconciliation',
                    'opened_at': datetime.now(UTC).isoformat(),
                    'discovered_untracked': True
                })
                updated.append(asset)

        if removed:
            self.logger.warning(f"⚠️ PRE-DECISION RECONCILE: Positions closed by exchange: {removed}")
            self._write_diary_entry({
                'timestamp': datetime.now(UTC).isoformat(),
                'action': 'reconcile_pre_decision',
                'removed_assets': removed,
                'note': 'Position closed by SL/TP before LLM decision'
            })
        
        if updated:
            self.logger.info(f"📊 PRE-DECISION RECONCILE: Updated tracking for: {updated}")

        return removed

    async def _verify_position_before_trade(self, asset: str, intended_action: str) -> Dict[str, Any]:
        """
        IMPROVED v3: Verify position state BEFORE executing a trade.
        Now also checks for UNTRACKED existing positions.
        """
        real_pos = await self._get_real_position(asset)
        tracked = next((t for t in self.active_trades if t['asset'] == asset), None)
        
        result = {
            'should_execute': True,
            'current_position': real_pos,
            'warning': None,
            'adjust_size': None
        }
        
        # ===== NEW: Check for untracked existing position =====
        if not tracked and real_pos:
            # We're NOT tracking this asset but exchange has a position!
            result['warning'] = f"UNTRACKED position exists: {real_pos['side']} {real_pos['size']}. Adding to tracking instead of executing."
            result['should_execute'] = False
            
            # Add to tracking
            self.active_trades.append({
                'asset': asset,
                'is_long': (real_pos['side'] == 'long'),
                'amount': real_pos['size'],
                'entry_price': real_pos['entry_price'],
                'tp_oid': None,
                'sl_oid': None,
                'exit_plan': 'Discovered during pre-trade verification',
                'opened_at': datetime.now(UTC).isoformat(),
                'discovered_untracked': True
            })
            
            self._write_diary_entry({
                'timestamp': datetime.now(UTC).isoformat(),
                'action': 'discovered_untracked_position',
                'asset': asset,
                'position': real_pos,
                'intended_action': intended_action,
                'note': 'Blocked trade due to untracked existing position'
            })
            
            return result
        
        if tracked and not real_pos:
            result['warning'] = f"Position was closed by TP/SL. Skipping {intended_action}."
            result['should_execute'] = False
            self.active_trades = [t for t in self.active_trades if t['asset'] != asset]
            
            self._write_diary_entry({
                'timestamp': datetime.now(UTC).isoformat(),
                'action': 'trade_skipped_position_closed',
                'asset': asset,
                'intended_action': intended_action,
                'note': 'Position was closed by TP/SL before trade execution'
            })
            
        elif tracked and real_pos:
            tracked_side = 'long' if tracked.get('is_long') else 'short'
            
            if tracked_side != real_pos['side']:
                result['warning'] = f"Side mismatch: tracked {tracked_side}, actual {real_pos['side']}"
                result['should_execute'] = False
                self.logger.error(f"❌ CRITICAL: {asset} side mismatch - aborting trade")
                
            elif abs(tracked.get('amount', 0) - real_pos['size']) > 0.0001:
                result['adjust_size'] = real_pos['size']
                result['warning'] = f"Size adjusted from {tracked.get('amount')} to {real_pos['size']}"
        
        return result

    async def _verify_post_trade(self, asset: str, expected_side: str, expected_size: float) -> bool:
        """Verify trade execution after placing orders."""
        try:
            await asyncio.sleep(1)
            state = await self.hyperliquid.get_user_state()
            
            for pos in state['positions']:
                if pos.get('coin') == asset:
                    actual_size = float(pos.get('szi', 0) or 0)
                    actual_side = 'long' if actual_size > 0 else 'short'
                    
                    size_match = abs(abs(actual_size) - expected_size) < 0.0001
                    side_match = actual_side == expected_side
                    
                    if size_match and side_match:
                        self.logger.info(f"✅ POST-TRADE VERIFY: {asset} position confirmed ({actual_side} {abs(actual_size)})")
                        return True
                    else:
                        self.logger.warning(f"⚠️ POST-TRADE VERIFY: {asset} mismatch - expected {expected_side} {expected_size}, got {actual_side} {abs(actual_size)}")
                        return False
            
            self.logger.warning(f"⚠️ POST-TRADE VERIFY: No position found for {asset}")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ POST-TRADE VERIFY failed: {e}")
            return False

    async def _main_loop(self):
        """Main trading loop - PATCHED v3"""
        try:
            while self.is_running:
                self.invocation_count += 1
                self.state.invocation_count = self.invocation_count

                try:
                    # ===== CHECK FOR TOO MANY RATE LIMIT ERRORS =====
                    if self._consecutive_429s >= self._max_consecutive_429s:
                        self.logger.warning(f"⏸️ Too many rate limit errors ({self._consecutive_429s}). Waiting 60s before retry...")
                        await asyncio.sleep(60)
                        self._consecutive_429s = 0
                        continue

                    # ===== PHASE 1: Fetch Account State =====
                    await asyncio.sleep(2)  # Rate limit protection - increased for testnet
                    state = await self.hyperliquid.get_user_state()
                    balance = state['balance']
                    total_value = state['total_value']

                    initial_balance = self.initial_account_value or total_value
                    if initial_balance > 0:
                        total_return_pct = ((total_value - initial_balance) / initial_balance) * 100
                    else:
                        total_return_pct = 0.0

                    sharpe_ratio = self._calculate_sharpe(self.trade_log)
                    
                    self.logger.debug(f"  Balance: ${balance:,.2f} | Return: {total_return_pct:+.2f}% | Sharpe: {sharpe_ratio:.2f}")

                    self.state.balance = balance
                    self.state.total_value = total_value
                    self.state.total_return_pct = total_return_pct
                    self.state.sharpe_ratio = sharpe_ratio

                    # ===== PHASE 2: Enrich Positions =====
                    enriched_positions = []
                    for pos in state['positions']:
                        symbol = pos.get('coin')
                        size = float(pos.get('szi', 0) or 0)
                        if abs(size) < 0.00001:
                            continue
                        try:
                            await asyncio.sleep(0.2)
                            current_price = await self.hyperliquid.get_current_price(symbol)
                            enriched_positions.append({
                                'symbol': symbol,
                                'quantity': size,
                                'entry_price': float(pos.get('entryPx', 0) or 0),
                                'current_price': current_price,
                                'liquidation_price': float(pos.get('liquidationPx', 0) or 0),
                                'unrealized_pnl': pos.get('pnl', 0.0),
                                'leverage': pos.get('leverage', {}).get('value', 1) if isinstance(pos.get('leverage'), dict) else pos.get('leverage', 1)
                            })
                        except Exception as e:
                            self.logger.error(f"Error enriching position for {symbol}: {e}")

                    self.state.positions = enriched_positions

                    # ===== PHASE 3: Load Recent Diary =====
                    recent_diary = self._load_recent_diary(limit=10)

                    # ===== PHASE 4: Fetch Open Orders =====
                    await asyncio.sleep(1)  # Rate limit protection
                    open_orders_raw = await self.hyperliquid.get_open_orders()
                    open_orders = []
                    for o in open_orders_raw:
                        order_type_obj = o.get('orderType', {})
                        trigger_price = None
                        order_type_str = 'limit'

                        if isinstance(order_type_obj, dict) and 'trigger' in order_type_obj:
                            order_type_str = 'trigger'
                            trigger_data = order_type_obj.get('trigger', {})
                            if 'triggerPx' in trigger_data:
                                trigger_price = float(trigger_data['triggerPx'])

                        open_orders.append({
                            'coin': o.get('coin'),
                            'oid': o.get('oid'),
                            'is_buy': o.get('side') == 'B',
                            'size': float(o.get('sz', 0)),
                            'price': float(o.get('limitPx', 0)),
                            'trigger_price': trigger_price,
                            'order_type': order_type_str
                        })

                    self.state.open_orders = open_orders

                    # ===== PHASE 5: Reconcile Active Trades =====
                    await self._reconcile_active_trades(state['positions'], open_orders_raw)

                    # ===== PHASE 6: Fetch Recent Fills =====
                    await asyncio.sleep(1)  # Rate limit protection
                    fills_raw = await self.hyperliquid.get_recent_fills(limit=50)
                    recent_fills = []
                    for fill in fills_raw[-20:]:
                        ts = fill.get('time')
                        if ts and ts > 1_000_000_000_000:
                            ts = ts / 1000
                        ts_str = datetime.fromtimestamp(ts, UTC).isoformat() if ts else ""

                        recent_fills.append({
                            'timestamp': ts_str,
                            'coin': fill.get('coin'),
                            'is_buy': fill.get('side') == 'B',
                            'size': float(fill.get('sz', 0)),
                            'price': float(fill.get('px', 0))
                        })

                    self.state.recent_fills = recent_fills

                    # ===== PHASE 7: Build Dashboard =====
                    dashboard = {
                        'total_return_pct': total_return_pct,
                        'balance': balance,
                        'account_value': total_value,
                        'sharpe_ratio': sharpe_ratio,
                        'positions': enriched_positions,
                        'active_trades': self.active_trades,
                        'open_orders': open_orders,
                        'recent_diary': recent_diary,
                        'recent_fills': recent_fills
                    }

                    # ===== PHASE 8: Gather Market Data =====
                    market_sections = []
                    for idx, asset in enumerate(self.assets):
                        try:
                            await asyncio.sleep(0.3)
                            current_price = await self.hyperliquid.get_current_price(asset)

                            self.price_history[asset].append({
                                't': datetime.now(UTC).isoformat(),
                                'mid': current_price
                            })

                            await asyncio.sleep(0.2)
                            oi = await self.hyperliquid.get_open_interest(asset)
                            
                            await asyncio.sleep(0.2)
                            funding = await self.hyperliquid.get_funding_rate(asset)

                            indicators = self.taapi.fetch_asset_indicators(asset)
                            
                            if idx < len(self.assets) - 1:
                                self.logger.info(f"Waiting 15s before fetching next asset (TAAPI rate limit)...")
                                await asyncio.sleep(15)
                            
                            ema20_5m_series = indicators["5m"].get("ema20", [])
                            macd_5m_series = indicators["5m"].get("macd", [])
                            rsi7_5m_series = indicators["5m"].get("rsi7", [])
                            rsi14_5m_series = indicators["5m"].get("rsi14", [])

                            interval = CONFIG.get("interval", "1h")
                            lt_indicators = indicators.get(interval, {})
                            lt_ema20 = lt_indicators.get("ema20")
                            lt_ema50 = lt_indicators.get("ema50")
                            lt_atr3 = lt_indicators.get("atr3")
                            lt_atr14 = lt_indicators.get("atr14")
                            lt_macd_series = lt_indicators.get("macd", [])
                            lt_rsi_series = lt_indicators.get("rsi14", [])

                            market_sections.append({
                                "asset": asset,
                                "current_price": current_price,
                                "intraday": {
                                    "ema20": ema20_5m_series[-1] if ema20_5m_series else None,
                                    "macd": macd_5m_series[-1] if macd_5m_series else None,
                                    "rsi7": rsi7_5m_series[-1] if rsi7_5m_series else None,
                                    "rsi14": rsi14_5m_series[-1] if rsi14_5m_series else None,
                                    "series": {
                                        "ema20": ema20_5m_series,
                                        "macd": macd_5m_series,
                                        "rsi7": rsi7_5m_series,
                                        "rsi14": rsi14_5m_series
                                    }
                                },
                                "long_term": {
                                    "ema20": lt_ema20,
                                    "ema50": lt_ema50,
                                    "atr3": lt_atr3,
                                    "atr14": lt_atr14,
                                    "macd_series": lt_macd_series,
                                    "rsi_series": lt_rsi_series
                                },
                                "open_interest": oi,
                                "funding_rate": funding,
                                "funding_annualized_pct": funding * 24 * 365 * 100 if funding else None,
                                "recent_mid_prices": [p['mid'] for p in list(self.price_history[asset])[-10:]]
                            })

                        except Exception as e:
                            self.logger.error(f"Error gathering market data for {asset}: {e}")

                    # ===== PHASE 8.5: PRE-DECISION SYNC =====
                    # CRITICAL: If sync fails, SKIP trading this iteration
                    sync_state = await self._sync_exchange_state()
                    
                    if sync_state is None:
                        self.logger.error("❌ PRE-DECISION SYNC FAILED - SKIPPING TRADING THIS ITERATION")
                        self.state.error = "Exchange sync failed - skipping trades"
                        self._notify_state_update()
                        await asyncio.sleep(self._get_interval_seconds())
                        continue
                    
                    # Update with fresh data
                    dashboard['positions'] = sync_state['positions']
                    dashboard['account_value'] = sync_state['total_value']
                    dashboard['balance'] = sync_state['balance']
                    dashboard['open_orders'] = sync_state['open_orders']
                    
                    # Reconcile
                    closed_by_exchange = await self._reconcile_pre_decision(
                        sync_state['positions_raw'], 
                        sync_state['open_orders_raw']
                    )
                    
                    dashboard['active_trades'] = self.active_trades
                    
                    for section in market_sections:
                        asset = section['asset']
                        for pos in sync_state['positions']:
                            if pos['symbol'] == asset:
                                section['current_price'] = pos['current_price']
                                break
                    
                    self.state.positions = sync_state['positions']
                    self.state.balance = sync_state['balance']
                    self.state.total_value = sync_state['total_value']
                    self.state.open_orders = sync_state['open_orders']
                    self.state.error = None  # Clear error on successful sync

                    # ===== PHASE 8.7: Enhanced Analysis =====
                    self.logger.info("Building enhanced market analysis...")
                    for section in market_sections:
                        asset = section["asset"]
                        current_price = section["current_price"]
                        
                        try:
                            await asyncio.sleep(0.3)
                            ohlcv_data = await self.hyperliquid.get_candles(
                                asset=asset, 
                                interval=self.interval,
                                limit=30
                            )
                            
                            if ohlcv_data and len(ohlcv_data) >= 6:
                                recent = ohlcv_data[-6:]
                                daily_ohlc = {
                                    "high": max(c["high"] for c in recent),
                                    "low": min(c["low"] for c in recent),
                                    "close": recent[-1]["close"]
                                }
                            else:
                                daily_ohlc = {
                                    "high": current_price * 1.02,
                                    "low": current_price * 0.98,
                                    "close": current_price
                                }
                            
                            enhanced = self.enhanced_context.build_context(
                                asset=asset,
                                current_price=current_price,
                                ohlcv_data=ohlcv_data,
                                daily_ohlc=daily_ohlc,
                                timeframe=self.interval
                            )
                            
                            section["enhanced_analysis"] = enhanced["prompt_text"]
                            section["composite_signal"] = enhanced["composite_signal"]
                            section["composite_confidence"] = enhanced["confidence"]
                            
                            self.logger.info(
                                f"Enhanced {asset}: {enhanced['composite_signal']} "
                                f"(conf: {enhanced['confidence']}%)"
                            )
                            
                        except Exception as e:
                            self.logger.warning(f"Enhanced analysis failed for {asset}: {e}")
                            section["enhanced_analysis"] = "Enhanced analysis unavailable"
                            section["composite_signal"] = "NEUTRAL"
                            section["composite_confidence"] = 50

                    # ===== PHASE 9: Build LLM Context =====
                    context_payload = OrderedDict([
                        ("invocation", {
                            "count": self.invocation_count,
                            "current_time": datetime.now(UTC).isoformat()
                        }),
                        ("account", dashboard),
                        ("market_data", market_sections),
                        ("instructions", {
                            "assets": self.assets,
                            "note": "Follow the system prompt guidelines strictly"
                        })
                    ])
                    context = json.dumps(context_payload, default=json_default, indent=2)

                    with open("data/prompts.log", "a", encoding="utf-8") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Invocation {self.invocation_count} - {datetime.now(UTC).isoformat()}\n")
                        f.write(f"{'='*80}\n")
                        f.write(context + "\n")

                    # ===== PHASE 10: Get LLM Decision =====
                    decisions = await asyncio.to_thread(
                        self.agent.decide_trade, self.assets, context
                    )

                    if not isinstance(decisions, dict) or 'trade_decisions' not in decisions:
                        self.logger.warning("Invalid decision format, retrying with strict prefix...")
                        strict_context = (
                            "Return ONLY the JSON object per the schema. "
                            "No markdown, no explanation.\n\n" + context
                        )
                        decisions = await asyncio.to_thread(
                            self.agent.decide_trade, self.assets, strict_context
                        )

                    trade_decisions = decisions.get('trade_decisions', [])
                    if all(
                        d.get('action') == 'hold' and 'parse error' in d.get('rationale', '').lower()
                        for d in trade_decisions
                    ):
                        self.logger.warning("All holds with parse errors, retrying...")
                        decisions = await asyncio.to_thread(
                            self.agent.decide_trade, self.assets, context
                        )
                        trade_decisions = decisions.get('trade_decisions', [])

                    reasoning = decisions.get('reasoning', '')
                    if reasoning:
                        self.logger.info(f"LLM Reasoning: {reasoning[:200]}...")

                    self.state.last_reasoning = decisions

                    # ===== PHASE 11: Execute Trades =====
                    for decision in trade_decisions:
                        asset = decision.get('asset')
                        if asset not in self.assets:
                            continue

                        action = decision.get('action')
                        rationale = decision.get('rationale', '')
                        if not rationale:
                            rationale = decision.get('exit_plan', '') or (reasoning[:200] if reasoning else '')
                        allocation = float(decision.get('allocation_usd') or 0)
                        tp_price = decision.get('tp_price')
                        sl_price = decision.get('sl_price')
                        exit_plan = decision.get('exit_plan', '')
                        confidence = decision.get('confidence', 75.0)

                        if action in ['buy', 'sell']:
                            # MANUAL MODE
                            if self.trading_mode == "manual":
                                try:
                                    await asyncio.sleep(0.2)
                                    current_price = await self.hyperliquid.get_current_price(asset)
                                    size = allocation / current_price if current_price > 0 else 0
                                    
                                    # Validate TP/SL
                                    validation = self._validate_tp_sl(action, current_price, tp_price, sl_price)
                                    for warning in validation['warnings']:
                                        self.logger.warning(f"⚠️ TP/SL VALIDATION: {warning}")
                                    
                                    risk_reward = None
                                    if validation['tp_price'] and validation['sl_price'] and current_price:
                                        potential_gain = abs(validation['tp_price'] - current_price) / current_price
                                        potential_loss = abs(validation['sl_price'] - current_price) / current_price
                                        if potential_loss > 0:
                                            risk_reward = potential_gain / potential_loss
                                    
                                    proposal = TradeProposal(
                                        asset=asset,
                                        action=action,
                                        confidence=confidence,
                                        risk_reward=risk_reward,
                                        entry_price=current_price,
                                        tp_price=validation['tp_price'],
                                        sl_price=validation['sl_price'],
                                        size=size,
                                        allocation=allocation,
                                        rationale=rationale,
                                        market_conditions={
                                            'current_price': current_price,
                                            'exit_plan': exit_plan,
                                            'tp_sl_warnings': validation['warnings']
                                        }
                                    )
                                    
                                    self.pending_proposals.append(proposal)
                                    self.logger.info(f"[PROPOSAL] Created: {action.upper()} {asset} @ ${current_price:,.2f} (ID: {proposal.id[:8]})")
                                    
                                    self.state.pending_proposals = [p.to_dict() for p in self.pending_proposals if p.is_pending]
                                    
                                except Exception as e:
                                    self.logger.error(f"Error creating proposal for {asset}: {e}")
                                    
                                continue

                            # AUTO MODE
                            try:
                                # PRE-TRADE VERIFICATION
                                pre_check = await self._verify_position_before_trade(asset, action)
                                
                                if not pre_check['should_execute']:
                                    self.logger.warning(f"⏭️ SKIPPING {action} {asset}: {pre_check['warning']}")
                                    continue
                                
                                if pre_check['warning']:
                                    self.logger.warning(f"⚠️ {pre_check['warning']}")
                                
                                await asyncio.sleep(0.3)
                                current_price = await self.hyperliquid.get_current_price(asset)
                                amount = allocation / current_price if current_price > 0 else 0
                                
                                if pre_check.get('adjust_size') and pre_check['current_position']:
                                    real_pos = pre_check['current_position']
                                    if (action == 'buy' and real_pos['side'] == 'short') or \
                                       (action == 'sell' and real_pos['side'] == 'long'):
                                        amount = real_pos['size']
                                        self.logger.info(f"📐 Using actual position size for close: {amount}")

                                # ===== VALIDATE TP/SL =====
                                validation = self._validate_tp_sl(action, current_price, tp_price, sl_price)
                                for warning in validation['warnings']:
                                    self.logger.warning(f"⚠️ TP/SL VALIDATION: {warning}")
                                tp_price = validation['tp_price']
                                sl_price = validation['sl_price']

                                if amount > 0:
                                    await asyncio.sleep(0.3)
                                    if action == 'buy':
                                        order_result = await self.hyperliquid.place_buy_order(asset, amount)
                                    else:
                                        order_result = await self.hyperliquid.place_sell_order(asset, amount)

                                    self.logger.info(f"Executed {action} {asset}: {amount:.6f} @ {current_price}")

                                    # POST-TRADE VERIFICATION
                                    expected_side = 'long' if action == 'buy' else 'short'
                                    verified = await self._verify_post_trade(asset, expected_side, amount)
                                    
                                    if not verified:
                                        self.logger.warning(f"⚠️ Trade verification failed for {asset} - checking state")
                                        real_pos = await self._get_real_position(asset)
                                        if real_pos:
                                            amount = real_pos['size']
                                            expected_side = real_pos['side']
                                        else:
                                            self.logger.error(f"❌ No position found after trade for {asset}")
                                            continue

                                    await asyncio.sleep(1)
                                    recent_fills_check = await self.hyperliquid.get_recent_fills(limit=5)
                                    filled = any(
                                        f.get('coin') == asset and
                                        abs(float(f.get('sz', 0)) - amount) < 0.0001
                                        for f in recent_fills_check
                                    )

                                    # CANCEL EXISTING ORDERS
                                    try:
                                        existing_orders = await self.hyperliquid.get_open_orders()
                                        for order in existing_orders:
                                            if order.get('coin') == asset:
                                                oid = order.get('oid')
                                                if oid:
                                                    await asyncio.sleep(0.3)
                                                    await self.hyperliquid.cancel_order(asset, oid)
                                                    self.logger.info(f"🧹 Cancelled existing order for {asset} before placing new TP/SL")
                                    except Exception as e:
                                        self.logger.warning(f"Failed to cancel existing orders: {e}")

                                    tp_oid = None
                                    sl_oid = None

                                    if tp_price:
                                        try:
                                            is_buy = (action == 'buy')
                                            await asyncio.sleep(0.3)
                                            tp_result = await self.hyperliquid.place_take_profit(
                                                asset, is_buy, amount, tp_price
                                            )
                                            # V4: Check structured response
                                            if isinstance(tp_result, dict) and tp_result.get('success'):
                                                tp_oid = tp_result.get('oid')
                                                self.logger.info(f"✅ Placed TP order for {asset} @ {tp_price} (oid: {tp_oid})")
                                            elif isinstance(tp_result, dict):
                                                self.logger.error(f"❌ TP order REJECTED for {asset}: {tp_result.get('error', 'unknown')}")
                                                tp_oid = None
                                            else:
                                                # Fallback for old API format
                                                oids = self.hyperliquid.extract_oids(tp_result)
                                                tp_oid = oids[0] if oids else None
                                                if tp_oid:
                                                    self.logger.info(f"✅ Placed TP order for {asset} @ {tp_price}")
                                                else:
                                                    self.logger.warning(f"⚠️ TP order for {asset} - no OID returned")
                                        except Exception as e:
                                            self.logger.error(f"Failed to place TP: {e}")

                                    if sl_price:
                                        try:
                                            is_buy = (action == 'buy')
                                            await asyncio.sleep(0.3)
                                            sl_result = await self.hyperliquid.place_stop_loss(
                                                asset, is_buy, amount, sl_price
                                            )
                                            # V4: Check structured response
                                            if isinstance(sl_result, dict) and sl_result.get('success'):
                                                sl_oid = sl_result.get('oid')
                                                self.logger.info(f"✅ Placed SL order for {asset} @ {sl_price} (oid: {sl_oid})")
                                            elif isinstance(sl_result, dict):
                                                self.logger.error(f"❌ SL order REJECTED for {asset}: {sl_result.get('error', 'unknown')}")
                                                sl_oid = None
                                            else:
                                                # Fallback for old API format
                                                oids = self.hyperliquid.extract_oids(sl_result)
                                                sl_oid = oids[0] if oids else None
                                                if sl_oid:
                                                    self.logger.info(f"✅ Placed SL order for {asset} @ {sl_price}")
                                                else:
                                                    self.logger.warning(f"⚠️ SL order for {asset} - no OID returned")
                                        except Exception as e:
                                            self.logger.error(f"Failed to place SL: {e}")

                                    self.active_trades = [
                                        t for t in self.active_trades if t['asset'] != asset
                                    ]
                                    self.active_trades.append({
                                        'asset': asset,
                                        'is_long': (action == 'buy'),
                                        'amount': amount,
                                        'entry_price': current_price,
                                        'tp_oid': tp_oid,
                                        'sl_oid': sl_oid,
                                        'exit_plan': exit_plan,
                                        'opened_at': datetime.now(UTC).isoformat()
                                    })

                                    self._write_diary_entry({
                                        'timestamp': datetime.now(UTC).isoformat(),
                                        'asset': asset,
                                        'action': action,
                                        'allocation_usd': allocation,
                                        'amount': amount,
                                        'size': amount,
                                        'entry_price': current_price,
                                        'tp_price': tp_price,
                                        'tp_oid': tp_oid,
                                        'sl_price': sl_price,
                                        'sl_oid': sl_oid,
                                        'exit_plan': exit_plan,
                                        'rationale': rationale,
                                        'order_result': str(order_result),
                                        'opened_at': datetime.now(UTC).isoformat(),
                                        'filled': filled
                                    })
                                    
                                    # POST-TRADE SYNC
                                    try:
                                        await asyncio.sleep(0.5)
                                        post_trade_state = await self.hyperliquid.get_user_state()
                                        post_positions = []
                                        for pos in post_trade_state['positions']:
                                            symbol = pos.get('coin')
                                            size = float(pos.get('szi', 0) or 0)
                                            if abs(size) < 0.00001:
                                                continue
                                            try:
                                                await asyncio.sleep(0.2)
                                                curr_price = await self.hyperliquid.get_current_price(symbol)
                                                post_positions.append({
                                                    'symbol': symbol,
                                                    'quantity': size,
                                                    'entry_price': float(pos.get('entryPx', 0) or 0),
                                                    'current_price': curr_price,
                                                    'liquidation_price': float(pos.get('liquidationPx', 0) or 0),
                                                    'unrealized_pnl': pos.get('pnl', 0.0),
                                                    'leverage': pos.get('leverage', {}).get('value', 1) if isinstance(pos.get('leverage'), dict) else pos.get('leverage', 1)
                                                })
                                            except Exception:
                                                pass
                                        self.state.positions = post_positions
                                        self.state.balance = post_trade_state['balance']
                                        self.state.total_value = post_trade_state['total_value']
                                        self.logger.info(f"📊 POST-TRADE SYNC: Updated state with {len(post_positions)} positions")
                                    except Exception as e:
                                        self.logger.warning(f"Post-trade state sync failed: {e}")

                                    if self.on_trade_executed:
                                        self.on_trade_executed({
                                            'asset': asset,
                                            'action': action,
                                            'amount': amount,
                                            'price': current_price,
                                            'timestamp': datetime.now(UTC).isoformat()
                                        })

                            except Exception as e:
                                self.logger.error(f"Error executing {action} for {asset}: {e}")
                                if self.on_error:
                                    self.on_error(f"Trade execution error: {e}")

                        elif action == 'hold':
                            self.logger.info(f"{asset}: HOLD - {rationale}")
                            
                            # ===== NEW: Set/Update TP/SL on HOLD if position exists =====
                            tracked_trade = next((t for t in self.active_trades if t['asset'] == asset), None)
                            
                            if tracked_trade and (tp_price or sl_price):
                                try:
                                    await asyncio.sleep(0.3)
                                    current_price = await self.hyperliquid.get_current_price(asset)
                                    is_long = tracked_trade.get('is_long', True)
                                    amount = tracked_trade.get('amount', 0)
                                    
                                    # Validate TP/SL
                                    action_for_validation = 'buy' if is_long else 'sell'
                                    validation = self._validate_tp_sl(action_for_validation, current_price, tp_price, sl_price)
                                    for warning in validation['warnings']:
                                        self.logger.warning(f"⚠️ TP/SL VALIDATION (HOLD): {warning}")
                                    
                                    validated_tp = validation['tp_price']
                                    validated_sl = validation['sl_price']
                                    
                                    if validated_tp or validated_sl:
                                        self.logger.info(f"📝 HOLD with TP/SL update for {asset}: TP={validated_tp}, SL={validated_sl}")
                                        
                                        # Cancel existing TP/SL orders first
                                        try:
                                            existing_orders = await self.hyperliquid.get_open_orders()
                                            for order in existing_orders:
                                                if order.get('coin') == asset:
                                                    oid = order.get('oid')
                                                    if oid:
                                                        await asyncio.sleep(0.3)
                                                        await self.hyperliquid.cancel_order(asset, oid)
                                                        self.logger.info(f"🧹 Cancelled existing order for {asset}")
                                        except Exception as e:
                                            self.logger.warning(f"Failed to cancel existing orders: {e}")
                                        
                                        # Place new TP if valid
                                        tp_oid = None
                                        if validated_tp:
                                            try:
                                                await asyncio.sleep(0.3)
                                                tp_result = await self.hyperliquid.place_take_profit(
                                                    asset, is_long, amount, validated_tp
                                                )
                                                # V4: Check structured response
                                                if isinstance(tp_result, dict) and tp_result.get('success'):
                                                    tp_oid = tp_result.get('oid')
                                                    self.logger.info(f"✅ Placed TP order for {asset} @ {validated_tp} (oid: {tp_oid})")
                                                elif isinstance(tp_result, dict):
                                                    self.logger.error(f"❌ TP order REJECTED for {asset}: {tp_result.get('error', 'unknown')}")
                                                    tp_oid = None
                                                else:
                                                    # Fallback for old API format
                                                    oids = self.hyperliquid.extract_oids(tp_result)
                                                    tp_oid = oids[0] if oids else None
                                                    if tp_oid:
                                                        self.logger.info(f"✅ Placed TP order for {asset} @ {validated_tp}")
                                                    else:
                                                        self.logger.warning(f"⚠️ TP order for {asset} - no OID returned")
                                            except Exception as e:
                                                self.logger.error(f"Failed to place TP: {e}")
                                        
                                        # Place new SL if valid
                                        sl_oid = None
                                        if validated_sl:
                                            try:
                                                await asyncio.sleep(0.3)
                                                sl_result = await self.hyperliquid.place_stop_loss(
                                                    asset, is_long, amount, validated_sl
                                                )
                                                # V4: Check structured response
                                                if isinstance(sl_result, dict) and sl_result.get('success'):
                                                    sl_oid = sl_result.get('oid')
                                                    self.logger.info(f"✅ Placed SL order for {asset} @ {validated_sl} (oid: {sl_oid})")
                                                elif isinstance(sl_result, dict):
                                                    self.logger.error(f"❌ SL order REJECTED for {asset}: {sl_result.get('error', 'unknown')}")
                                                    sl_oid = None
                                                else:
                                                    # Fallback for old API format
                                                    oids = self.hyperliquid.extract_oids(sl_result)
                                                    sl_oid = oids[0] if oids else None
                                                    if sl_oid:
                                                        self.logger.info(f"✅ Placed SL order for {asset} @ {validated_sl}")
                                                    else:
                                                        self.logger.warning(f"⚠️ SL order for {asset} - no OID returned")
                                            except Exception as e:
                                                self.logger.error(f"Failed to place SL: {e}")
                                        
                                        # Update tracked trade
                                        if tp_oid:
                                            tracked_trade['tp_oid'] = tp_oid
                                            tracked_trade['tp_price'] = validated_tp
                                        if sl_oid:
                                            tracked_trade['sl_oid'] = sl_oid
                                            tracked_trade['sl_price'] = validated_sl
                                        
                                        self._write_diary_entry({
                                            'timestamp': datetime.now(UTC).isoformat(),
                                            'asset': asset,
                                            'action': 'hold_update_tpsl',
                                            'tp_price': validated_tp,
                                            'tp_oid': tp_oid,
                                            'sl_price': validated_sl,
                                            'sl_oid': sl_oid,
                                            'rationale': rationale
                                        })
                                        
                                except Exception as e:
                                    self.logger.error(f"Error updating TP/SL on HOLD for {asset}: {e}")
                            else:
                                self._write_diary_entry({
                                    'timestamp': datetime.now(UTC).isoformat(),
                                    'asset': asset,
                                    'action': 'hold',
                                    'rationale': rationale
                                })

                    self.state.market_data = market_sections
                    self.state.last_update = datetime.now(UTC).isoformat()
                    self._notify_state_update()

                except Exception as e:
                    self.logger.error(f"Error in main loop iteration: {e}", exc_info=True)
                    self.state.error = str(e)
                    if self.on_error:
                        self.on_error(str(e))

                await asyncio.sleep(self._get_interval_seconds())

        except asyncio.CancelledError:
            self.logger.info("Bot loop cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in bot loop: {e}", exc_info=True)
            self.state.error = str(e)
            if self.on_error:
                self.on_error(str(e))

    async def _reconcile_active_trades(self, positions: List[Dict], open_orders: List[Dict]):
        """Reconcile local active_trades with exchange state."""
        exchange_assets = {pos.get('coin') for pos in positions if abs(float(pos.get('szi', 0) or 0)) > 0.00001}
        order_assets = {o.get('coin') for o in open_orders}
        tracked_assets = exchange_assets | order_assets

        removed = []
        for trade in self.active_trades[:]:
            if trade['asset'] not in tracked_assets:
                self.active_trades.remove(trade)
                removed.append(trade['asset'])

        if removed:
            self.logger.info(f"Reconciled: removed stale trades for {removed}")
            self._write_diary_entry({
                'timestamp': datetime.now(UTC).isoformat(),
                'action': 'reconcile',
                'removed_assets': removed,
                'note': 'Position no longer exists on exchange'
            })

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate naive Sharpe ratio from returns list"""
        if len(returns) < 2:
            return 0.0

        try:
            import statistics
            mean = statistics.mean(returns)
            stdev = statistics.stdev(returns)
            return mean / stdev if stdev > 0 else 0.0
        except Exception:
            return 0.0

    def _get_interval_seconds(self) -> int:
        """Convert interval string to seconds"""
        if self.interval.endswith('m'):
            return int(self.interval[:-1]) * 60
        elif self.interval.endswith('h'):
            return int(self.interval[:-1]) * 3600
        elif self.interval.endswith('d'):
            return int(self.interval[:-1]) * 86400
        return 300

    def _notify_state_update(self):
        """Notify GUI of state update via callback"""
        if self.on_state_update:
            try:
                self.on_state_update(self.state)
            except Exception as e:
                self.logger.error(f"Error in state update callback: {e}")

    def _write_diary_entry(self, entry: Dict):
        """Write entry to diary.jsonl"""
        try:
            with open(self.diary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=json_default) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write diary entry: {e}")

    def _load_recent_diary(self, limit: int = 10) -> List[Dict]:
        """Load recent diary entries"""
        if not self.diary_path.exists():
            return []

        try:
            entries = []
            with open(self.diary_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return entries[-limit:]
        except Exception as e:
            self.logger.error(f"Failed to load diary: {e}")
            return []

    def get_state(self) -> BotState:
        """Get current bot state"""
        return self.state

    def get_assets(self) -> List[str]:
        """Get configured assets"""
        return self.assets

    def get_interval(self) -> str:
        """Get configured interval"""
        return self.interval

    async def close_position(self, asset: str) -> bool:
        """Manually close a position for given asset."""
        try:
            await self.hyperliquid.cancel_all_orders(asset)

            for pos in self.state.positions:
                if pos['symbol'] == asset:
                    quantity = abs(pos['quantity'])
                    if quantity > 0:
                        if pos['quantity'] > 0:
                            await self.hyperliquid.place_sell_order(asset, quantity)
                        else:
                            await self.hyperliquid.place_buy_order(asset, quantity)

                        self.active_trades = [
                            t for t in self.active_trades if t['asset'] != asset
                        ]

                        self._write_diary_entry({
                            'timestamp': datetime.now(UTC).isoformat(),
                            'asset': asset,
                            'action': 'manual_close',
                            'quantity': quantity,
                            'note': 'Position closed manually via GUI'
                        })

                        self.logger.info(f"Manually closed position: {asset}")
                        return True

            self.logger.warning(f"No position found to close: {asset}")
            return False

        except Exception as e:
            self.logger.error(f"Failed to close position {asset}: {e}")
            if self.on_error:
                self.on_error(f"Failed to close position: {e}")
            return False
    
    def get_pending_proposals(self) -> List[TradeProposal]:
        """Get list of pending trade proposals"""
        return [p for p in self.pending_proposals if p.is_pending]
    
    def approve_proposal(self, proposal_id: str) -> bool:
        """Approve and execute a trade proposal."""
        proposal = next((p for p in self.pending_proposals if p.id == proposal_id), None)
        
        if not proposal or not proposal.is_pending:
            self.logger.warning(f"Proposal {proposal_id} not found or not pending")
            return False
        
        proposal.approve()
        self.logger.info(f"[APPROVED] Proposal: {proposal.action.upper()} {proposal.asset} (ID: {proposal_id[:8]})")
        
        asyncio.create_task(self._execute_proposal(proposal))
        
        self.state.pending_proposals = [p.to_dict() for p in self.pending_proposals if p.is_pending]
        self._notify_state_update()
        
        return True
    
    def reject_proposal(self, proposal_id: str, reason: Optional[str] = None) -> bool:
        """Reject a trade proposal."""
        proposal = next((p for p in self.pending_proposals if p.id == proposal_id), None)
        
        if not proposal or not proposal.is_pending:
            self.logger.warning(f"Proposal {proposal_id} not found or not pending")
            return False
        
        proposal.reject(reason or "Rejected by user")
        self.logger.info(f"[REJECTED] Proposal: {proposal.action.upper()} {proposal.asset} (ID: {proposal_id[:8]})")
        
        self._write_diary_entry({
            'timestamp': datetime.now(UTC).isoformat(),
            'asset': proposal.asset,
            'action': 'proposal_rejected',
            'proposal_id': proposal_id,
            'reason': reason,
            'rationale': proposal.rationale
        })
        
        self.state.pending_proposals = [p.to_dict() for p in self.pending_proposals if p.is_pending]
        self._notify_state_update()
        
        return True
    
    async def _execute_proposal(self, proposal: TradeProposal):
        """Execute an approved trade proposal."""
        try:
            self.logger.info(f"Executing proposal: {proposal.action.upper()} {proposal.asset}")
            
            pre_check = await self._verify_position_before_trade(proposal.asset, proposal.action)
            
            if not pre_check['should_execute']:
                self.logger.warning(f"⏭️ SKIPPING proposal {proposal.id[:8]}: {pre_check['warning']}")
                proposal.mark_failed(pre_check['warning'])
                return
            
            await asyncio.sleep(0.3)
            current_price = await self.hyperliquid.get_current_price(proposal.asset)
            amount = proposal.size
            
            if amount <= 0:
                raise ValueError(f"Invalid amount: {amount}")
            
            # Validate TP/SL
            validation = self._validate_tp_sl(proposal.action, current_price, proposal.tp_price, proposal.sl_price)
            for warning in validation['warnings']:
                self.logger.warning(f"⚠️ TP/SL VALIDATION: {warning}")
            
            await asyncio.sleep(0.3)
            if proposal.action == 'buy':
                order_result = await self.hyperliquid.place_buy_order(proposal.asset, amount)
            elif proposal.action == 'sell':
                order_result = await self.hyperliquid.place_sell_order(proposal.asset, amount)
            else:
                raise ValueError(f"Invalid action: {proposal.action}")
            
            self.logger.info(f"Order placed: {proposal.action} {proposal.asset}: {amount:.6f} @ {current_price}")
            
            expected_side = 'long' if proposal.action == 'buy' else 'short'
            await self._verify_post_trade(proposal.asset, expected_side, amount)
            
            await asyncio.sleep(1)
            recent_fills = await self.hyperliquid.get_recent_fills(limit=5)
            filled = any(
                f.get('coin') == proposal.asset and
                abs(float(f.get('sz', 0)) - amount) < 0.0001
                for f in recent_fills
            )
            
            try:
                existing_orders = await self.hyperliquid.get_open_orders()
                for order in existing_orders:
                    if order.get('coin') == proposal.asset:
                        oid = order.get('oid')
                        if oid:
                            await asyncio.sleep(0.3)
                            await self.hyperliquid.cancel_order(proposal.asset, oid)
            except Exception as e:
                self.logger.warning(f"Failed to cancel existing orders: {e}")
            
            tp_oid = None
            sl_oid = None
            
            if validation['tp_price']:
                try:
                    is_buy = (proposal.action == 'buy')
                    await asyncio.sleep(0.3)
                    tp_result = await self.hyperliquid.place_take_profit(
                        proposal.asset, is_buy, amount, validation['tp_price']
                    )
                    # V4: Check structured response
                    if isinstance(tp_result, dict) and tp_result.get('success'):
                        tp_oid = tp_result.get('oid')
                        self.logger.info(f"✅ Placed TP order @ {validation['tp_price']} (oid: {tp_oid})")
                    elif isinstance(tp_result, dict):
                        self.logger.error(f"❌ TP order REJECTED: {tp_result.get('error', 'unknown')}")
                        tp_oid = None
                    else:
                        # Fallback for old API format
                        oids = self.hyperliquid.extract_oids(tp_result)
                        tp_oid = oids[0] if oids else None
                        if tp_oid:
                            self.logger.info(f"✅ Placed TP order @ {validation['tp_price']}")
                        else:
                            self.logger.warning(f"⚠️ TP order - no OID returned")
                except Exception as e:
                    self.logger.error(f"Failed to place TP: {e}")
            
            if validation['sl_price']:
                try:
                    is_buy = (proposal.action == 'buy')
                    await asyncio.sleep(0.3)
                    sl_result = await self.hyperliquid.place_stop_loss(
                        proposal.asset, is_buy, amount, validation['sl_price']
                    )
                    # V4: Check structured response
                    if isinstance(sl_result, dict) and sl_result.get('success'):
                        sl_oid = sl_result.get('oid')
                        self.logger.info(f"✅ Placed SL order @ {validation['sl_price']} (oid: {sl_oid})")
                    elif isinstance(sl_result, dict):
                        self.logger.error(f"❌ SL order REJECTED: {sl_result.get('error', 'unknown')}")
                        sl_oid = None
                    else:
                        # Fallback for old API format
                        oids = self.hyperliquid.extract_oids(sl_result)
                        sl_oid = oids[0] if oids else None
                        if sl_oid:
                            self.logger.info(f"✅ Placed SL order @ {validation['sl_price']}")
                        else:
                            self.logger.warning(f"⚠️ SL order - no OID returned")
                except Exception as e:
                    self.logger.error(f"Failed to place SL: {e}")
            
            self.active_trades = [
                t for t in self.active_trades if t['asset'] != proposal.asset
            ]
            self.active_trades.append({
                'asset': proposal.asset,
                'is_long': (proposal.action == 'buy'),
                'amount': amount,
                'entry_price': current_price,
                'tp_oid': tp_oid,
                'sl_oid': sl_oid,
                'exit_plan': proposal.market_conditions.get('exit_plan', ''),
                'opened_at': datetime.now(UTC).isoformat(),
                'from_proposal': proposal.id
            })
            
            proposal.mark_executed(current_price)
            
            self._write_diary_entry({
                'timestamp': datetime.now(UTC).isoformat(),
                'asset': proposal.asset,
                'action': proposal.action,
                'allocation_usd': proposal.allocation,
                'amount': amount,
                'entry_price': current_price,
                'tp_price': validation['tp_price'],
                'tp_oid': tp_oid,
                'sl_price': validation['sl_price'],
                'sl_oid': sl_oid,
                'rationale': proposal.rationale,
                'order_result': str(order_result),
                'filled': filled,
                'from_proposal': proposal.id,
                'approved_manually': True
            })
            
            if self.on_trade_executed:
                self.on_trade_executed({
                    'asset': proposal.asset,
                    'action': proposal.action,
                    'amount': amount,
                    'price': current_price,
                    'timestamp': datetime.now(UTC).isoformat(),
                    'from_proposal': True
                })
            
            self.logger.info(f"[SUCCESS] Proposal executed: {proposal.id[:8]}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute proposal {proposal.id}: {e}")
            proposal.mark_failed(str(e))
            
            if self.on_error:
                self.on_error(f"Failed to execute trade: {e}")
        
        finally:
            self.state.pending_proposals = [p.to_dict() for p in self.pending_proposals if p.is_pending]
            self._notify_state_update()
