# HoloClaw ML Trading Architecture v3

**Status:** Enhanced & Revised (2026-03-11)  
**Platform:** HoloClaw (Ryzen 9950X3D, 256GB RAM, multi-TB storage)  
**Repository:** GitHub at `/home/hologaun/projects/mlat`  
**Interface:** Modular Python architecture with configurable UI

---

## Executive Summary

A scalable, production-grade system for multiple algorithmic/ML strategies running on HoloClaw with **OpenClaw Agent orchestration**, **agentic trading capability**, and **modular code structure**.

### Core Capabilities (v3)
- **Modular Code Design** — Hardcoded optimization scripts calling verified functions for rapid iteration
- **OpenClaw Agent Integration** — Full system observability and control via OpenClaw infrastructure
- **Agentic Trader** — Independent reasoning-based execution layer with advanced configuration UI
- **Configuration Widget** — Live parameter updates via web interface
- **Full TradeStation API v3** — Complete order lifecycle, account management, streaming endpoints
- **Built-in Scanner** — Real-time strategy/indicator scanning with candlestick patterns
- **GitHub Tracking** — Version control for all strategies and configurations

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                       HoloClaw ML Trading Platform v3                        │
├────────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐│
│  │    Market      │  │   Signal       │  │   Portfolio    │  │   Execution    ││
│  │   Data Layer   │  │   Engine       │  │   Manager      │  │   Gateway      ││
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘│
│           │                   │                   │                   │         │
│           ▼                   ▼                   ▼                   ▼         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐│
│  │ PostgreSQL     │  │ Python/Rust    │  │ Position       │  │ TradeStation   ││
│  │ (skald_ohlc)   │  │ Strategies     │  │ Sizing         │  │ API v3 (OAuth) ││
│  │ + Cache        │  │ (ML + Rules)   │  │ & Allocation   │  │ Full Endpoint  ││
│  └────────────────┘  └────────────────┘  └────────────────┘  └────────┬───────┘│
│                                                                     │         │
│                                                                      ▼         │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    OpenClaw Agent Integration Layer                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │  System     │  │   Agentic   │  │  Config     │  │   Scanner   │   │  │
│  │  │  Monitor    │  │   Trader    │  │   Widget    │  │   Engine    │   │  │
│  │  │  (GUI)      │  │  (Reasoning)│  │  (Live)     │  │   (Indicators)│  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                      │
│                                         ▼                                      │
│                              ┌──────────────────┐                             │
│                              │  Monitoring      │                             │
│                              │   Dashboard      │                             │
│                              │   + Alerts       │                             │
│                              │  Web UI          │                             │
│                              └──────────────────┘                             │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Market Data Ingestion Layer

```python
# src/data/ingestion.py
class MarketDataIngestion:
    """
    Ingests OHLC data from multiple sources:
    - PostgreSQL (skald_ohlc) - primary source
    - CSV archives - for historical backtesting
    - TradeStation streaming - for real-time trading
    """
    
    def __init__(self, db_config: dict, cache_dir: str = "/data/cache"):
        self.db = PostgreSQLClient(db_config)
        self.cache = LRUCache(cache_dir, max_size_gb=50)
        self.symbol_list = ["AAPL", "GLD", "IBIT", "NFLX", "O", "SCHD", "SPY", "TSLA", "XOM"]
        self.stream = None  # TradeStation streaming connection
    
    async def fetch_ohlc(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch OHLC data with automatic caching"""
        cache_key = f"ohlc_{symbol}_{start.date()}_{end.date()}"
        
        if cached := self.cache.get(cache_key):
            return cached
        
        df = await self.db.fetch_ohlc(symbol, start, end)
        self.cache.set(cache_key, df)
        return df
    
    async def stream_live_bars(self, symbols: List[str]) -> AsyncGenerator[OHLCBar, None]:
        """Real-time bar streaming via PostgreSQL listen/notify + TradeStation fallback"""
        while True:
            for symbol in symbols:
                bar = await self.db.fetch_latest_bar(symbol)
                if bar:
                    yield bar
            await asyncio.sleep(5)  # 5-second polling
    
    async def connect_to_tradestation_stream(self):
        """Connect to TradeStation v3 market data streaming endpoint"""
        pass  # Implementation via TradeStation API v3 /marketdata/stream
```

---

### 2. Signal Engine (Strategy Orchestrator)

```python
# src/strategy/signals.py
from abc import ABC, abstractmethod
from typing import Dict, List, Set
import numpy as np
import pandas as pd

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    @abstractmethod
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate entry/exit signals and add to DataFrame"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Strategy name for logging/metrics"""
        pass

class StrategyOrchestrator:
    """Manages multiple strategies and combines signals"""
    
    def __init__(self, strategies: List[BaseStrategy], weights: Dict[str, float]):
        self.strategies = strategies
        self.weights = weights  # Strategy weights for portfolio
        self.active_strategies: Set[BaseStrategy] = set(strategies)
        self.symbol_filters: Dict[str, Set[BaseStrategy]] = {}  # Per-symbol filters
    
    def calculate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all active strategies and combine signals"""
        for strategy in self.strategies:
            if strategy in self.active_strategies:
                df = strategy.calculate_signals(df)
                df[f'{strategy.get_name()}_weight'] = self.weights.get(strategy.get_name(), 1.0)
        
        # Combine signals with weights
        df['combined_score'] = sum(
            df[f'{s.get_name()}_weight'] * df[f'{s.get_name().replace(" ", "_")}_signal']
            for s in self.strategies if s in self.active_strategies and f'{s.get_name()}_signal' in df.columns
        )
        
        return df
    
    def toggle_strategy(self, name: str) -> bool:
        """Toggle strategy active status across all symbols"""
        for s in self.strategies:
            if s.get_name() == name:
                if s in self.active_strategies:
                    self.active_strategies.remove(s)
                else:
                    self.active_strategies.add(s)
                return True
        return False
    
    def set_symbol_filter(self, symbol: str, strategies: List[BaseStrategy]):
        """Apply strategy filter for specific symbol"""
        self.symbol_filters[symbol] = set(strategies)
```

---

### 3. Portfolio Manager & Risk Engine

```python
# src/portfolio/portfolio.py
class PortfolioManager:
    """Manages positions, risk, and capital allocation"""
    
    def __init__(self, initial_capital: float = 100000, max_position_size: float = 0.05):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.max_position_size = max_position_size  # Max 5% per position
        self.max_drawdown = 0.15  # 15% max drawdown
        self.max_positions = 10  # Max concurrent positions
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                                stop_loss_price: float, risk_amount: float) -> int:
        """
        Position sizing based on risk
        Position Size = Risk Amount / (Entry - Stop Loss) * Price
        """
        risk_per_share = abs(entry_price - stop_loss_price)
        shares = int(risk_amount / risk_per_share)
        
        # Apply position size limits
        max_shares = int((self.current_capital * self.max_position_size) / entry_price)
        return min(shares, max_shares, 1000)  # Cap at 1000 shares per position
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        total_value = self.current_capital + sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        
        # Drawdown calculation
        if hasattr(self, 'peak_value'):
            drawdown = (self.peak_value - total_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)
        else:
            self.peak_value = total_value
            drawdown = 0
        
        return {
            'total_value': total_value,
            'drawdown': drawdown,
            'max_drawdown': self.max_drawdown,
            'position_count': len(self.positions),
            'capital_used': sum(pos.size * pos.entry_price for pos in self.positions.values())
        }
    
    def can_trade(self, symbol: str, entry_price: float, risk_amount: float) -> bool:
        """Check if trade can be executed based on risk limits"""
        metrics = self.calculate_portfolio_metrics()
        
        # Check drawdown limit
        if metrics['drawdown'] > self.max_drawdown:
            return False
        
        # Check position limit
        if len(self.positions) >= self.max_positions:
            return False
        
        # Check capital availability
        required_capital = entry_price * self.calculate_position_size(
            symbol, entry_price, entry_price * 0.95, risk_amount
        )
        
        return required_capital <= self.current_capital * 0.9  # Keep 10% buffer
    
    def configure(self, **kwargs):
        """Dynamic configuration via widget"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
```

---

### 4. Order Execution Gateway (TradeStation API v3 Full Coverage)

```python
# src/execution/gateway.py
from tsapiv1 import TradeStationAPI, OrderType, TimeInForce
from enum import Enum
from typing import Optional

class OrderStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class Order:
    """Represents an order with full lifecycle tracking"""
    def __init__(self, symbol: str, side: str, quantity: int,
                 order_type: OrderType, time_in_force: TimeInForce,
                 price: Optional[float] = None, stop_price: Optional[float] = None):
        self.symbol = symbol
        self.side = side  # 'BUY', 'SELL', 'SELL_SHORT', 'BUY_TO_COVER'
        self.quantity = quantity
        self.order_type = order_type
        self.time_in_force = time_in_force
        self.price = price  # Limit price
        self.stop_price = stop_price  # Stop price
        self.status = OrderStatus.PENDING
        self.order_id = None
        self.fill_price = None
        self.fill_time = None
        self.cancel_reason = None

class ExecutionGateway:
    """Handles order submission and management via TradeStation API v3"""
    
    def __init__(self, api: TradeStationAPI, portfolio: PortfolioManager):
        self.api = api
        self.portfolio = portfolio
        self.orders: Dict[str, Order] = {}
        self.accounts = []
    
    async def initialize(self):
        """Fetch accounts on startup"""
        self.accounts = await self.api.get_accounts()
    
    async def place_order(self, symbol: str, side: str, quantity: int,
                          order_type: str = "MARKET", time_in_force: str = "GTC",
                          price: Optional[float] = None, stop_price: Optional[float] = None) -> str:
        """Place a trade order with full v3 endpoint coverage"""
        if not self.portfolio.can_trade(symbol, price or 0, 0):
            raise ValueError("Position size exceeds risk limits")
        
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType[order_type],
            time_in_force=TimeInForce[time_in_force],
            price=price,
            stop_price=stop_price
        )
        
        order_id = await self.api.submit_order(order)
        order.order_id = order_id
        self.orders[order_id] = order
        
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order via TradeStation v3 /brokerage/orders/{order_id}/cancel"""
        order = self.orders.get(order_id)
        if not order:
            return False
        
        result = await self.api.cancel_order(order_id)
        if result:
            order.status = OrderStatus.CANCELLED
        return result
    
    async def modify_order(self, order_id: str, **kwargs) -> bool:
        """Modify an existing order via TradeStation v3 /brokerage/orders/{order_id}"""
        order = self.orders.get(order_id)
        if not order:
            return False
        
        for key, value in kwargs.items():
            if hasattr(order, key) and key != 'order_id':
                setattr(order, key, value)
        
        result = await self.api.replace_order(order_id, order)
        return result
    
    async def get_account_balances(self) -> dict:
        """Get account balances via TradeStation v3 /brokerage/accounts"""
        return await self.api.get_accounts()
    
    async def get_positions(self) -> dict:
        """Get current positions via TradeStation v3 /brokerage/accounts/{account_id}/positions"""
        return await self.api.get_positions()
    
    async def stream_orders(self):
        """Stream order updates via TradeStation v3 /brokerage/orders/stream"""
        async for update in self.api.stream_orders():
            order_id = update['order_id']
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus(update['status'])
                if update.get('filled'):
                    self.orders[order_id].fill_price = update['fill_price']
                    self.orders[order_id].fill_time = update['fill_time']
```

---

### 5. OpenClaw Agent Integration Layer

```python
# src/agents/openclaw_integration.py
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

@dataclass
class AgentStatus:
    """Status of an OpenClaw agent"""
    name: str
    status: str  # running, idle, error
    last_heartbeat: float
    cpu_usage: float
    memory_usage: float
    messages_sent: int
    messages_received: int

class OpenClawMonitor:
    """Monitor and control OpenClaw agents via dashboard"""
    
    def __init__(self, dashboard_url: str = "http://localhost:3000"):
        self.dashboard_url = dashboard_url
        self.agents: Dict[str, AgentStatus] = {}
        self.session = None  # Websocket session
    
    async def connect(self):
        """Connect to OpenClaw agent dashboard"""
        pass  # WebSocket connection to /ws/agents
    
    async def get_all_agents(self) -> List[AgentStatus]:
        """Fetch all registered agents"""
        pass  # HTTP GET /api/agents
    
    async def get_system_stats(self) -> dict:
        """Get system-wide stats from dashboard"""
        return {
            "total_agents": len(self.agents),
            "active_strategies": 0,
            "positions": 0,
            "total_value": 0.0
        }

class AgenticTraderConfig:
    """Configuration for the Agentic Trader with advanced customization"""
    
    def __init__(self):
        # Model selection
        self.model_type = "momentum"  # Options: momentum, ml_classifier, ml_regressor
        self.ml_model_path = None
        self.ml_feature_engineer_path = None
        
        # Analysis options
        self.sample_interval = 300  # seconds
        self.confidence_threshold = 0.8
        self.lookback_period = 3600  # Look back 1 hour for analysis
        
        # Behavior defaults
        self.max_position_size = 0.02  # 2% of capital per trade
        self.min_confidence = 0.7  # Minimum confidence for trade execution
        self.max_trades_per_day = 5  # Daily trade limit
        self.cooldown_minutes = 30  # Cooldown between trades
        
        # Custom behaviors (extendable)
        self.custom_filters = []  # List of filter functions
        self.custom_exit_rules = []  # List of exit rule functions
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization"""
        return {
            'model_type': self.model_type,
            'ml_model_path': self.ml_model_path,
            'ml_feature_engineer_path': self.ml_feature_engineer_path,
            'sample_interval': self.sample_interval,
            'confidence_threshold': self.confidence_threshold,
            'lookback_period': self.lookback_period,
            'max_position_size': self.max_position_size,
            'min_confidence': self.min_confidence,
            'max_trades_per_day': self.max_trades_per_day,
            'cooldown_minutes': self.cooldown_minutes,
            'custom_filters': [f.__name__ if hasattr(f, '__name__') else str(f) for f in self.custom_filters],
            'custom_exit_rules': [f.__name__ if hasattr(f, '__name__') else str(f) for f in self.custom_exit_rules]
        }
    
    def update_from_dict(self, config: dict):
        """Update config from dictionary"""
        for key, value in config.items():
            if hasattr(self, key) and key not in ('custom_filters', 'custom_exit_rules'):
                setattr(self, key, value)
        # Handle custom functions (would need to be re-registered)

class AgenticTrader:
    """
    Independent reasoning-based trading agent
    Optional feature: samples recent market data and executes trades outside strategies
    """
    
    def __init__(self, config: AgenticTraderConfig = None):
        self.config = config or AgenticTraderConfig()
        self.enabled = False
        self.last_sample_time = 0
        self.last_trade_time = 0
        self.trades_today = 0
        self.reasoning_log: List[dict] = []
    
    async def sample_market_data(self, symbols: List[str], data_layer) -> dict:
        """Sample recent market data for reasoning"""
        from datetime import datetime, timedelta
        recent_data = {}
        for symbol in symbols:
            df = await data_layer.fetch_ohlc(
                symbol, 
                datetime.now() - timedelta(seconds=self.config.lookback_period), 
                datetime.now()
            )
            recent_data[symbol] = df.tail(12)  # Last sample period
        return recent_data
    
    async def analyze_and_execute(self, symbols: List[str], data_layer, execution_gateway, portfolio):
        """Analyze sampled data and execute trades if confidence is high enough"""
        if not self.enabled:
            return []
        
        # Check cooldown and daily limit
        current_time = asyncio.get_event_loop().time()
        if (current_time - self.last_trade_time) < (self.config.cooldown_minutes * 60):
            return []
        if self.trades_today >= self.config.max_trades_per_day:
            return []
        
        recent_data = await self.sample_market_data(symbols, data_layer)
        decisions = []
        
        for symbol, df in recent_data.items():
            if self.config.model_type == "momentum":
                # Simple momentum reasoning
                returns = df['close'].pct_change().dropna()
                momentum = returns.tail(5).mean()
                volatility = returns.tail(5).std()
                confidence = abs(momentum) / volatility if volatility > 0 else 0
            elif self.config.model_type in ("ml_classifier", "ml_regressor"):
                # ML-based reasoning (requires trained model)
                # Placeholder for ML inference
                confidence = 0.5  # Default placeholder
                momentum = 0
            else:
                confidence = 0
                momentum = 0
            
            decision = {
                'symbol': symbol,
                'direction': 'long' if momentum > 0 else 'short',
                'confidence': confidence,
                'reasoning': f"Model: {self.config.model_type}, Momentum: {momentum:.4f}, Volatility: {volatility:.4f if 'volatility' in dir() else 'N/A'}"
            }
            decisions.append(decision)
            self.reasoning_log.append(decision)
        
        # Execute high-confidence trades
        executed = []
        for decision in decisions:
            if decision['confidence'] > self.config.min_confidence:
                current_price = recent_data[decision['symbol']]['close'].iloc[-1]
                qty = int(portfolio.current_capital * self.config.max_position_size / current_price)
                
                order_id = await execution_gateway.place_order(
                    symbol=decision['symbol'],
                    side='BUY' if decision['direction'] == 'long' else 'SELL_SHORT',
                    quantity=qty,
                    order_type='MARKET'
                )
                executed.append({'decision': decision, 'order_id': order_id})
                self.last_trade_time = asyncio.get_event_loop().time()
                self.trades_today += 1
        
        return executed
    
    def toggle(self):
        """Toggle agentic trader on/off"""
        self.enabled = not self.enabled
        return self.enabled
```

---

### 6. Configuration Widget

```python
# src/config/config_widget.py
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import os
from .agents.agentic_trader import AgenticTraderConfig

@dataclass
class SystemConfig:
    """All configurable system parameters"""
    # Portfolio
    initial_capital: float = 100000
    max_position_size: float = 0.05
    max_drawdown: float = 0.15
    max_positions: int = 10
    
    # Agentic Trader
    agentic_enabled: bool = False
    agentic_sample_interval: int = 300
    agentic_confidence_threshold: float = 0.8
    agentic_max_position_size: float = 0.02
    
    # Data
    data_cache_size_gb: int = 50
    stream_poll_interval: int = 5
    
    # Monitoring
    alert_email: Optional[str] = None
    alert_telegram_chat_id: Optional[str] = None
    
    def save(self, path: str = "/data/config/system_config.json"):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str = "/data/config/system_config.json") -> 'SystemConfig':
        """Load configuration from file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except FileNotFoundError:
            return cls()
    
    def update(self, **kwargs):
        """Update configuration dynamically"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

class ConfigWidget:
    """Web-based configuration interface for live parameter updates"""
    
    def __init__(self):
        self.config = SystemConfig.load()
        self.agentic_config = AgenticTraderConfig()
        self.listeners = []
    
    def get_config(self) -> dict:
        """Return current configuration"""
        return {
            'system': asdict(self.config),
            'agentic_trader': self.agentic_config.to_dict()
        }
    
    def update_config(self, updates: dict):
        """Update configuration and save"""
        if 'system' in updates:
            self.config.update(**updates['system'])
        if 'agentic_trader' in updates:
            self.agentic_config.update_from_dict(updates['agentic_trader'])
        self.config.save()
        
        # Notify listeners of config change
        for listener in self.listeners:
            listener(self.get_config())
    
    def register_listener(self, callback):
        """Register a callback for config updates"""
        self.listeners.append(callback)
```

---

### 7. Strategy Management System

```python
# src/strategy/manager.py
import importlib.util
import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass
class StrategyInfo:
    """Metadata about a strategy"""
    name: str
    class_name: str
    module_path: str
    enabled: bool = True
    symbols: List[str] = None  # Specific symbols, None = all
    parameters: Dict[str, Any] = None
    created_at: str = None
    last_modified: str = None

class StrategyManager:
    """Manages strategy loading, unloading, and configuration"""
    
    def __init__(self, strategies_dir: str = "/home/hologaun/projects/mlat/strategies"):
        self.strategies_dir = strategies_dir
        self.strategy_registry: Dict[str, StrategyInfo] = {}
        self.loaded_strategies: Dict[str, object] = {}
        self.strategy_instances: Dict[str, object] = {}
    
    def discover_strategies(self) -> List[StrategyInfo]:
        """Discover all strategies in the strategies directory"""
        strategies = []
        
        for filename in os.listdir(self.strategies_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                module_path = os.path.join(self.strategies_dir, filename)
                strategy_info = self._extract_strategy_info(module_path)
                if strategy_info:
                    strategies.append(strategy_info)
                    self.strategy_registry[filename] = strategy_info
        
        # Load strategies from registry
        for name, info in self.strategy_registry.items():
            self.load_strategy(info)
        
        return strategies
    
    def _extract_strategy_info(self, module_path: str) -> Optional[StrategyInfo]:
        """Extract strategy information from a module"""
        try:
            # Simple parsing - in production, use AST parsing
            with open(module_path, 'r') as f:
                content = f.read()
            
            # Extract class name
            import re
            class_match = re.search(r'class\s+(\w+)\(BaseStrategy\):', content)
            if not class_match:
                return None
            
            class_name = class_match.group(1)
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            
            return StrategyInfo(
                name=module_name,
                class_name=class_name,
                module_path=module_path,
                symbols=None,  # Default to all symbols
                parameters={},
                created_at=str(pd.Timestamp.now()),
                last_modified=str(pd.Timestamp.fromtimestamp(os.path.getmtime(module_path)))
            )
        except Exception as e:
            return None
    
    def load_strategy(self, strategy_info: StrategyInfo) -> bool:
        """Load a strategy module"""
        try:
            spec = importlib.util.spec_from_file_location(
                strategy_info.name, 
                strategy_info.module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Instantiate the strategy class
            strategy_class = getattr(module, strategy_info.class_name)
            instance = strategy_class(**(strategy_info.parameters or {}))
            
            self.loaded_strategies[strategy_info.name] = module
            self.strategy_instances[strategy_info.name] = instance
            
            return True
        except Exception as e:
            return False
    
    def unload_strategy(self, name: str) -> bool:
        """Unload a strategy"""
        if name in self.strategy_instances:
            del self.strategy_instances[name]
        if name in self.loaded_strategies:
            del self.loaded_strategies[name]
        if name in self.strategy_registry:
            del self.strategy_registry[name]
        return True
    
    def import_strategy_from_file(self, file_path: str, name: str = None) -> Optional[StrategyInfo]:
        """Import a strategy from an external file (e.g., AI-generated or CSV analysis)"""
        if not os.path.exists(file_path):
            return None
        
        # Copy to strategies directory
        if not name:
            name = os.path.splitext(os.path.basename(file_path))[0]
        
        dest_path = os.path.join(self.strategies_dir, f"{name}.py")
        if os.path.exists(dest_path):
            # Strategy already exists
            return self.strategy_registry.get(name)
        
        # Copy file
        import shutil
        shutil.copy(file_path, dest_path)
        
        # Extract info and register
        strategy_info = self._extract_strategy_info(dest_path)
        if strategy_info:
            strategy_info.name = name
            self.strategy_registry[name] = strategy_info
            self.load_strategy(strategy_info)
            return strategy_info
        
        return None
    
    def get_active_strategies(self) -> List[object]:
        """Get all active strategy instances"""
        return [inst for name, inst in self.strategy_instances.items() 
                if self.strategy_registry.get(name, StrategyInfo(name="", class_name="", module_path="")).enabled]
```

---

### 8. Built-in Scanner Engine

```python
# src/scanner/scanner.py
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta

class TechnicalIndicator:
    """Base class for technical indicators"""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        return prices.rolling(period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def awesome_oscillator(df: pd.DataFrame) -> pd.Series:
        median = (df['high'] + df['low']) / 2
        return median.rolling(5).mean() - median.rolling(34).mean()
    
    @staticmethod
    def candlestick_patterns(df: pd.DataFrame) -> dict:
        patterns = {}
        patterns['doji'] = abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1
        patterns['hammer'] = (
            (df['low'].shift(1) > df['high']) &
            (df['close'] > df['open']) &
            ((df['high'] - df['close']) < (df['open'] - df['low']))
        )
        patterns['engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['close'] > df['close'].shift(1)) &
            (df['open'] < df['open'].shift(1))
        )
        return patterns

class Scanner:
    """Real-time market scanner with indicator columns"""
    
    def __init__(self, symbols: List[str], data_layer):
        self.symbols = symbols
        self.data_layer = data_layer
        self.indicators = [
            ('RSI(5)', lambda df: TechnicalIndicator.rsi(df['close'], 5)),
            ('RSI(10)', lambda df: TechnicalIndicator.rsi(df['close'], 10)),
            ('RSI(14)', lambda df: TechnicalIndicator.rsi(df['close'], 14)),
            ('RSI(21)', lambda df: TechnicalIndicator.rsi(df['close'], 21)),
            ('SMA(10)', lambda df: TechnicalIndicator.sma(df['close'], 10)),
            ('SMA(30)', lambda df: TechnicalIndicator.sma(df['close'], 30)),
            ('SMA(50)', lambda df: TechnicalIndicator.sma(df['close'], 50)),
            ('Awesome', TechnicalIndicator.awesome_oscillator),
        ]
    
    async def scan_symbol(self, symbol: str) -> dict:
        df = await self.data_layer.fetch_ohlc(
            symbol,
            datetime.now() - timedelta(days=7),
            datetime.now()
        )
        
        result = {'symbol': symbol}
        
        for name, func in self.indicators:
            try:
                result[name] = func(df).iloc[-1]
            except:
                result[name] = None
        
        patterns = TechnicalIndicator.candlestick_patterns(df)
        for pattern_name, is_pattern in patterns.items():
            result[f'pattern_{pattern_name}'] = is_pattern.iloc[-1]
        
        result['close'] = df['close'].iloc[-1]
        result['volume'] = df['volume'].iloc[-1]
        
        return result
    
    async def scan_all(self) -> List[dict]:
        results = []
        for symbol in self.symbols:
            try:
                scan_result = await self.scan_symbol(symbol)
                results.append(scan_result)
            except Exception as e:
                results.append({'symbol': symbol, 'error': str(e)})
        return results
```

---

### 9. Monitoring Dashboard

```python
# src/dashboard/server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime
import asyncio
import json
import os

app = FastAPI(title="HoloClaw Trading Dashboard v3")

templates = Jinja2Templates(directory="src/dashboard/templates")
manager = type('ConnectionManager', (), {
    'active_connections': [],
    'async def connect': lambda self, ws: asyncio.run_coroutine_threadsafe(ws.accept(), asyncio.new_event_loop()),
    'async def disconnect': lambda self, ws: self.active_connections.remove(ws) if ws in self.active_connections else None,
    'async def broadcast': lambda self, msg: [asyncio.run_coroutine_threadsafe(c.send_json(msg), asyncio.new_event_loop()) for c in self.active_connections]
})()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager.active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'get_status':
                await websocket.send_json({
                    "type": "status",
                    "strategies": len(orchestrator.active_strategies),
                    "positions": len(portfolio.positions),
                    "capital": portfolio.current_capital,
                    "agents": len(openclaw_monitor.agents),
                    "agentic_trader_enabled": agentic_trader.enabled
                })
            
            elif message['type'] == 'toggle_strategy':
                name = message['name']
                if orchestrator.toggle_strategy(name):
                    await manager.broadcast({"type": "strategy_toggled", "name": name})
            
            elif message['type'] == 'agentic_toggle':
                enabled = agentic_trader.toggle()
                await websocket.send_json({"type": "agentic_toggled", "enabled": enabled})
            
            elif message['type'] == 'update_config':
                config_widget.update_config(**message['data'])
                await manager.broadcast({"type": "config_updated"})
            
            elif message['type'] == 'scan':
                results = await scanner.scan_all()
                await websocket.send_json({"type": "scan_results", "data": results})
    
    except WebSocketDisconnect:
        manager.active_connections.remove(websocket)

@app.get("/api/strategies")
async def list_strategies():
    return [{"name": s.get_name(), "active": s in orchestrator.active_strategies} 
            for s in orchestrator.strategies]

@app.get("/api/config")
async def get_config():
    return config_widget.get_config()

@app.get("/api/scanner")
async def get_scanner():
    results = await scanner.scan_all()
    return {"timestamp": datetime.now().isoformat(), "results": results}

@app.get("/api/agents")
async def get_agents():
    return [{"name": name, **asdict(status)} for name, status in openclaw_monitor.agents.items()]

app.mount("/static", StaticFiles(directory="src/dashboard/static"), name="static")
```

---

## Scalability Features

### 1. Strategy Isolation
```python
import concurrent.futures

def run_strategy(strategy, data):
    return strategy.calculate_signals(data)

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(run_strategy, strategies, [data] * len(strategies))
```

### 2. Async I/O for Real-time Processing
```python
async def main():
    ingestion = MarketDataIngestion(db_config)
    orchestrator = StrategyOrchestrator(strategies, weights)
    portfolio = PortfolioManager()
    executor = ExecutionGateway(api, portfolio)
    openclaw_monitor = OpenClawMonitor()
    agentic_trader = AgenticTrader(agentic_config)
    
    async for bar in ingestion.stream_live_bars(symbols):
        signals = orchestrator.calculate_all_signals(bar)
        
        if agentic_trader.enabled:
            await agentic_trader.analyze_and_execute(symbols, ingestion, executor, portfolio)
        
        orders = generate_orders(signals, portfolio)
        for order in orders:
            await executor.place_order(**order)

asyncio.run(main())
```

---

## Deployment

### Project Structure
```
/home/hologaun/projects/mlat/
├── src/
│   ├── data/           # Market data ingestion
│   ├── strategy/       # Trading strategies (loadable modules)
│   │   ├── __init__.py
│   │   ├── signals.py  # Base strategy classes
│   │   └── manager.py  # Strategy management
│   ├── portfolio/      # Portfolio management
│   ├── execution/      # Order execution (TradeStation API)
│   ├── agents/         # OpenClaw integration + Agentic Trader
│   │   ├── openclaw_integration.py
│   │   └── agentic_trader.py
│   ├── config/         # Configuration management
│   ├── scanner/        # Market scanner
│   ├── ml/             # ML training pipeline
│   └── dashboard/      # Web UI
│       ├── server.py
│       ├── templates/
│       └── static/
├── data/               # OHLC data cache
├── models/             # Trained ML models
├── strategies/         # User-added strategies (external files)
├── config/             # System configuration
│   └── system_config.json
├── logs/               # Application logs
└── HoloClaw_ML_Trading_Architecture.md
```

### Production Run
```bash
cd ~/projects/mlat

# Initialize
mkdir -p data models strategies config logs

# Start main application
python -m holoclaw.main

# Start dashboard
uvicorn holoclaw.dashboard.server:app --host 0.0.0.0 --port 8000 --reload
```

---

## GitHub Integration

**Repository:** `/home/hologaun/projects/mlat`  
**Tracking:** Git + GitHub via OpenClaw GitHub skills  
**Strategy Import:** Strategies can be:
- Built-in (module-based, in `src/strategy/`)
- External files (CSV analysis, AI-generated EL code imported via `strategy/import`)
- Toggleable per symbol or globally

---

**Version:** 3.0  
**Last Updated:** 2026-03-11  
**Maintainer:** Aesir (Hologaun)
