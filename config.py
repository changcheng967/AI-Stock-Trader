"""
Configuration management for the algorithmic trading bot.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""

    # ========== API KEYS (Load from .env) ==========

    # Alpaca API (Required)
    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        env="ALPACA_BASE_URL"
    )
    alpaca_data_url: str = Field(
        default="https://data.alpaca.markets",
        env="ALPACA_DATA_URL"
    )

    # NVIDIA NIM (Optional - for LLM sentiment analysis)
    nim_api_key: str = Field(default="", env="NIM_API_KEY")
    nim_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1/chat/completions",
        env="NIM_BASE_URL"
    )

    # ========== TRADING CONFIGURATION (Hardcoded) ==========

    # Account Settings
    initial_capital: float = 1000.0  # Initial portfolio value (reference only)

    # Pattern Day Trading (PDT) Settings
    # If account balance < $25,000, you're limited to 3 day trades per 5-day period
    pdt_threshold: float = 25000.0
    respect_pdt_limits: bool = True  # Only trade 3 times per 5 days when account < $25k

    # Risk Management
    max_position_size: float = 0.15   # 15% of portfolio per position (higher for best ideas)
    max_positions: int = 3           # Maximum number of open positions (quality over quantity)
    stop_loss_pct: float = 0.025     # 2.5% stop loss (tighter for quick exits)
    take_profit_pct: float = 0.08    # 8% take profit base (will adapt based on volatility)
    use_trailing_stops: bool = True  # Enable trailing stop losses
    trailing_stop_pct: float = 0.02  # 2% trailing stop (lock in profits)

    # Scanning Schedule
    check_interval_minutes: int = 5  # Check positions every 5 minutes (active management)

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
