from pydantic import BaseModel

class SalesRequest(BaseModel):
    country: str
    marketing_spend: float
    discount_pct: float
    stock_available: float
    price: float
    month: int
    day_of_week: int
    is_weekend: int
    is_holiday: int
    is_non_labour: int
    lag_7: float