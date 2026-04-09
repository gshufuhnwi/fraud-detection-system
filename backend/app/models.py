from pydantic import BaseModel, Field


class CreateAccountRequest(BaseModel):
    name: str
    initial_balance: float = 0.0


class DepositWithdrawRequest(BaseModel):
    account_id: str
    amount: float = Field(gt=0)


class TransferRequest(BaseModel):
    from_account_id: str
    to_account_id: str
    amount: float = Field(gt=0)


class FraudTransactionRequest(BaseModel):
    transaction_amount: float
    merchant_category: str
    merchant_country: str
    device_type: str
    transaction_type: str
    hour: int
    distance_from_home: float
    transactions_last_24h: int
    merchant_risk_score: float
    is_international: int
    is_card_present: int
    device_trust_score: float
    account_balance: float
