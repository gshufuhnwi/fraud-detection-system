import uuid
from datetime import datetime


class Transaction:
    def __init__(self, tx_type: str, amount: float, details: str = ""):
        self.id = str(uuid.uuid4())[:8]
        self.tx_type = tx_type
        self.amount = amount
        self.details = details
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.id,
            "type": self.tx_type,
            "amount": self.amount,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class Account:
    def __init__(self, name: str, initial_balance: float = 0.0):
        self.account_id = str(uuid.uuid4())[:8]
        self.name = name
        self.balance = initial_balance
        self.transactions: list[Transaction] = []

    def deposit(self, amount: float) -> None:
        self.balance += amount
        self.transactions.append(Transaction("deposit", amount))

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        self.transactions.append(Transaction("withdraw", amount))

    def to_dict(self) -> dict:
        return {
            "account_id": self.account_id,
            "name": self.name,
            "balance": self.balance,
            "transactions": [t.to_dict() for t in self.transactions],
        }


class BankSystem:
    def __init__(self):
        self.accounts: dict[str, Account] = {}

    def create_account(self, name: str, initial_balance: float = 0.0) -> Account:
        account = Account(name, initial_balance)
        self.accounts[account.account_id] = account
        return account

    def get_account(self, account_id: str) -> Account:
        if account_id not in self.accounts:
            raise ValueError("Account not found")
        return self.accounts[account_id]

    def transfer(self, from_account_id: str, to_account_id: str, amount: float) -> None:
        sender = self.get_account(from_account_id)
        receiver = self.get_account(to_account_id)

        if sender.balance < amount:
            raise ValueError("Insufficient funds")

        sender.withdraw(amount)
        receiver.deposit(amount)

        sender.transactions.append(Transaction("transfer_out", amount, f"to {to_account_id}"))
        receiver.transactions.append(Transaction("transfer_in", amount, f"from {from_account_id}"))
