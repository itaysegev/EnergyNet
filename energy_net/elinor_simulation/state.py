
from datetime import datetime, timedelta
from typing import Optional
from typing import TypedDict

class ConsumptionStateDict(TypedDict):
    consumption: float 
    next_consumption: float
    date: datetime

class ConsumptionState:
    def __init__(self, consumption: float, next_consumption: float, date: Optional[datetime] = None):
        self.data: ConsumptionStateDict = {
            'consumption': consumption,
            'next_consumption': next_consumption,
            'date': date if date else None
        }

    def promote_date(self, minutes: int = 30):
        """Promote the date by a specified number of minutes."""
        self.data['date'] += timedelta(minutes=minutes)
        return self.data['date']

    def set_date(self, new_date: datetime):
        """Set the date to a new value."""
        if isinstance(new_date, str):
            new_date = datetime.fromisoformat(new_date)
        if not isinstance(new_date, datetime):
            raise ValueError("The date must be a datetime object.")
        self.data['date'] = new_date

    def get_date_in_month_hour_format(self) -> str:
        """Return the date in the 'month-hour' format."""
        return self.data['date'].strftime("%m-%H")

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return str(self.data)

    def items(self):
        return self.data.items()

class UnitState(TypedDict):
    storge:float
    consumption: float
    next_consumption: float


    
    
