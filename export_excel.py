import pandas as pd
import os
from datetime import datetime

os.makedirs("attendance", exist_ok=True)

data = {
    "Name": ["Yonas", "Student2"],
    "Date": [datetime.now().strftime("%Y-%m-%d")] * 2,
    "Time": [datetime.now().strftime("%H:%M:%S")] * 2,
    "Status": ["Present", "Present"]
}

df = pd.DataFrame(data)

excel_path = "attendance/attendance.xlsx"

df.to_excel(excel_path, index=False)

print(f"Excel report saved: {excel_path}")