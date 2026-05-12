import pandas as pd
import os

# Remove duplicates from attendance file
if os.path.exists('attendance/attendance.csv'):
    df = pd.read_csv('attendance/attendance.csv')
    # Keep only first occurrence per user per day
    df = df.drop_duplicates(subset=['id', 'date'], keep='first')
    df.to_csv('attendance/attendance.csv', index=False)
    print(f"Cleaned: {len(df)} unique records remain")
else:
    print("No attendance file found")