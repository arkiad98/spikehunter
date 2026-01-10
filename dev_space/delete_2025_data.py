import sqlite3
import os

# DB Path
db_path = r'd:\spikehunter\data\db\spikehunter_log.db'

def delete_2025_signals():
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check count before deletion
        cursor.execute("SELECT COUNT(*) FROM daily_signals WHERE date LIKE '2025%'")
        count_before = cursor.fetchone()[0]
        print(f"Records to be deleted: {count_before}")

        if count_before > 0:
            # Delete records
            cursor.execute("DELETE FROM daily_signals WHERE date LIKE '2025%'")
            conn.commit()
            print("Deletion executed.")

            # Check count after deletion
            cursor.execute("SELECT COUNT(*) FROM daily_signals WHERE date LIKE '2025%'")
            count_after = cursor.fetchone()[0]
            print(f"Records remaining after deletion: {count_after}")
        else:
            print("No records found for 2025.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    delete_2025_signals()
