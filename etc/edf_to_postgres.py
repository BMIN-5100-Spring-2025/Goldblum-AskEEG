import os
import mne
import psycopg2
from dotenv import load_dotenv
from datetime import datetime, timezone
import pandas as pd

load_dotenv()  # Load environment variables from .env


def get_db_connection():
    """Create and return a PostgreSQL database connection"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT")),
    )


def create_table(conn, channels):
    """Create table with dynamic columns based on EDF channels"""
    with conn.cursor() as cursor:
        # Create column definitions
        columns = ",\n".join([f'"{ch}" FLOAT' for ch in channels])
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS eeg_data (
                timestamp TIMESTAMP PRIMARY KEY,
                {columns}
            );
        """
        )
    conn.commit()


def store_edf_in_postgres(edf_path, conn):
    """Main function to process EDF and store in PostgreSQL"""
    # Load EDF file
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    df = raw.to_data_frame()

    # Create tables if they don't exist
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS eeg_metadata (
                id SERIAL PRIMARY KEY,
                edf_path VARCHAR NOT NULL UNIQUE,
                recording_start TIMESTAMPTZ NOT NULL,
                original_sfreq NUMERIC NOT NULL,
                channels TEXT[] NOT NULL,
                patient_id VARCHAR,
                patient_sex VARCHAR(1),
                patient_age VARCHAR,
                lowpass_filter NUMERIC,
                highpass_filter NUMERIC,
                original_file_creation TIMESTAMPTZ
            );
        """
        )

        # Fix the table creation query
        columns = [f'"{ch}" FLOAT NOT NULL' for ch in raw.ch_names]
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS eeg_data (
                timestamp TIMESTAMPTZ NOT NULL,
                {', '.join(columns)}
            );
        """
        cursor.execute(create_table_query)

    # Store metadata with timezone info preserved
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO eeg_metadata (
                edf_path, 
                recording_start,
                original_sfreq,
                channels,
                patient_id,
                patient_sex,
                patient_age,
                lowpass_filter,
                highpass_filter,
                original_file_creation
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
            (
                edf_path,
                raw.info["meas_date"],
                raw.info["sfreq"],
                raw.ch_names,
                raw.info.get("subject_info", {}).get("id"),
                raw.info.get("subject_info", {}).get("sex"),
                raw.info.get("subject_info", {}).get("age"),
                raw.info["lowpass"],
                raw.info["highpass"],
                datetime.fromtimestamp(os.path.getctime(edf_path), tz=timezone.utc),
            ),
        )

    # Data insertion with timezone info preserved
    with conn.cursor() as cursor:
        cols = ["timestamp"] + list(raw.ch_names)
        columns = ", ".join([f'"{c}"' for c in cols])
        placeholders = ", ".join(["%s"] * len(cols))

        insert_query = f"""
            INSERT INTO eeg_data ({columns})
            VALUES ({placeholders})
        """

        # Create timestamps properly preserving timezone info
        start_time = raw.info["meas_date"]
        # Convert index to timedelta and add to start time
        df["timestamp"] = pd.to_timedelta(df.index, unit="s") + pd.Timestamp(start_time)

        df = df.reindex(columns=["timestamp"] + list(raw.ch_names))
        records = [tuple(row) for row in df.values]

        print(f"Inserting {len(records)} records...")
        cursor.executemany(insert_query, records)
        conn.commit()


if __name__ == "__main__":
    edf_path = os.path.join(
        "data", "input", "EMU1371_Day02_1_5006_to_5491.edf"
    )

    try:
        conn = get_db_connection()
        store_edf_in_postgres(edf_path, conn)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if conn:
            conn.close()
