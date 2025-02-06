import os
import sys
import dotenv
import numpy as np
import pandas as pd
from ieeg.auth import Session
import argparse
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values


def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT")),
    )


def create_tables(conn):
    """Create the necessary database tables if they don't exist"""
    with conn.cursor() as cursor:
        # Drop existing tables to handle schema changes
        cursor.execute("DROP TABLE IF EXISTS eeg_data;")
        cursor.execute("DROP TABLE IF EXISTS eeg_metadata;")

        # Create metadata table with IEEG-specific fields
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS eeg_metadata (
                id SERIAL PRIMARY KEY,
                dataset_name VARCHAR NOT NULL UNIQUE,
                recording_start TIMESTAMPTZ NOT NULL,
                recording_end TIMESTAMPTZ NOT NULL,
                sample_rate NUMERIC NOT NULL,
                channels TEXT[] NOT NULL,
                voltage_conversion_factor NUMERIC,
                number_of_samples BIGINT,
                duration BIGINT  -- microseconds
            );
            """
        )

        conn.commit()


def create_data_table(conn, channel_labels):
    """Create the data table with all channels in one go"""
    with conn.cursor() as cursor:
        # Create channel columns dynamically
        channel_columns = [f'"{ch}" FLOAT' for ch in channel_labels]
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS eeg_data (
                timestamp TIMESTAMPTZ NOT NULL,
                {', '.join(channel_columns)}
            );
        """
        cursor.execute(create_table_query)
        conn.commit()


def add_channels_to_data_table(conn, channel_labels):
    """Add channel columns to the data table if they don't exist"""
    with conn.cursor() as cursor:
        for channel in channel_labels:
            cursor.execute(
                f"""
                ALTER TABLE eeg_data 
                ADD COLUMN IF NOT EXISTS "{channel}" FLOAT;
                """
            )
        conn.commit()


def main(total_mins, remove_nan=False):
    dotenv.load_dotenv()

    # Connect to IEEG
    session = Session(os.getenv("IEEG_USERNAME"), os.getenv("IEEG_PASSWORD"))
    dataset = session.open_dataset(os.getenv("IEEG_DATASET"))

    # Get the channel labels
    channel_labels = dataset.get_channel_labels()
    num_channels = len(channel_labels)

    # Get the temporal details
    timeseries = dataset.get_time_series_details(channel_labels[0])
    fs = int(timeseries.sample_rate)  # Hz
    start_time = datetime.fromtimestamp(
        dataset.start_time / 1e6, tz=timezone.utc
    )  # Convert μs to datetime

    print(f"Sampling rate: {fs} Hz")
    print(f"Recording start time: {start_time}")

    # Connect to PostgreSQL
    conn = get_db_connection()
    try:
        # Create tables with all channels at once
        create_tables(conn)
        create_data_table(conn, channel_labels)

        # Get detailed metadata from first channel
        ts_details = dataset.get_time_series_details(channel_labels[0])

        # Insert metadata with IEEG-specific information
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO eeg_metadata (
                    dataset_name,
                    recording_start,
                    recording_end,
                    sample_rate,
                    channels,
                    voltage_conversion_factor,
                    number_of_samples,
                    duration
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (dataset_name) DO UPDATE SET
                    recording_start = EXCLUDED.recording_start,
                    recording_end = EXCLUDED.recording_end,
                    sample_rate = EXCLUDED.sample_rate,
                    channels = EXCLUDED.channels,
                    voltage_conversion_factor = EXCLUDED.voltage_conversion_factor,
                    number_of_samples = EXCLUDED.number_of_samples,
                    duration = EXCLUDED.duration
                """,
                (
                    dataset.name,
                    datetime.fromtimestamp(dataset.start_time / 1e6, tz=timezone.utc),
                    datetime.fromtimestamp(dataset.end_time / 1e6, tz=timezone.utc),
                    ts_details.sample_rate,
                    channel_labels,
                    ts_details.voltage_conversion_factor,
                    ts_details.number_of_samples,
                    ts_details.duration,
                ),
            )

        # Calculate chunk size
        max_chunk_mins = 10
        samples_per_min = fs * 60
        max_samples = max_chunk_mins * samples_per_min
        valid_samples = max_samples - (max_samples % fs)
        chunk_mins = valid_samples / samples_per_min
        chunk_usec = chunk_mins * 60 * 1e6

        # Calculate number of chunks needed
        num_chunks = int(np.ceil(total_mins / chunk_mins))

        # Process chunks
        for i in range(num_chunks):
            start_usec = i * chunk_usec
            end_min = min((i + 1) * chunk_mins, total_mins)
            print(
                f"Processing chunk {i+1}/{num_chunks}: "
                f"Minutes {i*chunk_mins:.2f} to {end_min:.2f}"
            )

            # Get data chunk
            chunk_df = dataset.get_dataframe(
                start_usec, chunk_usec, np.arange(num_channels)
            )

            if remove_nan:
                chunk_df = chunk_df.dropna()

            if not chunk_df.empty:
                # Create timestamps
                chunk_timestamps = [
                    start_time
                    + pd.Timedelta(microseconds=start_usec + idx * (1e6 / fs))
                    for idx in range(len(chunk_df))
                ]

                # Prepare data for insertion
                columns = ["timestamp"] + list(chunk_df.columns)
                values = [
                    tuple([ts] + row.tolist())
                    for ts, row in zip(chunk_timestamps, chunk_df.values)
                ]

                # Insert data
                with conn.cursor() as cursor:
                    execute_values(
                        cursor,
                        f"""
                        INSERT INTO eeg_data ({', '.join(f'"{col}"' for col in columns)})
                        VALUES %s
                        """,
                        values,
                        page_size=1000,
                    )

            conn.commit()

        print("Data successfully stored in PostgreSQL")

    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pull data from IEEG dataset and store in PostgreSQL"
    )
    parser.add_argument("minutes", type=int, help="Number of minutes to pull")
    parser.add_argument(
        "--no-nan",
        action="store_true",
        help="Remove NaN values from the dataset (default: keep NaN)",
    )

    args = parser.parse_args()

    if args.minutes <= 0:
        print("Error: Number of minutes must be positive")
        sys.exit(1)

    main(args.minutes, remove_nan=args.no_nan)
