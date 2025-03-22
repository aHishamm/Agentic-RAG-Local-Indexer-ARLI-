def get_database_connection():
    import psycopg2
    from psycopg2 import sql
    import os

    DATABASE_URL = os.getenv('DATABASE_URL')

    try:
        connection = psycopg2.connect(DATABASE_URL)
        return connection
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def create_table_if_not_exists():
    connection = get_database_connection()
    if connection is None:
        return

    cursor = connection.cursor()
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS indexed_files (
        id SERIAL PRIMARY KEY,
        file_name VARCHAR(255) NOT NULL,
        indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    '''
    try:
        cursor.execute(create_table_query)
        connection.commit()
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        cursor.close()
        connection.close()