import os
import psycopg2
from dotenv import load_dotenv

def test_redshift_connection():
    """Test connection to Redshift database using credentials from .env file."""
    # Load environment variables
    load_dotenv()
    
    # Get credentials from environment variables
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    dbname = os.getenv('DB_DATABASE')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    
    # Print connection details (without password)
    print(f"Attempting to connect to Redshift:")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Database: {dbname}")
    print(f"User: {user}")
    
    try:
        # Attempt to connect
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        
        # Test the connection with a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        
        # Close the connection
        cursor.close()
        conn.close()
        
        # Print success message
        print("\n✅ Connection successful!")
        print("Your Redshift credentials are working correctly.")
        return True
    
    except Exception as e:
        # Print error message
        print("\n❌ Connection failed!")
        print(f"Error: {e}")
        print("\nPlease check your .env file and make sure your credentials are correct.")
        return False

if __name__ == "__main__":
    test_redshift_connection()