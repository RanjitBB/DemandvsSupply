#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found."
    echo "Please create a .env file with your Redshift credentials."
    echo "Example:"
    echo "DB_HOST=your_redshift_host"
    echo "DB_PORT=your_redshift_port"
    echo "DB_DATABASE=your_redshift_database"
    echo "DB_USER=your_redshift_username"
    echo "DB_PASSWORD=your_redshift_password"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Test database connection
echo "Testing database connection..."
python3 test_connection.py

# Check if connection test was successful
if [ $? -ne 0 ]; then
    echo "Database connection test failed. Please check your credentials and try again."
    exit 1
fi

# Run the dashboard
echo "Starting the dashboard..."
streamlit run main.py