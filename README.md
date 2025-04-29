# Supply-Demand Dashboard

A Streamlit-based dashboard for visualizing the supply-demand gap for tutors across different regions, grade levels, and time periods.

## Features

- Interactive heatmap visualization showing supply-demand metrics
- Filtering by:
  - Region (NAM, APAC, EMEA, IND-SUB)
  - Grade Level (k-2, 3-5, 6-8, 9-10, 11-12)
  - Time Period (Current Week, Past 2 Weeks, Past 4 Weeks)
  - Assignment Round (All, assignment_round_1, assignment_round_2, etc.)
  - Tutor Set Name (All, training_complete_tutors_with_zero_trials, etc.)
- Color-coded cells indicating surplus/shortage levels
- Detailed metrics for each time slot and day combination

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your Redshift credentials:
   ```
   DB_HOST=your_redshift_host
   DB_PORT=your_redshift_port
   DB_DATABASE=your_redshift_database
   DB_USER=your_redshift_username
   DB_PASSWORD=your_redshift_password
   ```

4. Test your database connection:
   ```
   python test_connection.py
   ```
   This will verify that your Redshift credentials are working correctly.

## Running the Dashboard

### Option 1: Using the shell script (recommended)

Run the following command in your terminal:

```
./run_dashboard.sh
```

This script will:
1. Check if Python and pip are installed
2. Verify that the .env file exists
3. Install the required dependencies
4. Test the database connection
5. Start the Streamlit server

### Option 2: Manual execution

Alternatively, you can run the dashboard manually:

```
streamlit run main.py
```

This will start the Streamlit server and open the dashboard in your default web browser.

## Dashboard Usage

1. Use the sidebar filters to select your desired parameters
2. The heatmap will update automatically based on your selections
3. Each cell in the heatmap shows:
   - S: Supply (eligible tutors)
   - D: Demand (trial requests)
   - Ratio: Supply ÷ Demand
   - Percentage: (Supply-Demand)/Demand × 100%
4. Color coding indicates the supply-demand gap:
   - Dark green: ≥ 50% surplus
   - Light green: 20% to 50% surplus
   - Yellow: -20% to 20% balanced
   - Orange: -50% to -20% shortage
   - Red: ≤ -50% severe shortage
5. Expand the "View Raw Data" section to see the underlying data

## Data Source

The dashboard connects to a Redshift database and executes a query to retrieve tutor supply and demand data. The data is then processed and visualized in the heatmap.
