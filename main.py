import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import psycopg2
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import calendar


# Set page configuration
st.set_page_config(
    page_title="Cuemath Supply-Demand Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Database connection function
def connect_to_redshift():
    try:
        # Check if all required environment variables are present
        required_vars = ['DB_HOST', 'DB_PORT', 'DB_DATABASE', 'DB_USER', 'DB_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return None
            
        # Add debug output for connection parameters (excluding password)
        conn_debug = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'database': os.getenv('DB_DATABASE'),
            'user': os.getenv('DB_USER')
        }
        st.sidebar.write("Connection parameters:", conn_debug)
        
        # Attempt connection
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            dbname=os.getenv('DB_DATABASE'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        return conn
        
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        return None

# Function to execute query and return dataframe
def execute_query(query):
    try:
        conn = connect_to_redshift()
        if conn is None:
            st.error("Failed to connect to database")
            return pd.DataFrame()
            
        # Add debug output for connection
        st.sidebar.write("Database connection successful")
        
        # Execute query with error handling
        try:
            df = pd.read_sql_query(query, conn)
            st.sidebar.write(f"Query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            st.error(f"Error executing SQL query: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()
            
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return pd.DataFrame()

# Function to get data based on filters
def get_filtered_data(region, grade_level, time_period, assignment_round, tutor_set_name):

    # Base query
    query = """
    WITH base_data AS (
      SELECT
        id as trial_id,
        meta.assessment_id as id,
        created_on,
        trial_slot,
        region,
        grade_group,
        meta
      FROM application_service_teacher.auto_mapping_trial_request
      WHERE created_on >= '{start_date}'
    ),
    all_rounds AS (
      SELECT
        id,   trial_id, created_on, trial_slot, region, grade_group, 'assignment_round_1' AS round_name,
        meta.teacher_logs.assignment_round_1 AS round_data
      FROM base_data
      WHERE meta.teacher_logs.assignment_round_1 IS NOT NULL
      UNION ALL
      SELECT
        id,   trial_id, created_on, trial_slot, region, grade_group, 'assignment_round_2',
        meta.teacher_logs.assignment_round_2
      FROM base_data
      WHERE meta.teacher_logs.assignment_round_2 IS NOT NULL
      UNION ALL
      SELECT
        id,   trial_id, created_on, trial_slot, region, grade_group, 'assignment_round_3',
        meta.teacher_logs.assignment_round_3
      FROM base_data
      WHERE meta.teacher_logs.assignment_round_3 IS NOT NULL
      UNION ALL
      SELECT
        id,   trial_id, created_on, trial_slot, region, grade_group, 'assignment_round_4',
        meta.teacher_logs.assignment_round_4
      FROM base_data
      WHERE meta.teacher_logs.assignment_round_4 IS NOT NULL
      UNION ALL
      SELECT
        id,   trial_id, created_on, trial_slot, region, grade_group, 'assignment_round_5',
        meta.teacher_logs.assignment_round_5
      FROM base_data
      WHERE meta.teacher_logs.assignment_round_5 IS NOT NULL
    ),
    all_sets AS (
      SELECT
        id,   trial_id, created_on, trial_slot, region, grade_group, round_name, 'training_complete_tutors_with_zero_trials' as tutor_set_name ,
        round_data.training_complete_tutors_with_zero_trials AS tutor_data
      FROM all_rounds
      WHERE round_data.training_complete_tutors_with_zero_trials.can_map is null
      UNION ALL
      SELECT
        id,   trial_id, created_on, trial_slot, region, grade_group, round_name, 'training_complete_tutors_with_one_or_two_trials' as tutor_set_name,
        round_data.training_complete_tutors_with_one_or_two_trials AS tutor_data
      FROM all_rounds
      WHERE round_data.training_complete_tutors_with_one_or_two_trials.can_map is null
      UNION ALL
      SELECT
        id,   trial_id, created_on, trial_slot, region, grade_group, round_name, 'active_tutors_older_version_training' AS tutor_set_name,
        round_data.active_tutors_older_version_training AS tutor_data
      FROM all_rounds
      WHERE round_data.active_tutors_older_version_training.can_map is null
      UNION ALL
      SELECT
        id,   trial_id, created_on, trial_slot, region, grade_group, round_name, 'active_tutors_latest_version_training' as tutor_set_name,
        round_data.active_tutors_latest_version_training AS tutor_data
      FROM all_rounds
      WHERE round_data.active_tutors_latest_version_training.can_map is null
    )
    SELECT
      id AS "assessment_id",
      trial_id,
      TO_CHAR(created_on, 'YYYY-MM-DD') AS "created_date",
      TO_CHAR(created_on, 'HH24') AS "Window Time",
      trial_slot.from AS "Trial Request At",
      region AS "Trial Region",
      grade_group AS "Trial Grade",
      round_name AS "Assignment Round",
      tutor_set_name AS "Tutor Set Name",
      tutor_data.total_teachers::INT AS total_teachers,
      tutor_data.after_previously_mapped_teachers,
      tutor_data.after_teachers_already_mapped_in_current_window,
      tutor_data.after_snoozed_teacher,
      tutor_data.after_ineligible_professional_review,
      tutor_data.after_disabled_region,
      tutor_data.after_max_open_trials,
      tutor_data.after_paused_on_trial_date,
      tutor_data.after_training_version_not_completed,
      tutor_data.after_max_demo_on_trial_day,
      COALESCE(tutor_data.availability_matched, 0) + COALESCE(tutor_data.availability_mismatched,0) as total_eligible_teachers,
      tutor_data.availability_matched,
      tutor_data.availability_mismatched

FROM all_sets 
  """
    
    # Calculate date range based on time period
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    if time_period == 'Current Week':
        start_date = yesterday - timedelta(days=6)
    elif time_period == 'Past 2 Weeks':
        start_date = yesterday - timedelta(days=13)
    elif time_period == 'Past 4 Weeks':
        start_date = yesterday - timedelta(days=27)
    
    # Add filters to query
    filters = []
    
    # Replace placeholder in base query
    query = query.format(start_date=start_date)
    
    # Add WHERE clause if there are filters
    if region != 'All':
        filters.append(f'"Trial Region" = \'{region}\'')
    
    if grade_level != 'All':
        filters.append(f'"Trial Grade" = \'{grade_level}\'')
    
    if assignment_round != 'All':
        filters.append(f'"Assignment Round" = \'{assignment_round}\'')
    
    if tutor_set_name != 'All':
        filters.append(f'"Tutor Set Name" = \'{tutor_set_name}\'')
    
    if filters:
        query += " WHERE " + " AND ".join(filters)
    
    query += " ORDER BY \"Trial Request At\" DESC"
    
        # Execute query and return dataframe
    try:
        # Create debug expander for query details
        query_debug = st.sidebar.expander("SQL Query Debug", expanded=True)
        
        # Show query parameters
        query_debug.markdown("### Query Parameters")
        query_debug.write({
            "Region": region,
            "Grade Level": grade_level,
            "Time Period": time_period,
            "Assignment Round": assignment_round,
            "Tutor Set Name": tutor_set_name,
            "Start Date": start_date
        })
        
        # Show applied filters
        query_debug.markdown("### Applied Filters")
        query_debug.write(filters if filters else "No filters applied")
        
        # Show final query
        query_debug.markdown("### Final SQL Query")
        query_debug.code(query, language="sql")
        
        # Execute query
        df = execute_query(query)
        
        # Show query results
        query_debug.markdown("### Query Results")
        query_debug.write({
            "Rows Retrieved": len(df),
            "Columns": df.columns.tolist(),
            "Data Types": {col: str(df[col].dtype) for col in df.columns}
        })
        
        # Show sample data
        if not df.empty:
            query_debug.markdown("### Sample Data")
            query_debug.dataframe(df.head(5))
            
            # Show unique values in key columns
            query_debug.markdown("### Unique Values in Key Columns")
            for col in ['Trial Region', 'Trial Grade', 'Assignment Round', 'Tutor Set Name']:
                if col in df.columns:
                    query_debug.write(f"{col}: {df[col].nunique()} unique values")
                    query_debug.write(df[col].unique().tolist())

        return df
    except Exception as e:
        st.error("Error executing query")
        error_debug = st.sidebar.expander("Error Details", expanded=True)
        error_debug.markdown("### Error Information")
        error_debug.error(f"Error Type: {type(e).__name__}")
        error_debug.error(f"Error Message: {str(e)}")
        error_debug.markdown("### Query that caused error")
        error_debug.code(query, language="sql")
        return pd.DataFrame()

# Function to process data for heatmap
def process_data_for_heatmap(df, supply_metric='availability_matched'):
    if df.empty:
        return pd.DataFrame()

    
    # Create debug expander
    debug_expander = st.sidebar.expander("Data Processing Debug", expanded=True)
    
    # Input data overview
    debug_expander.markdown("### 1. Input Data")
    debug_expander.write({
        "Number of rows": len(df),
        "Columns": df.columns.tolist(),
        "Data types": df.dtypes.to_dict()
    })
    
    # Convert all numeric columns to int
    numeric_columns = [
        'total_teachers', 'after_previously_mapped_teachers',
        'after_teachers_already_mapped_in_current_window', 'after_snoozed_teacher',
        'after_ineligible_trial_state', 'after_ineligible_professional_review',
        'after_disabled_region', 'after_max_open_trials', 'after_paused_on_trial_date',
        'after_training_version_not_completed', 'after_max_demo_on_trial_day',
        'total_eligible_teachers', 'availability_matched', 'availability_mismatched'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
    # Rename columns to match expected names (case-insensitive)
    column_mapping = {}
    for col in df.columns:
        if col.lower() == 'trial request at':
            column_mapping[col] = 'Trial Request At'
        elif col.lower() == 'assessment_id':
            column_mapping[col] = 'assessment_id'
        elif col.lower() == 'availability_matched':
            column_mapping[col] = 'availability_matched'
    
    # Create a copy of the dataframe with renamed columns
    df_renamed = df.copy()
    df_renamed.rename(columns=column_mapping, inplace=True)
    
    
    # Check if required columns exist after renaming
    required_columns = ['Trial Request At', 'assessment_id', 'availability_matched']
    missing_columns = [col for col in required_columns if col not in df_renamed.columns]
    
    if missing_columns:
        
        # Try to find columns by case-insensitive match
        for req_col in missing_columns:
            matches = [col for col in df_renamed.columns if col.lower() == req_col.lower()]
            if matches:
                print(f"Possible match for '{req_col}': {matches}")
        
        return pd.DataFrame()
    
    # Use the renamed dataframe for further processing
    df = df_renamed
    
    # Convert Unix timestamp to IST datetime
    df['Trial Request At'] = pd.to_datetime(df['Trial Request At'], unit='s')  # Convert Unix timestamp to datetime
    df['Trial Request At'] = df['Trial Request At'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')  # Convert to IST
    
    # Add debug output
    debug_expander = st.sidebar.expander("Timestamp Debug", expanded=True)
    debug_expander.write("Sample timestamps after conversion:")
    debug_expander.write(df['Trial Request At'].head().to_dict())

    
    # Extract day of week and hour
    df['Day'] = df['Trial Request At'].dt.day_name()
    df['Hour'] = df['Trial Request At'].dt.hour
    

    
    # Create comprehensive time slot mapping for all 24 hours
    hour_to_slot = {
        0: '11 PM - 1 AM',  # 12 AM falls in 11 PM - 1 AM slot
        1: '1 AM - 3 AM',
        2: '1 AM - 3 AM',
        3: '3 AM - 5 AM',
        4: '3 AM - 5 AM',
        5: '5 AM - 7 AM',
        6: '5 AM - 7 AM',
        7: '7 AM - 9 AM',
        8: '7 AM - 9 AM',
        9: '9 AM - 11 AM',
        10: '9 AM - 11 AM',
        11: '11 AM - 1 PM',
        12: '11 AM - 1 PM',
        13: '1 PM - 3 PM',
        14: '1 PM - 3 PM',
        15: '3 PM - 5 PM',
        16: '3 PM - 5 PM',
        17: '5 PM - 7 PM',
        18: '5 PM - 7 PM',
        19: '7 PM - 9 PM',
        20: '7 PM - 9 PM',
        21: '9 PM - 11 PM',
        22: '9 PM - 11 PM',
        23: '11 PM - 1 AM'
    }
    
    # Map hours to time slots - comprehensive version
    def map_hour_to_slot(hour):
        if hour is None or pd.isna(hour):
            # Default to a reasonable time slot instead of 'Other'
            debug_expander.write(f"WARNING: Found null hour value, defaulting to '7 AM - 9 AM'")
            return '7 AM - 9 AM'
        
        try:
            hour = int(hour)
            if hour < 0 or hour > 23:
                debug_expander.write(f"WARNING: Hour value {hour} out of range, defaulting to '7 AM - 9 AM'")
                return '7 AM - 9 AM'
            
            return hour_to_slot[hour]
        except (ValueError, TypeError):
            debug_expander.write(f"WARNING: Invalid hour value {hour}, defaulting to '7 AM - 9 AM'")
            return '7 AM - 9 AM'
    
    df['Time Slot'] = df['Hour'].apply(map_hour_to_slot)
    
    # Verify no 'Other' values in Time Slot
    other_count = (df['Time Slot'] == 'Other').sum()
    if other_count > 0:
        debug_expander.write(f"WARNING: {other_count} rows still have 'Other' as Time Slot!")
    
    # Check time slot mapping
    debug_expander.write("Time Slot mapping:")
    debug_expander.write(df['Time Slot'].value_counts().to_dict())
    
    # Group by day and time slot
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_slots_order = [
        '1 AM - 3 AM',
        '3 AM - 5 AM',
        '5 AM - 7 AM',
        '7 AM - 9 AM',
        '9 AM - 11 AM',
        '11 AM - 1 PM',
        '1 PM - 3 PM',
        '3 PM - 5 PM',
        '5 PM - 7 PM',
        '7 PM - 9 PM',
        '9 PM - 11 PM',
        '11 PM - 1 AM'
    ]
    
    # Calculate demand (unique assessment_id count)
    demand_df = df.groupby(['Day', 'Time Slot'])['assessment_id'].nunique().reset_index()
    demand_df.rename(columns={'assessment_id': 'Demand'}, inplace=True)
    
    debug_expander.write("### Demand Calculation")
    debug_expander.write("Grouped by Day and Time Slot:")
    debug_expander.dataframe(demand_df)
    
    # Use the selected supply metric for calculation
    debug_expander.write(f"Using supply metric: {supply_metric}")
    
    if supply_metric in df.columns:
        df['supply_value'] = pd.to_numeric(df[supply_metric], errors='coerce').fillna(0)
        sum_val = df['supply_value'].sum()
        debug_expander.write(f"Sum of {supply_metric}: {sum_val}")
    else:
        debug_expander.write(f"WARNING: Selected metric '{supply_metric}' not found in columns!")
        debug_expander.write("Available columns:", df.columns.tolist())
        df['supply_value'] = 0
    
    debug_expander.write("### Supply Values")
    debug_expander.write("First 10 supply values:")
    debug_expander.write(df['supply_value'].head(10).tolist())
    debug_expander.write(f"Total supply: {df['supply_value'].sum()}")
    
    # Calculate supply (average of supply_value)
    supply_df = df.groupby(['Day', 'Time Slot'])['supply_value'].mean().reset_index()
    supply_df.rename(columns={'supply_value': 'Supply'}, inplace=True)
    
    debug_expander.write("### Supply Calculation")
    debug_expander.write("Grouped by Day and Time Slot:")
    debug_expander.dataframe(supply_df)
    
    # Merge demand and supply
    result_df = pd.merge(demand_df, supply_df, on=['Day', 'Time Slot'], how='outer')
    
    # Fill NaN values with 0
    result_df.fillna(0, inplace=True)
    
    # Calculate ratio and percentage
    result_df['Ratio'] = result_df['Supply'] / result_df['Demand'].replace(0, np.nan)
    result_df['Percentage'] = (result_df['Supply'] - result_df['Demand']) / result_df['Demand'].replace(0, np.nan) * 100
    
    # Fill NaN values with 0 again
    result_df.fillna(0, inplace=True)
    
    # Round values
    result_df['Supply'] = result_df['Supply'].round(0).astype(int)
    result_df['Demand'] = result_df['Demand'].astype(int)
    result_df['Ratio'] = result_df['Ratio'].round(2)
    result_df['Percentage'] = result_df['Percentage'].round(0).astype(int)
    
    # Create a complete grid with all days and time slots
    grid = pd.DataFrame([(day, slot) for day in days_order for slot in time_slots_order], 
                        columns=['Day', 'Time Slot'])
    
    # Merge with the result to ensure all combinations exist
    final_df = pd.merge(grid, result_df, on=['Day', 'Time Slot'], how='left')
    final_df.fillna(0, inplace=True)
    
    # Convert to pivot table format for heatmap
    pivot_df = final_df.pivot(index='Time Slot', columns='Day',
                             values=['Supply', 'Demand', 'Ratio', 'Percentage'])
    
    # Reorder indices and columns
    pivot_df = pivot_df.reindex(index=time_slots_order)
    
    # Ensure all days are present in the correct order
    for metric in ['Supply', 'Demand', 'Ratio', 'Percentage']:
        for day in days_order:
            if (metric, day) not in pivot_df.columns:
                # Add missing columns with NaN values
                pivot_df[(metric, day)] = np.nan
    
    # Reorder columns to match days_order
    column_order = []
    for metric in ['Supply', 'Demand', 'Ratio', 'Percentage']:
        for day in days_order:
            column_order.append((metric, day))
    
    # Filter to only include columns that exist in the DataFrame
    existing_columns = [col for col in column_order if col in pivot_df.columns]
    pivot_df = pivot_df[existing_columns]
    
    return pivot_df

# Function to create heatmap
def create_heatmap(pivot_df):
    # Create debug expander
    debug_expander = st.sidebar.expander("Heatmap Debug", expanded=True)
    
    if pivot_df.empty:
        st.warning("No data available for the selected filters.")
        debug_expander.write("Pivot DataFrame is empty")
        return
        
    # Debug pivot table structure
    debug_expander.markdown("### Pivot Table Structure")
    debug_expander.write({
        "Index (Time Slots)": pivot_df.index.tolist(),
        "Column Levels": [list(level) for level in pivot_df.columns.levels],
        "Shape": pivot_df.shape
    })
    
    # Ensure days are in correct order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_slots_order = [
        '7 AM - 9 AM',
        '9 AM - 11 AM',
        '11 AM - 1 PM',
        '1 PM - 3 PM',
        '3 PM - 5 PM',
        '5 PM - 7 PM',
        '7 PM - 9 PM',
        '9 PM - 11 PM',
        '11 PM - 1 AM',
        '1 AM - 3 AM',
        '3 AM - 5 AM',
        '5 AM - 7 AM',
    ]
    time_slots_order = list(reversed(time_slots_order))

    
    # Create a table-like layout
    fig = go.Figure()
    
    # Get available days and time slots
    available_days = [day for day in days_order if day in pivot_df.columns.levels[1]]
    available_time_slots = [slot for slot in time_slots_order if slot in pivot_df.index]
    
    if not available_days or not available_time_slots:
        st.warning("No data available for the selected filters.")
        return
    
    # Create cells for each day and time slot
    for i, day in enumerate(days_order):
        if day not in pivot_df.columns.levels[1]:
            continue
            
        for j, time_slot in enumerate(time_slots_order):
            if time_slot not in pivot_df.index:
                continue
                
            try:
                supply = pivot_df[('Supply', day)][time_slot]
                demand = pivot_df[('Demand', day)][time_slot]
                ratio = pivot_df[('Ratio', day)][time_slot]
                percentage = pivot_df[('Percentage', day)][time_slot]
            except:
                # Skip if data is missing
                continue
            
            # Determine cell color based on percentage
            if percentage >= 50:
                color = 'rgb(0, 128, 0)'  # Dark green
            elif percentage >= 20:
                color = 'rgb(144, 238, 144)'  # Light green
            elif percentage >= 0:
                color = 'rgb(220, 255, 220)'  # Very light green
            elif percentage >= -25:
                color = 'rgb(255, 255, 0)'  # Yellow
            elif percentage >= -50:
                color = 'rgb(255, 165, 0)'  # Orange
            elif percentage >= -75:
                color = 'rgb(255, 99, 71)'  # Tomato
            else:
                color = 'rgb(255, 0, 0)'  # Red
            
            # Create text for cell with percentage in bold and larger
            text = f"<b style='font-size: 14px'>{percentage}%</b><br>S: {supply} | D: {demand}<br>Ratio: {ratio}"
            
            # Add colored rectangle as shape
            fig.add_shape(
                type="rect",
                x0=i-0.47,  # Make rectangle fill most of the cell
                y0=j-0.47,
                x1=i+0.47,
                y1=j+0.47,
                fillcolor=color,
                line=dict(color='white', width=1),
                layer='below'
            )
            
            # Add text annotation with HTML formatting
            fig.add_annotation(
                x=i,
                y=j,
                text=text,
                showarrow=False,
                font=dict(color="black", size=10),
                align='center',
                bgcolor='rgba(255,255,255,0.3)'  # Semi-transparent white background
            )
    
    # Update layout with adjusted cell sizes
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(days_order))),
            ticktext=days_order,
            tickangle=0,
            side='top',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            range=[-0.5, len(days_order)-0.5],
            fixedrange=True  # Prevent zooming
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(time_slots_order))),
            ticktext=time_slots_order,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            range=[-0.5, len(time_slots_order)-0.5],
            fixedrange=True  # Prevent zooming
        ),
        width=1000,
        height=800,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    
    # Add grid lines as shapes
    for i in range(len(days_order) + 1):
        fig.add_shape(
            type="line",
            x0=i - 0.5,
            y0=-0.5,
            x1=i - 0.5,
            y1=len(time_slots_order) - 0.5,
            line=dict(color="rgba(0,0,0,0.3)", width=1),
            layer="below"
        )
    
    for j in range(len(time_slots_order) + 1):
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=j - 0.5,
            x1=len(days_order) - 0.5,
            y1=j - 0.5,
            line=dict(color="rgba(0,0,0,0.3)", width=1),
            layer="below"
        )
    
    return fig

def generate_summary_from_original_data(df, day, time_slot):
    # Filter the original data to match the selected day and time slot
    filtered_data = df[(df['Day'] == day) & (df['Time Slot'] == time_slot)]

    if filtered_data.empty:
        return f"No data available for the selected combination: {day} - {time_slot}"

    # Extract relevant columns
    summary_columns = [
        'total_teachers', 'after_previously_mapped_teachers', 'after_teachers_already_mapped_in_current_window', 
        'after_snoozed_teacher', 'after_ineligible_trial_state', 'after_ineligible_professional_review',
        'after_disabled_region', 'after_max_open_trials', 'after_paused_on_trial_date',
        'after_training_version_not_completed', 'after_max_demo_on_trial_day',
        'total_eligible_teachers', 'availability_matched', 'availability_mismatched'
    ]

    # Ensure all columns are present in the DataFrame
    missing_columns = [col for col in summary_columns if col not in filtered_data.columns]
    if missing_columns:
        return f"Missing columns: {', '.join(missing_columns)}"

    # Create the summary table with remaining and eliminated tutors
    summary = {
        'Filter Stage': summary_columns,
        'Tutors Remaining': [filtered_data[col].sum() for col in summary_columns],
        '% of Total': [round((filtered_data[col].sum() / filtered_data['total_teachers'].sum()) * 100, 2) for col in summary_columns],
        'Eliminated': [filtered_data['total_teachers'].sum() - filtered_data[col].sum() for col in summary_columns]
    }
    
    summary_df = pd.DataFrame(summary)
    
    # Primary Bottleneck (find the filter stage with maximum elimination)
    eliminated_data = {col: filtered_data['total_teachers'].sum() - filtered_data[col].sum() for col in summary_columns}
    
    # Find the primary bottleneck (filter with the highest eliminated tutors)
    primary_bottleneck = max(eliminated_data, key=eliminated_data.get)
    
    # Return the summary as a string and DataFrame
    result = f"### Filter Breakdown: {time_slot} - {day}\n"
    result += f"Based on {filtered_data['total_teachers'].sum()} unique trial requests | {filtered_data['total_teachers'].sum()} total filter passes\n\n"
    
    result += summary_df.to_markdown(index=False)
    
    result += f"\n\nPrimary Bottleneck: **{primary_bottleneck}**\n"
    result += f"Top 3 filters by impact:\n"
    result += "\n".join([f"{i+1}. {filter_stage}: {eliminated_data[filter_stage]} tutors eliminated" for i, filter_stage in enumerate(sorted(eliminated_data, key=eliminated_data.get, reverse=True)[:3])])
    
    return result
    
# Main app
def main():
    # Title
    st.title("Cuemath Supply-Demand Analysis Dashboard")
    
    # Filters in main area
    with st.container():
        # Primary filters
        st.markdown("### Primary Filters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Region**")
            region_options = ['NAM', 'APAC', 'EMEA', 'IND-SUB']
            region = st.selectbox("Region", region_options, index=region_options.index('NAM'))

        with col2:
            st.markdown("**Grade Level**")
            grade_options = ['k-2', '3-5', '6-8', '9-10', '11-12']
            grade_level = st.selectbox("Grade", grade_options, index=grade_options.index('k-2'))

        with col3:
            st.markdown("**Time Period**")
            time_period_options = ['Current Week', 'Past 2 Weeks', 'Past 4 Weeks']
            time_period = st.selectbox("Time Period", time_period_options)

        with col4:
            st.markdown("**Supply Metric**")
            supply_metrics = {
                'availability_matched': 'Availability Matched',
                'total_teachers': 'Total Teachers',
                'total_eligible_teachers': 'Total Eligible Teachers',
                'after_previously_mapped_teachers': 'After Previously Mapped',
                'after_teachers_already_mapped_in_current_window': 'After Already Mapped',
                'after_snoozed_teacher': 'After Snoozed',
                'after_ineligible_trial_state': 'After Ineligible Trial',
                'after_ineligible_professional_review': 'After Ineligible Review',
                'after_disabled_region': 'After Disabled Region',
                'after_max_open_trials': 'After Max Open Trials',
                'after_paused_on_trial_date': 'After Paused',
                'after_training_version_not_completed': 'After Training Not Complete',
                'after_max_demo_on_trial_day': 'After Max Demo',
                'availability_mismatched': 'Availability Mismatched'
            }
            supply_metric = st.selectbox(
                "Supply Metric",
                options=list(supply_metrics.keys()),
                format_func=lambda x: supply_metrics[x],
                index=0
            )
        
    
    # Create expander for additional filters
    with st.expander("Additional Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # Assignment Round filter
            st.markdown("**Assignment Round**")
            assignment_round_options = ['All', 'assignment_round_1', 'assignment_round_2', 'assignment_round_3', 'assignment_round_4', 'assignment_round_5']
            assignment_round = st.selectbox("", assignment_round_options, label_visibility="collapsed")
        
        with col2:
            # Tutor Set Name filter
            st.markdown("**Tutor Set Name**")
            tutor_set_options = ['All', 'training_complete_tutors_with_zero_trials', 'training_complete_tutors_with_one_or_two_trials', 'active_tutors_older_version_training', 'active_tutors_latest_version_training']
            tutor_set_name = st.selectbox("", tutor_set_options, label_visibility="collapsed")
    
    # Get data based on filters
    with st.spinner("Loading data..."):
        # Add a button to use sample data instead
        use_sample_data = st.sidebar.checkbox("Use Sample Data for Testing", value=False)
        
        # Store in session state
        st.session_state['use_sample_data'] = use_sample_data
        
        # Initialize df
        df = pd.DataFrame()
        
        if not use_sample_data:
            df = get_filtered_data(region, grade_level, time_period, assignment_round, tutor_set_name)
        
        # Check if we should use sample data
        use_sample = use_sample_data or df.empty
        if use_sample:
            if df.empty and not use_sample_data:
                st.warning("No data returned from the query. Using sample data instead.")
            
            # Create sample data
            sample_data = []
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hours = list(range(24))  # Use all 24 hours to ensure coverage of all time slots
            
            for day_idx, day in enumerate(days):
                for hour in hours:
                    # Create a datetime for this day and hour
                    dt = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
                    if day_idx > 0:  # Adjust day
                        dt = dt - timedelta(days=day_idx)
                    
                    # Generate random supply and demand values that create different scenarios
                    # Some with surplus, some with shortage
                    if day_idx < 3:  # First 3 days have surplus
                        supply = np.random.randint(50, 100)
                        demand = np.random.randint(10, 50)
                    elif day_idx < 5:  # Next 2 days are balanced
                        supply = np.random.randint(40, 60)
                        demand = np.random.randint(40, 60)
                    else:  # Last 2 days have shortage
                        supply = np.random.randint(10, 40)
                        demand = np.random.randint(50, 100)
                    
                    # Create sample data with all supply metrics
                    sample_row = {
                        'Trial Request At': dt,
                        'assessment_id': f"sample_{day}_{hour}",
                        'Trial Region': region,
                        'Trial Grade': grade_level,
                        # Add all supply metrics with slightly different values
                        'availability_matched': supply,
                        'total_teachers': supply + np.random.randint(-5, 5),
                        'total_eligible_teachers': supply + np.random.randint(-3, 3),
                        'after_previously_mapped_teachers': supply * 0.9,
                        'after_teachers_already_mapped_in_current_window': supply * 0.8,
                        'after_snoozed_teacher': supply * 0.7,
                        'after_ineligible_trial_state': supply * 0.6,
                        'after_ineligible_professional_review': supply * 0.5,
                        'after_disabled_region': supply * 0.4,
                        'after_max_open_trials': supply * 0.3,
                        'after_paused_on_trial_date': supply * 0.2,
                        'after_training_version_not_completed': supply * 0.15,
                        'after_max_demo_on_trial_day': supply * 0.1,
                        'availability_mismatched': supply * 0.05
                    }
                    sample_data.append(sample_row)
            
            df = pd.DataFrame(sample_data)
            st.sidebar.success("Using sample data for visualization")
    
    # Process and display data
    if not df.empty:
        # Process data for heatmap using selected supply metric
        pivot_df = process_data_for_heatmap(df, supply_metric=supply_metric)
        
        if not pivot_df.empty:
            # Display heatmap title
            region_display = region if region != 'All' else 'All Regions'
            grade_display = grade_level if grade_level != 'All' else 'All Grades'
            st.markdown(f"### {region_display} Region, {grade_display} Grade: Supply-Demand Heatmap")
            
            # Display selected metric
            st.markdown(f"**Supply Metric:** {supply_metrics[supply_metric]}")
            
            st.markdown("Click on any cell to see detailed filter breakdown")
            
            # Display legend
            st.markdown("**Supply-Demand Gap:**")
            legend_cols = st.columns(7)
            with legend_cols[0]:
                st.markdown('<div style="background-color: rgb(0, 128, 0); color: white; padding: 5px; border-radius: 3px; text-align: center; font-size: 12px;">≥50% surplus</div>', unsafe_allow_html=True)
            with legend_cols[1]:
                st.markdown('<div style="background-color: rgb(144, 238, 144); padding: 5px; border-radius: 3px; text-align: center; font-size: 12px;">20-49% surplus</div>', unsafe_allow_html=True)
            with legend_cols[2]:
                st.markdown('<div style="background-color: rgb(220, 255, 220); padding: 5px; border-radius: 3px; text-align: center; font-size: 12px;">0-19% surplus</div>', unsafe_allow_html=True)
            with legend_cols[3]:
                st.markdown('<div style="background-color: rgb(255, 255, 0); padding: 5px; border-radius: 3px; text-align: center; font-size: 12px;">1-25% shortage</div>', unsafe_allow_html=True)
            with legend_cols[4]:
                st.markdown('<div style="background-color: rgb(255, 165, 0); padding: 5px; border-radius: 3px; text-align: center; font-size: 12px;">26-50% shortage</div>', unsafe_allow_html=True)
            with legend_cols[5]:
                st.markdown('<div style="background-color: rgb(255, 99, 71); color: white; padding: 5px; border-radius: 3px; text-align: center; font-size: 12px;">51-75% shortage</div>', unsafe_allow_html=True)
            with legend_cols[6]:
                st.markdown('<div style="background-color: rgb(255, 0, 0); color: white; padding: 5px; border-radius: 3px; text-align: center; font-size: 12px;">>75% shortage</div>', unsafe_allow_html=True)
            
            # Create and display the heatmap
            fig = create_heatmap(pivot_df)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
                        ##### Legend:
                        - **S**: Supply (eligible tutors)
                        - **D**: Demand (trial requests)
                        - **Ratio**: Supply ÷ Demand
                        - **Percentage** shows surplus/shortage relative to demand: (Supply - Demand) / Demand × 100%
                    """)

            with st.expander("Filter Breakdown"):
                col1, col2 = st.columns(2)
                with col1:
                    # Time Slot
                    st.markdown("**Time Slot**")
                    time_slots_order = ['7 AM - 9 AM','9 AM - 11 AM','11 AM - 1 PM','1 PM - 3 PM','3 PM - 5 PM','5 PM - 7 PM','7 PM - 9 PM','9 PM - 11 PM','11 PM - 1 AM','1 AM - 3 AM','3 AM - 5 AM','5 AM - 7 AM']
                    time_slot = st.selectbox("", time_slots_order, label_visibility="collapsed")
                with col2:
                    # Day
                    st.markdown("**Day**")
                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    days = st.selectbox("", days_order, label_visibility="collapsed")
                summary = generate_summary_from_original_data(df, days, time_slot)
                st.markdown(summary)
                    
                
            # Display raw data
            with st.expander("View Raw Data"):
                st.dataframe(df)
            
        else:
            st.warning("No data available for visualization after processing.")
    else:
        st.warning("No data available for the selected filters.")

if __name__ == "__main__":
    main()
