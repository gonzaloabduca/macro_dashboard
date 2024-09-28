import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf
import json
import requests
import streamlit as st

start = '1900-01-01'  
end = datetime.now().strftime('%Y-%m-%d')
# Initialize FRED with your API key
API_KEY = 'a2fb338b4ef6e2dcb7c667c21b2d1c4e'

# Define the FredPy class
class FredPy:

    def __init__(self, token=None):
        self.token = token
        self.url = "https://api.stlouisfed.org/fred/series/observations" + \
                    "?series_id={seriesID}&api_key={key}&file_type=json" + \
                    "&observation_start={start}&observation_end={end}&units={units}"

    def set_token(self, token):
        self.token = token

    def get_series(self, seriesID, start, end, units='lin'):
        # Format the URL with the values
        url_formatted = self.url.format(
            seriesID=seriesID, start=start, end=end, units=units, key=self.token
        )
        
        # Request the data from FRED API
        response = requests.get(url_formatted)
        
        if self.token:
            if response.status_code == 200:
                # Extract and format the data as a DataFrame
                data = pd.DataFrame(response.json()['observations'])[['date', 'value']] \
                        .assign(date=lambda cols: pd.to_datetime(cols['date'])) \
                        .assign(value=lambda cols: pd.to_numeric(cols['value'], errors='coerce')) \
                        .rename(columns={'value': seriesID})
                
                # This will convert non-numeric values (e.g., '.') to NaN
                return data
            else:
                print(f"Error: Bad response from API, status code = {response.status_code}")
                print(f"URL: {url_formatted}")
                print(f"Response content: {response.content}")
                raise Exception(f"Bad response from API, status code = {response.status_code}")
        else:
            raise Exception("You did not specify an API key.")

# Instantiate FredPy object
fredpy = FredPy()
fredpy.set_token(API_KEY)

def get_indicators(df, start, end):
    
    # Initialize an empty DataFrame to store all indicators
    macro_indicators = pd.DataFrame()

    # Loop through each indicator, fetch the data, and merge it into the main DataFrame
    for name, series_id in df.items():
        print(f"Fetching data for {name} ({series_id})")
        
        # Fetch the series data using the get_series method
        try:
            series_data = fredpy.get_series(
                seriesID=series_id,
                start=start,
                end=end
            )
            
            # Rename the 'value' column to the name of the series (key)
            series_data = series_data.rename(columns={series_id: name})
            
            # Merge the series data with the macro_indicators DataFrame
            if macro_indicators.empty:
                macro_indicators = series_data.set_index('date')
            else:
                macro_indicators = macro_indicators.merge(series_data.set_index('date'), on='date', how='outer')

        except Exception as e:
            print(f"Failed to fetch data for {name} ({series_id}): {e}")

    # Display the first few rows of the final DataFrame
    return macro_indicators


def generate_macro_board(macro_dict, start, end):
    """
    This function generates a macro performance board for given macro indicators over specific time periods.

    Parameters:
    - macro_dict: Dictionary where keys are macro indicator names and values are their FRED API codes.
    - start: Start date for downloading the data.
    - end: End date for downloading the data.

    Returns:
    - A styled DataFrame with macro performance metrics and percentile rank for 1-year performance.
    """
    # Download the macro indicator data
    macro_data = get_indicators(macro_dict, start=start, end=end)

    macro_data = macro_data.resample('M').last().ffill()

    # Initialize an empty DataFrame for the board
    macro_board = pd.DataFrame()

    # Add the 'Current Value' column
    macro_board['Current Value'] = macro_data.iloc[-1]

    current_performance = macro_data.pct_change(periods=12).iloc[-1] 
    mean_1y_perf = macro_data.pct_change(periods=12).mean()  # Mean of the 1 year performance
    std_1y_perf = macro_data.pct_change(periods=12).std()    # Standard deviation of the 1 year performance
    z_score = (current_performance - mean_1y_perf) / std_1y_perf
    

    # Add performance columns for different periods (1 month, 3 months, 1 year)
    macro_board = pd.concat([macro_board,
                              macro_data.pct_change(periods=1).iloc[-1].rename('1 month Perf'),
                              macro_data.pct_change(periods=3).iloc[-1].rename('3 month Perf'),
                              macro_data.pct_change(periods=12).iloc[-1].rename('1 year Perf'),
                              macro_data.pct_change(periods=12).rank(pct=True).iloc[-1].rename('1y Percentile Rank'),
                              z_score.rename('Z-Score')],
                             axis=1)
    
    # Reorder columns
    macro_board = macro_board[['Current Value', '1 month Perf', '3 month Perf', '1 year Perf', 'Z-Score']]

    # Format the DataFrame for better readability
    macro_board = macro_board.style.format({
        'Current Value': "{:.2f}",
        '1 month Perf': "{:.2%}",
        '3 month Perf': "{:.2%}",
        '1 year Perf': "{:.2%}",
        'Z-Score': "{:.2f}"
    })

    # Return the styled DataFrame
    return macro_board

def generate_sector_board(sectors_dict, start, end):
    """
    Generates a sector performance board for given sectors over specific time periods.
    """
    try:
        # Download adjusted close prices for the given tickers in sectors_dict
        sectors_data = yf.download(tickers=list(sectors_dict.keys()), start=start, end=end)['Adj Close']
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()

    # Handle missing data
    sectors_data = sectors_data.ffill()

    # Initialize an empty DataFrame for the board
    sector_board = pd.DataFrame()

    # Add the 'Current Price' column
    sector_board['Current Price'] = sectors_data.iloc[-1]

    # Calculate performance metrics
    try:
        current_performance = sectors_data.pct_change(periods=252).iloc[-1]
        mean_1y_perf = sectors_data.pct_change(periods=252).mean()  # Mean of the 1-year performance
        std_1y_perf = sectors_data.pct_change(periods=252).std()    # Standard deviation of the 1-year performance
        z_score = (current_performance - mean_1y_perf) / std_1y_perf
    except Exception as e:
        print(f"Error calculating Z-Score: {e}")
        return pd.DataFrame()

    # Add performance columns for different periods (1 week, 1 month, 3 months, 1 year)
    sector_board = pd.concat([sector_board,
                              sectors_data.pct_change(periods=6).iloc[-1].rename('1 week Perf'),
                              sectors_data.pct_change(periods=21).iloc[-1].rename('1 month Perf'),
                              sectors_data.pct_change(periods=63).iloc[-1].rename('3 month Perf'),
                              sectors_data.pct_change(periods=252).iloc[-1].rename('1 year Perf'),
                              sectors_data.pct_change(periods=252).rank(pct=True).iloc[-1].rename('1y Percentile Rank'),
                              z_score.rename('Z-Score')],
                             axis=1)

    # Create the 'Asset' column by mapping sector names from sectors_dict
    sector_board['Asset'] = sector_board.index.map(sectors_dict)

    # Reorder columns to have 'Asset' as the first column
    sector_board = sector_board[['Asset', 'Current Price', '1 week Perf', '1 month Perf', '3 month Perf', '1 year Perf', 'Z-Score', '1y Percentile Rank']]

    # Return the raw DataFrame, without applying styling yet
    return sector_board

macro_indicators_dict = {
    'Consumer Sentiment Index' : 'UMCSENT',
    'Building Permits': 'PERMIT',
    'Retail Money Market Funds': 'WRMFNS',
    'Capital Goods New Orders' : 'NEWORDER',
    'Manufacturers New Orders': 'AMTMNO',
    'Job Openings' : 'JTSJOL',
}

savings = {
    'Personal Savings Rate' : 'PSAVERT'
}

consumer_dict = {    
    'Consumer Sentiment Index' : 'UMCSENT',
    'Building Permits': 'PERMIT',
}


nowcast = {
    'Real GPD' : 'GDPC1',
    'Headline CPI' : 'CPIAUCSL',
    'Real PCE' : 'PCEC96',
    'PCE Deflator':'PCEPI',
}

foodnenergy = {
    "ZC=F": "Corn Futures",
    "ZS=F": "Soybean Futures",
    "KE=F": "Wheat Futures",
    "ZR=F": "Rough Rice Futures",
    "CL=F": "Crude Oil Futures",
    "NG=F": "Natural Gas Futures",
    "HO=F": "Heating Oil Futures",
    "RB=F": "RBOB Gasoline Futures",
    "CC=F": "Cocoa Futures",
    "CT=F": "Cotton Futures",
    "SB=F": "Sugar Futures",
    "LE=F": "Live Cattle Futures",
    "HE=F": "Lean Hogs Futures"
}


ism = pd.read_excel("ISM Data.csv", index_col='Date')

ism_roc = ism - ism.shift(3)

ism_roc_manufacturing = ism_roc[['ISM Manufacturing Index', 'Manufacturing New Orders',
       'Manufacturing Production', 'Manufacturing Employment',
       'Manufacturing Deliveries', 'Manufacturing Inventories']]

ism_roc_services = ism_roc [[
       'ISM Non-Manufacturing Index', 'Non-Manufacturing Business Activity',
       'Non-Manufacturing New Orders', 'Non-Manufacturing Employment',
       'Non-Manufacturing Deliveries', 'Non-Manufacturing Inventories'
]]

ism['ISM Composite'] = (ism['ISM Manufacturing Index'] + ism['ISM Non-Manufacturing Index']) / 2

manufacturing_ism = ism[['ISM Manufacturing Index',
                         'Manufacturing New Orders',
                         'Manufacturing Production',
                         'Manufacturing Employment',
                         'Manufacturing Deliveries',
                         'Manufacturing Inventories',
                         ]]

services_ism = ism[[ 'ISM Non-Manufacturing Index', 'Non-Manufacturing Business Activity',
                     'Non-Manufacturing New Orders', 'Non-Manufacturing Employment',
                     'Non-Manufacturing Deliveries', 'Non-Manufacturing Inventories'
                     ]]

services_ism = services_ism.dropna()


manufacturing_board = pd.DataFrame()

manufacturing_board['Current Value'] = manufacturing_ism.iloc[-1]
manufacturing_board['MoM Change'] = (manufacturing_ism.iloc[-1] - manufacturing_ism.iloc[-2])
manufacturing_board['3 month Change'] = (manufacturing_ism.iloc[-1] - manufacturing_ism.iloc[-4])

manufacturing_board['Mean'] = ism_roc_manufacturing.iloc[-120:].mean()
manufacturing_board['StdDev'] = ism_roc_manufacturing.iloc[-120:].std()

manufacturing_board['3 Month ROC Z-Score'] = (ism_roc_manufacturing.iloc[-1] - manufacturing_board['Mean']) / manufacturing_board['StdDev']
percentile_rank = ism_roc_manufacturing.rank(pct=True)
manufacturing_board['Percentile Rank'] = percentile_rank.iloc[-1]

manufacturing_board = manufacturing_board.drop(columns=['Mean', 'StdDev'])

manufacturing_board = manufacturing_board.style.format({
    'Current Value': "{:.2f}",
    'MoM Change': "{:.2f}",
    '3 month Change': "{:.2f}",
    '3 Month ROC Z-Score': "{:.2f}",
    'Percentile Rank' : "{:.2%}"
})


services_board = pd.DataFrame()

services_board['Current Value'] = services_ism.iloc[-1]
services_board['MoM Change'] = (services_ism.iloc[-1] - services_ism.iloc[-2])
services_board['3 month Change'] = (services_ism.iloc[-1] - services_ism.iloc[-4])

services_board['Mean'] = ism_roc_services.iloc[-120:].mean()
services_board['StdDev'] = ism_roc_services.iloc[-120:].std()

services_board['3 Month ROC Z-Score'] = (ism_roc_services.iloc[-1] - services_board['Mean']) / services_board['StdDev']
services_percentile_rank = ism_roc_services.rank(pct=True)
services_board['Percentile Rank'] = services_percentile_rank.iloc[-1]

services_board = services_board.drop(columns=['Mean', 'StdDev'])

services_board = services_board.style.format({
    'Current Value': "{:.2f}",
    'MoM Change': "{:.2f}",
    '3 month Change': "{:.2f}",
    '3 Month ROC Z-Score': "{:.2f}",
    'Percentile Rank' : "{:.2%}"
})

macros = get_indicators(macro_indicators_dict, start=start, end=end).resample('M').last().ffill()
personal_savings = get_indicators(savings, start=start, end=end)
consumer = macros['Consumer Sentiment Index'].to_frame(name='Consumer Sentiment Index')
macros = macros[['Building Permits', 'Retail Money Market Funds', 'Capital Goods New Orders', 'Manufacturers New Orders', 'Job Openings']]

macro_board = pd.DataFrame()

macro_board['Current Value'] = macros.iloc[-1]
macro_board['3m Change'] = macros.pct_change(periods=3).iloc[-1]
macro_board['1y Change'] = macros.pct_change(periods=12).iloc[-1]
macro_board['Percentile Rank'] = macros.pct_change(periods=12).rank(pct=True).iloc[-1]

macro_board.style.format({
    'Current Value': "{:.1f}",
    '3m Change': "{:.2%}",
    '1y Change': "{:.2%}",
    'Percentile Rank' : "{:.2%}"
})

c_macro_board = pd.DataFrame()

c_macro_board['Current Value'] = consumer.iloc[-1]
c_macro_board['3m Change'] = consumer.pct_change(periods=3).iloc[-1]
c_macro_board['1y Change'] = consumer.pct_change(periods=12).iloc[-1]
c_macro_board['Percentile Rank'] = consumer.rank(pct=True).iloc[-1]

c_macro_board.style.format({
    'Current Value': "{:.1f}",
    '3m Change': "{:.2f}",
    '1y Change': "{:.2f}",
    'Percentile Rank' : "{:.2%}"
})


savings_board = pd.DataFrame()

savings_board['Current Value'] = personal_savings.iloc[-1]
savings_board['3m Change'] = (personal_savings.iloc[-1] - personal_savings.iloc[-4])
savings_board['1y Change'] = (personal_savings.iloc[-1] - personal_savings.iloc[-13])
savings_board['Percentile Rank'] = personal_savings.rank(pct=True).iloc[-1]

savings_board.style.format({
    'Current Value': "{:.1f}",
    '3m Change': "{:.2f}",
    '1y Change': "{:.2f}",
    'Percentile Rank' : "{:.2%}"
})

macro_board = pd.concat([c_macro_board, macro_board, savings_board])

macro_board = macro_board.style.format({
    'Current Value': "{:.1f}",
    '3m Change': "{:.2%}",
    '1y Change': "{:.2%}",
    'Percentile Rank' : "{:.2%}"
})

food_energy = generate_sector_board(foodnenergy, start=start, end=end)
fe_data = yf.download(tickers=list(foodnenergy.keys()), start=start, end=end)['Adj Close']

st.set_page_config(layout="wide", page_title="Macro Dashboard")
# Title of the Dashboard
st.title("Macro Dashboard")

# Section 1: ISM Manufacturing and Services Survey Chart (centered)
st.header("ISM Manufacturing and Services Survey Chart")

# Create a full-width chart for ISM
selected_column_ism = st.selectbox("Select a value to plot", ism.columns)

# Create the Plotly figure
line_fig_ism = go.Figure()

# Add the selected ISM column's line chart (without moving average for slider interaction)
line_fig_ism.add_trace(go.Scatter(
    x=ism.index,
    y=ism[selected_column_ism],
    mode='lines',
    name=selected_column_ism
))

# Add a red dotted line at y = 50
line_fig_ism.add_shape(
    type="line",
    x0=ism.index[0],  # Start of the x-axis
    y0=50,  # Position on the y-axis
    x1=ism.index[-1],  # End of the x-axis
    y1=50,  # Same y position for the end of the line
    line=dict(
        color="red",
        width=2,
        dash="dot"  # Dotted line style
    )
)

# Calculate the default x-axis range for the last 10 years
last_10_years = ism.index[-120]  # Assuming the data is monthly, this slices the last 120 months (10 years)

# Add a range slider to the x-axis to allow interactive time range selection
line_fig_ism.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=12, label="1Y", step="month", stepmode="backward"),
                dict(count=60, label="5Y", step="month", stepmode="backward"),
                dict(count=120, label="10Y", step="month", stepmode="backward"),  # No active=True here
                dict(count=300, label="25Y", step="month", stepmode="backward"),
                dict(step="all", label="MAX")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date",
        range=[last_10_years, ism.index[-1]]  # Set default range to the last 10 years
    ),
    autosize=True,
    title=f"{selected_column_ism} over time",
    yaxis=dict(
        title=selected_column_ism,
        range=[25, 75]  # Cap the y-axis range between 30 and 70
    ),
    legend_title="Legend",
    template="plotly_white",
    height=700,  # Set the chart height to make it taller
    width=1800,  # Set the chart width to make it narrower
    showlegend=False
)

# Create three columns, and place the chart in the center column
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratio [1, 2, 1] to control centering

with col2:
    st.plotly_chart(line_fig_ism)

# Section 2: Two columns for ISM Manufacturing and Services Dashboards
st.header("ISM Manufacturing and Services Survey Dashboard")
col1_1, col1_2 = st.columns(2)

with col1_1:
    st.subheader("ISM Manufacturing Dashboard")
    st.table(manufacturing_board)

with col1_2:
    st.subheader("ISM Services Dashboard")
    st.table(services_board)
# Section 3: Leading Indicators
st.header("Leading Indicators")
col2_1, col2_2 = st.columns(2)

with col2_2:
    st.write("Leading Indicators Dashboard")
    st.table(macro_board)

with col2_1:
    st.subheader("Leading Indicators Chart")

    # Dropdown to select either Consumer Sentiment or other indicators for plotting
    selected_macro_indicator = st.selectbox(
        "Select an indicator to plot", 
        options=[
            "Consumer Sentiment Index", 
            "Building Permits", 
            "Retail Money Market Funds", 
            "Job Openings", 
            "Personal Savings Rate"
        ]
    )

    # Create the Plotly figure
    line_fig_macro = go.Figure()

    # Select the appropriate data to plot based on user selection
    if selected_macro_indicator == "Consumer Sentiment Index":
        data_to_plot = consumer
    elif selected_macro_indicator == "Personal Savings Rate":
        data_to_plot = personal_savings
    else:
        data_to_plot = macros[['Building Permits', 'Retail Money Market Funds', 'Job Openings']]

    # Add the selected indicator's line chart
    line_fig_macro.add_trace(go.Scatter(
        x=data_to_plot.index,
        y=data_to_plot[selected_macro_indicator],
        mode='lines',
        name=selected_macro_indicator
    ))

    # Calculate the default x-axis range for the last 10 years
    last_10_years_macro = data_to_plot.index[-120]  # Assuming the data is monthly, this slices the last 120 months (10 years)

    # Add a range slider to the x-axis to allow interactive time range selection
    line_fig_macro.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=12, label="1Y", step="month", stepmode="backward"),
                    dict(count=60, label="5Y", step="month", stepmode="backward"),
                    dict(count=120, label="10Y", step="month", stepmode="backward"),
                    dict(count=300, label="25Y", step="month", stepmode="backward"),
                    dict(step="all", label="MAX")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date",
            range=[last_10_years_macro, data_to_plot.index[-1]]  # Set default range to the last 10 years
        ),
        autosize=True,
        title=f"{selected_macro_indicator} over time",
        yaxis=dict(
            title=selected_macro_indicator,
            range=[min(data_to_plot[selected_macro_indicator]), 
                   max(data_to_plot[selected_macro_indicator])]  # Dynamic y-axis range based on data
        ),
        legend_title="Legend",
        template="plotly_white",
        height=700,  # Set the chart height to make it taller
        width=1200,  # Set the chart width to make it narrower
        showlegend=False
    )

    # Display the chart in Streamlit
    st.plotly_chart(line_fig_macro, use_container_width=True)

# Section 4: Food & Energy
st.header("Food & Energy")
col3_2, col3_1 = st.columns(2)

with col3_2:
    st.write("Food & Energy Dashboard")
    st.table(food_energy)

with col3_1:
    st.write("Food & Energy Charts")

    # Create a dropdown to select a commodity (column) to plot from `fe_data`
    selected_commodity = st.selectbox("Select a commodity to plot", fe_data.columns)

    # Create the Plotly figure for the commodity
    line_fig_commodity = go.Figure()

    # Add the selected commodity's price data to the plot
    line_fig_commodity.add_trace(go.Scatter(
        x=fe_data.index,
        y=fe_data[selected_commodity],
        mode='lines',
        name=selected_commodity
    ))

    # Optionally, add a horizontal red dotted line at the mean price
    mean_price = fe_data[selected_commodity].mean()  # Calculate the mean price for reference
    line_fig_commodity.add_shape(
        type="line",
        x0=fe_data.index[0],  # Start of the x-axis
        y0=mean_price,  # Position on the y-axis based on mean price
        x1=fe_data.index[-1],  # End of the x-axis
        y1=mean_price,  # Same y position for the end of the line
        line=dict(
            color="red",
            width=2,
            dash="dot"  # Dotted line style
        )
    )

    # Calculate the default x-axis range for the last 10 years
    last_10_years_commodity = fe_data.index[-120]  # Assuming monthly data; adjust if needed

    # Add a range slider to the x-axis to allow interactive time range selection
    line_fig_commodity.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=12, label="1Y", step="month", stepmode="backward"),  # 1 year back
                    dict(count=60, label="5Y", step="month", stepmode="backward"),  # 5 years back
                    dict(count=120, label="10Y", step="month", stepmode="backward"),  # 10 years back
                    dict(count=300, label="25Y", step="month", stepmode="backward"),  # 25 years back
                    dict(step="all", label="MAX")  # Full range
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date",
            range=[last_10_years_commodity, fe_data.index[-1]]  # Set default range to the last 10 years
        ),
        autosize=True,
        title=f"{selected_commodity} Price Over Time",
        yaxis=dict(
            title=f"{selected_commodity} Price",
            range=[fe_data[selected_commodity].min() * 0.95, fe_data[selected_commodity].max() * 1.05],  # Adding some padding around min/max
            tickprefix="$"  # Add a dollar sign as a prefix to represent prices
        ),
        legend_title="Legend",
        template="plotly_white",
        height=700,  # Chart height
        width=1200,  # Chart width
        showlegend=False
    )

    # Display the chart in Streamlit
    st.plotly_chart(line_fig_commodity, use_container_width=True)
