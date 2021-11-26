"""
Run this script to download the dataset and build the 
CSV files we'll be using for this project.
"""

"""
--------------------------------------------------------------------
    Loading the raw datset
--------------------------------------------------------------------
"""

# Imports
import pathlib, requests, csv

# Links to the dataset
base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
us_datasets = [
    "time_series_covid19_confirmed_US.csv",
    "time_series_covid19_deaths_US.csv"
]

# Create the 'raw' directory for the raw data
pathlib.Path('raw').mkdir(exist_ok=True)

# Download the datasets and write to the 'raw' folder
for ds_name in us_datasets:
    # Make a request for the dataset
    r = requests.get(base_url + ds_name)
    # Extract the text and save to csv
    with open('raw/%s' % ds_name, 'w') as f:
        for line in r.iter_lines():
            f.write('%s\n' % line.decode("utf-8"))
    # Close the requests object
    r.close()


"""
--------------------------------------------------------------------
Create country-wide, state-level, and county-level datasets
--------------------------------------------------------------------
"""

# Imports
import os
import pandas as pd

# Create the directory for the formatted data
pathlib.Path('formatted').mkdir(exist_ok=True)

"""
County and State Populations
"""

# Load the data using pandas
deaths_df = pd.read_csv(
    os.path.join('raw', 'time_series_covid19_deaths_US.csv'),
    index_col=False
)

# Determine which rows should be discarded
# (Rows for counties in US states and with population > 0)
include_row = (deaths_df.iso3 == 'USA') & (deaths_df.Population > 0)
deaths_df = deaths_df[include_row]

# Save the population by county and state
deaths_df.to_csv(
    'formatted/county_populations.csv',
    columns=['Admin2', 'Province_State', 'Population'],
    header=False, index=False
)
deaths_df[
        ['Province_State','Population']
    ].groupby('Province_State').sum().to_csv(
    'formatted/state_populations.csv',
    header=False
)


"""
Time Series
"""
# Columns to exclude from both datasets
shared_excluded_cols = [
    'UID', 'iso2', 'iso3', 'code3', 'FIPS',
    'Country_Region', 'Lat', 'Long_', 'Combined_Key'
]

for i, ds_name in enumerate(us_datasets):
    
    # Reload the data using pandas
    df = pd.read_csv(
        os.path.join('raw', ds_name),
        index_col=False
    )
    df = df[include_row]

    # Remove superfluous columns
    to_exclude = [] if i == 0 else ['Population']
    to_exclude += shared_excluded_cols
    df = df.drop(columns=to_exclude)

    # County-level
    dtype = 'cases' if i == 0 else 'deaths'
    df.to_csv(
        os.path.join('formatted', 'county_%s.csv' % dtype),
        index=False, header=False
    )

    # State-level: collapse rows by state
    df.groupby(df.Province_State).sum().to_csv(
        os.path.join('formatted', 'state_%s.csv' % dtype),
        header=False
    )
    
    # Country-level: collapse into a single timeseries
    df.drop(columns=['Admin2','Province_State']).sum(axis=0).to_csv(
        os.path.join('formatted', 'country_%s.csv' % dtype),
        header=False, index=False
    )
