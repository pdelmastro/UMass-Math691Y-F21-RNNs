JHU COVID-19 time series for the United States

---

Goals for the scripts
- Download the datasets and put them in the "raw" folder
- Function to build the dataset for each state and save as CSV
- Function to load the dataset for each state


---

Hi everyone, turns out the JHU Github has county level data (# of cases and deaths) for a lot (if not all) of the states.

I'm thinking of writing a function to load the data at different levels of granularity (country-wide, state-level, and county-level).


---

I'm thinking of writing function that allow us to do the following:
- Load at the state level, county level, and country level
  - Country level returns a single time series
  - State level returns a dictionary mapping from state to time series
  - County-level returns a dictionary mapping from state to another dictionary mapping county name to the time series
- Select a specific state - basically returns the second-level dictionary from 1

---

I want to exclude the lines that don't have any population. I'm thinking I need to do the following process:

- Read in the populations from the deaths dataset
- Figure out which rows I need to exlcude

