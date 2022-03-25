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

---

# Note to the group

Hi everyone,

I just added some functionality to the Github to download and load the JHU US dataset. Here are some instructions for how to get started with this update:

1. Pull the latest version of the repo to get the code I recently added.
2. Navigate to the directory 'datasets/jhu' and run the Python script ```build_dataset.py```. This script will download the csv files directly from the Github and then reformat them a bit so they'll be quicker to load.

Once the dataset has been downloaded and reformatted, you'll be able to load the data using the function ```JHU_US()``` in the file ```src/load_data.py```. Some "documentation" about this function is included in the file where the function is declared, but I will highlight a few things that might be particularly useful:

- In order to import ```JHU_US()``` from a different directory, you'll need to tell Python where to look for the ```src``` folder. For instance, if you're in 'experiments/DeepXDE', you can use the line ```sys.path.append(os.path.abspath('../../'))``` to tell Python to look back two directories, i.e. the root of the Github, when looking to import something. After you've included that line, ```from src.load_data import JHU_US``` will successfully load the function.

- The JHU dataset has county-level data for each state in the US. The function ```JHU_US``` has the ability to load these data at three levels of granularity: country-wide totals, state-level totals, and county-level. You can change the level of granularity of the data returned by the function using the ```granularity``` keyword. The valid options are 'country', 'state', and 'county'.

- When using state-level or county-level granularity, you can tell the function to return data only for a specific state by using the ```select_state``` argument. For instance, if ```granularity = 'state'``` and ```select_state = 'MA'```, the function will return single time series representing a sum over all counties in Massachusetts. On the other hand, if ```granularity = 'county'``` and ```select_state = 'MA'```, the function will return one time series for each county in Massachusetts.

- You can also tell the function to rescale the data based on populations by setting the keyword ```rescaling``` to ```'population'```.

- The argument ```smooth``` can be used to smooth each time series. No smoothing is perform is ```smooth=None```, but if this argument is integer, each data point in the time series will be an average over the last ```smooth``` dates.

There are a few more arguments to the function, but I suppose you can read the function declaration if you're interested. This should be enough to get started in the MA data for now, but let me know if you have any questions. We can also talk more on Tuesday about how to use the dataset and code.

Best,
Peter

