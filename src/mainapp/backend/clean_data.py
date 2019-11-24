import pandas as pd

# Function to drop rows with 0 number of tickets and 0 ticket cost
clean_dataframe = lambda raw_df, col_names_to_clean: raw_df[(raw_df[col_names_to_clean["num_of_tickets"]]>0)&(raw_df[col_names_to_clean["total_cost"]]>0)].reset_index().drop("index", axis=1)
# Function to strip spaces
strip_spaces = lambda x: x.strip()
# Replace cities
replace_remain_cities = {
    "From":{
        'GRP EXP CDN YVR': "Vancouver",
        'WEYBURN SASK': "Weyburn",
        'TORONTO  DTOWN': "Toronto",
        'Southn Lake': "South Lake",
        "Wha Ti": "WhaTi",
        "Wunnummin Lake": "Kenora",
        "PITTS MEADOW  BC": "Pitts Meadow"
        },
    "To":{
        'GRP EXP CDN YVR': "Vancouver",
        'WEYBURN SASK': "Weyburn",
        'TORONTO  DTOWN': "Toronto",
        'Southn Lake': "South Lake",
        "Wha Ti": "WhaTi",
        "Wunnummin Lake": "Kenora",
        "PITTS MEADOW  BC": "Pitts Meadow"
    }}
if __name__ == "__main__":
    # Generic object to clean the given columns
    col_names_to_clean = {
    "num_of_tickets": "Sum of Net Tickets",
    "total_cost": "Sum of Total $"
    }
    # Column names to be stripped of spaces
    col_names_to_strip = ['Major Class', 'Month of Travel Date', 'From', 'To']
    # Read excel to get raw df
    raw_df = pd.read_excel("../data/government_dataset.xlsx", skiprows=2)
    # Produce the clean df from raw df
    clean_df = clean_dataframe(raw_df, col_names_to_clean)
    # Drop the last row
    clean_df = clean_df.loc[:clean_df.shape[0]-2, :]
    # Strip the columns in df
    for each_col in col_names_to_strip:
        clean_df[each_col] = clean_df[each_col].apply(strip_spaces)
    # Replace erroneous city names
    clean_df = clean_df.replace(replace_remain_cities)
    # Write clean df onto CSV
    clean_df.to_csv("../data/cleaned_government_data.csv", index=False)