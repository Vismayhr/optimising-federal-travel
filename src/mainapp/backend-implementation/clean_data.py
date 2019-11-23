import pandas as pd

# Function to drop rows with 0 number of tickets and 0 ticket cost
clean_dataframe = lambda raw_df, col_names_to_clean: raw_df[(raw_df[col_names_to_clean["num_of_tickets"]]>0)&(raw_df[col_names_to_clean["total_cost"]]>0)].reset_index().drop("index", axis=1)

if __name__ == "__main__":
	# Generic object to clean the given columns
	col_names_to_clean = {
    	"num_of_tickets": "Sum of Net Tickets",
    	"total_cost": "Sum of Total $"
		}
	# Read excel to get raw df
	raw_df = pd.read_excel("../data/government_dataset.xlsx", skiprows=2)
	# Produce the clean df from raw df
	clean_df = clean_dataframe(raw_df, col_names_to_clean)
	# Drop the last row
	clean_df = clean_df.loc[:clean_df.shape[0]-2, :]
	# Write clean df onto CSV
	clean_df.to_csv("../data/cleaned_government_data.csv", index=False)