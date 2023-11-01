import pandas as pd
import numpy as np

def load_data(path):
    """
    Purpose: 
        - The purpose of this function is to load the various data sets into corresponding dataframes. 
    Parameters: 
        - Path: This is the path to the directory that contains all of the CSV files that we will be using.
    Returns: 
        - data: This is a dictionary that contains all of the datasets. 
    """
    # This is a dictionary that contains the names of the datasets we will be using. The variable name is the key, and the file name is the value. 
    orchestra_files = {
        'subscriptions_df': 'subscriptions.csv',
        'accounts_df': 'account.csv',
        'tickets_df': 'tickets.csv',
        'test_df': 'test.csv',
        'concerts_df': 'concerts.csv',
        'concerts2_df': 'concerts_2014-15.csv',
        'training_df': 'training.csv',
        'zipcodes_df': 'zipcodes.csv'
    }
    # Here we store the dataframes
    data= {}
    # The for loop goes through each key and value pair of the orchestra_files dictionary we made above
    for key, value in orchestra_files.items():
        # We will attempt to use the read_csv pandas function using the defauly encoding. If that does not work, then 
        # we will use the encoding ISO-8859-1
        try:
            data[key] = pd.read_csv(f"{path}/{value}")
        # If there is an error, we use the encoding ISO-8859-1
        except UnicodeDecodeError: 
            data[key] = pd.read_csv(f"{path}/{value}", encoding='ISO-8859-1')
    return data


def test_data_processing(path):
    """
    Purpose: 
        - The purpose of this function is to merge the dataframes together and also merge rows based on common account.id (primary key) values.

    Parameters: 
        - Path: This is the path to the directory that contains all of the CSV files that we will be using.

    Returns: 
        - training_aggregate_final: This is the final dataframe that contains all of the features that we will be using for our model.
    
    """
    # During the merging process, whenever there are many NaN values, we will attempt to fill them in with the mode, especially with categorical variables
    data = load_data(path)
    mode = lambda x: x.mode()[0] if not x.mode().empty else np.nan 
    accounts_df = data['accounts_df']
    # Here we convert the first.donated column to a datetime object using pandas to_datetime function. 
    accounts_df['first.donated'] = pd.to_datetime(accounts_df['first.donated'], errors='coerce')
    # We can now convert the first.donated column to an ordinal value using the toordinal() function
    accounts_df['first.donated_ordinal'] = accounts_df['first.donated'].apply(lambda x: x.date().toordinal() if not pd.isnull(x) else np.nan)
    test_df = data['test_df']
    test_df = test_df.rename(columns={"ID": "account.id"})
    # We can now merge the test_df with the accounts_df using the account.id as the primary key
    test_accounts = pd.merge(test_df, data['accounts_df'], on='account.id', how='left')
    # We can now merge test_accounts with the subscriptions_df using the account.id as the primary key
    test_accounts_subscriptions = pd.merge(test_accounts, data['subscriptions_df'], on='account.id', how='left')
    
    # Often times during the merging process, the relationship between the two dataframes is not 1 to 1, meaning one primary key in one dataframe can be associated with multiple primary keys in the other dataframe.
    # For this reason, we can take the aggregate of each feature when we are merging so that we keep the test rows at the same length (6491 rows)
    aggregation_dictionary_1 = {
            'amount.donated.2013': mode, 
            'amount.donated.lifetime': mode,
            'no.donations.lifetime': mode,
            'first.donated_ordinal': mode,
            'subscription_tier': mode,
            'multiple.subs': mode,
            'package': mode,
            'section': mode,
            'season': mode,
            'billing.city': mode
        }
    # Now we can create an aggregated dataframe by grouping by the account.id and aggregating using the aggregation_dictionary_1
    test_aggregate_1 = test_accounts_subscriptions.groupby('account.id').agg(aggregation_dictionary_1).reset_index()
    # Now we can merge with the tickets dataframe with test_aggregate_1 on the account.id column
    test_aggregate_tickets = pd.merge(test_aggregate_1, data['tickets_df'], on='account.id', how='left')
    # We need to drop the season_y column due to redundancy 
    test_aggregate_tickets.drop(columns=['season_y'], inplace=True)
    # We need to also rename season_x to season
    test_aggregate_tickets.rename(columns={'season_x': 'season'}, inplace=True)
    
    # Now we can create our second aggregation dictionary
    aggregation_dictionary_2 = {
        'amount.donated.2013': mode, 
        'amount.donated.lifetime': mode,
        'no.donations.lifetime': mode,
        'first.donated_ordinal': mode,
        'subscription_tier': mode,
        'multiple.subs': mode,
        'price.level': mode,
        'no.seats': mode,
        'multiple.tickets': mode,
        'package': mode,
        'set': mode,
        'section': mode,
        'season': mode,
        'billing.city': mode

    }
    # We can group by the account.id and aggregate using the aggregation_dictionary_2
    test_aggregate_2 = test_aggregate_tickets.groupby('account.id').agg(aggregation_dictionary_2).reset_index()
    # We can merge with the zipcodes dataframe now 
    test_aggregate_zipcodes = pd.merge(test_aggregate_2, data['zipcodes_df'], left_on='billing.city', right_on='City', how='left')

    # Now we can create our third aggregation dictionary
    aggregation_dictionary_3 = {
        'amount.donated.2013': mode,
        'amount.donated.lifetime': mode,
        'no.donations.lifetime': mode,
        'first.donated_ordinal': mode,
        'subscription_tier': mode,
        'multiple.subs': mode,
        'price.level': mode,
        'no.seats' : mode,
        'multiple.tickets': mode,
        'package': mode,
        'section': mode,
        'no.seats': mode,
        'TotalWages': mode,
        'set': mode,
        'season': mode,
        'State' : mode,
        'billing.city': mode
    }
    
    # We can group by the account.id and aggregate using the aggregation_dictionary_3
    test_aggregate_3 = test_aggregate_zipcodes.groupby('account.id').agg(aggregation_dictionary_3).reset_index()
    # We can slighly change the concerts_df to match what we want 
    concerts_df = data['concerts_df'][['season', 'concert.name', 'set', 'who','location', 'what']]
    
    test_aggregate_concerts = pd.merge(test_aggregate_3, concerts_df, on='set', how='left').set_index('account.id')
    
    # We can now do the final aggregation
    aggregation_dictionary_final = {
        'amount.donated.2013': mode,
        'amount.donated.lifetime': mode,
        'no.donations.lifetime': mode,
        'first.donated_ordinal': mode,
        'subscription_tier': mode,
        'multiple.subs': mode,
        'price.level': mode,
        'no.seats' : mode,
        'multiple.tickets': mode,
        'package': mode,
        'section': mode,
        'no.seats': mode,
        'TotalWages': mode,
        'set': mode,
        'season_x': mode,
        'season_y': mode,
        'State' : mode,
        'billing.city': mode,
        'concert.name': mode,
        'who': mode, 
        'what': mode,
        'location': mode
    }

    # We group by the account.id and aggregate using the aggregation_dictionary_final. This will be our final time aggregating. 
    test_aggregate_3 = test_aggregate_concerts.groupby('account.id').agg(aggregation_dictionary_final).reset_index()
    # We can now merge with the concerts2_df dataframe on the set column
    test_aggregate_final = pd.merge(test_aggregate_3, data['concerts2_df'] [['set', 'concert.name', 'season', 'who']], on='set', how='left')
    

    # Return the final dataframe
    return test_aggregate_final

# Now we need to make a function that is responsible for filling in any missing values within the trainign and test dataframes. 
def missing_values_filler(dataframe):
    """
    Purpose: 
        - The purpose of this function is to fill in any missing values within the training and test dataframes.
    Parameters: 
        - dataframe: This is the input dataframe that we are going to be filling in the missing values for.
    Returns:
        - dataframe: This is the dataframe that has been filled in with the missing values.
    
    """
    # First we will fill in any NaN in numerical values with 0. Essentially 0 is meant to represent the absence of a value
    numerical_columns = ['TotalWages', 'first.donated_ordinal']
    dataframe[numerical_columns] = dataframe[numerical_columns].fillna(0)

    # Now we need to do the same thing for the categorical variables. This time, the only difference is that we will fill the NaN values with 'Unknown'
    categorical_columns = ['amount.donated.2013', 'amount.donated.lifetime', 'no.donations.lifetime', 'first.donated_ordinal', 'subscription_tier', 'multiple.subs', 'price.level',	
    'no.seats', 'multiple.tickets', 'package', 'section', 'TotalWages',	'set', 'season_x', 'season_y', 'State', 'billing.city', 'concert.name_x', 'who_x', 'location',
    'concert.name_y', 'season', 'who_y', 'what']
    dataframe[categorical_columns] = dataframe[categorical_columns].fillna('Unknown')

    # Return the non NaN dataframe 
    return dataframe


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_data_folder>")
        sys.exit(1)

    path = sys.argv[1]
    result = test_data_processing(path)
    result = missing_values_filler(result)
    # Save the result dataframe in the same directory as the input data
    result.to_csv(f"{path}/test_aggregate_final.csv")

