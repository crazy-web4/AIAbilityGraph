import pandas as pd
import ast
import os


"""
Knowledge base text filtering approach:
- Implement access control for knowledge base texts based on company positions
- A knowledge base text may be associated with N positions

Simulate database storage using CSV files
- kb_topK.csv: Top K texts retrieved from the knowledge base
- kb_position_ref.csv: Association information between knowledge base texts and positions
- user.csv: User information and their company positions
- position.csv: Position information
"""

# Get the absolute path of the current script
base_path = os.path.dirname(os.path.abspath(__file__))

# Build the absolute paths for the CSV files
kb_topK_path = os.path.join(base_path, 'db', 'kb_topK.csv')
user_path = os.path.join(base_path, 'db', 'user.csv')
kb_position_ref_path = os.path.join(base_path, 'db', 'kb_position_ref.csv')
position_path = os.path.join(base_path, 'db', 'position.csv')


# Read the CSV files
kb_topK_table = pd.read_csv(kb_topK_path)
user_table = pd.read_csv(user_path)
kb_position_ref_table = pd.read_csv(kb_position_ref_path)
position_table = pd.read_csv(position_path)


def get_filter_contents(user_id):
    # Query the user's position (permissions)
    user_position_id = user_table[user_table['user_id'] == user_id]['position_id'].values[0]
    position_name = position_table.loc[position_table['id'] == 1, 'position_name'].values[0]
    print("Current user's position: {}\n".format(position_name))

    # Query the positions (permissions) corresponding to the top K texts
    topK_position_table = pd.merge(kb_topK_table, kb_position_ref_table, on='kb_id')
    # Print the retrieved texts
    print("==========Retrieved Texts==========")
    for content in topK_position_table["content"].tolist():
        print(content)
    print("==========Retrieved Texts==========\n")

    # Iterate through the merged table and find matching positions
    matching_kb_ids = []

    for index, row in topK_position_table.iterrows():
        # Convert the string to a list
        position_ids = ast.literal_eval(row['position_ids'])
        if user_position_id in position_ids:
            matching_kb_ids.append(row['kb_id'])

    # Filter out the corresponding rows based on kb_id
    filtered_data = kb_topK_table[kb_topK_table['kb_id'].isin(matching_kb_ids)]
    # Get the content column and convert it to a list
    content_list = filtered_data['content'].tolist()
    print("Retrieved texts that the user has permission to access:", content_list)


if __name__ == "__main__":
    user_id = 201
    get_filter_contents(user_id)