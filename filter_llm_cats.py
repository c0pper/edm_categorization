import pandas as pd


def keep_n_samples(df: pd.DataFrame, n_samples: int):
    return df.sample(n_samples, random_state=69420)


def filter_out_exact_matches():
    # Load the CSV file
    input_file_path = 'full_SN_categorization_all_methods.xlsx'
    output_file_path = 'full_sn_cat_evaluation_samples1.xlsx'

    # Read the CSV file
    df = pd.read_excel(input_file_path)

    # Define a function to check if the ground truth category is in the other columns
    def category_not_in_columns(row):
        ground_truth_category = row['Ground truth category']
        llm_chatgpt_categories = eval(row['LLM ChatGPT Categorization'])
        llm_elmi_categories = eval(row['LLM ELMI Categorization'])
        
        if ground_truth_category not in llm_chatgpt_categories and ground_truth_category not in llm_elmi_categories:
            return True
        return False

    # Apply the function to filter the dataframe
    columns_to_keep = ['text', 'Ground truth category', 'LLM ChatGPT Categorization', 'LLM ELMI Categorization']
    filtered_df = df[df.apply(category_not_in_columns, axis=1)][columns_to_keep]
    filtered_df = keep_n_samples(filtered_df, 100)

    # Extract the first element from each list in the 'LLM ELMI Categorization' column
    filtered_df['LLM ChatGPT Categorization'] = filtered_df['LLM ChatGPT Categorization'].apply(lambda x: eval(x)[0] if isinstance(eval(x), list) and len(eval(x)) > 0 else None)
    filtered_df['LLM ELMI Categorization'] = filtered_df['LLM ELMI Categorization'].apply(lambda x: eval(x)[0] if isinstance(eval(x), list) and len(eval(x)) > 0 else None)
    
    # Save the filtered dataframe to a new CSV file
    filtered_df.to_excel(output_file_path, index=False)

    print(f"Filtered rows have been saved to {output_file_path}")
    
    
if __name__ == "__main__":
    filter_out_exact_matches()


