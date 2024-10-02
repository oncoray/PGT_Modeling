# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:59:51 2024

@author: Phase
"""
import pandas as pd
from pmma.cmd_args import cML_final_result_parser
from PIL import Image, ImageDraw, ImageFont
import textwrap

def get_final_results_table(df_cross_val, df_ex_val, final_model_setup):
    """
    Generates a summary table for cross-validation and external validation results for a specific model setup.

    Parameters:
    - df_cross_val: DataFrame containing cross-validation results.
    - df_ex_val: DataFrame containing external validation results.
    - final_model_setup: Dictionary specifying the model setup to summarize.

    Returns:
    - A DataFrame representing summarized results for a specific condition.
    """
    summary_rows = []

    # Filter DataFrames based on the final model setup
    df_cross_val_filtered = df_cross_val[
        (df_cross_val["feature_type"] == final_model_setup["feature_type"]) & 
        (df_cross_val["feature_selection_method"] == final_model_setup["feature_selection_method"]) & 
        (df_cross_val["model_learner"] == final_model_setup["model_learner"])
    ]
    df_ex_val_filtered = df_ex_val[
        (df_ex_val["feature_type"] == final_model_setup["feature_type"]) & 
        (df_ex_val["feature_selection_method"] == final_model_setup["feature_selection_method"]) & 
        (df_ex_val["model_learner"] == final_model_setup["model_learner"])
    ]

    # Assert that there is only one unique value for metrics in each group to validate the use of iloc[0]
    assert df_cross_val_filtered.groupby(['cv_data_set', 'spot_type', 'proton_energy'])['RMSE'].nunique().max() == 1, "Multiple RMSE values found for a group in cross-validation."
    assert df_ex_val_filtered.groupby(['cohort', 'spot_type', 'proton_energy'])['RMSE'].nunique().max() == 1, "Multiple RMSE values found for a group in external validation."

    # Helper function to append rows
    def append_rows(df_grouped, step):
        for name, group in df_grouped:
            
            if name[0] == "development" or name[0] == "training":
                cohort_name = "training"
            else:
                cohort_name = "validation"
                
            summary_rows.append({
                **final_model_setup,  # Unpack all key-value pairs from final_model_setup
                'signature': group['signature'].iloc[0],
                'sign_size': group['sign_size'].iloc[0],
                'step': step,
                'cohort': cohort_name,
                'spot_type': name[1],
                'proton_energy': name[2],
                'RMSE': group['RMSE'].iloc[0],
                'RMSE CI Low': group['RMSE CI Low'].iloc[0],
                'RMSE CI High': group['RMSE CI High'].iloc[0],
                'R2': group['R2'].iloc[0],
                'R2 CI Low': group['R2 CI Low'].iloc[0],
                'R2 CI High': group['R2 CI High'].iloc[0]
            })

    # Process cross-validation and external validation results
    append_rows(df_cross_val_filtered.groupby(['cv_data_set', 'spot_type', 'proton_energy']), 'cross_validation')
    append_rows(df_ex_val_filtered.groupby(['cohort', 'spot_type', 'proton_energy']), 'external_validation')

    # Convert the list of dictionaries into a DataFrame
    final_df = pd.DataFrame(summary_rows)

    return final_df

def summarize_results_to_string(final_df):
    """
    Summarizes the results from the final DataFrame into a structured string format.

    Parameters:
    - final_df: The final DataFrame containing the model results and setup.

    Returns:
    - A string summarizing the results.
    """
    # Initialize an empty string to hold the summary
    summary_str = ""

    # Add final model setup to the summary string
    model_setup_keys = ['feature_type', 'feature_selection_method', 'model_learner', 'signature', 'sign_size']
    model_setup = {key: final_df.iloc[0][key] for key in model_setup_keys}  # Assuming all rows have the same model setup
    summary_str += "Final Model Setup:\n" + "\n".join([f"{key}: {value}" for key, value in model_setup.items()]) + "\n\n"

    # Iterate over each unique step in the DataFrame
    for step in final_df['step'].unique():
        summary_str += f"Step: {step}\n"
        
        # Filter DataFrame by the current step
        df_step = final_df[final_df['step'] == step]
        
        # Sort dataset combinations, ensuring (combined, combined) comes first
        datasets = list(df_step.groupby(['spot_type', 'proton_energy']).groups.keys())
        datasets.sort(key=lambda x: (x != ('combined', 'combined'), x))
        
        # Iterate over each sorted dataset combination
        for dataset in datasets:
            df_dataset = df_step[(df_step['spot_type'] == dataset[0]) & (df_step['proton_energy'] == dataset[1])]
            dataset_name = f"{dataset[0]}, {dataset[1]}"
            summary_str += f"  Dataset: {dataset_name}\n"
            
            # Iterate over each cohort within the current dataset
            for cohort in df_dataset['cohort'].unique():
                df_cohort = df_dataset[df_dataset['cohort'] == cohort]
                
                # Assuming one row per cohort, if more, adjust as needed
                for _, row in df_cohort.iterrows():
                    rmse = f"RMSE: {round(row['RMSE'], 2)} (CI: {round(row['RMSE CI Low'], 2)} - {round(row['RMSE CI High'], 2)})"
                    r2 = f"R2: {round(row['R2'], 2)} (CI: {round(row['R2 CI Low'], 2)} - {round(row['R2 CI High'], 2)})"
                    summary_str += f"    Cohort: {cohort}\n      {rmse}\n      {r2}\n"

    return summary_str
        


def save_string_to_image_and_text(string, image_path, text_path):
    """
    Saves a given string to a high-quality PNG image and a text file, preserving line breaks in the string.

    Parameters:
    - string: The string to be saved.
    - image_path: Path where the PNG image will be saved.
    - text_path: Path where the text file will be saved.
    """
    # Save string to text file
    with open(text_path, 'w') as text_file:
        text_file.write(string)

    # Define higher image size and background color for better quality
    img_width, img_height = 1600, 1200
    background_color = 'white'
    font_color = 'black'
    
    # Create an image with white background
    image = Image.new('RGB', (img_width, img_height), background_color)
    draw = ImageDraw.Draw(image)
    
    # Use a better-quality font
    font_size = 28  # Larger font size for better readability
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Prepare to wrap and maintain line breaks
    lines = string.split('\n')  # Split the input string by line breaks
    wrapped_lines = [textwrap.fill(line, width=120) for line in lines]  # Wrap each line to fit wider image
    wrapped_text = '\n'.join(wrapped_lines)  # Reassemble the wrapped lines, preserving original line breaks

    # Calculate text height and start drawing text on image
    current_h, pad = 50, 10
    for line in wrapped_text.split('\n'):
        draw.text((10, current_h), line, fill=font_color, font=font)
        current_h += pad + font.getsize(line)[1]
    
    # Save image
    image.save(image_path)
    

def find_experiment_setup_from_cross_val(df):
    """
    Finds the row with the minimum RMSE for the validation cv_data_set and returns selected information.
    
    Parameters:
    - df: The DataFrame to process.
    
    Returns:
    - A dictionary with the information of interest from the row with the minimum RMSE in the validation cross validation set.
    """
    # Filter the DataFrame for the validation set
    validation_df = df[(df['cv_data_set'] == "validation") & (df['spot_type'] == 'combined') & (df['proton_energy'] == 'combined')]
    
    # Find the row with the minimum RMSE
    min_rmse_row = validation_df.loc[validation_df['RMSE'].idxmin()]
    
    # Extract the required information and rename RMSE and R2 related keys
    setup = {
        'feature_type': min_rmse_row['feature_type'],
        'feature_selection_method': min_rmse_row['feature_selection_method'],
        'model_learner': min_rmse_row['model_learner']
    }
    
    return setup



if __name__ == "__main__":
    # Parsing command line arguments for feature ranking
    parser = cML_final_result_parser("cML final result")
    args = parser.parse_args()
    
    cv_results = pd.read_csv(args.cross_val_performance_file_path, sep = ";")
    ex_val_results = pd.read_csv(args.ex_val_performance_file_path, sep = ";")
    
    print("Find final model setup")
    final_model_setup = find_experiment_setup_from_cross_val(cv_results)
    print("Final model setup is: \n", final_model_setup)
    
    print("Calculate final results table")
    df_final = get_final_results_table(cv_results, ex_val_results, final_model_setup)
    
    df_final.to_csv(args.final_result_csv, sep = ";", index = False)
    
    print("Summarise results in string...")
    summary_str = summarize_results_to_string(df_final)
    
    print(summary_str)
    
    save_string_to_image_and_text(summary_str, args.final_result_png, args.final_result_txt)
    
    print("Everything finished!")
    
    
    
    