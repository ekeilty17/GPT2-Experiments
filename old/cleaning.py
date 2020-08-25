import pandas as pd

def clean_reflection(generated_reflection):
    lines = generated_reflection.split('\n')
    return lines[0]
"""
if __name__ == "__main__":
    df = pd.read_csv("data/prepended_definition.csv", index_col=0)

    cleaned_reflections = []
    for index, row in df.iterrows():
        new_reflection = row["new_reflection"]
        cleaned_reflections.append( clean_reflection(new_reflection) )
    
    df.insert(len(df.columns), "cleaned_reflection", cleaned_reflections, True)

    df.to_csv('data/prepended_definition_cleaned.csv', index=False)
"""
def is_duplicate(df, prompt, response):
    return (prompt in df['prompt'].values) and (response in df['response'].values)

def remove_duplicates(primer_df, full_df):
    filtered_df = pd.DataFrame(columns=full_df.columns)

    for index, row in full_df.iterrows():
        prompt = row['prompt']
        response = row['response']
        if is_duplicate(primer_df, prompt, response):
            continue 
        if is_duplicate(filtered_df, prompt, response):
            continue
        filtered_df = filtered_df.append(row)
    
    return filtered_df

if __name__ == "__main__":
    df = remove_duplicates()
    print(df.head(10))
    df.to_csv('data/test_reflection_data.csv', index=True)