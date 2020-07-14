import pandas as pd

def clean_reflection(generated_reflection):
    lines = generated_reflection.split('\n')
    return lines[0]

if __name__ == "__main__":
    df = pd.read_csv("data/prepended_definition.csv", index_col=0)

    cleaned_reflections = []
    for index, row in df.iterrows():
        new_reflection = row["new_reflection"]
        cleaned_reflections.append( clean_reflection(new_reflection) )
    
    df.insert(len(df.columns), "cleaned_reflection", cleaned_reflections, True)

    df.to_csv('data/prepended_definition_cleaned.csv', index=False)