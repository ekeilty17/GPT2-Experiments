def convert_example_to_formatted_string(inp, reflection='', delimiter='\n'):
    prompt, response = inp

    out  = f"Interviewer: {prompt}{delimiter}"
    out += f"Client: {response}{delimiter}"
    out += f"Reflection: {reflection}"
    return out

def reflection_definition():
    return  "Make a short statement about smoking that reflects the meaning of the Client:"

def clean_reflection(generated_reflection):
    lines = generated_reflection.split('\n')
    return lines[0]

def add_column_to_dataframe(df, data, column_name):
    if len(data) > len(df):
        data = data[:len(df)]
    elif len(data) < len(df):
        data += [''] * (len(df) - len(data))
    
    df.insert(len(df.columns), column_name, data)
    return df

def log_print(string=''):
    print(string)
    return string + '\n'