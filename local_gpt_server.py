from flask import Flask, request, jsonify
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)  # Corrected _name_ to __name__

# Load GPT-Neo model locally
model_name = "EleutherAI/gpt-neo-125M"  # Replace with larger models like "EleutherAI/gpt-neo-1.3B" if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text_description(df):
    """
    Generate a natural language description of the dataset.
    """
    description = f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
    
    for column in df.columns:
        unique_values = df[column].dropna().unique()[:5]
        num_unique = df[column].nunique()
        missing_values = df[column].isnull().sum()
        data_type = str(df[column].dtype)
        
        description += f"The column '{column}' has a data type of {data_type}. "
        if num_unique <= 10:
            description += f"It contains {num_unique} unique values, including {', '.join(map(str, unique_values))}. "
        else:
            description += f"It contains {num_unique} unique values. Some example values are {', '.join(map(str, unique_values))}. "
        
        if missing_values > 0:
            description += f"There are {missing_values} missing values in this column. "
    
    return description

def enhance_description_with_gpt_neo(description):
    """
    Use GPT-Neo to enhance the dataset description into human-readable insights.
    """
    prompt = (
        f"Dataset Analysis:\n\n{description}\n\n"
        "Provide an insightful and human-readable summary explaining what the dataset is about, "
        "what the columns represent, and any notable patterns or observations."
    )
    
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(
        inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        num_return_sequences=1
    )
    enhanced_description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return enhanced_description.strip()

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Load the dataset into a Pandas DataFrame
        df = pd.read_csv(file)
        
        # Generate textual description of the dataset
        dataset_description = generate_text_description(df)
        
        # Enhance description using GPT-Neo
        enhanced_description = enhance_description_with_gpt_neo(dataset_description)
        
        return jsonify({
            "dataset_insights": enhanced_description
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':  # Corrected _name_ to __name__
    app.run(host='0.0.0.0', port=5000)
