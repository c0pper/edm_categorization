import csv
import os
import uuid
import requests
import json

from tqdm import tqdm

# Function to read the contents of a text file
def read_txt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def load_cpk():
    endpoint = "http://localhost:8000/apis/cogito"
    headers = {
        'accept': 'application/json',
        'content-type': 'application/json'
    }
    data = {
        "path": "C:\\Users\\smarotta\\OneDrive - Expert.ai S.p.A\\MIRCO\\pywaldoservice\\pywaldo\\resources\\cpk\\edm_sn_experiment_483_model_(en)\\standard_en_16.4.0_4.14.0"
    }
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        print("Cogito endpoint called successfully.")
    else:
        print("Error calling cogito endpoint:", response.text)


# Function to load the model
def load_model():
    endpoint = "http://localhost:8000/apis/model"
    headers = {
        'accept': 'application/json',
        'content-type': 'application/json'
    }
    data = {
        "path": "C:\\Users\\smarotta\\OneDrive - Expert.ai S.p.A\\MIRCO\\pywaldoservice\\pywaldo\\resources\\models\\edm_sn_experiment_483_model_(en)\\model.mod",
        "model_id": "ee69d4ab-ef9c-456a-9e2d-3a8c71441dea"
    }
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        print("Model loaded successfully.")
    else:
        print("Error loading model:", response.text)

# Function to post data to the endpoint
def post_data_to_endpoint(text):
    endpoint = "http://localhost:8000/apis/analyze_and_apply/sync/ee69d4ab-ef9c-456a-9e2d-3a8c71441dea"
    payload = {
        "text_document": text,
        "configuration": {
            "load_from_model": False,
            "full_prediction_output": True
        }
    }
    response = requests.post(endpoint, json=payload).json()
    print("Posted:", text)
    categories = list(json.loads(response["results"])[0]["annotations_list"]["winners"].keys())
    categories_str = ", ".join(categories) if categories else ""
    print("Response:", categories_str)
    return categories_str

# Main function
def main():
    load_cpk()
    load_model()
    directory = r'platform_lib\SN_20k.csv_balanced\test\test'
    ann_directory = r'platform_lib\SN_20k.csv_balanced\test\ann'
    output_file = f'experiment_linear_svm_{str(uuid.uuid4())[:8]}.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'categories', 'expected_category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for filename in tqdm(os.listdir(directory)[:200]):
            cache_filename = f"cache\sn_llm_cat_cache\linearsvm\{filename[:-3] + 'cache'}"
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                text = read_txt_file(filepath)
                ann_filepath = os.path.join(ann_directory, filename)[:-3] + "ann"
                ann = read_txt_file(ann_filepath).split("\t\t")[1].strip()
                
                if os.path.exists(cache_filename):
                    with open(cache_filename, "r") as cache_file:
                        cached_json = json.load(cache_file)
                        categories = cached_json["categories"]
                else:
                    categories = post_data_to_endpoint(text)
                    json_dict = {'text': text, 'categories': categories, 'expected_category': ann}
                    with open(cache_filename, "w", encoding="utf8") as cachefile:
                        json.dump(json_dict, cachefile, indent=4)
                writer.writerow({'text': text, 'categories': categories, 'expected_category': ann})
                print("Processed:", filename)

if __name__ == "__main__":
    main()
