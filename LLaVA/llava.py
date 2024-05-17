from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import pandas as pd
import math
import torch

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

def predict(prompt, image_path):
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_path,
        "sep": ",",
        "temperature":0,
        "top_p":1.0,
        "num_beams":5,
        "max_new_tokens":200
    })()

    return eval_model(args)

# Function for calculating accuracy of LLaVA predictions
def calculate_accuracies(filename):
    llava_results = pd.read_csv(filename)
    correct_llava_preds = llava_results[llava_results["correct_answer"].str.lower() == llava_results["llava_generated_answers"].str.lower()]
    print(f"Number of correct predictions: {len(correct_llava_preds)}")
    print(f"Accuracy: {len(correct_llava_preds)/len(llava_results)}")
    
def generate_answers(input_filename, result_filename, img_description_provided=False):
    image_qa_df = pd.read_csv(input_filename)
    image_qa_df["llava_generated_answers"] = None
    for idx, question in enumerate(image_qa_df["question_name"]):
        if pd.isna(image_qa_df.loc[idx, "llava_generated_answers"]):
            answer_choice_1 = image_qa_df.loc[idx, "answer_choice_1"]
            answer_choice_2 = image_qa_df.loc[idx, "answer_choice_2"]
            answer_choice_3 = image_qa_df.loc[idx, "answer_choice_3"]
            answer_choice_4 = image_qa_df.loc[idx, "answer_choice_4"]
            answer_string = f"{answer_choice_1}\n{answer_choice_2}\n{answer_choice_3}\n{answer_choice_4}\n"
            
            prompt = None
            if img_description_provided:
                image_description = image_qa_df.loc[idx, "image_description"]
                prompt = f"Image Description: {image_description}\n\nChoose only one option below as the answer for the following question. An explanation is not needed.\n\n{question}\n\n{answer_string}"
            else:
                prompt = f"Choose only one option below as the answer for the following question. An explanation is not needed.\n\n{question}\n\n{answer_string}"
            image_path = image_qa_df.loc[idx, "image_path"]
            image_qa_df.loc[idx, "llava_generated_answers"] = predict(prompt, image_path)
            image_qa_df.to_csv(result_filename, index=False)
    
    # Calculate prediction accuracy
    calculate_accuracies(result_filename)
            
# Generate predictions for images without image descriptions
generate_answers("../Dataset/test/DiagramQuestionsData.csv", "llava_7B_answers.csv")

# Generate predictions for images with image descriptions
generate_answers("common_blip2_llava_gpt4_preds.csv", "common_blip2_llava_gpt4_preds.csv", True)