import torch
from PIL import Image
import pandas as pd
from lavis.models import load_model_and_preprocess

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)

# Function for calculating accuracy of BLIP-2 predictions
def calculate_accuracies(filename):
    blip2_answers = pd.read_csv(filename)
    def map_predicted_to_actual(row):
        if row['blip_2_generated_answers'] in ['a.', 'a']:
            return row['answer_choice_1']
        elif row['blip_2_generated_answers'] in ['b', 'b.']:
            return row['answer_choice_2']
        elif row['blip_2_generated_answers'] in ['c', 'c.']:
            return row['answer_choice_3']
        elif row['blip_2_generated_answers'] in ['d.', '(d)', 'd']:
            return row['answer_choice_4']
    
    blip2_answers['mapped_predictions'] = blip2_answers.apply(map_predicted_to_actual, axis=1)
    correct_blip2_preds = blip2_answers[blip2_answers["mapped_predictions"].str.lower() == blip2_answers["correct_answer"].str.lower()]
    print(f"Number of correct predictions: {len(correct_blip2_preds)} out of a total of {len(blip2_answers)} questions.")
    print(f"Accuracy: {len(correct_blip2_preds) / len(blip2_answers) * 100}")

# Function for making predictions using BLIP-2
def generate_answers(input_filename, result_filename, img_description_provided=False):
    image_qa_df = pd.read_csv(input_filename)
    image_qa_df["blip_2_generated_answers"] = None
    for idx, question in enumerate(image_qa_df["question_name"]):
        if pd.isna(image_qa_df.loc[idx, "blip_2_generated_answers"]):
            # Build answer choice string
            answer_choice_1 = image_qa_df.loc[idx, "answer_choice_1"]
            answer_choice_2 = image_qa_df.loc[idx, "answer_choice_2"]
            answer_choice_3 = image_qa_df.loc[idx, "answer_choice_3"]
            answer_choice_4 = image_qa_df.loc[idx, "answer_choice_4"]
            answer_string = f"\na. {answer_choice_1}\nb. {answer_choice_2}\nc. {answer_choice_3}\nd. {answer_choice_4}"

            # prepare the image
            raw_image = Image.open(image_qa_df.loc[idx, "image_path"]).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            # Build prompt
            prompt = None
            if img_description_provided:
                image_description = image_qa_df.loc[idx, "image_description"]
                prompt = f"Image Description: {image_description}\n\nQuestion: {question} Choose only one option.\n{answer_string}\n\nAnswer:"
            else:
                prompt = f"\nQuestion: {question} Choose only one option.\n{answer_string}\n\nAnswer:"
            image_qa_df.loc[idx, "blip_2_generated_answers"] = model.generate({"image": image, "prompt": prompt})[0]

            # Continuously store answers in a CSV file in-case of a session timeout
            image_qa_df.to_csv(result_filename, index=False)
    
    # Calculate accuracy of predictions
    calculate_accuracies(result_filename)

# Generate predictions for images without image descriptions
generate_answers("../Dataset/test/DiagramQuestionsData.csv", "blip2_answers.csv")

# Generate predictions for images with image descriptions
generate_answers("common_blip2_llava_gpt4_preds.csv", "common_blip2_llava_gpt4_preds.csv", True)