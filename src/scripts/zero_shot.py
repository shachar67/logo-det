from pathlib import Path
from PIL import Image
import google.generativeai as genai

API_KEY = "AIzaSyDQqB_HCz6CtgFJ4TllwkFJ3Usu4d109jQ"
BASE_PROMPT = """
Please provide bounding box and classification for logos in this image.
Return the answer as a plain text response where each row represents a bounding box in the format "class_name x_center y_center width height".
The coordinates must be normalized (between 0 and 1).
- Make sure to include all instances of logos.
- The logo can be the brand's symbol or its name written in its branded font and color.
- Be as inclusive as possible, including all brand markings.
"""

JUDGE_PROMPT = """
As an expert data scientist and annotator, your task is to evaluate the quality of an AI model's logo detection predictions.
You will be given the ground truth labels and the model's predicted labels for an image.

Ground Truth Labels:
{ground_truth}

Model's Predicted Labels:
{predictions}

Your evaluation should be thorough. Analyze the predictions and provide the following:
1.  **Correct Classifications**: List the labels that the model correctly identified.
2.  **False Positives**: List any labels predicted by the model that are not in the ground truth.
3.  **Missed Detections (False Negatives)**: List the labels from the ground truth that the model failed to detect.
4.  **Overall Accuracy**: Provide a brief summary of the model's performance on this image.
5.  **Suggestions for Human Labeler**: Note if the human labeler might have missed any logos that the model correctly identified.

Provide your evaluation in a clear, structured format.
"""


class GeminiAPI:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.vision_model = genai.GenerativeModel('models/gemini-2.5-pro')
        self.text_model = genai.GenerativeModel('models/gemini-pro-latest')

    def get_completion(self, prompt: str, image: Image.Image) -> str:
        response = self.vision_model.generate_content([prompt, image])
        return response.text


class SimpleAI:
    def __init__(self, api_key: str, prompt: str):
        self.api = GeminiAPI(api_key)
        self.prompt = prompt

    def get_labels(self, image_path: Path) -> str:
        image = Image.open(image_path)
        return self.api.get_completion(self.prompt, image)


class AIJudge:
    def __init__(self, api_key: str, judge_prompt: str):
        # We can reuse the GeminiAPI instance's text model
        genai.configure(api_key=api_key)
        self.text_model = genai.GenerativeModel('models/gemini-pro-latest')
        self.prompt = judge_prompt

    def evaluate(self, ground_truth: str, predictions: str) -> str:
        formatted_prompt = self.prompt.format(
            ground_truth=ground_truth,
            predictions=predictions
        )
        response = self.text_model.generate_content(formatted_prompt)
        return response.text


class ZeroShotPredictor:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def run(self, image_path: str, label_path: str = None):
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"Error: Image file not found at {image_path}")
            return

        # 1. Get AI predictions
        simple_ai = SimpleAI(api_key=self.api_key, prompt=BASE_PROMPT)
        predicted_labels = simple_ai.get_labels(image_path)
        print("\n--- Model Predictions ---")
        print(predicted_labels)

        # 2. If a label file is provided, get Ground Truth and AI Judge Evaluation
        if label_path:
            label_path = Path(label_path)
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as f:
                    ground_truth_labels = f.read()
                print("\n--- Ground Truth Labels ---")
                print(ground_truth_labels)

                ai_judge = AIJudge(api_key=self.api_key, judge_prompt=JUDGE_PROMPT)
                evaluation = ai_judge.evaluate(
                    ground_truth=ground_truth_labels,
                    predictions=predicted_labels
                )
                print("\n--- AI Judge Evaluation ---")
                print(evaluation)
            else:
                print(f"\nWarning: Label file not found at {label_path}")


def main():
    # This main function is for standalone execution of the script
    predictor = ZeroShotPredictor(api_key=API_KEY)
    image_path = r"C:\projects\logo-det\data\logos-dataset-section-2\Adidas_38.jpg"
    label_path = r"C:\projects\logo-det\data\logos-dataset-section-2\Adidas_38.txt"
    predictor.run(image_path, label_path)


if __name__ == "__main__":
    main()
    
