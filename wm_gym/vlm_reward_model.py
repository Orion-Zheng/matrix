import base64
import os
import io
from typing import List, Tuple, Union
from openai import OpenAI
from PIL import Image

class OpenAIRewardModel:
    def __init__(self, api_key: str, reward_query: str, reward_criteria: str = None, model: str = "gpt-4o"):
        os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI(api_key=api_key)
        self.query_prompt = reward_query
        self.summary_prompt = reward_criteria
        self.model = model

    def encode_image_file(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def encode_pil_image(self, pil_image: Image.Image) -> str:
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _build_contents(self, image_sources: List[Union[str, Image.Image]]) -> List[dict]:
        contents = []
        for idx, image in enumerate(image_sources):
            contents.append({"type": "text", "text": f"Image {idx + 1}:"})
            if isinstance(image, str):
                img_base64 = self.encode_image_file(image)
            elif isinstance(image, Image.Image):
                img_base64 = self.encode_pil_image(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}",
                    "detail": "high"
                }
            })
        contents.append({"type": "text", "text": self.query_prompt})
        return contents

    def analyze(self, images: List[Union[str, Image.Image]], temperature: float = 0.0) -> Tuple[str, str]:
        contents = self._build_contents(images)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": contents}],
            temperature=temperature,
            max_tokens=1000
        )
        answer = response.choices[0].message.content.strip()

        if self.summary_prompt:
            extracted = self.extract_answer(answer)
        else:
            extracted = self.simple_extract_yes_no(answer)

        return answer, extracted

    def extract_answer(self, full_answer: str, temperature: float = 0.0) -> str:
        prompt = self.summary_prompt.format(full_answer)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    def simple_extract_yes_no(self, text: str) -> str:
        lowered = text.lower()
        if "no" in lowered or "not" in lowered or "false" in lowered:
            return "No"
        elif "yes" in lowered or "true" in lowered:
            return "Yes"
        else:
            return "Unclear"
        
if __name__ == "__main__":
    from PIL import Image
    import os

    analyzer = OpenAIRewardModel(
        api_key=os.environ["OPENAI_API_KEY"],
        query_prompt="In these images, did the car crash into any obstacles?",
        summary_prompt="Please summarize the following response with a simple Yes or No:\n\"{}\"\n\nAnswer:"
    )

    # Example 1: Using file paths
    image_paths = ["img1.jpg", "img2.jpg"]
    full_response, short_answer = analyzer.analyze(image_paths)
    print("Full response:", full_response)
    print("Extracted answer:", short_answer)

    # Example 2: Using PIL images
    img1 = Image.open("img1.jpg")
    img2 = Image.open("img2.jpg")
    full_response_pil, short_answer_pil = analyzer.analyze([img1, img2])
    print("Full response (PIL):", full_response_pil)
    print("Extracted answer (PIL):", short_answer_pil)

