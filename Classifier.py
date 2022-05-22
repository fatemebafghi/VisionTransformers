
from turtle import shape
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
from typing import Dict,List
from logzero import logger

class Model:
    """ Pass """
    def __init__(self) -> None:
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

    def predict(self,image_path:str) -> Dict:

        image = Image.open(image_path)
        # logger.debug(f"shape = {shape()}")
        input = self._preprocess(image)
        output = self.model(**input)
        logits = output.logits
        predicted_class_id = logits.argmax(-1).item()
        predicted_class = self.model.config.id2label[predicted_class_id]
        return {"class_id":predicted_class_id, "class_name":predicted_class}


    def _preprocess(self,image:Image) -> List:
        input = self.feature_extractor(images=image, return_tensors="pt")
        return input

