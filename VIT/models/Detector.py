from base_model import BaseModel
from logzero import logger
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from transformers import YolosModel, YolosConfig
from typing import List,Dict
from PIL import Image


class Model(BaseModel):

    """This Model tries to detect objects which are located in a picture """

    def __init__(self, model = "Yolos") -> None:
        self.Model = model
        # self.configs = YolosConfig()
        # self.model = YolosModel.config(self.configs)
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
        # self.configs = self.model(self.config)
        self.feature_extractor = YolosFeatureExtractor.from_pretrained("hustvl/yolos-small")


    def predict(self,image_path:str) -> Dict:
        image = Image.open(image_path)
        inputs = self._preprocess(image)
        outputs = self.model(**inputs)
        # model predicts bounding boxes and corresponding COCO classes
        logits = outputs.logits
        bboxes = outputs.pred_boxes
        logger.debug(f"bboxes = {bboxes} and logits = {logits}")
     
    def _preprocess(self,image:Image) -> List:
        input = self.feature_extractor(images=image, return_tensors="pt")
        return input