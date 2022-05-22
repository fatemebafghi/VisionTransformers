# from transformers import pipeline
# import tensorflow as tf
from logzero import logger
from model import Model

# classifier = pipeline("sentiment-analysis")
# classifier("We are very happy to show you the 🤗 Transformers library.")

# image_classifier = pipeline("image-classification")
# image_path = "index.jpeg"
# result = image_classifier(image_path)
# logger.info(result)

def main():
    model = Model()
    image_path = "index.jpeg"
    result = model.predict(image_path)
    logger.info(result)


if __name__ == '__main__':
    main()
