from abc import ABC
import json
import logging
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class TransformersClassifierHandler(BaseHandler, ABC):
    """
    The handler takes an inout string and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        
        # If you would like to use JIT trace instead, 
        # please uncomment the following line
        # self.model = torch.jit.load(model_pt_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        #Make sure to use the same tokenizer that we used during training
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True

    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes.
            Extend with your own preprocessing steps as needed.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info("Received text: '%s'", sentences)

        inputs = self.tokenizer.encode_plus(
                  sentences,
                  max_length=128,
                  add_special_tokens=True,
                  return_token_type_ids=False,
                  pad_to_max_length=True,
                  return_attention_mask=False,
                  return_tensors='pt',
        )
        return inputs

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        prediction = self.model(
            inputs['input_ids'].to(self.device)
        )[0].argmax().item()
        print(prediction)
        logger.info("Model predicted: '%s'", prediction)

        if self.mapping:
            prediction = self.mapping[str(prediction)]

        return [prediction]

    def postprocess(self, inference_output):
        return inference_output


