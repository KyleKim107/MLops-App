from contextlib import asynccontextmanager

from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from mlopslib import MLOpsGCSClient
import logging


class Input(BaseModel):
    text: str="""summarize: Twitter’s interim resident grievance officer for India has stepped down, leaving the micro-blogging site without a grievance official as mandated by the new IT rules to address complaints from Indian subscribers, according to a source.

The source said that Dharmendra Chatur, who was recently appointed as interim resident grievance officer for India by Twitter, has quit from the post.

The social media company’s website no longer displays his name, as required under Information Technology (Intermediary Guidelines and Digital Media Ethics Code) Rules 2021.

Twitter declined to comment on the development.

The development comes at a time when the micro-blogging platform has been engaged in a tussle with the Indian government over the new social media rules. The government has slammed Twitter for deliberate defiance and failure to comply with the country’s new IT rules.
"""

GCP_KEY_FILE = {
    #private info
  "type": "service_account",
  "project_id": "mlops-projects-456022",
  "private_key_id": "73397d9a1f507285ff262047c03ea8e43d183b0c",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCwIKjvhC6fnG2w\nP1kek2ww6rjvJjjgjbley9VEB4OrStvup0KFawz81Ubp7gzbzaZ54MG0n3LmPFBB\nZURlPJAXG7jdWOdM8LE/oIBq6BNqMpl9KpRz9CjGZj/wIn9whUFkjeDBl+d+vGOR\nlZZtHv2abU4YHkY2Ta56PzzPPjZ8E5RDn+DOpUPITGhbMh6hMB0SQvhycUiPVfax\n6bcP+/2zOb+uV4q5ZVU6vXSbRiwhFTJkPu2g7zJMJIch2Wx6XugTrJax9x1TNMZp\nEve0QeT8Z3BmVKrAfpoc/6RV+4tRVFX7U+StomjBY0r3/ltUVrhSgHIHPIgmG/P0\nU0W6sAhtAgMBAAECggEAIQLmFwp74J+lsXiWIBR6nUqJQg20cGYv3dQvk1IjPCpK\nQZdo8O/JHeRtp6/eCh6n+sUFGXEfTIb2deNgp+brjOE9y4L7y2B7oFTZbPA1n1x/\nh0XBUGBteg41IrGcmD7prEqpWy+5DHiL4wAaEzbRSq39Cqk1+LyjnLcYkzFybdu/\nIM5LnIyJd6ojZUf2+Y8ppFIY+3GYfT+XXR03Fj09kUccnxxEL9UX2d5BpuwKCPpM\nHuSdI4QtWxK9zL2lHFCMdUqjymFFoA/agSDDtaBbxVowSgvdTHwQZoC8uxEBk2r6\nnNvsjU6kOsELIrbZexkExcRPZIoU4UBonbuy8ZuKwQKBgQD2JS7GDo8zr74/77kM\nIYITS/ZbstDhy6jySxCfgUBDLgaBA7GLLjwhzoXV2jtwp1KUyHDxUXCuChSzSN1N\na7MsinDba7ZxutyNxzWdbiUXC9KFPyzj+8TVIwaIERaoFXSJtkx9+r3DO40qW2CO\nPb6HycheSOBt3ldzrQC0/pR4DQKBgQC3LdhKMrKusQPjIr2qOxw1soF6wY2ABDVN\n6fpMQ+r3+t1R0Jd8mQZXLRvMCaswFJoGnKIlXVDLUg4kXfY3S9NIdKFHBepOIyKt\n7PWgFT0xhiB80OQz7Rjz2K1BPr22zXJ8hmqrJWkEZ0nOtjryx74iHdhIlB2c2jtc\nIIKUSs5Z4QKBgQCQdsI8+86QR1UhsVCqDgCJ0suEhi6SVzWQ/v2CoBlXVGv8zG4n\ngbtJHCzwlmTMeWe+auXevC2l98SudNVt2wgN3DLglRc/KKzJiWGuRDcAj6+/BJUU\nOXAMp8HtmhnyDKWZLaU3SzKUox72/SpuUU7B42v16k/OK/Cns6H2mHQLAQKBgQCC\nu5hzFA3MeY9xqti8RqWSeUzJhJvtcpNsgQHfLlXjF8qe2oVTKo+I5ivBimD/1qQ6\n9c0PZ7MC+RQYxmqz+kzXmE2GN+WKUK8ufECtOJtM3hIPorJLkPrPBaRZPWL+tU/x\nKam1beSufN0DR7F4li/Pf4dd/T/JM0NMM/OdzGnLYQKBgBN3dla1iRZ9F6HegmWH\nMEEf+9GE1RvDPDLdpKNg3WoGWRFOE1HeLg4B05DVYrqnaumM030Nkdsb9bdoiuFu\nWw/GzjizNAfVnXKSkRr/J0quoQ65b2dgUhVHJT6WDs4+jrzQlxw5YwO/j6fjHqDH\njYzADISWmMYlIyldJj2a7U7Q\n-----END PRIVATE KEY-----\n",
  "client_email": "mlops-service-account@mlops-projects-456022.iam.gserviceaccount.com",
  "client_id": "112355588665505574695",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/mlops-service-account%40mlops-projects-456022.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}


def load_model():
    client = MLOpsGCSClient(GCP_KEY_FILE) # gs://mlops-library/nlp-model/config.json
    model_list = ['config.json' , 'pytorch_model.bin', 'special_tokens_map.json', 'spiece.model', 'tokenizer.json', 'tokenizer_config.json']
    blob_base = 'nlp-model'

    for model_name in model_list:
        client.download_model(bucket_name="mlops-library",
                            blob_name=f"{blob_base}/{model_name}",
                            dest_file_path=f"./model/{model_name}")
    print("start model load")
    model = pipeline("summarization", model="./model", framework="pt")
    print("finish model load")
    return model

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logging.basicConfig(level=logging.DEBUG)
    ml_models["nlp_model"] = load_model()
    yield

    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(input: Input):
    print(f"Input to predict: {input}, Type: {type(input)}")
    result = ml_models["nlp_model"](input.text, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]
    return {"result": result}
    