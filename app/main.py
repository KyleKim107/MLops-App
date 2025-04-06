from contextlib import asynccontextmanager

from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
import logging


class Input(BaseModel):
    text: str="""summarize: Twitter’s interim resident grievance officer for India has stepped down, leaving the micro-blogging site without a grievance official as mandated by the new IT rules to address complaints from Indian subscribers, according to a source.

The source said that Dharmendra Chatur, who was recently appointed as interim resident grievance officer for India by Twitter, has quit from the post.

The social media company’s website no longer displays his name, as required under Information Technology (Intermediary Guidelines and Digital Media Ethics Code) Rules 2021.

Twitter declined to comment on the development.

The development comes at a time when the micro-blogging platform has been engaged in a tussle with the Indian government over the new social media rules. The government has slammed Twitter for deliberate defiance and failure to comply with the country’s new IT rules.
"""


def load_model():
    print("start model load")
    model = pipeline("summarization", model="./model", framework="pt")
    # model.load_model("t5", "./model", use_gpu =False)
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
    