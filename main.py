from typing import Optional
from fastapi import FastAPI
from magicsim import main_calculation_func

app = FastAPI()

@app.get("/")
def hello(text: Optional[str] = None):
    return "It works!"


@app.get("/recommendations/")
def make_some_recommendations(text: Optional[str] = None):
    return main_calculation_func(text)
