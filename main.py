from typing import Optional
from fastapi import FastAPI
from magicsim import main_calculation_func

app = FastAPI()
'''
демо-функция
@app.get("/")
def hello(text: Optional[str] = None):
    return "It works!"
'''

@app.get("/recommendations/")
def make_some_recommendations(rows: Optional[int] = 10, vacancy_age: Optional[int] = 120, text: Optional[str] = None):
    """Запускает осовную функцию подбора"""
    return main_calculation_func(text, rows, vacancy_age)
