# main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import numpy as np

# Инициализация приложения и шаблонов
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Загрузка модели (выполняется один раз при старте приложения)
model = load_model('iris_model.h5')
# Названия классов для вывода
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


# Эндпоинт для отображения формы
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


# Эндпоинт для обработки данных формы
@app.post("/predict", response_class=HTMLResponse)
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    predicted_class_name = class_names[np.argmax(prediction)]

    # Ключевое изменение: request передаётся как первый параметр
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": predicted_class_name}
    )