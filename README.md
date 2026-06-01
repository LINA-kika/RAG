# README

### Что поменялось

#### 1. Модели

| Что | Было | Стало |
|-----|------|-------|
| Эмбеддинг-модель | `text-embedding-nomic-embed-text-v1.5` | `bge-m3` |
| LLM | `google/gemma-4-e4b` | `qwen/qwen3-1.7b` |

`text-embedding-nomic-embed-text-v1.5` в задании №1 показала плохое косинусное сходство, поэтому она была заменена. Возможно, `text-embedding-nomic-embed-text-v1.5` плохо работает с русским языком.

![Косинусное сходство nomic-embed-text-v1.5](https://drive.google.com/file/d/1cgFeRcIrOBF9vx6vdpGusevemwpktyGs/view?usp=sharing)

![Косинусное сходство bge-m3](https://drive.google.com/file/d/1jp2hMYOLe61jJG0cNx9P9z4qnufdSQkC/view?usp=sharing)

#### 2. Добавлено уточнение неопределённостей

Функция ищет в ответе LLM маркеры неуверенности и вызывает функцию, которая просит пользователя уточнить свой вопрос.

![Пример уточнения](https://drive.google.com/file/d/1uF-QvT9fP2Xfgb-2HubSCaXUYMGzWpTt/view?usp=sharing)

```python
def detect_uncertainty(answer: str) -> tuple[bool, list[str]]
```
#### 3. Добавлена функция байесовского сюрприза

Считает, насколько текущий запрос отличается от среднего. Формула:
```python
def bayesian_surprise(query: str, results: list, index: np.ndarray, chunks: list[str]) -> float```

сюрприз = (лучшее_сходство - среднее_сходство) / среднее_сходство

