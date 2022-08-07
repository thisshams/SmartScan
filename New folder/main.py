from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def root():
    return {'message': 'hello world'}


@app.get("/name/{name}")
def root(name):
    return {'Name': name}


@app.get("/rno/{rno}")
def root(rno: int):
    return {'Rno': rno}


class Blog(BaseModel):
    title: str
    body: str
    published_at: Optional[bool]


@app.post('/blog')
def create_blog(request: Blog):
    return {'data': 'data is created'}
