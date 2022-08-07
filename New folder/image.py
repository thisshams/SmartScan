from fastapi import FastAPI, File, UploadFile
# from deta import Drive

app = FastAPI()
# files=Drive("myfiles")


@app.post("/")
def upload(file: UploadFile = File(...)):
    # return files.put(file.filename,file.file)
    return list(file)
