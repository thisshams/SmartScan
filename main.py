import pyrebase
from fastapi import FastAPI, File, UploadFile, Form
# from deta import Drive
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import cv2
import base64

app = FastAPI()


@app.get("/")
def home():
    return {"ThePixels": "SIH"}
# files=Drive("myfiles")


# @app.post("/")
# def upload(file: bytes = File(...)):
#     # return files.put(file.filename,file.file)
#     # api keys
#     config = {
#         "apiKey": "AIzaSyD3cZX1RNOYr2S15FIfzxHm8CKANpYocD4",
#         "authDomain": "pyrebase-63e5e.firebaseapp.com",
#         "projectId": "pyrebase-63e5e",
#         "storageBucket": "pyrebase-63e5e.appspot.com",
#         "messagingSenderId": "835721530466",
#         "appId": "1:835721530466:web:e3da318f01ce46f3d4153b",
#         "databaseURL": "",
#         "serviceAccount":  "serviceAppKey.json"
#     }

#     firebase_storage = pyrebase.initialize_app(config)
#     storage = firebase_storage.storage()

#     # upload
#     # storage.child(file).put(file.filename+".jpg")
#     return FileResponse(file)

# class Analyzer(BaseModel):
#     uid: str


@app.post("/analyze")  # , response_model=Analyzer)
async def analyze_route(uid: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_dimensions = str(img.shape)
    return_img = processImage(img)
    print("Image recieved")

    config = {
        "apiKey": "AIzaSyD3cZX1RNOYr2S15FIfzxHm8CKANpYocD4",
        "authDomain": "pyrebase-63e5e.firebaseapp.com",
        "projectId": "pyrebase-63e5e",
        "storageBucket": "pyrebase-63e5e.appspot.com",
        "messagingSenderId": "835721530466",
        "appId": "1:835721530466:web:e3da318f01ce46f3d4153b",
        "databaseURL": "",
        "serviceAccount":  {
            "type": "service_account",
            "project_id": "pyrebase-63e5e",
            "private_key_id": "52cfbdd207c1a65ea8412e5ef4fb1981dd0e6fe9",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDRjAGbY9XP67K2\nNIl8FYb1Z7PsfZgeSdxtbuOmZga4Nrc4vlq5QP6wATK7ouea0zKCsXgHdXfZdh5O\nyB5toCj9fjmQ1Q8zMNxZ8aDNjuFFxiywd71JSSm1Z11K0GyfEsLsFLt/I3YSnsc7\nfR0wHHCaeQUzMa4y8vIa11mT++OidD6jebUbPK8WHr8+7ui95jAQrniiVkdHMNPA\njBZs9uTtH9K5A1m6iMRxLKarNB/ifm243aNAqDs6i+CDtJOafw9z6mBvfULAtBLN\nCqVbZOXYl1sEwjNVSc2kOd+4We3+AvPcd+mKwKvjX0ZFHxBljuxzXy9ff/NzEJSZ\ndDowUlodAgMBAAECggEASkHztdXgyNBtYe1nRU45p9CqraVqWV1yXGN+EBM60WAu\neNDYRKsYNlYOXPkjWzX5wd/TOc1yfeFCZ0e6pL2rmP14t+8Q9mKby9H5Wq8F/Nx4\nLMcaEJT5T1xEbsVCoPKcmd3U/f6akomq79VsGQcTDmLRqW0zHsICcumayHrslb44\n2k/QU/T+QyzU/tq44k9b/fpUUEZgVf5xeEt6QqYVRrV16RWRI77DHvTXft18yk58\n2RP7znWFLDb8AMZcuAFnIfM8uzDR6Nh4leUrga9E9M0qF2jSyusp3F7oFzsdv0/x\nNcfGqkCI9KIcKM1WDnl4ypv4sewhw/RawNaRTrBD/wKBgQDtPcRi4eqVJLVZJTtK\nwSQXndajsL5oaOQfkcCoI83pU81CZNC/r5O4GiKL8YMB0Nr+bqKLdWNexWmeY5U5\naEF+q8osiaGbLzl/zGfE2Y+ZFwa9qIJAl5TUeNJJ0xdJ0ESjzfpQOAo+9fZ3TFkW\nZdaVt7p0fIAbN8uuOZHe24zgnwKBgQDiHaZu7KXVfUNS9QmJneSWkXjW2XqLB7Fd\nZbdvTGnendx/Lq/q/yLkyDjw5DI1stmVLiWAHs647xEX5OSI1dnoEVdIKJ4HvwsE\nsCOdJIWUwax+HYOkjD93fEVQ4UjIAlxlhWkkIL/WMv2JVf/D7xeAZ2jW7hRKOpLI\n9qau2TAfwwKBgG6FXu799i3C6yT7flLGBY1m/65EUYAMlXHLkegCvhOb8byjgMg8\nAGI88qklOvXmmY78dYboigGFkD20gLk8w35Cg64Z9Ap6hpvt2s3O2OHl40MJtJxo\nwXH2U2kHCQtfFgsFkz45zTQlm8tZ6wrPKJeY/yjzMy764E1rDnS27TtbAoGAOgrw\naTL4EshX2tipvRi5z+jBwy0KZtvvrJDquHg+CPYu3rrmT4V0uJOpAjUhqmUhs0io\nOa4u1IwRsDeCbpmumQKyjARZJJXmxypLyg/Q9nGMzMbYvwl9VTeiN3PNEgKBI1JO\nvgZmGB74tCNOR0Z9mulwoRN2Q+OrQEkIWoPH9a8CgYBPlYF/oNlTH4aBsizeVDMu\n9PahfC22GswxUfJ/Ry5IEEkTQgsBCehK0ObIAeFwpjtNf18sv2PM20wYo4kt/nn+\nVUaNfcwnpTkbSw63XsCsH34+drVITKb4ksbda3oGif7R/Us1PwOrpLzVmdXhYjl5\ng8jXtHQswtRc51GoBgiUcA==\n-----END PRIVATE KEY-----\n",
            "client_email": "firebase-adminsdk-hf0vj@pyrebase-63e5e.iam.gserviceaccount.com",
            "client_id": "109875827564040494833",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-hf0vj%40pyrebase-63e5e.iam.gserviceaccount.com"
        }
    }

    firebase_storage = pyrebase.initialize_app(config)
    storage = firebase_storage.storage()

    # upload
    # storage.child(return_img).put(file.filename)
    # blob = storage.blob(file.filename, chunk_size=262144)  # 256KB
    # blob.upload_from_file(return_img)
    return_img = stretch_near = cv2.resize(
        return_img, (1240, 1754), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("output.jpg", return_img)
    try:
        storage.child(uid+"/output.jpg").put("output.jpg")
        print("firebase upload successfull")
    except:
        print("Unsuccessfull !!!")
        return {"Error": "FireBase Upload"}
    # line that fixed it
    # _, encoded_img = cv2.imencode('.PNG', return_img)

    # encoded_img = base64.b64encode(encoded_img)

    return{
        'filename': file.filename,
        'dimensions': img_dimensions,
        # 'encoded_img': endcoded_img,
    }


def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def processImage(img):
    img = img.copy()
    kSize = 51
    whitePoint = 127
    blackPoint = 66

    print("applying high pass filter")

    if not kSize % 2:
        kSize += 1

    kernel = np.ones((kSize, kSize), np.float32)/(kSize*kSize)

    filtered = cv2.filter2D(img, -1, kernel)

    filtered = img.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127*np.ones(img.shape, np.uint8)

    filtered = filtered.astype('uint8')

    img = filtered

    print("white point selection running ...")

    # refer repository's wiki page for detailed explanation

    _, img = cv2.threshold(img, whitePoint, 255, cv2.THRESH_TRUNC)

    img = img.astype('int32')
    img = map(img, 0, whitePoint, 0, 255)
    img = img.astype('uint8')

    print("adjusting black point for final output ...")

    # refer repository's wiki page for detailed explanation

    img = img.astype('int32')

    img = map(img, blackPoint, 255, 0, 255)

    # if cv.__version__ == '3.4.4':
    #img = img.astype('uint8')

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)

    img = img.astype('uint8')
    print("image scan complete")
    return img

# image = img.copy()
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# blurred = cv.GaussianBlur(gray, (5, 5), 0)
# # cv2.imshow(fixColor(blurred))

# canny = cv.Canny(blurred, 1, 200)
# # plt.imshow(fixColor(canny))
# cv.imwrite("canny.jpg", canny)
# (cnts, _) = cv.findContours(canny.copy(),
#                             cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# maxw = 0
# maxh = 0
# fx = 0
# fy = 0
# for cnt in cnts:
#     x, y, w, h = cv.boundingRect(cnt)
#     if w > maxw and h > maxh:
#         maxw = w
#         maxh = h
#         fx = x
#         fy = y

#     cv.rectangle(coins, (x, y), (x+w, y+h), (0, 255, 0), 2)

# # cv.imwrite("rect.jpg", coins)
# oh, ow = image.shape[:2]
# if maxw*maxh < (oh*ow)//2:
#     fimg = img
# else:
#     fimg = img[fy:fy+maxh, fx:fx+maxw]
# cv.imwrite("output.jpg", fimg)
# # cv.waitKey(0)

# print("\ndone.")
    # return img
