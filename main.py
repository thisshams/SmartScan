import pyrebase
from fastapi import FastAPI, File, UploadFile, Form
# from deta import Drive
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import cv2
import base64
import time
from PIL import Image


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
async def analyze_route(uid: str = Form(...), requestCode: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    pstart = time.time()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("Image recieved")
    img_dimensions = str(img.shape)

    # image = cv2.GaussianBlur(img, (5, 5), 0)
    image = img.copy()

#     image = cv2.GaussianBlur(img, (5, 5), 0)
    image = img.copy()

    scan = processImage(image)
    edgeDetected = edge(img)
    houghP = houghLinesP(edgeDetected)
#     croppedImg = contourCrop(houghP, scan)
    compress(scan)
    print("Program Total RunTime :", time.time()-pstart)

    config = {
        "apiKey": "AIzaSyAaa8Fv5r8l3AjLfFlw4CZk48oNLk-iXfc",
        "authDomain": "smartscan-7d1a3.firebaseapp.com",
        "projectId": "smartscan-7d1a3",
        "storageBucket": "smartscan-7d1a3.appspot.com",
        "messagingSenderId": "427150561575",
        "appId": "1:427150561575:web:ded4c273c16eba27d85902",
        "databaseURL": "",
        "serviceAccount":  {
            "type": "service_account",
            "project_id": "smartscan-7d1a3",
            "private_key_id": "c2387e82c1ca27062a570fbc6c0d15fe133e7bae",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDCPYct1GO/FL0X\nWeryId95vXXnWfBhjZW/CZRjMSB41Qi774s+CvgVnAqjYiDoyRaWW01DZ3nQmYgd\n2zZpb4qOUK9/YBkNURzeaS7yeSv2zp2bdnbfOqYp65sod49Pxisf2xwVYHeVcwmH\nbd6sND6ke33trHQvf3v8aBhFI729/nmj/O3DHkd2pEF0ytK9OM7efLQTArntfZcr\nhULVoNBYwpWakIUtUHTba8ohPBGGWRxC5RkAyLSe/4iB230JDLcLPiRUgGVvkB4a\nzoq1ImKE/V11BBI48nU7HVDgTiJ3powjlyMsRxog1GNBCR/JIjButyDfZRILb5TR\nXOMkJE61AgMBAAECggEAFB92sS9HNCUMX+5vUi1jLwQLQnAFYL3fzR5LcqlEwW/k\niz/KT5+oSujTC1Efsq4eem40B28hZhR5zwoGTY8CLjM6szn77m7ATGReOj2GaffG\nyTPRJdg4HbCsbtlQDgYsMo0rECXhzahQzOh7gKCa+sRWa/iJQuB0slYAaa1Fu3iC\nii+arUmPYK2usprVTFhQGSMINfSo4JbNF1VzozgeLp4EGkivexuMcCKJC3oqT9/t\n+4p/OZjvNSdhlIp0O7e+yx6VYFeEQ0PXD2LeDbukhnTTBKq8LbrNuzJiQqydrMcI\n7v7wmKLLttIU79g7LsOpsNnq1ZvgNCIWZQfNkTx4YQKBgQD78SDmkXlP/y9wrB5T\nex2N0ELdRbK3KkrMcoUud/uiDOTmMTl0lgX6hhQFRxZOKXAmoaKhK1ZKicHhUp0x\nuSeQhy0m/R4BLRo1CInJDOYCIR0TZ8akN2zwRy5WlDeHJvNJer4vWLM6YiRUTFAP\nO0Y6IJeUkUOYxWHY5EA039JdEQKBgQDFXng6Npf9ZW5Q6hkvrK0egCXESgcTg1tM\npmQhsCwmjtK0ZWhrCO77+d7vg+OF3TZCvQLxivkWDa11kTzyACsJaZoWD1o5eOwH\nYLxqnFntf5heDgDcrBnVpGPPJC6S97BjB0Qp9+kmcBlIvffv656F57r/++gOPzL8\nYQK3ingnZQKBgGS4SUbj8XOhuP16UcVd+rqu/4wmSQQgzDZfsg6ZuOdX8Ep2c1nA\ngDNfVrGlca1ds5A+Hh4AjUbPO8swk9dFBiQpZkun9U7TER8SgsL1fR5szorreeY8\nojiMvGGwb2KAl9JQV6fl9gDpK0zoFTmBoNmsHe0vBa8VecCTv3dj412BAoGAMrlL\nttJPD42g42S2ol0DhQI0MpU/6lDpBvMAavQG9MXh+wDQ7Ck4mkOmevHvaHjouBAx\nkHhB+dv8B2oTOrK2XM3qDt9VNc4RAvhmlBOovPP86bc5m30TiqecCyFmYtkLWPgG\nGa8gGYPXy60e6mcor4tVsPJBul+dr+USuK76oE0CgYBPVbcLaN3heGgklIGtBiHC\nWC8cPWieltTJal7F24HWYKmcaJNHQMvbPZPfc3kAIrFWjhqNduU62p1g1dfpEClJ\nY4C3adEbaiG2eh4AbH1AFZbW0235f1U7yRkzEi9fXfETkrnm4NEVUeIEU5Z12GE7\nRHRMn7sKVqA/ViySgGXPPQ==\n-----END PRIVATE KEY-----\n",
            "client_email": "firebase-adminsdk-qzn7o@smartscan-7d1a3.iam.gserviceaccount.com",
            "client_id": "104110910319662337608",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-qzn7o%40smartscan-7d1a3.iam.gserviceaccount.com"
        }
    }

    firebase_storage = pyrebase.initialize_app(config)
    storage = firebase_storage.storage()

    # upload
    # storage.child(return_img).put(file.filename)
    # blob = storage.blob(file.filename, chunk_size=262144)  # 256KB
    # blob.upload_from_file(return_img)
    # return_img = stretch_near = cv2.resize(
    #     return_img, (1240, 1754), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite("output.jpg", return_img)
    try:
        storage.child(uid+"/"+requestCode+"/output.jpg").put("output.jpg")
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


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def processImage(img):
    img = img.copy()
    kSize = 51
    whitePoint = 127
    blackPoint = 66

    if not kSize % 2:
        kSize += 1

    kernel = np.ones((kSize, kSize), np.float32)/(kSize*kSize)

    filtered = cv2.filter2D(img, -1, kernel)

    filtered = img.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127*np.ones(img.shape, np.uint8)

    filtered = filtered.astype('uint8')

    img = filtered

    _, img = cv2.threshold(img, whitePoint, 255, cv2.THRESH_TRUNC)

    img = img.astype('int32')
    img = map(img, 0, whitePoint, 0, 255)
    img = img.astype('uint8')

    print("Scanning Done.")

    # refer repository's wiki page for detailed explanation

    img = img.astype('int32')

    img = map(img, blackPoint, 255, 0, 255)

    # if cv.__version__ == '3.4.4':
    #img = img.astype('uint8')

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)

    img = img.astype('uint8')
    print("image scan complete")
    return img


def edge(image):
    start = time.time()
    print("Edge Detection started")
    # Load the model.
    net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt", "hed_pretrained_bsds.caffemodel")
    cv2.dnn_registerLayer('Crop', CropLayer)

    # image = cv2.imread(r"test1.jpg")
    image = cv2.resize(image, (0, 0), fx=0.07, fy=0.07)
    # image = cv.GaussianBlur(image, (5, 5), 3)

    h = image.shape[0]
    w = image.shape[1]

    inp = cv2.dnn.blobFromImage(image, scalefactor=5, size=(w, h),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(inp)
    # edges = cv.Canny(image,image.shape[1],image.shape[0])
    out = net.forward()

    out = out[0, 0]
    out = cv2.resize(out, (image.shape[1], image.shape[0]))
    cv2.dnn_unregisterLayer('Crop')

    print(out.shape)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)

    print(type(out))
    print(np.max(out))
    print(np.min(out))
    print(out.shape)
    print(image.shape)
    # con = np.concatenate((image, out), axis=1)
    # cv.imwrite('out.jpg', con)
    # cv2.imwrite("output.jpg", out)

    print("Time Taken For Edge Detection :", time.time()-start)
    return out


def houghLinesP(dst):
    # dst = cv2.Canny(src, 180, 200)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    linesP = cv2.HoughLinesP(dst, 3, np.pi / 180, 50, None, 100, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]),
                     (255, 255, 255), 3, cv2.LINE_AA)

    # plt.imshow(src[..., ::-1])
    # plt.show()
    ret, thresh1 = cv2.threshold(cdstP, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh1


def contourCrop(img, scan):
    img = cv2.resize(img, (0, 0), fx=14.28, fy=14.28)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    # threshold to get just the signature (INVERTED)
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255,
                                        type=cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx = (0, 0, 0, 0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
    x, y, w, h = mx

    # Output to files
    image = scan
    roi = image[y:y+h, x:x+w]
    print(image.shape)
    print(img.shape)
    return roi


def compress(img):
    img = Image.fromarray(img)
    w, h = img.size
    print(w, h)
    myheight, mywidth = img.size
    img = img.resize((w, h), Image.ANTIALIAS)
    img.save("output.jpg")
#     img = img.resize((w//2, h//2), Image.ANTIALIAS)
#     img.save("compressedHalfSize.jpg")
