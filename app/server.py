from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
from fastai.vision import *
import base64

model_file_url = 'https://github.com/simongrest/fastai-vision-app/blob/master/app/models/model.pkl'
model_file_name = 'model'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pkl')
    learn = load_learner(path/'models', f'{model_file_name}.pkl')
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

PREDICTION_FILE_SRC = path/'static'/'predictions.txt'

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = data['img']
    bytes = base64.b64decode(img_bytes)
    return predict_from_bytes(bytes)

def predict_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    predictions = learn.predict(img)
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    
    if str(predictions[0]) == 'POSITIVES':
        pred = 'The test was read as <b>Positive</b> with a '
    elif str(predictions[0]) == 'NEGATIVES':
        pred = 'The test was read as <b>Negative</b> with a '
    else:
        pred = 'The test was read as <b>Invalid</b> with a '
        
        
    result_html = str(result_html1.open().read() + pred + str(round(float(predictions[2].max())*100,2)) +'%  probability' + result_html2.open().read())
    return HTMLResponse(result_html)

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    #print('in main')
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
