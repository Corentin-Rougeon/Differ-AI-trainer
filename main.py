import queue

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from scrapper.main import run_image_scraper
from run_model.process import process
from model_trainer.train import train
from pathlib import Path
from queue import Queue
import threading
import multiprocessing
import os
import time
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)

request_queue = Queue()
current_proccess = []


def process_requests():
    while True:
        if not request_queue.empty():
            global current_proccess
            # Process the request


            request_data = request_queue.get()
            current_proccess = [request_data]
            socketio.emit("update_queue", current_proccess + list(request_queue.queue))

            # Do something with the request
            print("Processing request:", request_data)

            if request_data["func"] == "scrape":
                run_image_scraper(dirpath=f"data/{request_data['current_project']}",class1=request_data["class1"],class2=request_data["class2"],n=int(request_data["n"]))
            elif request_data["func"] == "train":
                dir = f"data/{request_data['current_project']}"
                train_thread = multiprocessing.Process(target=train,args=(dir,int(request_data["epoch"])))
                train_thread.start()
                train_thread.join()
            elif request_data["func"] == "test":
                dir = f"data/{request_data['current_project']}"
                train_thread = multiprocessing.Process(target=process,args=(dir,request_data["img_url"]))
                train_thread.start()
                train_thread.join()
                get_lastest_test(request_data['current_project'])



            get_project_meta(request_data["current_project"])
            print("process done")
            del current_proccess[0]
            socketio.emit("update_queue",current_proccess + list(request_queue.queue))
            #socketio.emit('response', {'message': 'Request processed'}, namespace='/')
        else:
            # Wait for 1 second before checking the queue again
            time.sleep(1)




def send_project_list():
    folder_path = Path('data')

    folders = []

    for item in folder_path.iterdir():
        if item.is_dir():
            folders.append(item.name)


    socketio.emit("update_project_list", folders)


def get_project_meta(project):
    with open(f"./data/{project}/meta.json") as f:
        data = json.loads(f.read())

        socketio.emit("get_project_meta",data)


def get_lastest_test(project):
    if os.path.isfile(f"./data/{project}/last_test.json"):
        with open(f"./data/{project}/last_test.json") as f:
            data = json.loads(f.read())

            socketio.emit("get_last_test",data)
    else:
        socketio.emit("get_last_test","no_test")





@app.route('/')
def index():


    return render_template('index.html')


@socketio.on("loaded")
def loaded():
    send_project_list()
    print(current_proccess + list(request_queue.queue))
    socketio.emit("update_queue",current_proccess + list(request_queue.queue))


@socketio.on("get_project")
def get_project(data):
    get_project_meta(data)
    get_lastest_test(data)


@socketio.on("train_model")
def train_model(data):
    print(data)
    if data["current_project"] != "":
        request_queue.put({"func": "train",
                           "current_project":data["current_project"],
                           "epoch":data["epoch"]})

        socketio.emit("update_queue", current_proccess + list(request_queue.queue))


@socketio.on("test_model")
def test_model(data):
    print(data)
    if data["current_project"] != "":
        request_queue.put({"func": "test",
                           "current_project": data["current_project"],
                           "img_url": data["img_url"]})
        socketio.emit("update_queue", current_proccess + list(request_queue.queue))


@socketio.on("scrape")
def scrape_images(data):
    print(data)
    if data["current_project"] != "":
        request_queue.put({"func": "scrape",
                           "current_project":data["current_project"],
                           "class1":data["class1"],
                           "class2":data["class2"],
                           "n":data["n"]})

        socketio.emit("update_queue", current_proccess + list(request_queue.queue))

@socketio.on("create_project")
def create_project(data):

    print(data)

    if not os.path.isdir(f"data/{data}"):
        os.mkdir(f"data/{data}",mode=0o777)

        os.mkdir(f"data/{data}/img",mode=0o777)

        os.mkdir(f"data/{data}/img/train",mode=0o777)
        os.mkdir(f"data/{data}/img/validation", mode=0o777)

        with open(f"data/{data}/meta.json","w", encoding='utf-8') as f:
            data = {"name":data,"has_model":False,"has_img":False}
            f.write(json.dumps(data,indent=4))

    send_project_list()



@socketio.on('submit_request', namespace='/')
def submit_request(data):
    prompt1 = data['prompt1']
    prompt2 = data['prompt2']
    request_queue.put((prompt1, prompt2))

if __name__ == '__main__':
    # Start a separate thread to process the requests
    processing_thread = threading.Thread(target=process_requests)
    processing_thread.daemon = True
    processing_thread.start()

    socketio.run(app,allow_unsafe_werkzeug=True)