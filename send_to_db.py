from flask import Flask, request, jsonify
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from acquire_signal import acquire_signal
from daq import tdoa
import requests

app = Flask(__name__)

client = MongoClient('mongodb+srv://wwagud01:wwowA2002@testcluster0.z0psvfl.mongodb.net/?retryWrites=true&w=majority&appName=TestCluster0', server_api=ServerApi('1'))

db = client['test']

instruments = db['instruments']

 

if __name__ == '__main__':
    try:
        client.admin.command('ping')
        print("Pinged!")
    except Exception as e:
        raise e
    
    instruments.drop()

    time = int(input("How long do you want to record (in seconds)? "))
    data, num_chans = acquire_signal(time)

    r, angle = tdoa(data, num_chans, time)
    r = r[0]
    angle = angle[0]



    data = {

        'name': 'Unknown',

        'angleOfArrival': int(angle),

        'distance': int(r),

        'photoUrl': ''

    }

    instruments.insert_one(data)