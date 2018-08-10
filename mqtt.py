import paho.mqtt.client as mqtt
import json
from datetime import timedelta, datetime


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))


def on_publish(client, userdata, result):
    print("data published")


def on_disconnect(client, userdata, rc):
    print("client disconnected ok")


class Prediction:

    def __init__(self, day, time, prediction):
        self.day = day
        self.time = time
        self.prediction = prediction


class MqttConnect:

    def __init__(self):
        self.client = mqtt.Client("temperature-prediction")
        self.client.on_connect = on_connect
        self.client.on_publish = on_publish
        self.client.on_disconnect = on_disconnect

    def publish_json(self, prediction):
        today = datetime.today().strftime("%Y-%m-%d")
        time = datetime.now()
        delta = timedelta(minutes=15)
        l = {time.strftime("%H:%M"): prediction[0]}
        for i in range(12):
            time += delta
            l[time.strftime("%H:%M")] = prediction[i]
        js = {"date": today, "predictions": l}
        self.client.connect("fractal.tools", 1883, 60)
        self.client.publish("1/2/T1/PR/1/0", json.dumps(js))
        self.client.disconnect()
