import AWSIoTPythonSDK.MQTTLib as awsmqtt
import json
import time
import pickle
from sklearn.linear_model import LinearRegression

rootCAFile = "AmazonRootCA1.pem"
certFile = "0154ac910202c2a7bd11d5643037594dd97db3a45c7fa9c57b0764bdd38eed69-certificate.pem.crt.txt"
privCertFile = "0154ac910202c2a7bd11d5643037594dd97db3a45c7fa9c57b0764bdd38eed69-private.pem.key"
endpoint = "a3m5m42t28eosr-ats.iot.us-west-2.amazonaws.com"
client = "mlpy"

subTopic = "esp32/pub"
pubTopic = "mlpy/pub"


def on_message(client, userdata, message):
    data = json.loads(message.payload)
    print(data)
    predGrowth = LRmodel.predict([[data["Temperature"], data["Moisture"], (90 - data["Moisture"]), 0, 0]])
    pubData = {
        "Room": data["Room"],
        "prediction": predGrowth[0]
    }
    mqttClient.publish(pubTopic, json.dumps(pubData), 0)


# initialize connection
mqttClient = awsmqtt.AWSIoTMQTTClient(client)
mqttClient.configureEndpoint(endpoint, 8883)
mqttClient.configureCredentials(rootCAFile, privCertFile, certFile)

# read in model
with open("model.pkl", 'rb') as model_file:
    LRmodel = pickle.load(model_file)

mqttClient.connect()
mqttClient.subscribe(subTopic, 0, on_message)

# while loop
while True:
    time.sleep(1)
