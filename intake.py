import AWSIoTPythonSDK.MQTTLib as awsmqtt
import json
import time
import pickle
import math
from sklearn.linear_model import LinearRegression

rootCAFile = "AmazonRootCA1.pem"
certFile = "0154ac910202c2a7bd11d5643037594dd97db3a45c7fa9c57b0764bdd38eed69-certificate.pem.crt.txt"
privCertFile = "0154ac910202c2a7bd11d5643037594dd97db3a45c7fa9c57b0764bdd38eed69-private.pem.key"
endpoint = "a3m5m42t28eosr-ats.iot.us-west-2.amazonaws.com"
client = "mlpy"

subTopic = "esp32/pub"
pubTopic = "mlpy/pub"

CRIT_RH = 80
TIME_INTERVAL = 1 / 24  # put numerator in hrs, denominator is constant DO NOT CHANGE

def on_message(client, userdata, message):
    data = json.loads(message.payload)
    print(data)

    # assume category 2 sensitivity for material
    # (drywall, paper-based products & films, planed wood, wood-based panels)
    mmax = mmax_model.predict([[data["Temperature"], data["Moisture"], (CRIT_RH - data["Moisture"])]])
    growth = 0
    if (data["Moisture"] > 50):
        growth = TIME_INTERVAL * 1/(7 * math.exp((-.68 * math.log(data["Temperature"])
                                                  - (13.9 * math.log(data["Moisture"])) + 66.02)))
    recession = 0
    if (data["DryTime"] > 24):
        recession = -.016 * TIME_INTERVAL

    predM = m_model.predict([[growth, recession, mmax[0]]])
    pubData = {
        "Time": data["Time"],
        "Room": data["Room"],
        "Temperature": data["Temperature"],
        "Humidity": data["Moisture"],
        "Prediction": predM[0]
    }
    mqttClient.publish(pubTopic, json.dumps(pubData), 0)


# initialize connection
mqttClient = awsmqtt.AWSIoTMQTTClient(client)
mqttClient.configureEndpoint(endpoint, 8883)
mqttClient.configureCredentials(rootCAFile, privCertFile, certFile)

# read in models
with open("model.pkl", 'rb') as mmax_model_file:
    mmax_model = pickle.load(mmax_model_file)

with open("model.pkl", 'rb') as m_model_file:
    m_model = pickle.load(m_model_file)

mqttClient.connect()
mqttClient.subscribe(subTopic, 0, on_message)

# while loop
while True:
    time.sleep(1)
