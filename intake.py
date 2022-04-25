import AWSIoTPythonSDK.MQTTLib as awsmqtt
import json
import time
import pickle
import math
from sklearn.linear_model import LinearRegression

rootCAFile = "AmazonRootCA1.pem"
certFile = "04936d9a74a104fde368e3d39c59fa69867b42c1b1087822afa6e0e0012a48a7-certificate.pem.crt"
privCertFile = "04936d9a74a104fde368e3d39c59fa69867b42c1b1087822afa6e0e0012a48a7-private.pem.key"
endpoint = "a31fy3n3mv508f-ats.iot.us-west-2.amazonaws.com"
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
    rhdiff = 0
    if (data["Moisture"] > CRIT_RH):
        rhdiff = (data["Moisture"] - CRIT_RH)

    growth = 0
    if (data["Moisture"] > 50):
        growth = TIME_INTERVAL * 1/(7 * math.exp((-.68 * math.log(data["Temperature"])
                                                  - (13.9 * math.log(data["Moisture"])) + 66.02)))

    recession = 0
    if (data["DryTime"] > 6):
        recession = -.016 * TIME_INTERVAL

    predM = m_model.predict([[data["Temperature"], rhdiff, growth, recession]])
    pubData = {
        "Time": data["Time"],
        "Room": data["Room"],
        "Temperature": data["Temperature"],
        "Humidity": data["Humidity"],
        "Moisture": data["Moisture"],
        "Prediction": predM[0]
    }
    mqttClient.publish(pubTopic, json.dumps(pubData), 0)


# initialize connection
mqttClient = awsmqtt.AWSIoTMQTTClient(client)
mqttClient.configureEndpoint(endpoint, 8883)
mqttClient.configureCredentials(rootCAFile, privCertFile, certFile)

# read in models
with open("m_model.pkl", 'rb') as m_model_file:
    m_model = pickle.load(m_model_file)

mqttClient.connect()
mqttClient.subscribe(subTopic, 0, on_message)

# while loop
while True:
    time.sleep(1)
