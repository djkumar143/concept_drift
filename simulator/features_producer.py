import time
import json
import pandas as pd
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient

TOPIC = "market-features"
while True:
    admin = KafkaAdminClient(
        bootstrap_servers="kafka1:29092"
    )

    if TOPIC in admin.list_topics():
        admin.close()
        break

    admin.close()
    print("Waiting for topic...")
    time.sleep(2)
    

producer = KafkaProducer(
    bootstrap_servers=['kafka1:29092', 'kafka1:29092', 'kafka1:29092'],
    value_serializer = lambda v: json.dumps(v).encode('utf-8')
)
print("Producer is running.")

data_path = "/app/data/elec2_real_time_features.csv"
df = pd.read_csv(data_path)
df["eventID"] = df["eventID"].astype(str)

try:
    for _, row in df.iterrows():
        producer.send("market-features", value=row.to_dict())
        print(f"Sent:{row.to_dict()}")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Producer stopped.")
finally:
    print("Flushing remaining messages in buffer.")
    producer.flush()
    producer.close()