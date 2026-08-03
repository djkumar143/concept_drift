CREATE TABLE market_features(
    eventID VARCHAR(20) PRIMARY KEY,
    nswprice DOUBLE PRECISION,
    nswdemand DOUBLE PRECISION,
    vicprice DOUBLE PRECISION,
    vicdemand DOUBLE PRECISION,
    transfer DOUBLE PRECISION,
    predicted_label VARCHAR(10),
    true_label VARCHAR(10),
    prediction_time TIMESTAMP,
    label_arrival_time TIMESTAMP,
    model_version VARCHAR(20)
);

CREATE TABLE retraining_status (
    id INT PRIMARY KEY,
    running BOOLEAN NOT NULL,
    started_at TIMESTAMP
);

INSERT INTO retraining_status(
    id,
    running,
    started_at
)
VALUES
(1, FALSE, NULL);