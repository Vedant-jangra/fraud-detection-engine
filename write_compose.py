from pathlib import Path


content = """services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    container_name: fraud_zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    container_name: fraud_kafka
    depends_on:
      - zookeeper
    ports:
      - 9092:9092
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT_INTERNAL:PLAINTEXT,PLAINTEXT_EXTERNAL:PLAINTEXT
      KAFKA_LISTENERS: PLAINTEXT_INTERNAL://0.0.0.0:29092,PLAINTEXT_EXTERNAL://0.0.0.0:9092
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT_INTERNAL://kafka:29092,PLAINTEXT_EXTERNAL://localhost:9092
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT_INTERNAL
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
      KAFKA_LOG_RETENTION_HOURS: 168

  timescaledb:
    image: timescale/timescaledb:latest-pg16
    container_name: fraud_timescaledb
    ports:
      - 5432:5432
    environment:
      POSTGRES_PASSWORD: fraud_engine_secret
      POSTGRES_DB: fraud_db
      POSTGRES_USER: postgres
    volumes:
      - tsdb_data:/var/lib/postgresql/data
      - ./feature_store/migrations:/docker-entrypoint-initdb.d:ro

  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: fraud_prometheus
    ports:
      - 9090:9090
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:10.3.3
    container_name: fraud_grafana
    ports:
      - 3000:3000
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: 'false'
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

  spark:
    build:
      context: .
      dockerfile: docker/Dockerfile.spark
    container_name: fraud_spark
    depends_on:
      - kafka
      - timescaledb
    environment:
      - KAFKA_BOOTSTRAP=kafka:29092
      - JDBC_URL=jdbc:postgresql://timescaledb:5432/fraud_db
    volumes:
      - spark_checkpoints:/tmp/spark-checkpoints

volumes:
  tsdb_data:
  grafana_data:
  spark_checkpoints:
"""

Path("docker-compose.yml").write_text(content, encoding="utf-8", newline="\n")
print("docker-compose.yml written successfully")
