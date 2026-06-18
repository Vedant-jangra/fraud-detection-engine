from locust import HttpUser, task, between


class FraudScoringUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task(10)
    def score_transaction(self):
        self.client.post(
            "/v1/score",
            json={
                "transaction_id": "load-test-txn",
                "user_id": "user_00001",
                "timestamp": "2025-01-15T14:30:00Z",
                "amount": 1250.00,
                "merchant_cat": "electronics",
                "country": "IN",
            },
        )

    @task(1)
    def explain_transaction(self):
        self.client.post(
            "/v1/explain",
            json={
                "transaction_id": "load-test-explain",
                "user_id": "user_00001",
                "timestamp": "2025-01-15T14:30:00Z",
                "amount": 45000.00,
                "merchant_cat": "crypto",
                "country": "AE",
            },
        )

    @task(5)
    def health_check(self):
        self.client.get("/health")
