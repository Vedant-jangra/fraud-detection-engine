from data.generators.transaction_simulator import generate_dataset, generate_transaction


def test_generate_transaction_returns_expected_shape():
    transaction = generate_transaction("user_00001", is_fraud=True)

    assert transaction["user_id"] == "user_00001"
    assert transaction["is_fraud"] == 1
    assert transaction["transaction_id"]
    assert transaction["timestamp"].endswith("Z")
    assert transaction["amount"] > 0
    assert len(transaction["country"]) == 2


def test_generate_dataset_honors_requested_size_and_fraud_count():
    dataset = generate_dataset(n_transactions=100, fraud_rate=0.1)

    assert len(dataset) == 100
    assert dataset["is_fraud"].sum() == 10
    assert set(dataset["is_fraud"].unique()) <= {0, 1}
