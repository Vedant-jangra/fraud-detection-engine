import numpy as np
import pandas as pd

from model.graph_features import (
    GRAPH_FEATURE_COLS,
    build_fraud_graph,
    extract_graph_features,
)


def _make_transaction_df() -> pd.DataFrame:
    """Small transaction set with a known card testing ring."""
    return pd.DataFrame(
        {
            "card1": [1000, 1001, 1002, 1003, 2000, 2001],
            "DeviceInfo": [
                "ring_device",  # card 1000 on ring device
                "ring_device",  # card 1001 on ring device
                "ring_device",  # card 1002 on ring device
                "ring_device",  # card 1003 on ring device (4 cards, 1 device)
                "clean_device",  # card 2000 on clean device
                "clean_device2",  # card 2001 on separate device
            ],
            "TransactionAmt": [10, 15, 20, 25, 500, 800],
            "isFraud": [1, 1, 0, 1, 0, 0],
        }
    )


class TestBuildFraudGraph:
    def test_creates_bipartite_graph(self):
        df = _make_transaction_df()
        G = build_fraud_graph(df)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_card_and_device_nodes_have_types(self):
        df = _make_transaction_df()
        G = build_fraud_graph(df)
        for node, data in G.nodes(data=True):
            assert data["node_type"] in ("card", "device")

    def test_ring_device_has_high_degree(self):
        df = _make_transaction_df()
        G = build_fraud_graph(df)
        ring_degree = G.degree("device_ring_device")
        clean_degree = G.degree("device_clean_device")
        assert ring_degree == 4  # connected to 4 cards
        assert clean_degree == 1  # connected to 1 card

    def test_handles_missing_values(self):
        df = pd.DataFrame(
            {
                "card1": [1000, np.nan, 1002],
                "DeviceInfo": ["dev1", "dev2", None],
                "TransactionAmt": [10, 20, 30],
                "isFraud": [0, 0, 0],
            }
        )
        G = build_fraud_graph(df)
        # Only the first row has both card and device
        assert G.number_of_edges() == 1


class TestExtractGraphFeatures:
    def test_returns_correct_columns(self):
        df = _make_transaction_df()
        G = build_fraud_graph(df)
        result = extract_graph_features(df, G)
        for col in GRAPH_FEATURE_COLS:
            assert col in result.columns

    def test_ring_device_flagged_as_shared(self):
        df = _make_transaction_df()
        G = build_fraud_graph(df)
        result = extract_graph_features(df, G)
        # First 4 rows use ring_device (degree=4, >3 threshold)
        assert result.iloc[0]["is_shared_device"] == 1
        # Last row uses clean_device2 (degree=1)
        assert result.iloc[5]["is_shared_device"] == 0

    def test_device_degree_matches_graph(self):
        df = _make_transaction_df()
        G = build_fraud_graph(df)
        result = extract_graph_features(df, G)
        # ring_device connects to 4 cards
        assert result.iloc[0]["device_degree"] == 4
        assert result.iloc[4]["device_degree"] == 1

    def test_device_fraud_rate_computed_correctly(self):
        df = _make_transaction_df()
        G = build_fraud_graph(df)
        result = extract_graph_features(df, G)
        # ring_device: 3 fraud edges out of 4 = 0.75
        assert result.iloc[0]["device_fraud_rate"] == 0.75

    def test_output_index_matches_input(self):
        df = _make_transaction_df()
        G = build_fraud_graph(df)
        result = extract_graph_features(df, G)
        assert list(result.index) == list(df.index)


class TestGraphFeatureCols:
    def test_expected_feature_names(self):
        assert GRAPH_FEATURE_COLS == [
            "device_degree",
            "card_degree",
            "device_fraud_rate",
            "is_shared_device",
            "is_card_ring_member",
        ]
