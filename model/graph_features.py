"""
Graph-based fraud detection features using NetworkX.

Builds a bipartite card ↔ device graph from transaction data.
High-degree device nodes signal card testing rings.
High-degree card nodes signal account takeover.

These features feed into XGBoost alongside tabular features,
demonstrating AUPRC improvement from graph signals.
"""

import logging

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_fraud_graph(
    df: pd.DataFrame,
    card_col: str = "card1",
    device_col: str = "DeviceInfo",
    amount_col: str = "TransactionAmt",
    fraud_col: str = "isFraud",
) -> nx.Graph:
    """
    Build bipartite graph: card nodes <-> device nodes.
    Edge = a transaction occurred between card and device.
    High-degree device nodes = card testing rings.

    Args:
        df: Transaction DataFrame (must have card_col, device_col, fraud_col)
        card_col: Column for card identifier
        device_col: Column for device identifier
        amount_col: Column for transaction amount
        fraud_col: Column for fraud label

    Returns:
        NetworkX undirected graph with card and device nodes
    """
    G = nx.Graph()

    # Filter rows that have both card and device info
    mask = df[card_col].notna() & df[device_col].notna()
    subset = df.loc[mask, [card_col, device_col, amount_col, fraud_col]]

    logger.info(
        "Building graph from %d transactions (%d have both card + device)",
        len(df),
        len(subset),
    )

    for _, row in subset.iterrows():
        card_node = f"card_{int(row[card_col])}"
        device_node = f"device_{row[device_col]}"

        if card_node not in G:
            G.add_node(card_node, node_type="card")
        if device_node not in G:
            G.add_node(device_node, node_type="device")

        # Store fraud label on edge
        G.add_edge(
            card_node,
            device_node,
            amount=float(row[amount_col]),
            is_fraud=int(row[fraud_col]),
        )

    card_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "card")
    device_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "device")
    logger.info("Graph: %d card nodes, %d device nodes, %d edges", card_nodes, device_nodes, G.number_of_edges())

    return G


def extract_graph_features(
    df: pd.DataFrame,
    G: nx.Graph,
    card_col: str = "card1",
    device_col: str = "DeviceInfo",
) -> pd.DataFrame:
    """
    For each transaction, compute graph-based risk signals.
    Returns a DataFrame aligned with the input index.

    Features:
        device_degree: How many cards share this device (ring signal)
        card_degree: How many devices this card has been used on (takeover signal)
        device_fraud_rate: % of device's connected cards that are fraudulent
        is_shared_device: Flag if device_degree > 3
        is_card_ring_member: Flag if device_degree > 10
    """
    device_degrees = []
    card_degrees = []
    device_fraud_rates = []

    for _, row in df.iterrows():
        card_node = f"card_{int(row[card_col])}" if pd.notna(row[card_col]) else None
        device_raw = row[device_col]
        device_node = f"device_{device_raw}" if pd.notna(device_raw) else None

        # Device degree — how many cards share this device
        if device_node and device_node in G:
            d_degree = G.degree(device_node)
        else:
            d_degree = 0
        device_degrees.append(d_degree)

        # Card degree — how many devices this card has used
        if card_node and card_node in G:
            c_degree = G.degree(card_node)
        else:
            c_degree = 0
        card_degrees.append(c_degree)

        # Device fraud rate — % of neighbors that are fraudulent
        if device_node and device_node in G and G.degree(device_node) > 0:
            neighbors = list(G.neighbors(device_node))
            fraud_edges = sum(
                1
                for n in neighbors
                if G[device_node][n].get("is_fraud", 0) == 1
            )
            fraud_rate = fraud_edges / len(neighbors)
        else:
            fraud_rate = 0.0
        device_fraud_rates.append(fraud_rate)

    result = pd.DataFrame(
        {
            "device_degree": device_degrees,
            "card_degree": card_degrees,
            "device_fraud_rate": device_fraud_rates,
            "is_shared_device": [1 if d > 3 else 0 for d in device_degrees],
            "is_card_ring_member": [1 if d > 10 else 0 for d in device_degrees],
        },
        index=df.index,
    )

    return result


GRAPH_FEATURE_COLS = [
    "device_degree",
    "card_degree",
    "device_fraud_rate",
    "is_shared_device",
    "is_card_ring_member",
]


def add_graph_features(
    df: pd.DataFrame,
    sample_size: int | None = 50_000,
    card_col: str = "card1",
    device_col: str = "DeviceInfo",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Convenience function: build graph from full dataset, extract features.

    For large datasets, builds graph from full data but can extract features
    from a sample for speed. Set sample_size=None for full extraction.

    Returns:
        (df_with_features, graph_feature_col_names)
    """
    print("\nBuilding card-device fraud graph...")

    # Build graph from ALL data (captures full network structure)
    G = build_fraud_graph(df, card_col=card_col, device_col=device_col)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Extract features (can be slow for >100K rows due to iterrows)
    if sample_size and len(df) > sample_size:
        print(f"Extracting graph features for sample of {sample_size:,} rows...")
        print("(Set sample_size=None for full extraction — takes ~10 min for 590K rows)")

        # Stratified sample to preserve fraud rate
        fraud_df = df[df["isFraud"] == 1]
        legit_df = df[df["isFraud"] == 0]
        fraud_n = min(len(fraud_df), int(sample_size * df["isFraud"].mean()))
        legit_n = sample_size - fraud_n

        sample_df = pd.concat([
            fraud_df.sample(n=fraud_n, random_state=42),
            legit_df.sample(n=legit_n, random_state=42),
        ]).sort_index()

        graph_feats = extract_graph_features(sample_df, G, card_col=card_col, device_col=device_col)

        # For rows not in sample, fill with 0
        full_graph_feats = pd.DataFrame(0, index=df.index, columns=GRAPH_FEATURE_COLS)
        full_graph_feats.loc[graph_feats.index] = graph_feats
    else:
        print(f"Extracting graph features for all {len(df):,} rows...")
        full_graph_feats = extract_graph_features(df, G, card_col=card_col, device_col=device_col)

    df_out = pd.concat([df, full_graph_feats], axis=1)
    print(f"Added {len(GRAPH_FEATURE_COLS)} graph features")

    return df_out, GRAPH_FEATURE_COLS


if __name__ == "__main__":
    from data.loaders.ieee_cis_loader import prepare_ieee_cis

    df, feature_cols = prepare_ieee_cis()
    df, graph_cols = add_graph_features(df, sample_size=10_000)

    print("\nGraph feature summary:")
    for col in graph_cols:
        print(f"  {col}: mean={df[col].mean():.4f}, max={df[col].max()}")
