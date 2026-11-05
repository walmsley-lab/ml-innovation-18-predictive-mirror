# Roadmap 62 — connectome source contract

The biological program must distinguish **measured neural connectivity** from an illustrative graph. `python -m experiments.roadmap_62_connectome_contract` produces an explicitly synthetic directed ring reference. To ingest external data, supply separate `--matrix` NPZ (`adjacency`, unique `node_ids`) and `--metadata` JSON with source ID, measured/synthetic label, species, citation, license, preparation, version and exact SHA256.

Directed synapse matrices are not silently converted to undirected. A measured label is metadata supplied by the operator, **not independent verification** of acquisition or licensing. No measured connectome data have been bundled with this PR. Later roadmap steps must maintain the source classification, and real neural-response agreement requires independent neural recordings and source verification.
