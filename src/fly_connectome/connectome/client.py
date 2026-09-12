"""Lazy neuPrint access; credentials never enter saved configs or logs."""

import os


def create_client():
    """Connect to an explicit dataset using NEUPRINT_TOKEN from the environment."""
    from neuprint import Client

    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError("Set NEUPRINT_TOKEN from your neuPrint account; see .env.example")
    return Client(
        os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org"),
        dataset=os.environ.get("NEUPRINT_DATASET", "male-cns:v1.0"),
        token=token,
    )
