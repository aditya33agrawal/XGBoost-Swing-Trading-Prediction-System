"""Model registry — versioned bundles, prod pointer, champion/challenger gate."""
from src.registry.bundle import (
    save_bundle,
    load_bundle,
    load_prod_bundle,
    set_prod_pointer,
    get_prod_bundle_dir,
    rollback_prod,
    list_bundles,
    prune_old_bundles,
    features_hash,
)
from src.registry.promotion import evaluate_promotion

__all__ = [
    "save_bundle", "load_bundle", "load_prod_bundle", "set_prod_pointer",
    "get_prod_bundle_dir", "rollback_prod", "list_bundles", "prune_old_bundles",
    "features_hash", "evaluate_promotion",
]
