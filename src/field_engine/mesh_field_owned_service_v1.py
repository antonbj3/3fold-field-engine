"""Independent full snapshots over the explicitly borrowed shared transport."""
import numpy as np
from mesh_field_shared_service_v1 import MeshFieldService as BorrowedService


class MeshFieldService(BorrowedService):
    def query(self):
        """Return (independent full field, worker ms); copies are inside the call."""
        with super().query() as lease:
            gmin, shape, surface, solid, distance = lease.result
            result = (gmin, shape, np.array(surface, copy=True),
                      np.array(solid, copy=True), np.array(distance, copy=True))
            return result, lease.worker_ms
