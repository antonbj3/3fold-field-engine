"""Frozen mesh-stage geometry/sign policy with persistent paired EDT."""
from mesh_field_native_mask_v1 import PreparedMeshField as ReferenceField
from edt_persistent_pair_v1 import PersistentPairEDT


class _SignedDistance:
    def __init__(self,handle):self.handle=handle
    def signed_distance(self,mask,pitch):return self.handle.query(mask,pitch)


class PreparedMeshField(ReferenceField):
    def __init__(self,*args,pair_library,**kwargs):
        self._pair=None
        super().__init__(*args,**kwargs)
        try:
            self._pair=PersistentPairEDT(self.shape,library=pair_library)
            self._edt=_SignedDistance(self._pair)
        except Exception:
            super().close()
            raise

    def close(self):
        if self._pair is not None:self._pair.close()
        super().close()


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_service_persistent_probe import main
    raise SystemExit(main())
