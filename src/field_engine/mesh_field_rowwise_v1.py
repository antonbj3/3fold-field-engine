"""Frozen native geometry/EDT with row-local hit ordering and checked ordered mask."""
from mesh_field_persistent_edt_v1 import PreparedMeshField as ReferenceField
from rt_columns_rowwise_handle_v1 import ColumnsHandle
from column_mask_ordered_v1 import OrderedColumnMask


class _Rows:
    def __init__(self,handle):self.handle=handle
    def query(self,xy):return ColumnsHandle.query(self.handle,xy)
    def close(self):return self.handle.close()


class PreparedMeshField(ReferenceField):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        try:
            if self._rt is not None:self._rt=_Rows(self._rt)
            self._mask=OrderedColumnMask(kwargs['column_library'])
        except Exception:
            super().close()
            raise


if __name__=='__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_rowwise_stage_probe import main
    raise SystemExit(main())
