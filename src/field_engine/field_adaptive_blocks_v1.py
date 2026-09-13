"""Packed adaptive block tree with exact fine-grid narrow-band samples.

The existing classifier determines all subdivisions. Far-field leaves retain its
signed bound, not the dense distance. Dense construction memory is not eliminated.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOOKUP = np.array([[sum(((byte >> (2*i)) & 3) == code for i in range(4))
                    for byte in range(256)] for code in range(4)], dtype=np.int32)


@dataclass
class AdaptiveBlocks:
    shape: tuple
    pitch: float
    root_shape: tuple
    root_count: int
    node_count: int
    packed: np.ndarray
    branch_prefix: np.ndarray
    value_prefix: np.ndarray
    values: np.ndarray

    def kinds(self, nodes):
        return (self.packed[nodes//4] >> (2*(nodes%4))) & 3

    def rank_before(self, nodes, code):
        # Rank checkpoints every64 packed bytes =256 nodes; at most63 byte
        # lookups and three bit-field checks. No full-node expansion is required.
        byte = nodes//4
        group = byte//64
        start = group*64
        prefix = self.branch_prefix if code == 2 else self.value_prefix
        rank = prefix[group].copy()
        length = byte-start
        for offset in range(int(length.max(initial=0))):
            take = length > offset
            rank[take] += LOOKUP[code, self.packed[start[take]+offset]]
        partial = self.packed[byte]
        for slot in range(3):
            rank += ((nodes%4 > slot) & (((partial >> (2*slot)) & 3) == code))
        return rank

    def sample(self, indices):
        ijk = np.asarray(indices, dtype=np.int64)
        if ijk.ndim != 2 or ijk.shape[1] != 3 or np.any(ijk < 0) or np.any(ijk >= self.shape):
            raise ValueError("Expected in-domain integer grid coordinates[N,3]")
        root = ijk//8
        nodes = (root[:,0]*self.root_shape[1]+root[:,1])*self.root_shape[2]+root[:,2]
        size = np.full(len(ijk),8,dtype=np.int64)
        output = np.empty(len(ijk),dtype=np.float32)
        pending = np.arange(len(ijk))
        while len(pending):
            current = nodes[pending]
            kind = self.kinds(current)
            inactive = kind < 2
            ids = pending[inactive]
            output[ids] = np.where(kind[inactive] == 0, 1., -1.)*np.sqrt(3.)*size[ids]*self.pitch
            active = kind == 3
            ids = pending[active]
            output[ids] = self.values[self.rank_before(current[active],3)]
            branch = kind == 2
            ids = pending[branch]
            if not len(ids):
                break
            half = size[ids]//2
            if np.any(half < 1):
                raise RuntimeError("Invalid tree depth")
            octant = (ijk[ids]//half[:,None])%2
            child = 4*octant[:,0]+2*octant[:,1]+octant[:,2]
            nodes[ids] = self.root_count+8*self.rank_before(current[branch],2)+child
            size[ids] = half
            pending = ids
        return output

    def stored_bytes(self):
        # Include a fixed numeric header: shapes, counts, pitch and root size.
        return 80+sum(a.nbytes for a in (self.packed,self.branch_prefix,self.value_prefix,self.values))

    def fingerprint(self):
        h = hashlib.sha256()
        h.update(np.array([*self.shape,*self.root_shape,self.root_count,self.node_count],np.int64).tobytes())
        h.update(np.float64(self.pitch).tobytes())
        for a in (self.packed,self.branch_prefix,self.value_prefix,self.values):
            h.update(a.tobytes())
        return h.hexdigest()


def build(wp, dense, pitch):
    import faltkarna_v1_mesh_to_sdf as reference
    shape = tuple(dense.shape)
    levels = {}
    for side in (8,4,2,1):
        sf = reference.klassificera_och_evaluera_fran_tatt_falt(wp,dense,pitch,side,"cpu")
        levels[side] = sf["kind"].reshape(sf["n_bx"],sf["n_by"],sf["n_bz"])
    root_shape = levels[8].shape
    nodes = [(8*x,8*y,8*z,8) for x,y,z in np.ndindex(root_shape)]
    root_count = len(nodes)
    kinds,values = [],[]
    histogram = {}
    # Breadth-first allocation: the children of the b-th branch always start at
    # root_count+8*b. Node types plus rank therefore replace explicit pointers.
    for x,y,z,side in nodes:
        if x>=shape[0] or y>=shape[1] or z>=shape[2]:
            code = 0
        else:
            kind = int(levels[side][x//side,y//side,z//side])
            if kind:
                code = 0 if kind > 0 else 1
            elif side == 1:
                code = 3
                values.append(dense[x,y,z])
            else:
                code = 2
                half = side//2
                nodes.extend((x+i*half,y+j*half,z+k*half,half) for i,j,k in np.ndindex(2,2,2))
        kinds.append(code)
        if code != 2:
            label = f"side{side}_kind{code}"
            histogram[label] = histogram.get(label,0)+1
    kind = np.asarray(kinds,dtype=np.uint8)
    padded = np.pad(kind,(0,(-len(kind))%4))
    packed = np.bitwise_or.reduce(padded.reshape(-1,4) << np.array([0,2,4,6],np.uint8),axis=1)
    prefixes=[]
    for code in (2,3):
        cumulative=np.concatenate(([0],np.cumsum(LOOKUP[code,packed],dtype=np.int32)))
        prefixes.append(cumulative[np.arange(0,len(packed)+1,64)].astype(np.int32))
    tree=AdaptiveBlocks(shape,float(pitch),root_shape,root_count,len(kind),packed,
                        *prefixes,np.asarray(values,dtype=np.float32))
    return tree,histogram


def run_leg(wp):
    import faltkarna_v1_multires as reference
    with tempfile.TemporaryDirectory() as tmp:
        v,t=reference._bracket_mesh(str(Path(tmp)/"bracket.stl"))
    pitch,_=reference.M2S.valj_pitch_for_feature(v,t,2.0,"auto")
    fine=reference._global_fall(wp,"cpu",v,t,pitch,v.min(axis=0)-12*pitch,"fine")
    dense=fine["sd"]
    tree,histogram=build(wp,dense,pitch)
    coordinates=np.indices(dense.shape,dtype=np.int64).reshape(3,-1).T
    reconstructed=tree.sample(coordinates).reshape(dense.shape)
    narrow=np.abs(dense)<=pitch
    return {"fine_active_voxels":fine["aktiva_voxlar"],"active_voxels":len(tree.values),
            "ratio":len(tree.values)/fine["aktiva_voxlar"],"stored_bytes":tree.stored_bytes(),
            "fine_stored_bytes":sum(fine["sf"][k].nbytes for k in ("kind","active_ids","tiles")),
            "node_count":tree.node_count,"leaf_histogram":histogram,
            "sign_mismatches":int(np.count_nonzero((dense<0)!=(reconstructed<0))),
            "narrow_band_samples":int(narrow.sum()),
            "narrow_band_byte_identity":dense[narrow].tobytes()==reconstructed[narrow].tobytes(),
            "finite":bool(np.isfinite(reconstructed).all()),"tree_sha256":tree.fingerprint(),
            "reconstruction_sha256":hashlib.sha256(reconstructed.tobytes()).hexdigest()}


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import warp as wp
    wp.init()
    legs=[run_leg(wp),run_leg(wp)]
    gates={"two_run_identity":legs[0]==legs[1],
           "half_fine_voxels":all(r["ratio"]<=.5 for r in legs),
           "storage_no_larger":all(r["stored_bytes"]<=r["fine_stored_bytes"] for r in legs),
           "exact_sign":all(r["sign_mismatches"]==0 for r in legs),
           "exact_narrow_band":all(r["narrow_band_byte_identity"] for r in legs),
           "finite":all(r["finite"] for r in legs)}
    result={"legs":legs,"gates":gates,"timing_measured":False}
    dest=ROOT/"artifacts"/"field_adaptive_blocks_v1.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=="__main__":
    raise SystemExit(main())
