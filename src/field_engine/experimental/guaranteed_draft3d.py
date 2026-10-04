"""Iterative draft fields; exact repair and certificate remain mandatory.

Assembly matches guaranteed_scalar3d.solve_fields. CG/MINRES can warm start on
previous fields, with diagonal/block preconditioners. Numerical convergence is
not a certificate. Failure to converge falls back to the sparse direct solve.
"""
from itertools import combinations
import time
from .guaranteed_scalar3d import (require_mesh,potential_layout,_coefficients,_electrodes,repair_flux)
from .guaranteed_scalar import rat
def solve_fields_warm(mesh, source=1, coefficient=1, electrodes=None, potential_degree=1, previous=None, rtol=1e-10, maxiter=500):
    """SciPy P1 and constrained minimum RT0 energy; solves suggest fields only."""
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve, cg, minres, LinearOperator
    require_mesh(mesh); start = time.perf_counter(); iteration_counts={'primal':0,'dual':0}; fallbacks=[]
    def count(name):
        def callback(x): iteration_counts[name]+=1
        return callback
    dofs,boundary_dofs,edge_nodes,nv=potential_layout(mesh,potential_degree)
    coeff = _coefficients(mesh, coefficient)
    k = np.array(coeff, dtype=float); p = np.array(mesh.points, dtype=float); ts = np.array(mesh.tets)
    coords = p[ts]; V = np.array(mesh.volumes, dtype=float); f = float(rat(source))
    mat = coords[:, 1:]-coords[:, :1]; inv = np.linalg.inv(mat)
    g = np.concatenate([-inv.sum(axis=2)[:, None, :], np.swapaxes(inv, 1, 2)], axis=1)
    # g[t,i,:] is gradient lambda_i; exact integration follows independently.
    pdofs=np.array(dofs);nd=pdofs.shape[1]
    if potential_degree == 1:
        ke = V[:, None, None]*k[:, None, None]*np.einsum('tik,tjk->tij', g, g)
        load=np.repeat(V*f/4,4)
    else:
        # Basis gradients at the four vertices determine affine gradients.
        vertex_g = g[:,:,None,:]*(4*np.eye(4)[None,:,:,None]-1)
        edge_g = np.stack([4*(np.eye(4)[i][None,:,None]*g[:,j,None,:]+
                              np.eye(4)[j][None,:,None]*g[:,i,None,:]) for i,j in combinations(range(4),2)],axis=1)
        ng=np.concatenate([vertex_g,edge_g],axis=1);sg=ng.sum(axis=2)
        ke=V[:,None,None]*k[:,None,None]/20*(np.einsum('tiaj,tbaj->tib',ng,ng)+np.einsum('tij,tbj->tib',sg,sg))
        load=(V[:,None]*f*np.array([-1/20]*4+[1/5]*6)[None,:]).ravel()
    rows=np.repeat(pdofs,nd,axis=1).ravel();cols=np.tile(pdofs,(1,nd)).ravel()
    K=sparse.coo_matrix((ke.ravel(),(rows,cols)),shape=(nv,nv)).tocsr()
    b=np.bincount(pdofs.ravel(),weights=load,minlength=nv);v=np.zeros(nv)
    if electrodes is None:
        fixed = np.array(boundary_dofs, dtype=int)
    else:
        e, nodes = _electrodes(mesh, electrodes)
        if potential_degree == 2:
            for name in nodes:
                nodes[name].update(edge_nodes[tuple(sorted(pair))] for face in e[name] for pair in combinations(mesh.faces[face],2))
        if f:
            raise ValueError('resistance has zero source')
        v[list(nodes['right'])] = 1; fixed = np.array(sorted(nodes['left']|nodes['right']), dtype=int)
    free = np.setdiff1d(np.arange(nv), fixed)
    assembly_primal = time.perf_counter()-start; t0 = time.perf_counter()
    if len(free):
        Kf=K[free][:,free]; bf=(b-K@v)[free]
        pv=None if previous is None or len(previous['potential'])!=nv else np.asarray(previous['potential'])[free]
        diagonal=Kf.diagonal()
        preconditioner=LinearOperator(Kf.shape,matvec=lambda x:x/diagonal)
        proposal,info=cg(Kf,bf,x0=pv,rtol=rtol,atol=0,maxiter=maxiter,M=preconditioner,callback=count('primal'))
        if info!=0 or not np.all(np.isfinite(proposal)):
            proposal=spsolve(Kf,bf); fallbacks.append('primal')
        v[free]=proposal
    primal_s = time.perf_counter()-t0; t0 = time.perf_counter()
    ids = np.array([[fi for fi, _ in row] for row in mesh.tet_faces]); signs = np.array([[s for _, s in row] for row in mesh.tet_faces])
    phi = (coords[:, None, :, :]-coords[:, :, None, :])/(3*V[:, None, None, None])
    sp = phi.sum(axis=2)
    me = V[:, None, None]/(20*k[:, None, None])*(np.einsum('tiaj,tbaj->tib', phi, phi)+np.einsum('tij,tbj->tib', sp, sp))
    me *= signs[:, :, None]*signs[:, None, :]
    M = sparse.coo_matrix((me.ravel(), (np.repeat(ids, 4, axis=1).ravel(), np.tile(ids, (1, 4)).ravel())),
                          shape=(len(mesh.faces), len(mesh.faces))).tocsr()
    B = sparse.coo_matrix((signs.ravel(), (np.repeat(np.arange(len(ts)), 4), ids.ravel())),
                          shape=(len(ts), len(mesh.faces))).tocsr()
    rhs = -f*V
    if electrodes is not None:
        face_free = np.array(sorted(set(range(len(mesh.faces)))-set(e['wall'])), dtype=int)
        constraint = sparse.coo_matrix((np.ones(len(e['right'])), (np.zeros(len(e['right']), dtype=int), np.array(e['right']))),
                                       shape=(1, len(mesh.faces))).tocsr()
        B = sparse.vstack([B, constraint]).tocsr(); rhs = np.r_[rhs, 1.]
    else:
        face_free = np.arange(len(mesh.faces))
    Mf = M[face_free][:, face_free]; Bf = B[:, face_free]
    system = sparse.bmat([[Mf, Bf.T], [Bf, None]], format='csc')
    assembly_dual = time.perf_counter()-t0; t0 = time.perf_counter()
    right=np.r_[np.zeros(len(face_free)),rhs]
    old=None if previous is None or len(previous['dual_solution'])!=len(right) else previous['dual_solution']
    md=Mf.diagonal(); schur=np.asarray(Bf.multiply(Bf).dot(1/md)).ravel()
    diagonal=np.r_[md,np.maximum(schur,np.finfo(float).tiny)]
    preconditioner=LinearOperator(system.shape,matvec=lambda x:x/diagonal)
    solution,info=minres(system,right,x0=old,rtol=rtol,maxiter=maxiter,M=preconditioner,callback=count('dual'))
    if info!=0 or not np.all(np.isfinite(solution)):
        solution=spsolve(system,right); fallbacks.append('dual')
    z=np.zeros(len(mesh.faces));z[face_free]=solution[:len(face_free)]
    dual_s = time.perf_counter()-t0
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(z)):
        raise ValueError('nonfinite proposed solver fields')
    residual = float(np.max(np.abs(B@z-rhs))); t0 = time.perf_counter()
    exact_z = repair_flux(mesh, z, source, electrodes)
    repair_s = time.perf_counter()-t0
    return v, exact_z, {'assembly_s': assembly_primal+assembly_dual, 'primal_s': primal_s,
                        'dual_s': dual_s, 'balance_repair_s': repair_s, 'proposed_balance_max': residual,
                        'repair_max': max(float(abs(rat(a)-b)) for a, b in zip(z, exact_z)),
                        'iterations':iteration_counts,'fallbacks':fallbacks,
                        'state':{'potential':v.copy(),'dual_solution':solution.copy()}}


def solve_primal_warm(mesh, source=1, coefficient=1, electrodes=None, potential_degree=1, previous=None, rtol=1e-10, maxiter=500):
    """SciPy P1 and constrained minimum RT0 energy; solves suggest fields only."""
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve, cg, minres, LinearOperator
    require_mesh(mesh); start = time.perf_counter(); iteration_counts={'primal':0,'dual':0}; fallbacks=[]
    def count(name):
        def callback(x): iteration_counts[name]+=1
        return callback
    dofs,boundary_dofs,edge_nodes,nv=potential_layout(mesh,potential_degree)
    coeff = _coefficients(mesh, coefficient)
    k = np.array(coeff, dtype=float); p = np.array(mesh.points, dtype=float); ts = np.array(mesh.tets)
    coords = p[ts]; V = np.array(mesh.volumes, dtype=float); f = float(rat(source))
    mat = coords[:, 1:]-coords[:, :1]; inv = np.linalg.inv(mat)
    g = np.concatenate([-inv.sum(axis=2)[:, None, :], np.swapaxes(inv, 1, 2)], axis=1)
    # g[t,i,:] is gradient lambda_i; exact integration follows independently.
    pdofs=np.array(dofs);nd=pdofs.shape[1]
    if potential_degree == 1:
        ke = V[:, None, None]*k[:, None, None]*np.einsum('tik,tjk->tij', g, g)
        load=np.repeat(V*f/4,4)
    else:
        # Basis gradients at the four vertices determine affine gradients.
        vertex_g = g[:,:,None,:]*(4*np.eye(4)[None,:,:,None]-1)
        edge_g = np.stack([4*(np.eye(4)[i][None,:,None]*g[:,j,None,:]+
                              np.eye(4)[j][None,:,None]*g[:,i,None,:]) for i,j in combinations(range(4),2)],axis=1)
        ng=np.concatenate([vertex_g,edge_g],axis=1);sg=ng.sum(axis=2)
        ke=V[:,None,None]*k[:,None,None]/20*(np.einsum('tiaj,tbaj->tib',ng,ng)+np.einsum('tij,tbj->tib',sg,sg))
        load=(V[:,None]*f*np.array([-1/20]*4+[1/5]*6)[None,:]).ravel()
    rows=np.repeat(pdofs,nd,axis=1).ravel();cols=np.tile(pdofs,(1,nd)).ravel()
    K=sparse.coo_matrix((ke.ravel(),(rows,cols)),shape=(nv,nv)).tocsr()
    b=np.bincount(pdofs.ravel(),weights=load,minlength=nv);v=np.zeros(nv)
    if electrodes is None:
        fixed = np.array(boundary_dofs, dtype=int)
    else:
        e, nodes = _electrodes(mesh, electrodes)
        if potential_degree == 2:
            for name in nodes:
                nodes[name].update(edge_nodes[tuple(sorted(pair))] for face in e[name] for pair in combinations(mesh.faces[face],2))
        if f:
            raise ValueError('resistance has zero source')
        v[list(nodes['right'])] = 1; fixed = np.array(sorted(nodes['left']|nodes['right']), dtype=int)
    free = np.setdiff1d(np.arange(nv), fixed)
    assembly_primal = time.perf_counter()-start; t0 = time.perf_counter()
    if len(free):
        Kf=K[free][:,free]; bf=(b-K@v)[free]
        pv=None if previous is None or len(previous)!=nv else np.asarray(previous)[free]
        diagonal=Kf.diagonal()
        preconditioner=LinearOperator(Kf.shape,matvec=lambda x:x/diagonal)
        proposal,info=cg(Kf,bf,x0=pv,rtol=rtol,atol=0,maxiter=maxiter,M=preconditioner,callback=count('primal'))
        if info!=0 or not np.all(np.isfinite(proposal)):
            proposal=spsolve(Kf,bf); fallbacks.append('primal')
        v[free]=proposal
    primal_s = time.perf_counter()-t0; t0 = time.perf_counter()
    return v, {'assembly_s':assembly_primal,'primal_s':primal_s,
               'iterations':iteration_counts['primal'],'fallbacks':fallbacks}

def solve_dual_warm(mesh, source=1, coefficient=1, electrodes=None, previous=None, rtol=1e-10, maxiter=500):
    """Suggest ONLY a dual field, then enforce every flux balance exactly."""
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve,minres,LinearOperator
    require_mesh(mesh); start=time.perf_counter(); t0=start
    coeff=_coefficients(mesh,coefficient);k=np.array(coeff,dtype=float)
    p=np.array(mesh.points,dtype=float);ts=np.array(mesh.tets);coords=p[ts]
    V=np.array(mesh.volumes,dtype=float);f=float(rat(source))
    if electrodes is not None:
        e,_=_electrodes(mesh,electrodes)
        if f:raise ValueError('resistance has zero source')
    iteration_counts={'dual':0};fallbacks=[]
    def count(name):
        def callback(x):iteration_counts[name]+=1
        return callback
    ids = np.array([[fi for fi, _ in row] for row in mesh.tet_faces]); signs = np.array([[s for _, s in row] for row in mesh.tet_faces])
    phi = (coords[:, None, :, :]-coords[:, :, None, :])/(3*V[:, None, None, None])
    sp = phi.sum(axis=2)
    me = V[:, None, None]/(20*k[:, None, None])*(np.einsum('tiaj,tbaj->tib', phi, phi)+np.einsum('tij,tbj->tib', sp, sp))
    me *= signs[:, :, None]*signs[:, None, :]
    M = sparse.coo_matrix((me.ravel(), (np.repeat(ids, 4, axis=1).ravel(), np.tile(ids, (1, 4)).ravel())),
                          shape=(len(mesh.faces), len(mesh.faces))).tocsr()
    B = sparse.coo_matrix((signs.ravel(), (np.repeat(np.arange(len(ts)), 4), ids.ravel())),
                          shape=(len(ts), len(mesh.faces))).tocsr()
    rhs = -f*V
    if electrodes is not None:
        face_free = np.array(sorted(set(range(len(mesh.faces)))-set(e['wall'])), dtype=int)
        constraint = sparse.coo_matrix((np.ones(len(e['right'])), (np.zeros(len(e['right']), dtype=int), np.array(e['right']))),
                                       shape=(1, len(mesh.faces))).tocsr()
        B = sparse.vstack([B, constraint]).tocsr(); rhs = np.r_[rhs, 1.]
    else:
        face_free = np.arange(len(mesh.faces))
    Mf = M[face_free][:, face_free]; Bf = B[:, face_free]
    system = sparse.bmat([[Mf, Bf.T], [Bf, None]], format='csc')
    assembly_dual = time.perf_counter()-t0; t0 = time.perf_counter()
    right=np.r_[np.zeros(len(face_free)),rhs]
    old=None if previous is None or len(previous['dual_solution'])!=len(right) else previous['dual_solution']
    md=Mf.diagonal(); schur=np.asarray(Bf.multiply(Bf).dot(1/md)).ravel()
    diagonal=np.r_[md,np.maximum(schur,np.finfo(float).tiny)]
    preconditioner=LinearOperator(system.shape,matvec=lambda x:x/diagonal)
    solution,info=minres(system,right,x0=old,rtol=rtol,maxiter=maxiter,M=preconditioner,callback=count('dual'))
    if info!=0 or not np.all(np.isfinite(solution)):
        solution=spsolve(system,right); fallbacks.append('dual')
    z=np.zeros(len(mesh.faces));z[face_free]=solution[:len(face_free)]
    dual_s = time.perf_counter()-t0
    if not np.all(np.isfinite(z)):
        raise ValueError('nonfinite proposed solver fields')
    residual = float(np.max(np.abs(B@z-rhs))); t0 = time.perf_counter()
    exact_z = repair_flux(mesh, z, source, electrodes)
    repair_s = time.perf_counter()-t0
    return exact_z, {'assembly_s':assembly_dual,'dual_s':dual_s,
                      'balance_repair_s':repair_s,'iterations':iteration_counts['dual'],
                      'fallbacks':fallbacks,'proposed_balance_max':residual,
                      'state':{'dual_solution':solution.copy()}}
