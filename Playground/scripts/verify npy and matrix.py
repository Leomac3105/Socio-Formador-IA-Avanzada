import numpy as np, glob, os
p = '/home/sformador/equipo2/Socio-Formador-IA-Avanzada/Playground/data/graph/with_objects'
files = sorted(glob.glob(os.path.join(p,'*_X.npy')))
if not files:
    print('NO_X_FILES')
    raise SystemExit(1)
fn = files[0]
base = fn[:-6]
A0 = base + '_A0.npy'
A_intra = base + '_A_intra.npy'
A_inter = base + '_A_inter.npy'
print('Sample X file:', fn)
X = np.load(fn)
print('X.shape', X.shape, 'dtype', X.dtype)
print('X min/mean/max:', float(X.min()), float(X.mean()), float(X.max()))
# compute per-channel stats
for i in range(X.shape[0]):
    ch = X[i]
    print(f' ch{i} mean {ch.mean():.6f} min {ch.min():.6f} max {ch.max():.6f}')
for path in (A0, A_intra, A_inter):
    if os.path.exists(path):
        A = np.load(path)
        print(os.path.basename(path), 'shape', A.shape, 'nnz', int((A!=0).sum()), 'dtype', A.dtype)
    else:
        print('Missing', path)
# Also report a few entries
print('\nA0 sample (first 8x8):')
A = np.load(A0)
print(A[:8,:8])
# show mapping: V' and original M from filename? try to infer M from X shape last dim
if X.ndim==4:
    # expecting (C,T,Vp,M)
    C,T,Vp,M = X.shape
    print('\nInterpreted dims: C,T,Vp,M =', (C,T,Vp,M))
else:
    print('Unexpected X ndim:', X.ndim)
# quick check: whether joint channels are in [-1,1] or centered around 0
# compute mean of joint channels (0..1 mapping?)
jx = X[0]
jy = X[1]
print('\njoint x stats overall mean', jx.mean(), 'std', jx.std())
print('joint y stats overall mean', jy.mean(), 'std', jy.std())