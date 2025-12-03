from pathlib import Path
import numpy as np
from collections import Counter
p=Path('Playground/data/npy_filled')
files=sorted(list(p.glob('*_win*.npy')))
print('Found npy windows:', len(files))
if len(files)==0:
    raise SystemExit(0)

T_counts=Counter(); K_counts=Counter(); V_counts=Counter()
min_vals=[]; max_vals=[]; nan_counts=[]
allzero_frames_counts=[]; frames_nonzero_counts=[]; persons_with_nonzero=[]
meta_present=0; meta_missing=0; meta_pelvis_counts=[]; meta_torso_stats=[]

for f in files:
    try:
        arr=np.load(f)
    except Exception as e:
        print('ERROR loading', f, e)
        continue
    if arr.ndim==5:
        window=arr[0]
    elif arr.ndim==4:
        window=arr
    else:
        print('unexpected shape', f, arr.shape)
        continue
    T,K,V,_ = window.shape
    T_counts[T]+=1; K_counts[K]+=1; V_counts[V]+=1
    try:
        min_vals.append(float(np.nanmin(window)))
        max_vals.append(float(np.nanmax(window)))
    except Exception:
        min_vals.append(float('nan')); max_vals.append(float('nan'))
    nan_counts.append(int(np.isnan(window).sum()))
    # frames
    allzero=0
    for t in range(T):
        if np.allclose(window[t],0):
            allzero+=1
    allzero_frames_counts.append(allzero)
    frames_nonzero_counts.append(T-allzero)
    p_nonzero=0
    for m in range(K):
        if not np.allclose(window[:,m,:,:],0):
            p_nonzero+=1
    persons_with_nonzero.append(p_nonzero)
    # meta
    meta_path = f.with_suffix('')
    meta_path = meta_path.parent / (f.stem + '_meta.npz')
    if meta_path.exists():
        meta_present+=1
        try:
            mm=np.load(meta_path)
            if 'pelvis' in mm:
                meta_pelvis_counts.append(mm['pelvis'].shape)
            if 'torso' in mm:
                try:
                    meta_torso_stats.append(float(np.median(mm['torso'])))
                except Exception:
                    pass
        except Exception:
            pass
    else:
        meta_missing+=1

# summary
print('\n--- Shape distribution ---')
print('T counts (frames):', dict(T_counts))
print('K counts (persons):', dict(K_counts))
print('V counts (joints):', dict(V_counts))

print('\nCoordinate ranges across windows:')
if len(min_vals):
    finite_mins=[x for x in min_vals if not np.isnan(x)]
    finite_maxs=[x for x in max_vals if not np.isnan(x)]
    print('min overall:', min(finite_mins), 'max overall:', max(finite_maxs))
    print('median min:', float(np.median(finite_mins)), 'median max:', float(np.median(finite_maxs)))

print('\nNaN counts (per window): median', int(np.median(nan_counts)), 'max', int(max(nan_counts)))
print('\nFrames non-empty per window: median', int(np.median(frames_nonzero_counts)), 'min', int(min(frames_nonzero_counts)), 'max', int(max(frames_nonzero_counts)))
print('All-zero frames counts: median', int(np.median(allzero_frames_counts)))
print('\nPersons with any nonzero per window: median', int(np.median(persons_with_nonzero)), 'min', int(min(persons_with_nonzero)), 'max', int(max(persons_with_nonzero)))

print('\nMeta files: present', meta_present, 'missing', meta_missing)
if meta_present>0:
    print('Example pelvis shapes (first 10):', meta_pelvis_counts[:10])
    print('Torso median stats (sample): median of medians', float(np.median(meta_torso_stats)))

# T>=48
count_Tge48 = sum(1 for f in files if ((np.load(f).shape[0] if np.load(f).ndim==4 else np.load(f).shape[1])>=48))
print('\nWindows with T>=48:', count_Tge48, '/', len(files))

# useful windows
count_useful = sum(1 for x in persons_with_nonzero if x>0)
print('Windows with at least one person present:', count_useful, '/', len(files))

print('\nDone')