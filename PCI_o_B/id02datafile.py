import h5py, hdf5plugin
import numpy as np
import pandas as pd
import posixpath


class ID02DataFileReader:
    
    def __init__(self, filename):
        self._filename = filename
        self._fd = None
        
        
    def open(self):
        self._fd = h5py.File(self._filename, 'r', locking=False)
        
    def close(self):
        if self._fd is not None:
            self._fd.close()
        self._fd = None
        
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
        
    def __del__(self):
        self.close()
        
    def read_data(self, n=None):
        """
        Read the data from an ID02 NeXuS file
        
        :param n: The n-th dataset. Return all data if None.
        """
        
        default_ds_ = self._get_default_data()
        default_ds = self._fd[default_ds_]
        
        axes_ = [ self._str(a) for a in default_ds.attrs['axes'] ] if 'axes' in default_ds.attrs else None
        signal_ = self._str(default_ds.attrs['signal'])
        signalds = self._get_abs_path(default_ds_, signal_)
        
        axes = None
        
        if axes_ is not None:
            axes = []
            for a in axes_:
                if a == '.':
                    continue
                
                ads = self._get_abs_path(default_ds_, a)
                adata = np.array(self._fd[ads])
                
                axes += [{'name': a, 'data': adata}, ]
            
        if n is None:
            data = np.array(self._fd[signalds])
        else:
            data = np.array(self._fd[signalds][n])
            
        return axes, data
        
    def get_counters(self):
        default_ds_ = self._get_default_data()
        
        mcs_ds_ = '/'.join(default_ds_.split('/')[:-1])+'/MCS/interpreted'
        
        print(mcs_ds_)
        
        if mcs_ds_ not in self._fd:
            return None
        
        ds = pd.DataFrame()
        
        for k in self._fd[mcs_ds_]:
            ds[k] = np.array(self._fd[f"{mcs_ds_}/{k}"])
            
        return ds
    
    def get_headers(self):
        
        default_ds_ = self._get_default_data()
        
        for suff in ('../parameters', '../header','header_array'):
            headerds = self._get_abs_path(default_ds_, suff)
            #print(headerds)
            if headerds in self._fd:
                break
        else:
            return {}
        
        r = {}
        for k in self._fd[headerds]:
            r[k] = self._str(self._fd[f"{headerds}/{k}"][()])
            
        return r
        
        
    def _is_NXdata(self, node):
        return self._str(self._fd[node].attrs['NX_class']) == 'NXdata'
    
    def _get_abs_path(self, root, path):
        if path.startswith('/'):
            apth = path
        else:
            if not root.endswith('/'):
                root += '/'
            apth = root+path
            
        return posixpath.normpath(apth)
        
        
    def _get_default_data(self, node='/'):
        
       # print(node)
        
        if self._is_NXdata(node):
            return node
        else:
            dnode = self._get_abs_path(node, self._str(self._fd[node].attrs['default']))
            return self._get_default_data(dnode)
        
    def _str(self, s):
        if isinstance(s, str):
            return s
        elif isinstance(s, bytes):
            return s.decode()
        else:
            return str(s)
