# USPS data set in HDF5 format

To load using PyTables:

```py
import tables

f = tables.open_file('usps.h5', mode='r')
```

Training images, dtype = `float32`, shape = `(7291, 1, 16, 16)`

```train_X = f.root.usps.train_X```

Training classes; dtype = `int32`, shape = `(7291,)`

```train_y = f.root.usps.train_y```


Testing images; dtype = `float32`, shape = `(2007, 1, 16, 16)`

```test_X = f.root.usps.test_X```

Testing classes; dtype = `int32`, shape = `(2007,)`

```test_y = f.root.usps.test_y```

