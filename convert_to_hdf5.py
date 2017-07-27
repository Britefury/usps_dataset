import click


@click.command()
@click.argument('output_path', type=str)
@click.argument('train_path', type=str)
@click.argument('test_path', type=str)
def convert(output_path, train_path, test_path):
    import gzip
    import numpy as np
    import tables

    # Function for loading data in original GZIP compressed text format
    def _load_usps_file(path):
        # Each file is a GZIP compressed text file, each line of which consists
        # of:
        # ground truth class (as a float for some reason) followed by 256 values
        # that are the pixel values in the range [-1, 1]
        X = []
        y = []
        # Open file via gzip
        with gzip.open(path, 'r') as f:
            for line in f.readlines():
                sample = line.strip().split()
                y.append(int(float(sample[0])))
                flat_img = [float(val) for val in sample[1:]]
                flat_img = np.array(flat_img, dtype=np.float32)
                X.append(flat_img.reshape((1, 1, 16, 16)))
        y = np.array(y).astype(np.int32)
        X = np.concatenate(X, axis=0).astype(np.float32)
        # Scale from [-1, 1] range to [0, 1]
        return X * 0.5 + 0.5, y

    # Load
    print('Loading training data...')
    train_X, train_y = _load_usps_file(train_path)
    print('Loading test data...')
    test_X, test_y = _load_usps_file(test_path)

    print('Creating output file')
    f_out = tables.open_file(output_path, mode='w')
    g_out = f_out.create_group(f_out.root, 'usps', 'USPS data')
    filters = tables.Filters(complevel=9, complib='blosc')
    train_X_arr = f_out.create_earray(g_out, 'train_X', tables.Float32Atom(), (0, 1, 16, 16), filters=filters)
    train_y_arr = f_out.create_earray(g_out, 'train_y', tables.Int32Atom(), (0,), filters=filters)
    test_X_arr = f_out.create_earray(g_out, 'test_X', tables.Float32Atom(), (0, 1, 16, 16), filters=filters)
    test_y_arr = f_out.create_earray(g_out, 'test_y', tables.Int32Atom(), (0,), filters=filters)

    print('Adding data')
    train_X_arr.append(train_X)
    train_y_arr.append(train_y)
    test_X_arr.append(test_X)
    test_y_arr.append(test_y)

    assert train_X.shape == (7291, 1, 16, 16)
    assert train_y.shape == (7291,)
    assert test_X.shape == (2007, 1, 16, 16)
    assert test_y.shape == (2007,)

    assert train_X_arr.shape == (7291, 1, 16, 16)
    assert train_y_arr.shape == (7291,)
    assert test_X_arr.shape == (2007, 1, 16, 16)
    assert test_y_arr.shape == (2007,)

    f_out.close()
    print('Done.')





if __name__ == '__main__':
    convert()