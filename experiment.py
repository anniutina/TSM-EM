

def read_exper_data(fnames, distance=True):
    data = []
    ds, ws = {}, {}
    for fname in fnames:
        simfile = open(fname)
        for line in simfile.readlines():
            ln = line.split()
            data.append([round(float(ln[0]), 2), int(ln[1]), float(ln[2]),
                        int(ln[3]), float(ln[4]), int(ln[5]), float(ln[6])])
        simfile.close()
        for d in data:
            if d[0] in ds:
                ds[d[0]].append(d[1:])
            else:
                ds[d[0]] = [d[1:]]
    for d in ds:
        dds = 0
        for values in ds[d]:
            if distance:
                dds += values[1] - values[3] - values[5]
            else:
                dds += values[2] # number of conventional vehicles used for deliveries
        ws[d] = dds / len(ds[d])
    return dict(sorted(ws.items()))
