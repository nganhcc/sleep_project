import mne


def load_psg(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    return raw


def load_hyp(path):
    annots = mne.read_annotations(path)
    return annots


def attach(raw, annots):
    raw.set_annotations(annots)
    return raw
