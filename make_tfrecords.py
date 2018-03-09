from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wavfile
import argparse
import timeit
import toml
import sys
import os


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def slice_signal(signal, window_size, stride=0.5):
    """ Return windows of the given signal by sweeping in stride fractions
        of window
    """
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset),
                            range(window_size, n_samples + offset,
                                  offset)):
        if end_i - beg_i < window_size:
            break
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.int32)

def read_and_slice(filename, wav_canvas_size, stride=0.5):
    fm, wav_data = wavfile.read(filename)
    if fm != 16000:
        raise ValueError('Sampling rate is expected to be 16kHz!')
    signals = slice_signal(wav_data, wav_canvas_size, stride)
    return signals


def encoder_proc(wav_file, out_file, wav_canvas_size):
    """ Read and slice the wav and noisy files and write to TFRecords.
        out_file: TFRecordWriter.
    """
    clean_filename, noisy_filename, ref_filename = wav_file
    noisy_filename = wav_file[1]

    wav_signals = read_and_slice(clean_filename, wav_canvas_size)
    noisy_signals = read_and_slice(noisy_filename, wav_canvas_size)
    assert wav_signals.shape == noisy_signals.shape, noisy_signals.shape

    if ref_filename is not None:
        ref_signals = read_and_slice(ref_filename, wav_canvas_size)
        assert wav_signals.shape == ref_signals.shape, ref_signals.shape
        for (wav, noisy, ref) in zip(wav_signals, noisy_signals, ref_signals):
            wav_raw = wav.tostring()
            noisy_raw = noisy.tostring()
            ref_raw = ref.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'wav_raw': _bytes_feature(wav_raw),
                'noisy_raw': _bytes_feature(noisy_raw),
                'ref_raw': _bytes_feature(ref_raw)}))
            out_file.write(example.SerializeToString())
    else:
        for (wav, noisy) in zip(wav_signals, noisy_signals):
            wav_raw = wav.tostring()
            noisy_raw = noisy.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'wav_raw': _bytes_feature(wav_raw),
                'noisy_raw': _bytes_feature(noisy_raw)}))
            out_file.write(example.SerializeToString())

def main(opts):
    if not os.path.exists(opts.save_path):
        # make save path if it does not exist
        os.makedirs(opts.save_path)
    # set up the output filepath
    out_filepath = os.path.join(opts.save_path, opts.out_file)
    if os.path.splitext(out_filepath)[1] != '.tfrecords':
        # if wrong extension or no extension appended, put .tfrecords
        out_filepath += '.tfrecords'
    else:
        out_filename, ext = os.path.splitext(out_filepath)
        out_filepath = out_filename + ext
    # check if out_file exists and if force flag is set
    if os.path.exists(out_filepath) and not opts.force_gen:
        raise ValueError('ERROR: {} already exists. Set force flag (--force-gen) to '
                         'overwrite. Skipping this speaker.'.format(out_filepath))
    elif os.path.exists(out_filepath) and opts.force_gen:
        print('Will overwrite previously existing tfrecords')
        os.unlink(out_filepath)
    with open(opts.cfg) as cfh:
        # read the configuration description
        cfg_desc = toml.loads(cfh.read())
        beg_enc_t = timeit.default_timer()
        out_file = tf.python_io.TFRecordWriter(out_filepath)
        # process the acoustic and textual data now
        assert 'noises' in cfg_desc['segan'], "dBs must be specified"
        assert 'dBs' in cfg_desc['segan'], "dBs must be specified"
        print('-' * 50)
        print("Collecting information")
        clean_dir = cfg_desc['segan']['clean']
        noisy_dir = cfg_desc['segan']['noisy']
        ref_use =  'ref' in cfg_desc['segan']
        if ref_use:
            ref_dir = cfg_desc['segan']['ref']

        wav_files=[]
        for dataname in [wav for wav in os.listdir(clean_dir) if wav.endswith('.wav')]:
            clean_wav_file =  "{}//{}".format(clean_dir, dataname)
            #for noise in cfg_desc['segan']['noises'].split(','):
            #    for dB in cfg_desc['segan']['dBs'].split(','):
                    #noisy_wav_file = "{}//noise{}/{}/{}".format(noisy_dir, noise, dB, dataname)
            noisy_wav_file = "{}/{}".format(noisy_dir, dataname)
            if ref_use:
                ref_wav_file =  "{}/{}".format(ref_dir, dataname)
            else:
                ref_wav_file = None
            if not os.path.exists(noisy_wav_file):
                if opts.verbose:
                    print("Warning: noisy file is not found and skipped: {}".format(noisy_wav_file))
                continue
            if ref_wav_file is not None and not os.path.exists(ref_wav_file):
                if opts.verbose:
                    print("Warning reference file is not found and skipped: {}".format(ref_wav_file))
                continue
            wav_files.append((clean_wav_file, noisy_wav_file, ref_wav_file))
        nfiles = len(wav_files)

        #import pdb; pdb.set_trace()
        for m, wav_file in enumerate(wav_files):
            print('Processing wav file {}/{} {}{}'.format(m + 1,
                                                          nfiles,
                                                          wav_file[0],
                                                          ' ' * 10),
                  end='\r')
            sys.stdout.flush()
            encoder_proc(wav_file, out_file, 2 ** 14)
        out_file.close()
        end_enc_t = timeit.default_timer() - beg_enc_t
        print('')
        print('*' * 50)
        print('Total processing and writing time: {} s'.format(end_enc_t))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the set of txt and '
                                                 'wavs to TFRecords')
    parser.add_argument('--cfg', type=str, default='cfg/e2e_maker.cfg',
                        help='File containing the description of datasets '
                             'to extract the info to make the TFRecords.')
    parser.add_argument('--save_path', type=str, default='data2/',
                        help='Path to save the dataset')
    parser.add_argument('--out_file', type=str, default='segan.tfrecords',
                        help='Output filename')
    parser.add_argument('--force-gen', dest='force_gen', action='store_true',
                        help='Flag to force overwriting existing dataset.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Show warnings when file is not found.')
    parser.set_defaults(force_gen=False)
    opts = parser.parse_args()
    main(opts)
