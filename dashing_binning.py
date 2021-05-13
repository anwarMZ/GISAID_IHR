#!/usr/bin/env python

import argparse as ap
import pandas as pd
import subprocess
import os
import tempfile
import statistics
import logging
import numpy as np

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Runs MASH and computes '
                                           'pairwise distance and '
                                           'distance against the '
                                           'reference')

    parser.add_argument('-ll', '--loglevel', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR',
                                 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument('-t', '--time', type=int, default=7,
                        help='Number of days within a bin (default: 7)')
    parser.add_argument('-w', '--window', type=int, default=1,
                        help='Number of days between bins (default: 1)')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='Remove all temporary files after '
                             'execution')
    # parser.add_argument('-k', '--kmer', type=int, default=17,
    #                     help='kmer size for MASH (default: 17)')
    parser.add_argument('-p', '--threads', type=int, default=1,
                        help='Number of threads to use for Dashing ('
                             'default: 1)')
    parser.add_argument('--prefix', type=str, default='out',
                        help='Output file prefix (default: out)')
    parser.add_argument('meta', metavar='METADATA', help='GISAID '
                                                         'metadata')
    parser.add_argument('queries', metavar='TXT', help='Text file with paths '
                                                       'to fasta files')
    parser.add_argument('ref', metavar='TXT', help='Text file with path ' 
                                                   'to the reference genome')
    parser.add_argument('outdir', metavar='DIR',
                        help='Output directory')

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel, format='%(asctime)s (%('
                                                    'relativeCreated'
                                                    ')d ms) -> %('
                                                    'levelname)s: %('
                                                    'message)s',
                        datefmt='%I:%M:%S %p')

    logger = logging.getLogger()

    # Check if Dashing is installed
    mash_rc = subprocess.call(['dashing'], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    if mash_rc == 127:
        raise FileNotFoundError("Unable to locate \'Dashing\'")

    # Check if output directory exists
    tmpdir = os.path.join(args.outdir, 'tmp_dashing')
    tmpdir_obj = None
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    if args.clean:
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
    elif not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    # Read and preprocess metadata file
    logger.info('Reading and preprocessing metadata file')
    metadata = pd.read_csv(args.meta, sep='\t', header=0)
    metadata['date'] = pd.to_datetime(metadata['date'])
    metadata = metadata.sort_values('date')
    metadata = metadata.reset_index()
    metadata.strain = metadata.strain.str.replace('/', '_')

    paths_df = pd.read_csv(args.queries, header=None)
    paths_df.columns = ['paths']

    start_date = metadata['date'][0]
    end_date = start_date + pd.DateOffset(days=args.time)

    out_dates = []
    out_ref_dist_mean = []
    out_ref_dist_median = []
    out_ref_min = []
    out_ref_max = []
    out_pair_dist_mean = []
    out_pair_min = []
    out_pair_max = []
    out_pair_dist_median = []
    out_pair_dist_Q1 = []
    out_pair_dist_Q3 = []
    bin_size = []

    warning_bins = {
        '0': [],
        '1': []
    }

    prefix = os.path.join(tmpdir, args.prefix)

    if not os.path.exists('{0}.dist'.format(prefix)):
        # Run Dashing against reference
        logger.info('Run Dashing to obtain distance against reference')
        subprocess.run(['dashing dist --containment-index -O {0}.dist -Q {1} '
                        '--full-mash-dist -F {2} -p {3}'.format(prefix, args.queries,
                                                                args.ref, args.threads)], shell=True)

    logger.info('Loading distance matrix against reference')
    ref_dist_mat = pd.read_csv('{0}.dist'.format(prefix), sep='\t',
                               index_col=None, header=0)
    ref_dist_mat.columns = ['strain', 'distance']

    logger.info(
        'Going through each time bin to run pairwise Dashing and '
        'aggregate results')
    while end_date <= max(metadata['date']):
        sub_meta = metadata.query(
            'date >= @start_date and date <= @end_date')
        ids = sub_meta['strain']

        pattern = '|'.join(r"\b{}\b".format(x) for x in ids)

        if len(ids) == 0:
            out_ref_dist_mean.append(np.nan)
            out_ref_dist_median.append(np.nan)
            out_ref_max.append(np.nan)
            out_ref_min.append(np.nan)

            #out_dates.append(
            #    '{0} - {1}'.format(start_date.strftime('%Y-%m-%d'),
            #                       end_date.strftime('%Y-%m-%d')))
            out_dates.append(format(end_date.strftime('%Y-%m-%d')))
            out_pair_dist_mean.append(np.nan)
            out_pair_dist_median.append(np.nan)
            out_pair_dist_Q1.append(np.nan)
            out_pair_dist_Q3.append(np.nan)
            out_pair_min.append(np.nan)
            out_pair_max.append(np.nan)
            bin_size.append(0)

            #warning_bins['0'].append(
            #    '{0} - {1}'.format(start_date.strftime('%Y-%m-%d'),
            #                       end_date.strftime('%Y-%m-%d')))
            warning_bins['0'].append(format(end_date.strftime(
                '%Y-%m-%d')))

        elif len(ids) == 1:
            out_ref_dist_mean.append(0)
            out_ref_dist_median.append(0)
            out_ref_max.append(0)
            out_ref_min.append(0)

            # out_dates.append(
            #    '{0} - {1}'.format(start_date.strftime('%Y-%m-%d'),
            #                       end_date.strftime('%Y-%m-%d')))
            out_dates.append(format(end_date.strftime('%Y-%m-%d')))
            out_pair_dist_mean.append(0)
            out_pair_dist_median.append(0)
            out_pair_dist_Q1.append(0)
            out_pair_dist_Q3.append(0)
            out_pair_min.append(0)
            out_pair_max.append(0)
            bin_size.append(1)

            #warning_bins['1'].append(
            #    '{0} - {1}'.format(start_date.strftime('%Y-%m-%d'),
            #                       end_date.strftime('%Y-%m-%d')))
            warning_bins['0'].append(format(end_date.strftime(
                '%Y-%m-%d')))

        else:
            prefix = '{0}/{1}_{2}'.format(tmpdir, start_date.strftime(
                '%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            paths_df_subset = paths_df[paths_df['paths'].str.contains(pattern)]
            paths_df_subset.to_csv('{0}_paths.txt'.format(prefix), index=False, header=False)

            if not os.path.exists('{0}.dist'.format(prefix)):
                # Run Dashing distance calculation
                subprocess.run(['dashing dist -p {0} --full-mash-dist -O {1}.dist '
                                '-F {2}_paths.txt'.format(args.threads, prefix,
                                                          prefix)], shell=True,
                            stderr=subprocess.DEVNULL)

            # Load distance matrix
            pair_dist_mat = pd.read_csv('{0}.dist'.format(prefix),
                                        sep='\t', index_col=0, header=0)
            pair_distances = []

            pair_dist_mat = pair_dist_mat.apply(pd.to_numeric, errors='coerce')
            pair_distances = pair_dist_mat.to_numpy().flatten()
            pair_distances = pair_distances[~np.isnan(pair_distances)].tolist()
            
            logger.info("Current bin has {0} isolates, {1} pairwise "
                        "comparisons".format(len(ids),
                                             len(pair_distances)))

            sub_ref_dist = ref_dist_mat[
                ref_dist_mat['strain'].isin(ids)]
            out_ref_dist_mean.append(sub_ref_dist['distance'].mean())
            out_ref_dist_median.append(sub_ref_dist['distance'].median())
            out_ref_max.append(sub_ref_dist['distance'].max())
            out_ref_min.append(sub_ref_dist['distance'].min())

            bin_count = len(ids)
            # out_dates.append(
            #    '{0} - {1}'.format(start_date.strftime('%Y-%m-%d'),
            #                       end_date.strftime('%Y-%m-%d')))
            out_dates.append(format(end_date.strftime('%Y-%m-%d')))
            out_pair_dist_mean.append(statistics.mean(pair_distances))
            out_pair_min.append(min(pair_distances))
            out_pair_max.append(max(pair_distances))
            out_pair_dist_median.append(np.percentile(pair_distances,
                                                      50))
            out_pair_dist_Q1.append(np.percentile(pair_distances,
                                                  25))
            out_pair_dist_Q3.append(np.percentile(pair_distances,
                                                  75))
            bin_size.append(bin_count)

        start_date += pd.DateOffset(days=args.window)
        end_date += pd.DateOffset(days=args.window)

    if len(warning_bins['0']) > 0:
        logger.warning('The following time bins have 0 sequences:')
        for i in warning_bins['0']:
            print(i)

    if len(warning_bins['1']) > 0:
        logger.warning('The following time bins have only 1 sequence:')
        for i in warning_bins['1']:
            print(i)

    out_df = pd.DataFrame({'date': out_dates,
                           'average_ref_distance': out_ref_dist_mean,
                           'min_ref_distance': out_ref_min,
                           'max_ref_distance': out_ref_max,
                           'median_ref_distance': out_ref_dist_median,
                           'average_pair_distance': out_pair_dist_mean,
                           'min_pair_distance': out_pair_min,
                           'max_pair_distance': out_pair_max,
                           'median_pair_distance': out_pair_dist_median,
                           'Q1_pair_distance': out_pair_dist_Q1,
                           'Q3_pair_distance': out_pair_dist_Q3,
                           'sample_size': bin_size})
    out_df.to_csv(
        os.path.join(args.outdir, '{0}.tsv'.format(args.prefix)),
        sep='\t', index=False)

    if args.clean and tmpdir_obj is not None:
        tmpdir_obj.cleanup()
