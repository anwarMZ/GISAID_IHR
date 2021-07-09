#!/usr/bin/env python

import argparse as ap
import pandas as pd
import subprocess
import os
import tempfile
import statistics
import logging
import numpy as np

from itertools import combinations
from Bio import AlignIO


def hamming_dist(s1, s2):
    if len(s1) != len(s2):
        raise ValueError('Undefined for sequences of unequal length')
    return sum(ch1 != ch2 for ch1,ch2 in zip(s1, s2))


if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Computes pairwise distance '
                                           'from a multiple sequence alignment')

    parser.add_argument('-ll', '--loglevel', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR',
                                 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument('-t', '--time', type=int, default=7,
                        help='Number of days within a bin (default: 7)')
    parser.add_argument('-w', '--window', type=int, default=1,
                        help='Number of days between bins (default: 1)')
    parser.add_argument('--prefix', type=str, default='out',
                        help='Output file prefix (default: out)')
    parser.add_argument('meta', metavar='METADATA', help='GISAID '
                                                         'metadata')
    parser.add_argument('fasta', metavar='FASTA', help='MSA FASTA')
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

    alignment = AlignIO.read(args.fasta, 'fasta')
    
    align_dict = dict()
    for seq in alignment:
        align_dict[seq.id] = seq

    # Check if output directory exists
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # Read and preprocess metadata file
    logger.info('Reading and preprocessing metadata file')
    metadata = pd.read_csv(args.meta, sep='\t', header=0)
    metadata['date'] = pd.to_datetime(metadata['date'])
    metadata = metadata.sort_values('date')
    metadata = metadata.reset_index()
    # metadata.strain = metadata.strain.str.replace('/', '_')

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
            pair_distances = []
            ref_distances = []

            all_pairs_ids = list(combinations(ids, 2))
            for pair in all_pairs_ids:
                pair_distances.append(hamming_dist(align_dict[pair[0]], 
                                                   align_dict[pair[1]]))
            for one_id in ids:
                ref_distances.append(hamming_dist(align_dict['MN908947.3'],
                                                  align_dict[one_id]))
            
            logger.info("Current bin has {0} isolates, {1} pairwise "
                        "comparisons".format(len(ids),
                                             len(pair_distances)))

            pair_distances = pair_distances
            ref_distances = np.array(ref_distances)

            out_ref_dist_mean.append(ref_distances.mean())
            out_ref_dist_median.append(np.median(ref_distances))
            out_ref_max.append(ref_distances.max())
            out_ref_min.append(ref_distances.min())

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