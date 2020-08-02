import numpy as np
from bwmorph_thin import bwmorph_thin
from so import correspond_pixels
from thin import * 
def compute_rec_prec_f1(count_r, sum_r, count_p, sum_p):
	rec = count_r / (sum_r + (sum_r == 0))
	prec = count_p / (sum_p + (sum_p == 0))
	f1_denom = (prec + rec + ((prec+rec) == 0))
	f1 = 2.0 * prec * rec / f1_denom
	return rec, prec, f1
def PR_core(det,gt,thresholds):
	thresholds = np.linspace(1.0 / (thresholds + 1),1.0 - 1.0 / (thresholds + 1), thresholds)
	sum_p = np.zeros(thresholds.shape)
	count_p = np.zeros(thresholds.shape)
	sum_r = np.zeros(thresholds.shape)
	count_r = np.zeros(thresholds.shape)
	rec = np.zeros(thresholds.shape)
	prec = np.zeros(thresholds.shape)
	f1 = np.zeros(thresholds.shape)
	for i_t, thresh in enumerate(thresholds):
		bmap = det>=thresh
		bmap = bmap.astype(np.int)
		acc_prec = np.zeros(bmap.shape, dtype=bool)
		bmap = binary_thin(bmap)
		match1, match2 = correspond_pixels.correspond_pixels(bmap,gt, max_dist=0.01)
		match1 = match1 > 0
		match2 = match2 > 0
		acc_prec = acc_prec | match1
		sum_r[i_t] += gt.sum()
		count_r[i_t] += match2.sum()
		sum_p[i_t] = bmap.sum()
		count_p[i_t] = acc_prec.sum()
		rec[i_t], prec[i_t], f1[i_t] = compute_rec_prec_f1(count_r[i_t], sum_r[i_t], count_p[i_t], sum_p[i_t])

		
	return rec,prec,f1,thresholds
	
