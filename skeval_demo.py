import os.path
import numpy as np
from PR_core import *
from scipy.io import loadmat
import matplotlib.image as mpimg
from collections import namedtuple
from bwmorph_thin import bwmorph_thin
from so import correspond_pixels
SampleResult = namedtuple('SampleResult', ['idd','threshold','recall', 'precision', 'f1'])
Result = namedtuple('Result', ['thsh','Rec','Pre','F1'])
Score = namedtuple('Result', ['thsh','Rec','Pre','F1'])
sample_results = []
results = []
score = []
def skeval_demo():
	opts = {'nthresh':10,'score_path':'scores_txt'}
	if not os.path.exists(opts.get('score_path')):
		os.makedirs(opts.get('score_path'))
	gt_dir = '/home/lb/python_work/SYMMAX300/human-gt/test'
	sk_dir = 'DeepFlux_SYMMAX300_optim_ske_eps_8'
	opts['method'] = sk_dir
	pr_path = opts.get('score_path') + '/'+ sk_dir + '_pr.txt'
	score_path = opts.get('score_path') + '/'+ sk_dir + '_score.txt'
	scores_path = opts.get('score_path') + '/'+ sk_dir + '_scores.txt'
	if not os.path.exists(pr_path):
		f = open(pr_path,'a')
	if not os.path.exists(score_path):
		f = open(score_path,'a')
	if not os.path.exists(scores_path):
		f = open(scores_path,'a')
	items = os.listdir(gt_dir) 
	for j,i in enumerate(items):
		gt = loadmat(gt_dir + '/'+ i)
		gt = gt.get('gt')
		i = os.path.splitext(i)[0]
		det = mpimg.imread(sk_dir+ '/'+ i +'.png')
		if det.ndim > 2:
			det = np.squeeze(sum(det[1:end, :, :], 1))
		rec,prec,f1,thresholds = PR_core(det,gt,opts.get('nthresh'))
		best_ndx = np.argmax(f1)
		sample_results.append(SampleResult(j+1,thresholds[best_ndx],rec[best_ndx], prec[best_ndx],f1[best_ndx]))
	np.savetxt(scores_path,sample_results,fmt = '%d %f %f %f %f')
	thresholds = np.linspace(1.0 / (opts.get('nthresh') + 1),1.0 - 1.0 / (opts.get('nthresh') + 1), opts.get('nthresh'))
	sumP = np.zeros(len(items))
	cntP = np.zeros(len(items))
	sumR = np.zeros(len(items))
	cntR = np.zeros(len(items))
	threshs = np.zeros(10)
	Rec = np.zeros(thresholds.shape)
	Prec = np.zeros(thresholds.shape)
	F1 = np.zeros(thresholds.shape)
	for t,thresh in enumerate(thresholds):
		for j,i in enumerate(items):
			gt = loadmat(gt_dir + '/'+ i)
			gt = gt.get('gt')
			i = os.path.splitext(i)[0]
			det = mpimg.imread(sk_dir+ '/'+ i +'.png')
			if det.ndim > 2:
				det = np.squeeze(sum(det[1:end, :, :], 1))
			bmap = det>=thresh
			bmap = bmap.astype(np.int)
			acc_prec = np.zeros(bmap.shape, dtype=bool)
			bmap = bwmorph_thin(bmap)
			match1, match2 = correspond_pixels.correspond_pixels(bmap,gt, max_dist=0.01)
			match1 = match1 > 0
			match2 = match2 > 0
			acc_prec = acc_prec | match1
			sumR[j] += gt.sum()
			cntR[j] += match2.sum()
			sumP[j] = bmap.sum()
			cntP[j] = acc_prec.sum()
		sumR_toal = sumR.sum()
		cntR_toal = cntR.sum()
		sumP_toal = sumP.sum()
		cntP_toal = cntP.sum()
		Rec[t], Prec[t], F1[t] = compute_rec_prec_f1(cntR_toal, sumR_toal, cntP_toal, sumP_toal)
		threshs[t] = thresh
		results.append(Result(threshs[t],Rec[t], Prec[t], F1[t]))
		np.savetxt(pr_path,results,fmt = '%f')
	if t==9:
		Best_ndx = np.argmax(F1)
		score.append(Score(threshs[Best_ndx],Rec[Best_ndx], Prec[Best_ndx], F1[Best_ndx]))
		np.savetxt(score_path,score,fmt = '%f')
		
	
	
skeval_demo()
