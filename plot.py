import os.path
import numpy as np
import matplotlib.pyplot as plt
def plot():
	opts = {'nthresh':10,'score_path':'scores_txt'}
	gt_dir = '/home/lb/python_work/SYMMAX300/human-gt/test'
	sk_dir = 'DeepFlux_SYMMAX300_optim_ske_eps_8'
	opts['method'] = sk_dir
	pr_path = opts.get('score_path') + '/'+ sk_dir + '_pr.txt'
	score_path = opts.get('score_path') + '/'+ sk_dir + '_score.txt'
	scores_path = opts.get('score_path') + '/'+ sk_dir + '_scores.txt'
	rec = []
	prec = []
	with open(pr_path) as file_object:
		for line in file_object.readlines():
			linestr = line.strip()
			linestrlist = linestr.split(" ")
			linelist = map(float,linestrlist)
			rec.append(linelist[1])
			prec.append(linelist[2])
		plt.plot(rec,prec,linewidth=2)
		plt.title('Precision-Recall of' + sk_dir )
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.show()
				
plot()
