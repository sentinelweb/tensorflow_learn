import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

# TIP use 'with plt.xkcd():' for xkcd type plots

def multi_page(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf', bbox_inches='tight', fig_size=(10, 8))
    pp.close()
    print "wrote %s" % filename

