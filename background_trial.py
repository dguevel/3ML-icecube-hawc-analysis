#!/usr/bin/env python3

import os, sys, glob
import numpy as np
from matplotlib import pyplot as plt, colors
import mla
from mla.threeml import spectral
from mla.threeml.IceCubeLike import IceCubeLike
import astromodels
import warnings
import pandas as pd
import threeML

import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
warnings.simplefilter("always")


from threeML import *
import numpy.lib.recfunctions as rf

def drop_from_first_dataset(df1, df2):
    '''
    Remove from ana1 the overlapping events found in ana2. 
    Credit Leo Seen for this function
    '''
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    idx_to_remove = df1[df1.set_index(['event', 'run']).index.isin(df2.set_index(['event', 'run']).index)].index
    return df1.drop(idx_to_remove).to_records()

def read(filelist):
    data = []

    for f in sorted(filelist):
        x = np.load(f)
        if len(data) == 0: data = x.copy()
        else: data = np.concatenate([data, x])

    try:
        data=rf.append_fields(data, 'sindec',
                              np.sin(data['dec']),
                              usemask=False)
    except:
        pass
    return data

def load_IceCubeLike(data, sim, grl, cascade=False):
    data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
    sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
    #np.random.seed(2) #for reproduce
    data['ra'] = np.random.uniform(0, 2*np.pi, size=len(data))
    livetime = np.sum(grl['livetime'])
    bkg_days = np.sort(grl['stop'])[-1]-np.sort(grl['start'])[0]
    background_time_profile = mla.time_profiles.UniformProfile({'start':np.sort(grl['start'])[0], 'length':bkg_days})
    inject_signal_time_profile = mla.time_profiles.UniformProfile({'start':np.sort(grl['start'])[0], 'length':bkg_days})

    if 'sindec' not in data.dtype.names:
        data = rf.append_fields(
            data,
            'sindec',
            np.sin(data['dec']),
            usemask=False,
        )
    if 'sindec' not in sim.dtype.names:
        sim = rf.append_fields(
            sim,
            'sindec',
            np.sin(sim['dec']),
            usemask=False,
        )

    if extension == 0:
        config = mla.generate_default_config([
            mla.threeml.data_handlers.ThreeMLDataHandler,
            mla.PointSource,
            mla.SpatialTermFactory,
            mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory,
            mla.TimeTermFactory,
            mla.LLHTestStatisticFactory
        ])
        config['PointSource']['name'] = 'temp'
        config['PointSource']['ra'] = np.deg2rad(ra)
        config['PointSource']['dec'] = np.deg2rad(dec)
        if cascade:
            DNN_sin_dec_bin = np.linspace(-1, 1, 11)
            DNN_log_energy_bins = np.linspace(2, 8.01, 15)
            #DNN_sin_dec_bin = np.linspace(-1, 1, 11)
            #DNN_log_energy_bins = np.linspace(2, 8.01, 5)
            ### add the below in load_IceCubeLike for cascade
            config['ThreeMLPSIRFEnergyTermFactory']["list_sin_dec_bins"] = DNN_sin_dec_bin
            config['ThreeMLPSIRFEnergyTermFactory']["list_log_energy_bins"] = DNN_log_energy_bins
            config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(5)
            config['ThreeMLDataHandler']["reco_sampling_width"] = np.deg2rad(90)
        else:
            ESTES_sin_dec_bin = np.linspace(-1, 1, 10+1)
            ESTES_log_energy_bins = np.linspace(3, 7, 18+1)
            config['ThreeMLPSIRFEnergyTermFactory']["list_sin_dec_bins"] = ESTES_sin_dec_bin
            config['ThreeMLPSIRFEnergyTermFactory']["list_log_energy_bins"] = ESTES_log_energy_bins
            config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(2)
        config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)

        source = mla.PointSource(config['PointSource'])

    else:
        config = mla.generate_default_config([
            mla.threeml.data_handlers.ThreeMLDataHandler,
            mla.GaussianExtendedSource,
            mla.SpatialTermFactory,
            mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory,
            mla.TimeTermFactory,
            mla.LLHTestStatisticFactory
        ])
        config['GaussianExtendedSource']['name'] = 'temp'
        config['GaussianExtendedSource']['ra'] = np.deg2rad(ra)
        config['GaussianExtendedSource']['dec'] = np.deg2rad(dec)
        config['GaussianExtendedSource']['sigma'] = np.deg2rad(extension)
        config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(1)
        config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
        source = mla.GaussianExtendedSource(config['GaussianExtendedSource'])
        
    config['ThreeMLPSIRFEnergyTermFactory']['backgroundSOBoption']=energysob
    data_handler = mla.threeml.data_handlers.ThreeMLDataHandler(
        config['ThreeMLDataHandler'], sim, (data, grl))
    data_handler.injection_spectrum = injection_spectrum
    spatial_term_factory = mla.SpatialTermFactory(config['SpatialTermFactory'], data_handler, source)
    energy_term_factory = mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory(config['ThreeMLPSIRFEnergyTermFactory'], data_handler,source)
    time_term_factory = mla.TimeTermFactory(config['TimeTermFactory'],background_time_profile,inject_signal_time_profile)
    llh_factory = mla.LLHTestStatisticFactory(config['LLHTestStatisticFactory'],[spatial_term_factory,energy_term_factory])#,time_term_factory])
    icecube=IceCubeLike("temp",data,data_handler,llh_factory,source,verbose=False,livetime = livetime)
    return icecube

import argparse

p = argparse.ArgumentParser(description="Do background trial for a declination",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--index", default=2.0, type=float,
               help="Spectral Index (default=2.0)")
p.add_argument("--dataset", choices=["estes", "dnn", "nt", "all", "southern"], default="all")
p.add_argument("--estes-datapath", default="/data/ana/analyses/estes_ps/version-001-p03/", type=str,
               help="data path (default:/data/ana/analyses/estes_ps/version-001-p03/)")
p.add_argument("--dnn-datapath", default="/data/ana/analyses/dnn_cascades/version-001-p01/", type=str,
               help="data path (default:/data/ana/analyses/dnn_cascades/version-001-p01/)")
p.add_argument("--nt-datapath", default="/data/ana/analyses/northern_tracks/version-005-p01/", type=str,
               help="data path (default:/data/ana/analyses/northern_tracks/version-005-p01)")
p.add_argument("--wrkdir", default="./result_dir/", type=str,
               help="Output directory (default:./result_dir/)")    
p.add_argument("--nscramble", default=1000, type=int,
               help="Number of background only scrambles used "
               "to measure TS distribution (default=1000)")
p.add_argument("--dec", type=float,
               help="Dec")
p.add_argument("--extension", default=0, type=float,
               help="extension(degree)")
p.add_argument("--surfix",default="", type=str,
               help="surfix")
p.add_argument("--energysob", default=0, type=int,
               help="energysob")
p.add_argument("--overwrite", action="store_true",
               help="Overwrite existing files")
args = p.parse_args()               
index = args.index ###gamma
wrkdir = args.wrkdir
nscramble = args.nscramble
dec=args.dec
surfix=args.surfix
extension=args.extension
energysob = args.energysob
estes_data_path = args.estes_datapath
dnn_data_path = args.dnn_datapath
nt_data_path = args.nt_datapath
ra = 0
if extension == 0:
    surfixextension = ""
else:
    surfixextension = "_"+str(extension)


#injection_spectrum = spectral.PowerLaw(1e3,1,-index)
injection_spectrum = spectral.PowerLaw(1e3,1,-index)

# Remove overlapping events
ESTES_DATA_PATH = estes_data_path
estes_data_files = ESTES_DATA_PATH + "/IC86_*exp.npy"
listofdata = []
estes_data_all = read([i for i in glob.glob(estes_data_files)])

DNN_DATA_PATH = dnn_data_path
dnn_data_files = DNN_DATA_PATH + "/IC86_*exp.npy"
listofdata = []
dnn_data_all = read([i for i in glob.glob(dnn_data_files)])

NT_DATA_PATH = nt_data_path
nt_data_files = NT_DATA_PATH + "/IC86_*exp.npy"
listofdata = []
nt_data_all = read([i for i in glob.glob(nt_data_files)])

# credit leo seen for overlap code
########################### ESTES v DNN -> Remove From ESTES
estes_data = drop_from_first_dataset(estes_data_all, dnn_data_all)
########################### DNN v NT -> Remove From DNN
dnn_data = drop_from_first_dataset(dnn_data_all, nt_data_all)
########################### NT v ESTES -> Remove From NT
nt_data = drop_from_first_dataset(nt_data_all, estes_data_all)
# save some memory by deleting the full data arrays
del estes_data_all, dnn_data_all, nt_data_all
#estes_data = estes_data_all
#dnn_data = dnn_data_all


# Load ESTES data
sim_files = ESTES_DATA_PATH + "/MC_All_Combined.npy"
sim = np.load(sim_files)
sim=rf.append_fields(sim, 'sindec',
                     np.sin(sim['dec']),
                     usemask=False)
grlfile = ESTES_DATA_PATH + "/GRL/IC86_*_exp.npy"
grl = read([i for i in glob.glob(grlfile)])
icecube_estes = load_IceCubeLike(estes_data, sim, grl)

# Load DNN Cascades data
sim_files = DNN_DATA_PATH + "/MC_NuGen_bfrv1_2153x.npy"
sim = np.load(sim_files)
sim=rf.append_fields(sim, 'sindec',
                     np.sin(sim['dec']),
                     usemask=False)
grlfile = DNN_DATA_PATH + "/GRL/IC86_*_exp.npy"
grl = read([i for i in glob.glob(grlfile)])
icecube_dnn_cascade = load_IceCubeLike(dnn_data, sim, grl, cascade=True)

# Load NT data
sim_files = NT_DATA_PATH + "/IC86_pass2_MC.npy"
sim = np.load(sim_files)
sim=rf.append_fields(sim, 'sindec',
                    np.sin(sim['dec']),
                    usemask=False)
grlfile = NT_DATA_PATH + "/GRL/IC86_*_exp.npy"
grl = read([i for i in glob.glob(grlfile)])
icecube_nt = load_IceCubeLike(nt_data, sim, grl)

warnings.filterwarnings("ignore")
result=np.empty((nscramble,4))
if args.dataset == "estes":
    analysislist = mla.threeml.IceCubeLike.icecube_analysis([icecube_estes])
elif args.dataset == "dnn":
    analysislist = mla.threeml.IceCubeLike.icecube_analysis([icecube_dnn_cascade])
elif args.dataset == "nt":
    analysislist = mla.threeml.IceCubeLike.icecube_analysis([icecube_nt])
elif args.dataset == "all":
    analysislist = mla.threeml.IceCubeLike.icecube_analysis([icecube_estes, icecube_dnn_cascade, icecube_nt])
elif args.dataset == "southern":
    analysislist = mla.threeml.IceCubeLike.icecube_analysis([icecube_estes, icecube_dnn_cascade])
else:
    raise ValueError("Invalid dataset")
fitfailed=0
folder="/BG"+surfixextension
try:
    os.mkdir(wrkdir+folder)
except FileExistsError:
    pass

if os.path.exists(wrkdir+folder+"/BGTrial_dec"+str(dec)+surfixextension+"_" + surfix+".npy"):
    if args.overwrite:
        os.remove(wrkdir+folder+"/BGTrial_dec"+str(dec)+surfixextension+"_" + surfix+".npy")
    else:
        sys.exit("File exists")

#analysislist.newton_flux_norm = True
analysislist.newton_flux_norm = Truncated_gaussian
grid_minimizer = GlobalMinimization("grid")
if analysislist.newton_flux_norm:
    local_minimizer = LocalMinimization("scipy")
    my_grid = {'TXS.spectrum.main.Powerlaw.index': np.linspace(-1.1, -3.99, 5)}
else:
    local_minimizer = LocalMinimization("minuit")
    my_grid = {'TXS.spectrum.main.Powerlaw.index': np.linspace(-2, -3.99, 3)}

grid_minimizer.setup(
    second_minimization=local_minimizer, grid=my_grid)

for ntrial in range(nscramble):
    analysislist.injection(poisson=True)
    TXS_sp = astromodels.Powerlaw(piv=100e3) #In GeV.Setting a pivot energy is important
    #TXS_sp = astromodels.Powerlaw(piv=1e3) #In GeV.Setting a pivot energy is important

    TXS_sp.K.fix = analysislist.newton_flux_norm
    #TXS_sp.K.free = True

    TXS_sp.K.bounds = (1e-50,1e-15)
    TXS_sp.index.bounds = (-4,-1)
    TXS_sp.K = 1e-23
    TXS_sp.index = -2
    if extension == 0:
        TXS =  mla.threeml.IceCubeLike.NeutrinoPointSource("TXS", ra=ra, dec=dec, spectral_shape=TXS_sp)
    else:
        TXS_spatial = astromodels.Gaussian_on_sphere()

        TXS = mla.threeml.IceCubeLike.NeutrinoExtendedSource("TXS",spatial_shape = TXS_spatial, spectral_shape=TXS_sp)
        TXS.spatial_shape.lon0=0
        TXS.spatial_shape.lon0.fix=True
        TXS.spatial_shape.lat0=dec
        TXS.spatial_shape.lat0.fix=True
        TXS.spatial_shape.sigma=extension
        TXS.spatial_shape.sigma.fix=True
    model = astromodels.Model(TXS)

    IceCubedata = threeML.DataList(analysislist)
    jl = threeML.JointLikelihood(model, IceCubedata)
    jl.set_minimizer(grid_minimizer)

    try:
        jl.fit(quiet=True,compute_covariance=False)

        ns = 0
        for objecticecube in analysislist.listoficecubelike:
            ns += objecticecube.get_current_fit_ns()
        if -jl.current_minimum < 0:
            result[ntrial,0] = 0
            result[ntrial,1] = 0
            result[ntrial,2] = 4
            result[ntrial,3] = 0
            fitfailed +=1
        result[ntrial,0] = -jl.current_minimum
        result[ntrial,1] = analysislist.get_current_fit_ns()
        result[ntrial,2] = -jl.likelihood_model.TXS.spectrum.main.Powerlaw.index.value
        #result[ntrial,3] = jl.likelihood_model.TXS.spectrum.main.Powerlaw.K.value
        result[ntrial,3] = analysislist.cal_injection_fluxnorm(result[ntrial,1])
    except:
        try:
            if -jl.current_minimum < 0:
                result[ntrial,0] = 0
                result[ntrial,1] = 0
                result[ntrial,2] = 4
                result[ntrial,3] = 0
                fitfailed +=1
            result[ntrial,0] = -jl.current_minimum
            result[ntrial,1] = analysislist.get_current_fit_ns()
            result[ntrial,2] = -jl.likelihood_model.TXS.spectrum.main.Powerlaw.index.value
            #result[ntrial,3] = jl.likelihood_model.TXS.spectrum.main.Powerlaw.K.value
            result[ntrial,3] = analysislist.cal_injection_fluxnorm(result[ntrial,1])
        except:
            result[ntrial,0] = analysislist.get_log_like()
            result[ntrial,1] = analysislist.get_current_fit_ns()
            result[ntrial,2] = -jl.likelihood_model.TXS.spectrum.main.Powerlaw.index.value
            result[ntrial,3] = analysislist.cal_injection_fluxnorm(result[ntrial,1])
            fitfailed +=1
        print("failed : " + str(fitfailed) +" in " + str(ntrial+1) + "trials")

np.save(wrkdir+folder+"/BGTrial_dec"+str(dec)+surfixextension+"_" + surfix+".npy",result)
