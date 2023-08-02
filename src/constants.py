
# folders where data is saved
raw_data_folder = '../data'
preprocessed_data_folder = '../data_preprocessed'
models_folder = '../models'
predictions_folder = '../predictions'

# TCP montage (22 channels)
active_electrodes =    ['FP1', 'F7', 'T3', 'T5', 'FP2', 'F8', 'T4', 'T6', 'T3', 'C3', 'CZ', 'C4', 'FP1', 'F3', 'C3', 'P3', 'FP2', 'F4', 'C4', 'P4', 'A1', 'T4'] 
reference_electrodes = ['F7',  'T3', 'T5', 'O1', 'F8',  'T4', 'T6', 'O2', 'C3', 'CZ', 'C4', 'T4', 'F3',  'C3', 'P3', 'O1', 'F4',  'C4', 'P4', 'O2', 'T3', 'A2'] 

# double banana - neonatal (18 channels)
active_electrodes_neonates =    ['FP2', 'F4', 'C4', 'P4', 'FP1', 'F3', 'C3', 'P3', 'FP2', 'F8', 'T4', 'T6', 'FP1', 'F7', 'T3', 'T5', 'FZ', 'CZ'] 
reference_electrodes_neonates = ['F4',  'C4', 'P4', 'O2', 'F3',  'C3', 'P3', 'O1', 'F8',  'T4', 'T6', 'O2', 'F7',  'T3', 'T5', 'O1', 'CZ', 'PZ'] 

# plot
import matplotlib.pyplot as plt
import matplotlib.font_manager

fig_width_pt  = 110 * 2.8346438836889                          
inches_per_pt = 1.0/72.27                            
aspect_ratio  = 0.80                                 
fig_scale     = 1.0                                  
fig_width     = fig_width_pt*inches_per_pt*fig_scale 
fig_height    = fig_width*aspect_ratio               
fig_size      = [fig_width,fig_height]               
fs = 8
rcparams= {                      
 "text.usetex": True,             
 "font.family": "DejaVu Sans",	
 "font.serif": [],                
 "font.sans-serif": [],
 "font.monospace": [],
 "axes.labelsize": fs+2,           
 "axes.linewidth": 0.8,
 "font.size": fs+2,
 "legend.fontsize": fs,            
 "xtick.labelsize": fs,
 "ytick.labelsize": fs,
 "figure.figsize": fig_size,
 "xtick.major.width" : 0.8,
 "ytick.major.width" : 0.8
 }
matplotlib.rcParams.update(rcparams)
formatter = matplotlib.ticker.FuncFormatter(lambda x, p: '{:g}'.format(x))

cmap = plt.cm.get_cmap('tab10')
color1 = cmap(8) #'#99A3A4'
color2 = cmap(6) #'#CACFD2'
color3 = cmap(1) #'#AF7AC5'
color4 = cmap(7) #'#45B39D'
color5 = cmap(0) #'#5DADE2'
color6 = cmap(3) #'#CD6155'
color_white = '#ffffff'
light_color = { color1: '#f0f0a8'
	      , color2: '#e995d0'
	      , color3: '#ffd6b3'
	      , color4: '#a6a6a6'
	      , color5: '#B2CFE2'#'#bdddf4'
              , color6: '#eb9393' }

fig_pad_percentage = 0.05
