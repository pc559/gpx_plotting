## https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775

## # TO-DO:
## # Comment properly
## # Tests?
## # Animation would be cute
## # Scale histograms
## # Plot std vs mean for batches?
## # Plot std vs time for minutes

## Latitude: 1 deg = 110.574 km
## Longitude: 1 deg = 111.320*cos(latitude*np.pi/180.) km

#from statsmodels.tsa.seasonal import seasonal_decompose
import gpxpy
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import sys

#fname = sys.argv[1]
#fname = 'backs_fens_2021_04_20.gpx'
fname = 'hill_10km_2_5_2021.gpx'
gpx_file = open(fname, 'r')

gpx = gpxpy.parse(gpx_file)

long_points = []
lat_points = []
## # Time in minutes since start,
## # for each point.
times = []
for track in gpx.tracks:
    for segment in track.segments:
        starting_time = segment.points[0].time
        time_init = starting_time.hour*60+starting_time.minute+starting_time.second/60.

for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            lat_points.append(point.latitude)
            long_points.append(point.longitude)
            times.append(point.time.hour*60+point.time.minute+point.time.second/60.-time_init)
            #print('Point at ({0},{1}) -> {2}'.format(point.latitude, point.longitude, point.elevation))

long_points = np.array(long_points)
lat_points = np.array(lat_points)
times = np.array(times)

xpos = long_points*111.320*np.cos(lat_points*np.pi/180.)
ypos = lat_points*110.574

xdispl = xpos-xpos[0]
ydispl = ypos-ypos[0]

xdiffs = np.diff(xpos)
ydiffs = np.diff(ypos)
diffs = np.sqrt(xdiffs**2+ydiffs**2)
distances = np.array([0]+list(np.cumsum(diffs)))
speeds = diffs*(1000/60.)/np.diff(np.array(times))
speeds = np.append(speeds, speeds[-1])

#vmin, vmax = speeds.min(), speeds.max()
med = np.median(speeds)
width = speeds.max()-med#0.01
vmin, vmax = med-width, med+width
c_speeds = np.clip(speeds, vmin, vmax)
c_speeds = (c_speeds-vmin)/(vmax-vmin)
colour_by_speed = np.array([c_speeds, 0.5-np.abs(c_speeds-0.5), 1-c_speeds, 0.25+1.5*np.abs(c_speeds-0.5)]).T

fig, axs = plt.subplots(1, 2)
axs = axs.flatten()
#axs[0].scatter(long_points, lat_points, c=speeds, s=0.5)
for i in range(len(xdispl)-1):
    x1, y1, c1 = xdispl[i], ydispl[i], colour_by_speed[i]
    x2, y2 = xdispl[i+1], ydispl[i+1]
    offset = 1e-3*(times[i]>np.mean(times))
    axs[0].plot([x1-offset, x2-offset], [y1-offset, y2-offset], '-', color=c1)
for i in range(0, len(xdispl), len(xdispl)//5-1):
    offset = 1e-3*(times[i]>np.mean(times))
    axs[0].annotate(str(int(times[i]))+'min', (xdispl[i]-offset, ydispl[i]-offset))
    axs[0].plot(xdispl[i]-offset, ydispl[i]-offset, 'kx')

axs[1].plot(times, distances, label='Distance [km]')
axs[1].plot(times, speeds, label='Speed [m/s]', alpha=0.5)
smoothed_speeds1 = savgol_filter(speeds, len(times)//10+1, 3)
smoothed_speeds2 = savgol_filter(speeds, len(times)//8+1, 2)
axs[1].plot(times, smoothed_speeds1, label='Smoothing 1 [m/s]')
axs[1].plot(times, smoothed_speeds2, label='Smoothing 2 [m/s]')

'''
speeds_time_df = pd.DataFrame(speeds)
speeds_time_df['Datetime'] = pd.to_datetime(["2021-1-1 "+str(int(t//60))+":"+str(int(round(t%60)))+":"+str(int(round((t*60)%60))) for t in times])#df['date'] + ' ' + df['time'])
speeds_time_df = speeds_time_df.set_index('Datetime')
print(speeds_time_df)
speeds_time_df = speeds_time_df[~speeds_time_df.index.duplicated()]
speeds_time_df = speeds_time_df.asfreq("2S")
result_add = seasonal_decompose(speeds_time_df, model='additive', extrapolate_trend='freq')
'''

axs[0].set_xlabel('x displacement [km]')
axs[0].set_ylabel('y displacement [km]')
axs[1].set_xlabel('Time [min]')
#axs[1].set_ylabel('Distance [km]')
#secax_y = axs[1].secondary_yaxis('right')
#secax_y.set_ylabel('Speed [m/s]')
axs[1].legend()
axs[0].set_aspect('equal')
plt.show()
speed_bin_size = 0.25
counts, bins, patches = plt.hist(speeds, bins=np.arange(0, 6, speed_bin_size), density=True)
plt.xlabel('Speed [m/s]')
scaling = times[-1]/np.sum(counts)
plt.ylabel('(Time [min] at this speed)/'+str(round(scaling, 2)))
plt.show()

N = 5
fig, axs = plt.subplots(N, 1)
d = len(speeds)//N
for i in range(N):
    batch_mean = np.mean(speeds[d*i:d*(i+1)])
    batch_std = np.sqrt(np.sum((speeds[d*i:d*(i+1)]-batch_mean)**2))
    axs[i].hist(speeds[d*i:d*(i+1)], bins=np.arange(0, 6, speed_bin_size), density=True, alpha=1.)
    axs[i].set_ylim([0, 2])
    axs[i].set_ylabel(round(times[d*i], 2))
    axs[i].vlines(batch_mean, 0, 2, 'r')
    print('Mean '+str(i)+':\t', batch_mean, end='\t')
    print('std '+str(i)+':\t', batch_std, end='\n')
plt.xlabel('Speed [m/s]')
plt.show()
speeds_df = pd.DataFrame(speeds)

print('Mean:\t', speeds_df.mean()[0])
print('')
print('25%:\t', speeds_df.quantile(q=0.25)[0])
print('Median:\t', speeds_df.median()[0])
print('75%:\t', speeds_df.quantile(q=0.75)[0])
print('')
print('Std:\t', speeds_df.std()[0])

'''
## # Animation: dots moving around the route in real time, separated by 1min, color graded (by speed?).
fig, ax = plt.subplots(1, 1)

ims = []
#ax.grid('off')
#ax.axis('off')
for i in range(len(long_points)-1):
    x1, y1, c1 = long_points_m[i], lat_points_m[i], colour_by_speed[i]
    x2, y2 = long_points_m[i+1], lat_points_m[i+1]
    ax.plot([x1, x2], [y1, y2], '-', color=c1)
for ii in range(0, len(long_points), 1):
    N_x = 10
    ind_diff = len(long_points_m)//N_x
    temp = []
    for shift in range(N_x):
        jj = (ii+ind_diff*shift)%len(long_points)
        print(jj)
        temp.append( ax.plot(long_points_m[jj], lat_points_m[jj], 'kx') )
    ims.append(temp)
    #ims.append([*temp])

im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=300, blit=True)
# To save this second animation with some metadata, use the following command:
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#im_ani.save('im.mp4', writer=writer)

plt.show()
'''
# To save the animation, use the command: line_ani.save('lines.mp4')
'''
#fig2 = plt.figure()
fig, ax = plt.subplots(1, 1)

ims = []
#ax.grid('off')
#ax.axis('off')
for ii in range(0, len(speeds), 1):#len(speeds)//100):
    print(ii, end='\t', flush=True)
    n, bins, patches = ax.hist(speeds[:ii], bins=np.arange(0, 6, 0.05), density=False, color='k')
    print(len(patches), len(ims), sep='\t', flush=True)
    ims.append([*patches])

im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=300, blit=True)
# To save this second animation with some metadata, use the following command:
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#im_ani.save('im.mp4', writer=writer)

plt.show()'''

