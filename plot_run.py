## https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775

## Latitude: 1 deg = 110.574 km
## Longitude: 1 deg = 111.320*cos(latitude*np.pi/180.) km

import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

gpx_file = open('backs_fens_2021_04_20.gpx', 'r')

gpx = gpxpy.parse(gpx_file)

xpoints = []
ypoints = []
times = []
time_init = 0
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            ypoints.append(point.latitude)
            xpoints.append(point.longitude)
            if len(times)==0:
                times.append(0)
                time_init = point.time.minute+point.time.second/60.
            else:
                times.append(point.time.minute+point.time.second/60.-time_init)
            #print('Point at ({0},{1}) -> {2}'.format(point.latitude, point.longitude, point.elevation))

xpoints = np.array(xpoints)
ypoints = np.array(ypoints)
xdiffs = np.diff(xpoints)*111.320*np.cos(ypoints[0]*np.pi/180.)
ydiffs = np.diff(ypoints)*110.574
diffs = np.sqrt(xdiffs**2+ydiffs**2)
distances = np.array([0]+list(np.cumsum(diffs)))
speeds = diffs*(1000/60.)/np.diff(np.array(times))
speeds = np.append(speeds, speeds[-1])
#vmin, vmax = speeds.min(), speeds.max()
med = np.median(speeds)
wid = 0.01
vmin, vmax = med-wid, med+wid
c_speeds = np.clip(speeds, vmin, vmax)
c_speeds = (c_speeds-vmin)/(vmax-vmin)
c_speeds = np.array([c_speeds, np.zeros_like(c_speeds), 1-c_speeds]).T

fig, axs = plt.subplots(1, 2)
axs = axs.flatten()
#axs[0].scatter(xpoints, ypoints, c=speeds, s=0.5)
for i in range(len(xpoints)-1):
    x1, y1, c1 = xpoints[i], ypoints[i], c_speeds[i]
    x2, y2 = xpoints[i+1], ypoints[i+1]
    offset = 1e-3*(times[i]>np.mean(times))
    axs[0].plot([x1-offset, x2-offset], [y1-offset, y2-offset], '-', color=c1)
for i in range(0, len(xpoints), len(xpoints)//5-1):
    offset = 1e-3*(times[i]>np.mean(times))
    axs[0].annotate(round(times[i], 2), (xpoints[i]-offset, ypoints[i]-offset))
    axs[0].plot(xpoints[i]-offset, ypoints[i]-offset, 'kx')
axs[1].plot(times, distances, label='Distance [km]')
axs[1].plot(times, speeds, label='Speed [m/s]')
axs[0].set_xlabel('long')
axs[0].set_ylabel('lat')
axs[1].set_xlabel('Time [min]')
#axs[1].set_ylabel('Distance [km]')
#secax_y = axs[1].secondary_yaxis('right')
#secax_y.set_ylabel('Speed [m/s]')
axs[1].legend()
plt.show()
plt.hist(speeds, bins=np.arange(0,6,0.05), density=True)
plt.xlabel('Speed [m/s]')
plt.show()

speeds = pd.DataFrame(speeds)
print('Mean:\t', speeds.mean()[0])
print('')
print('25%:\t', speeds.quantile(q=0.25)[0])
print('Median:\t', speeds.median()[0])
print('75%:\t', speeds.quantile(q=0.75)[0])
print('')
print('Std:\t', speeds.std()[0])

'''
for waypoint in gpx.waypoints:
    print('waypoint {0} -> ({1},{2})'.format(waypoint.name, waypoint.latitude, waypoint.longitude))

for route in gpx.routes:
    print('Route:')
    for point in route.points:
        print('Point at ({0},{1}) -> {2}'.format(point.latitude, point.longitude, point.elevation))

# There are many more utility methods and functions:
# You can manipulate/add/remove tracks, segments, points, waypoints and routes and
# get the GPX XML file from the resulting object:

#print('GPX:', gpx.to_xml())

# Creating a new file:
# --------------------

gpx = gpxpy.gpx.GPX()

# Create first track in our GPX:
gpx_track = gpxpy.gpx.GPXTrack()
gpx.tracks.append(gpx_track)

# Create first segment in our GPX track:
gpx_segment = gpxpy.gpx.GPXTrackSegment()
gpx_track.segments.append(gpx_segment)

# Create points:
gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(2.1234, 5.1234, elevation=1234))
gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(2.1235, 5.1235, elevation=1235))
gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(2.1236, 5.1236, elevation=1236))

# You can add routes and waypoints, too...

#print('Created GPX:', gpx.to_xml())'''
