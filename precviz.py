"""
created by: Quincy Cambrel
"""

import os
import cv2
import time
import datetime
import calendar
import numpy as np
import netCDF4 as nc4
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import sunpy.image.resample as sir

from PIL import Image, ImageDraw, ImageFont

def plot_precip(image, img_path, cmap, norm):
    fig = plt.figure(dpi=1500)
    ax = plt.axes(projection=ccrs.PlateCarree())
    img = ax.imshow(image, interpolation='nearest', extent=(-180, 180, -90, 90), cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    ax.add_feature(cf.GSHHSFeature(scale='full', levels=[1, 2, 3, 4]), edgecolor='black', facecolor='none', linewidth=0.25)
    ax.add_feature(cf.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '50m'), edgecolor='black', facecolor='none', linewidth=0.25)
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def create_colorbar(cmap, img_path, title, n_labels, scale_factor):
    bar_width = 750
    bar_height = 25
    canvas_height = bar_height + 50

    canvas = Image.new('RGB', (bar_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    segment_width = bar_width / (len(cmap) - 1)
    for i in range(len(cmap) - 1):
        segment_start = i * segment_width
        segment_end = (i + 1) * segment_width
        draw.rectangle([segment_start, 30, segment_end, bar_height + 30], fill=tuple(cmap[i]))

    label_interval = bar_width / n_labels
    for i in range(n_labels + 1):
        label_pos = i * label_interval
        if i > 0:
            draw.text((label_pos, bar_height + 35), str(i * scale_factor), fill=(0, 0, 0))

    font = ImageFont.truetype('fonts/arial-font/arial.ttf', 20)
    draw.text((10, 0, 300, 30), title, fill=(0, 0, 0), font=font)

    canvas.save(img_path)

def annotate_image(img_path, cbar, report, log):
    canvas = Image.new('RGB', (5760, 3060), (255, 255, 255))
    plot = Image.open(f'{img_path}').resize((5760, 2760))
    colorbar = Image.open(cbar).resize((2880, 295))
    canvas.paste(plot, (0, 0))
    canvas.paste(colorbar, (1440, 2760))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('fonts/arial-font/G_ari_bd.TTF', 72)
    draw.text((10, 2760), report, fill=(0, 0, 0), font=font)
    draw.text((canvas.width - 1250, 2760), log, fill=(0, 0, 0), font=font, align='right')

    canvas.save(img_path)

def animate_frames(frame_dirs, fps, dimensions):
    width, height = dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('precip.mp4', fourcc, fps, (width, height))
    
    for frame_dir in frame_dirs:
        frame = cv2.imread(frame_dir)
        video_writer.write(frame)
    
    video_writer.release()


timer = time.time()

infile = 'datasets/precip.mon.mean.nc'
dataset = nc4.Dataset(infile)
precip = np.array(dataset.variables['precip'])
timescale = np.array(dataset.variables['time'])

cmap = mpl.colors.ListedColormap(
    np.array([
        [255, 255, 255],
        [175, 210, 235],
        [130, 160, 230],
        [90, 105, 220],
        [65, 175, 45],
        [95, 215, 75],
        [140, 240, 130],
        [175, 255, 165],
        [250, 250, 35],
        [250, 215, 30],
        [250, 190, 30],
        [245, 150, 30],
        [240, 110, 30],
        [230, 30, 30],
        [220, 0, 25],
        [200, 0, 25],
        [180, 0, 15],
        [140, 0, 10],
        [170, 50, 60],
        [205, 130, 130],
        [230, 185, 185],
        [245, 225, 225],
        [215, 200, 230],
        [185, 165, 210],
        [155, 125, 185],
        [140, 100, 170],
        [120, 70, 160]
    ]) / 255
)

norm = mpl.colors.Normalize(vmin=precip.min(), vmax=precip.max())

t0 = datetime.datetime(1800, 1, 1, 0, 0, 0)

create_colorbar(
    cmap, 
    'precip_colorbar.png',
    'Average Monthly Rate of Precipitation (mm/day)',
    10,
    5
)

for i in range(precip.shape[0]):
    if not os.path.exists(f'frames/precip-{i+1}.png'):
        hires = sir.resample(precip[i, :, :], (2880, 1441), center=True, method='linear')
        num_days = timescale[i]
        date = t0 + datetime.timedelta(days=num_days)
        month = calendar.month_name[date.month]

        plot_precip(hires, f'frames/precip-{i+1}.png', cmap, norm)

        report = f'Month: {month} {date.year}\n'
        report += f'Max: {hires.max():.4f}\n'
        report += f'Min: {hires.min():.4f}\n'
        report += f'Mean: {hires.mean():.4f}'

        log = f'Monthly Average {str(i+1).zfill(3)}\n'
        log += 'From January 1979\n'
        log += 'Surface Level\n'
        log += '/Datasets/gpcp/precip.mon.mean.nc'

        

        annotate_image(f'frames/precip-{i+1}.png', 'precip_colorbar.png', report, log)

        print(f'frame {i+1} saved')
    else:
        print(f'frame {i+1} already exists')


if not os.path.exists('precip.mp4'):
    frame_dirs = [f'frames/precip-{i+1}.png' for i in range(len(timescale))]
    fps = 5
    width, height = 5760, 3060
    animate_frames(frame_dirs, fps, (width, height))

    print('video saved')
else:
    print('video already exists')


elapsed = time.time() - timer
print(f'{elapsed // 60} minutes {(elapsed % 60):.3f} seconds')


span = timescale[-1] - timescale[0]
acc_precip = precip.sum(axis=0) * span / 1000000
hires = sir.resample(acc_precip, (2880, 1441), center=True, method='linear')

norm = mpl.colors.Normalize(vmin=acc_precip.min(), vmax=acc_precip.max())

plot_precip(hires, 'acc_precip.png', cmap, norm)

num_days = timescale[-1]
date = t0 + datetime.timedelta(days=num_days)
month = calendar.month_name[date.month]

report = f'Month: {month} {date.year}\n'
report += f'Max: {hires.max():.4f}\n'
report += f'Min: {hires.min():.4f}\n'
report += f'Mean: {hires.mean():.4f}'

log = 'Scaled Sum\n'
log += 'From January 1979\n'
log += 'Surface Level\n'
log += '/Datasets/gpcp/precip.mon.mean.nc'

create_colorbar(
    cmap, 
    'acc_precip_colorbar.png',
    'Accumulated Precipitation (km)',
    12,
    10
)

annotate_image('acc_precip.png', 'acc_precip_colorbar.png', report, log)