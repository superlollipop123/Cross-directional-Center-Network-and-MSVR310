import colorsys
import random

# https://www.cnblogs.com/hyhy904/p/10977554.html

# matplotlib.pyplot.cm.Set1
PLT_COLOR_MAP1 = (
    (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0), 
    (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0), 
    (0.30196078431372547, 0.6862745098039216, 0.2901960784313726, 1.0), 
    (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0), 
    (1.0, 0.4980392156862745, 0.0, 1.0), 
    # (1.0, 1.0, 0.2, 1.0), 
    (1.0, 0.8, 0.1, 1.0), 
    (0.6509803921568628, 0.33725490196078434, 0.1568627450980392, 1.0), 
    (0.9686274509803922, 0.5058823529411764, 0.7490196078431373, 1.0)
)

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def ncolors(num):
    assert num > 0
    if num <= 8:
        return PLT_COLOR_MAP1[:num]
    rgb_colors = []
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        # r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append((_r, _g, _b, 1.0))

    return rgb_colors