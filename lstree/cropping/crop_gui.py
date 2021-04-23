import os
import numpy as np
import pandas as pd
import luigi

import holoviews as hv
import panel as pn
import param
from holoviews import opts, streams
from holoviews.operation.datashader import rasterize
from improc.io import parse_collection, DCAccessor

DCAccessor.register()

from lstree.cropping.candidates import link_crop_candidates

hv.extension('bokeh', width=100)
opts.defaults(
    opts.Layout(sizing_mode='fixed'),
    opts.Image(aspect='equal',
               axiswise=False,
               framewise=False,
               cmap='greys_r',
               xaxis='bare',
               yaxis='bare'),
    opts.Rectangles(aspect='equal',
                    color=None,
                    nonselection_fill_color=None,
                    selection_fill_color=None,
                    cmap=['white', 'orange'],
                    line_width=3,
                    line_dash='dashed',
                    nonselection_line_alpha=1,
                    selection_fill_alpha=0.,
                    active_tools=['tap'],
                    tools=['tap']),
    opts.Rectangles('movie',
                    aspect='equal',
                    line_color='red',
                    line_width=3,
                    line_dash='solid'),
    opts.Curve(default_tools=['xwheel_zoom', 'xpan', 'reset'],
               active_tools=['xwheel_zoom', 'xpan']),
    opts.Overlay(axiswise=False),
)


class CropReviewer(param.Parameterized):
    outdir = param.Parameter()
    spacing = param.Tuple(None, 3)
    margin = param.Integer(10, bounds=(0, 250))
    frame_width = param.Integer(600)

    tp = param.ObjectSelector(default=1, objects=[1])
    x_range = param.Range(default=(0, 1), bounds=(0, 1))
    y_range = param.Range(default=(0, 1), bounds=(0, 1))
    z_range = param.Range(default=(0, 1), bounds=(0, 1))

    object_id = param.ObjectSelector(default=0,
                                     objects=list(range(10)),
                                     label='')

    selection_xy = param.Parameter(streams.Selection1D(), instantiate=True)
    selection_zx = param.Parameter(streams.Selection1D(), instantiate=True)
    selection_zy = param.Parameter(streams.Selection1D(), instantiate=True)

    df_mips = param.DataFrame(None)
    dfb = param.DataFrame(None)

    _relink_counter = param.Integer(0)

    _updating_ranges = param.Boolean(False)
    xy_proj_ratio = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._relink()

        # init slider bounds
        height, width = self.df_mips.dc['xy'].iloc[0:1].dc.read()[0].shape
        depth, _ = self.df_mips.dc['zy'].iloc[0:1].dc.read()[0].shape
        self.param.x_range.bounds = (0, width)
        self.param.y_range.bounds = (0, height)
        self.param.z_range.bounds = (0, depth)

        tps = self.dfb.index.levels[0].tolist()
        self.param.tp.objects = tps
        self.tp = tps[-1]

    @param.depends('margin', watch=True)
    def _update_margin(self):
        self._relink(start_tp=self.param.tp.objects[0])

    def _relink(self, start_tp=None):
        if start_tp is None:
            start_tp = self.tp

        margin = np.broadcast_to(np.asarray(self.margin), 3)
        margin = np.rint(margin / self.spacing).astype(int)
        self.dfb = link_crop_candidates(self.dfb,
                                        self.spacing,
                                        margin,
                                        start_tp=start_tp)
        self._relink_counter += 1

    @param.depends('tp', watch=True)
    def _update_ranges(self):
        self._updating_ranges = True

        filt_dfb = self.dfb[self.dfb.linked == True]
        filt_dfb = filt_dfb.loc[self.tp, :].iloc[0, :]
        self.x_range = tuple(filt_dfb[['x_start', 'x_stop']])
        self.y_range = tuple(filt_dfb[['y_start', 'y_stop']])
        self.z_range = tuple(filt_dfb[['z_start', 'z_stop']])

        self._updating_ranges = False

    @param.depends('x_range', 'y_range', 'z_range', watch=True)
    def _correct_roi(self):

        if not self._updating_ranges:
            self.dfb.loc[(self.tp, self.dfb.linked == True),
                         ['x_start', 'x_stop']] = np.asarray(self.x_range)
            self.dfb.loc[(self.tp, self.dfb.linked == True),
                         ['y_start', 'y_stop']] = np.asarray(self.y_range)
            self.dfb.loc[(self.tp, self.dfb.linked == True),
                         ['z_start', 'z_stop']] = np.asarray(self.z_range)
            self._relink()

    @param.depends('tp')
    def _plot_xy_proj(self):
        img = self.df_mips.dc['xy', self.tp].dc.read()[0][::-1]
        self.xy_proj_ratio = img.shape[0] / img.shape[1]
        bounds = (0, 0, img.shape[0] * self.spacing[2],
                  img.shape[1] * self.spacing[1])
        return hv.Image(img, bounds=bounds,
                        kdims=['y', 'x']).opts(frame_width=self.frame_width)

    @param.depends('tp')
    def _plot_zx_proj(self):
        img = self.df_mips.dc['zx', self.tp].dc.read()[0][::-1]
        bounds = (0, 0, img.shape[1] * self.spacing[2],
                  img.shape[0] * self.spacing[0])

        frame_height = int(np.rint(self.frame_width * self.xy_proj_ratio))

        return hv.Image(img, bounds=bounds,
                        kdims=['x', 'za']).opts(invert_axes=True,
                                                frame_width=None,
                                                frame_height=frame_height)

    @param.depends('tp')
    def _plot_zy_proj(self):
        img = self.df_mips.dc['zy', self.tp].dc.read()[0][::-1]
        bounds = (0, 0, img.shape[1] * self.spacing[1],
                  img.shape[0] * self.spacing[0])
        return hv.Image(img, bounds=bounds,
                        kdims=['y', 'zb']).opts(invert_yaxis=True,
                                                frame_width=self.frame_width)

    @param.depends('selection_xy.index', watch=True)
    def _watch_xy_click(self):
        if len(self.selection_xy.index) > 0:
            # reset linked
            self.dfb.loc[self.tp, 'linked'] = 0
            # set new linked
            self.dfb.loc[(self.tp, self.selection_xy.index[0]), 'linked'] = 1
            self._relink()

    @param.depends('selection_zx.index', watch=True)
    def _watch_zx_click(self):
        if len(self.selection_zx.index) > 0:
            # reset linked
            self.dfb.loc[self.tp, 'linked'] = 0
            # set new linked
            self.dfb.loc[(self.tp, self.selection_zx.index[0]), 'linked'] = 1
            self._relink()

    @param.depends('selection_zy.index', watch=True)
    def _watch_zy_click(self):
        if len(self.selection_zy.index) > 0:
            # reset linked
            self.dfb.loc[self.tp, 'linked'] = 0
            # set new linked
            self.dfb.loc[(self.tp, self.selection_zy.index[0]), 'linked'] = 1
            self._relink()

    @param.depends('tp', '_relink_counter')
    def _plot_xy_candidates(self, index):
        rects = hv.Rectangles(self.dfb.loc[self.tp, [
            'y_start_phy', 'x_start_phy', 'y_stop_phy', 'x_stop_phy', 'linked'
        ]],
                              vdims='linked')
        return rects.opts(line_color='linked')

    @param.depends('tp', '_relink_counter')
    def _plot_zx_candidates(self, index):
        rects = hv.Rectangles(self.dfb.loc[self.tp, [
            'x_start_phy', 'z_start_phy', 'x_stop_phy', 'z_stop_phy', 'linked'
        ]],
                              vdims='linked')
        return rects.opts(line_color='linked')

    @param.depends('tp', '_relink_counter')
    def _plot_zy_candidates(self, index):
        rects = hv.Rectangles(self.dfb.loc[self.tp, [
            'y_start_phy', 'z_start_phy', 'y_stop_phy', 'z_stop_phy', 'linked'
        ]],
                              vdims='linked')
        return rects.opts(line_color='linked')

    @param.depends('tp', '_relink_counter')
    def _plot_xy_box_movie(self):
        filt_dfb = self.dfb[self.dfb.linked == True]
        return hv.Rectangles(filt_dfb.loc[self.tp, [
            'y_start_movie_phy', 'x_start_movie_phy', 'y_stop_movie_phy',
            'x_stop_movie_phy'
        ]],
                             group='movie')

    @param.depends('tp', '_relink_counter')
    def _plot_zx_box_movie(self):
        filt_dfb = self.dfb[self.dfb.linked == True]
        return hv.Rectangles(filt_dfb.loc[self.tp, [
            'x_start_movie_phy', 'z_start_movie_phy', 'x_stop_movie_phy',
            'z_stop_movie_phy'
        ]],
                             group='movie')

    @param.depends('tp', '_relink_counter')
    def _plot_zy_box_movie(self):
        filt_dfb = self.dfb[self.dfb.linked == True]
        return hv.Rectangles(filt_dfb.loc[self.tp, [
            'y_start_movie_phy', 'z_start_movie_phy', 'y_stop_movie_phy',
            'z_stop_movie_phy'
        ]],
                             group='movie')

    @param.depends('_relink_counter')
    def _plot_positions(self):
        filt_dfb = self.dfb[self.dfb.linked == True]
        cm_pos = hv.Curve(filt_dfb, 'time', 'x_center', label='x') * \
                 hv.Curve(filt_dfb, 'time', 'y_center', label='y') * \
                 hv.Curve(filt_dfb, 'time', 'z_center', label='z')

        cm_pos.opts(opts.Curve(frame_width=self.frame_width - 100))
        return cm_pos.opts(legend_position='top').opts(
            opts.Curve(ylabel='center of mass'))

    @param.depends('_relink_counter')
    def _plot_box_size(self):
        filt_dfb = self.dfb[self.dfb.linked == True]
        box_size = hv.Curve(filt_dfb, 'time', 'x_box_size', label='x') * \
                   hv.Curve(filt_dfb, 'time', 'y_box_size', label='y') * \
                   hv.Curve(filt_dfb, 'time', 'z_box_size', label='z')

        box_size.opts(opts.Curve(frame_width=self.frame_width - 100))
        return box_size.opts(legend_position='top').opts(
            opts.Curve(ylabel='box size'))

    @param.depends('tp')
    def _plot_vline(self):
        return hv.VLine(self.tp).opts(color='black',
                                      line_dash='dashed',
                                      line_width=2)

    def save_as(self):
        '''exports a csv file with linked crops'''
        filt_dfb = self.dfb[self.dfb.linked == True].reset_index('bb_id',
                                                                 drop=True)
        filt_dfb = filt_dfb[[
            'z_start', 'z_stop', 'x_start', 'x_stop', 'y_start', 'y_stop',
            'z_start_movie', 'z_stop_movie', 'x_start_movie', 'x_stop_movie',
            'y_start_movie', 'y_stop_movie'
        ]]

        filt_dfb.to_csv(os.path.join(self.outdir,
                                     'crop_roi_{}.csv'.format(self.object_id)),
                        index=True)

    save_button = param.Action(save_as, label='save as')

    @param.depends()
    def dmap(self):
        ortho = rasterize(hv.DynamicMap(self._plot_xy_proj), dynamic=False) * hv.DynamicMap(self._plot_xy_candidates, streams=[self.selection_xy]) * hv.DynamicMap(self._plot_xy_box_movie) + \
                 rasterize(hv.DynamicMap(self._plot_zx_proj), dynamic=False) * hv.DynamicMap(self._plot_zx_candidates, streams=[self.selection_zx]) * hv.DynamicMap(self._plot_zx_box_movie) + \
                 rasterize(hv.DynamicMap(self._plot_zy_proj), dynamic=False) * hv.DynamicMap(self._plot_zy_candidates, streams=[self.selection_zy]) * hv.DynamicMap(self._plot_zy_box_movie)

        ortho = ortho.cols(2).opts(normalize=True)
        return ortho

    def summary_plots(self):
        return hv.DynamicMap(self._plot_positions) * hv.DynamicMap(self._plot_vline) +\
               hv.DynamicMap(self._plot_box_size) * hv.DynamicMap(self._plot_vline)

    def widgets(self):
        tp_slider = pn.Param(self.param.tp,
                             widgets={
                                 'tp': {
                                     'type': pn.widgets.DiscreteSlider,
                                     'sizing_mode': 'stretch_width'
                                 }
                             },
                             sizing_mode='stretch_width')
        link_wgs = pn.WidgetBox(
            self.param.margin,
            pn.Row(self.param.save_button,
                   pn.Row(self.param.object_id, width=100)))

        return pn.Row(
            pn.Column(tp_slider,
                      self.param.x_range,
                      self.param.y_range,
                      self.param.z_range,
                      sizing_mode='stretch_width'), link_wgs)

    def panel(self):
        return pn.Column(self.dmap(), self.widgets(), self.summary_plots())
