import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import datetime, date

from thesis_plots.ztf.data import data_dir
from thesis_plots.plotter import Plotter


logger = logging.getLogger(__name__)


def get_n_years():
    start_of_v2 = date(year=2019, month=6, day=17)
    today = date.today()
    d = today - start_of_v2
    return d.days / 364.25


def get_files():
    gcn_fn = data_dir / "neutrino_too_followup - OVERVIEW GCNed.csv"
    fu_fn = data_dir / "neutrino_too_followup - OVERVIEW_FU.csv"
    not_fu_fn = data_dir / "neutrino_too_followup - OVERVIEW_NOT_FU.csv"
    return gcn_fn, fu_fn, not_fu_fn


use_cols_gcn = [
    'IceCube event',
    'Classification',
    'Max Brightness g',
    'Max brightness r',
    'Brightness at neutrino arrival r',
    'Brightness at neutrino arrival g',
    'number of detection days',
    'Notes'
]


@Plotter.register("half")
def average() -> None:
    """
    Histogram summary of follow up since start of v2 alerts.
    """

    N_years = get_n_years()
    logger.info(f'V2 alerts running for {N_years} years')

    # Read csv file
    gcn_fn, fu_fn, not_fu_fn = get_files()
    orig_tab = pd.read_csv(gcn_fn, header=3)

    tab = orig_tab[use_cols_gcn]
    # select only those that have max brightness
    tab = tab[~(tab['Max brightness r'].isna() & tab['Max Brightness g'].isna())]

    # In the file the IC event is only listed for the first candidate
    # Fill in for the others as well
    m = tab['IceCube event'].isna()
    for i, v in enumerate(~m):
        if v:
            current_ic = tab['IceCube event'].iloc[i]
        tab['IceCube event'].iloc[i] = current_ic

    # set >20 to 20 to make everything numeric
    tab.loc[tab['number of detection days'] == '>20', 'number of detection days'] = '20'
    multiple_detections = np.array(['no'] * len(tab), dtype='U3')
    multiple_detections[np.array([float(n) for n in tab['number of detection days']]) > 1] = 'yes'
    tab['multiple detections'] = multiple_detections
    # if the event is observed more than twice, assime it's MSIP detected
    msip_detections = np.array(['no'] * len(tab), dtype='U3')
    msip_detections[np.array([float(n) for n in tab['number of detection days']]) > 2] = 'yes'
    tab['MSIP detected'] = msip_detections

    not_classified_mask = [isinstance(cl, str) and ('?' in cl) for cl in tab.Classification]
    tab.loc[not_classified_mask, 'Classification'] = np.nan

    # check if we triggered the classification ourselves
    check_strings = ["triggered NOT", "triggered Keck"]
    class_by_us = np.array([False] * len(tab))

    for c in check_strings:
        new_m = np.array([c in note if isinstance(note, str) else False for note in tab.Notes])
        class_by_us = class_by_us | new_m

    tab["classified by us"] = class_by_us

    classified = np.array(['no'] * len(tab), dtype='U3')
    classified[~tab.Classification.isna()] = 'yes'
    tab['classified'] = classified

    hue = [c + ' / ' + m for c, m in zip(classified, multiple_detections)]
    tab['classified / multiple detections'] = hue

    hue2 = [c + ' / ' + m for c, m in zip(classified, msip_detections)]
    tab['classified / MSIP detected'] = hue2

    tab.loc[tab['Max brightness r'].isna(), 'Max brightness r'] = tab.loc[
        tab['Max brightness r'].isna(), 'Max Brightness g']
    tab.loc[tab['Brightness at neutrino arrival r'].isna(), 'Brightness at neutrino arrival r'] = tab.loc[
        tab['Brightness at neutrino arrival r'].isna(), 'Brightness at neutrino arrival g']

    # N_alerts_total = np.nan
    N_alerts = len(tab['IceCube event'].unique())
    n_alerts_yr = N_alerts / N_years
    n_alerts_se = n_alerts_yr / 2

    N_candidates = len(tab)
    n_candidates_alert = N_candidates / N_alerts
    N_candidates_inmagrange = len(tab[(tab['Max brightness r'] > 19) & (tab['Max brightness r'] < 20.5)])
    n_candidates_inmagrange_alert = N_candidates_inmagrange / N_alerts

    logger.info(
        # f"{N_alerts_total} alerts in total\n "
        # f"{N_alerts_total / N_years / 2} per semester\n"
        f"{N_alerts} alerts followed up in {N_years:.2f} years"
        f"\n {n_alerts_yr:.2f} per year \n "
        f"{n_alerts_se:.2f} per semester \n"
        f"{N_candidates} candidates in total \n "
        f"{n_candidates_alert:.2f} per alert\n "
        f"{N_candidates_inmagrange} in Mag range\n "
        f"{n_candidates_inmagrange_alert} per alert"
    )

    # ---------------------------   make the summary plot --------------------------- #

    col = 'Max brightness r'
    alpha = 1
    intervals = [(17.5, 19), (19, 20.5), (20.5, np.inf)]
    plot_dictionary = dict()

    for i in intervals:
        o = dict()

        events_mask = (tab[col] > i[0]) & (tab[col] < i[1]) & ~tab["classified by us"]
        events = tab[events_mask]
        for p in tab['classified / MSIP detected'].unique():
            o[p] = len(events[events['classified / MSIP detected'] == p])

        plot_dictionary[i] = o

    classified_colors = {'yes': "C0", 'no': "C3"}
    detections_hatch = {'yes': '', 'no': '//'}

    fig, ax = plt.subplots()

    for i in intervals:
        Nclass = plot_dictionary[i]['yes / yes']  # + d[i]['yes / no']
        N_noclass = plot_dictionary[i]['no / yes'] + plot_dictionary[i]['no / no']
        ax.bar(i[0], Nclass / N_years / 2, 1.5, color=classified_colors['yes'], alpha=alpha)
        ax.bar(i[0], N_noclass / N_years / 2, 1.5, bottom=Nclass / N_years / 2, color=classified_colors['no'],
               alpha=alpha)
        ax.bar(i[0], plot_dictionary[i]['no / no'] / N_years / 2, 1.5, bottom=Nclass / N_years / 2, color=classified_colors['no'],
               hatch=detections_hatch['no'], alpha=alpha)

    class_head, = ax.plot([], [], ls='', label='classified')
    classp = mpatches.Patch(facecolor=classified_colors['yes'], label='yes', alpha=alpha)
    not_classp = mpatches.Patch(facecolor=classified_colors['no'], label='no', alpha=alpha)

    md_head, = ax.plot([], [], ls='', label='detected twice')
    mdp = mpatches.Patch(facecolor='grey', hatch=detections_hatch['yes'], label='yes', alpha=alpha)
    nmdp = mpatches.Patch(facecolor='grey', hatch=detections_hatch['no'], label='no', alpha=alpha)

    ax.set_xticks([17.5, 19, 20.5])
    ax.set_xticklabels(['17.5 - 19', '19 - 20.5', '>20.5'])

    ax.legend(handles=[class_head, md_head, classp, mdp, not_classp, nmdp], ncol=3,
              loc='lower center', bbox_to_anchor=(0.43, 1))
    # ax.set_ylim(0, 4)

    ax.set_xlabel('Peak $r$-band magnitude')
    ax.set_ylabel('candidates per semester')

    return fig


@Plotter.register("half")
def timeresolved():
    gcn_fn, fu_fn, not_fu_fn = get_files()
    fu = pd.read_csv(fu_fn, skiprows=3, skipfooter=4, engine='python')
    not_fu = pd.read_csv(not_fu_fn, skiprows=2)

    not_fu["dates"] = [datetime.strptime(ev[2:-1], "%y%m%d").date() for ev in not_fu.Event]
    fu["dates"] = [datetime.strptime(ev[2:-1], "%y%m%d").date() for ev in fu.Event.iloc]
    not_fu["maintenance"] = not_fu["Code"] == 3

    bins = np.array([datetime.strptime(f"{y}0{m}01", "%y%m%d").date() for y in range(17, 25) for m in [1, 7]])
    bindif = bins[1:] - bins[:-1]
    binmids = bins[:-1] + (bindif) / 2
    colors = ["C2", "C3", "grey"]

    fig, ax = plt.subplots(nrows=2, sharex='all', gridspec_kw={'hspace': .1})
    h, b, p = ax[0].hist([fu.dates, not_fu.dates[not_fu.maintenance], not_fu.dates[~not_fu.maintenance]],
                         histtype='barstacked',
                         label=["followed-up", "ZTF down", "other"],
                         bins=bins,
                         color=colors)

    total = h[-1]
    perc_not_fu_main = (h[1] - h[0]) / total

    for i in range(len(h)):
        ax[1].bar(binmids, h[len(h)-i-1] / h[-1], width=bindif, color=colors[len(h)-i-1])

    for i, p in enumerate(perc_not_fu_main):
        if p:
            logger.info(f"{p*100:.0f}% missed at {binmids[i]}")

    ax[0].set_ylabel("count")
    ax[1].set_ylabel("percentage")
    ax[1].set_xlabel("year")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax[1].xaxis.set_major_locator(locator)
    ax[1].xaxis.set_major_formatter(formatter)
    ax[0].legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)

    return fig
