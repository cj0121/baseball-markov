# import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time
# import timeit


# from tqdm.auto import tqdm
# from scipy.stats import *

def get_vid_link(data):
    def get_vid_link(team_abb, date, topbot_input, outs, inning, balls, strikes, pitcher_id, batter_id):
        team_id = {'KC': 118, 'CHC': 120, 'TOR': 141, 'NYM': 121, 'HOU': 117, 'TEX': 140, 'LAD': 119,
                   'STL': 138, 'TB': 139, 'ATL': 144, 'SEA': 136, 'BAL': 110, 'PHI': 143, 'MIN': 142,
                   'ARI': 109, 'CWS': 145, 'SF': 137, 'PIT': 134, 'MIL': 158, 'CLE': 114, 'SD': 135,
                   'CIN': 113, 'NYY': 147, 'LAA': 108, 'WSH': 120, 'OAK': 133, 'DET': 116, 'COL': 115,
                   'BOS': 111, 'MIA': 146}
        topbot_dict = {'Top': 'TOP', 'Bot': 'BOTTOM'}

        team = team_id[team_abb]
        topbot = topbot_dict[topbot_input]

        url = f"https://www.mlb.com/video/search?q=HomeTeamId+%3D+%5B{team}%5D+AND+Date+%3D+%5B%22{date}%22%5D+AND+Inning+%3D+%5B{inning}%5D+AND+TopBottom+%3D+%5B%22{topbot}%22%5D+AND+Outs+%3D+%5B{outs}%5D+AND+Balls+%3D+%5B{balls}%5D+AND+Strikes+%3D+%5B{strikes}%5D+AND+PitcherId+%3D+%5B{pitcher_id}%5D+AND+BatterId+%3D+%5B{batter_id}%5D+Order+By+Timestamp+DESC"

        return url

    data['url'] = data.apply(lambda x: get_vid_link(x['home_team'], x['game_date'], x['inning_topbot'],
                                                    x['outs_when_up'], x['inning'], x['balls'],
                                                    x['strikes'], x['pitcher'], x['batter']), axis=1).copy()
    return data




def get_brier_score(X, Y):
    return ((X-Y)**2).sum(axis=1).mean()