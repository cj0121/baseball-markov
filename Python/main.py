import pandas as pd
import numpy as np
from tqdm.auto import tqdm

B_COUNT_LIST = [0, 1, 2, 3]
S_COUNT_LIST = [0, 1, 2]
EVENT_LIST = ['1B', '2B', '3B', 'HR', 'FO', 'BB', 'K', 'other']
COUNT_LIST = [(b, s) for s in S_COUNT_LIST for b in B_COUNT_LIST]
PROBA_LIST = ['1B_proba', '2B_proba', '3B_proba', 'HR_proba', 'FO_proba', 'BB_proba', 'K_proba', 'other_proba']
RESID_COL = ['1B_resid', '2B_resid', '3B_resid', 'HR_resid', 'FO_resid', 'BB_resid', 'K_resid', 'other_resid']

PF_LIST = ['1B_pf', '2B_pf', '3B_pf', 'HR_pf', 'BB_pf', 'K_pf']


def extract_proba(arr) -> np.array:
    arr_len = len(arr[0])

    single_arr = np.reshape(arr[0][:, 1], [arr_len, 1])
    double_arr = np.reshape(arr[1][:, 1], [arr_len, 1])
    triple_arr = np.reshape(arr[2][:, 1], [arr_len, 1])
    hr_arr = np.reshape(arr[3][:, 1], [arr_len, 1])
    fo_arr = np.reshape(arr[4][:, 1], [arr_len, 1])
    bb_arr = np.reshape(arr[5][:, 1], [arr_len, 1])
    k_arr = np.reshape(arr[6][:, 1], [arr_len, 1])
    other_arr = np.reshape(arr[7][:, 1], [arr_len, 1])
    proba_arr = np.concatenate([single_arr, double_arr, triple_arr, hr_arr, fo_arr, bb_arr, k_arr, other_arr], axis=1)

    return proba_arr


def normalize_P(P_arr):
    devider = P_arr.sum(axis=1)
    devider = np.reshape(devider, [len(devider), 1])
    return P_arr / devider

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

def get_brier_score(y_pred, y_test, multi=True):
    if multi:
        return ((y_pred - y_test)**2).sum(axis=1).mean()
    else:
        return ((y_pred - y_test)**2).mean()


def get_count_prob(count_from, count_to, matchup_df):
    matchup_df.reset_index(inplace=True, drop=True)
    cur_count_total = matchup_df[(matchup_df['balls'] == count_from[0]) & (matchup_df['strikes'] == count_from[1])].shape[0]
    if cur_count_total == 0:
        proba = 0
        return proba
    cur_idx = matchup_df[(matchup_df['balls'] == count_from[0]) & (matchup_df['strikes'] == count_from[1])].index
    if type(count_to) == str:
        event_total = matchup_df.iloc[cur_idx].loc[matchup_df['event_type'] == count_to].shape[0]
        proba = event_total / cur_count_total
        return proba
    else:
        if 0 in cur_idx:
            cur_idx = cur_idx.drop(0)
        cur_at_bat_number = matchup_df.iloc[cur_idx]['at_bat_number'].to_list()
        cur_game_pk = matchup_df.iloc[cur_idx]['game_pk'].to_list()
        next_idx = cur_idx - 1
        next_at_bat_number = matchup_df.iloc[next_idx]['at_bat_number'].to_list()
        next_game_pk = matchup_df.iloc[next_idx]['game_pk'].to_list()
        pitch_pairs = list(zip(cur_idx, cur_game_pk, cur_at_bat_number, next_idx, next_game_pk, next_at_bat_number))
        pitch_pairs_keep = [p for p in pitch_pairs if p[1] == p[4] and p[2] == p[5]]
        if pitch_pairs_keep == []:
            next_count_total = 0
            proba = next_count_total / cur_count_total
        else:
            a, b, c, d, e, f = zip(*pitch_pairs_keep)
            next_count = matchup_df.iloc[list(d)][['balls', 'strikes']]
            next_count_total = next_count[(next_count['balls'] == count_to[0]) & (next_count['strikes'] == count_to[1])].shape[0]
            proba = next_count_total / cur_count_total
        return proba


def get_player_T(player_id, is_pitcher=False, data_df=None):
    if is_pitcher:
        player_df = data_df[data_df['pitcher'] == player_id].copy()
        if player_df.empty:
            raise Exception("player_id not found as pitcher")
    else:
        player_df = data_df[data_df['batter'] == player_id].copy()
        if player_df.empty:
            raise Exception("player_id not found as batter")
    player_df.reset_index(inplace=True, drop=True)
    T_list = []
    for count_from in COUNT_LIST:
        T_list_row = []
        for count_to in COUNT_LIST:
            T_list_row.append(get_count_prob(count_from, count_to, player_df))
        for event in EVENT_LIST:
            T_list_row.append(get_count_prob(count_from, event, player_df))
        T_list.append(T_list_row)

    T1_arr = np.array(T_list)
    B = np.zeros([8, 12])
    C = np.zeros([8, 8])
    np.fill_diagonal(C, 1)
    T = np.block([
        [T1_arr],
        [B, C]
    ])

    return T


def convert_T_df(T):
    T_df = pd.DataFrame(T)

    T_df.columns = COUNT_LIST + EVENT_LIST
    T_df.index = COUNT_LIST + EVENT_LIST

    return T_df


def get_allplayers_T(player_id_list, data_df, is_pitcher=False):
    T_dict = {}
    for player_id in tqdm(player_id_list):
        T = get_player_T(player_id, is_pitcher, data_df)
        T_dict[player_id] = T

    return T_dict


def sim_steady_state(T, n_sim=100):
    T_cur = T
    for i in range(n_sim):
        T_new = np.dot(T_cur, T)
        T_cur = T_new
    return T_cur


def sim_steady_state_matchUp(pitcher_id, batter_id, p_weight=0.5, _return_df=True, batter_dict=None, pitcher_dict=None):
    T_batter = batter_dict[batter_id]
    T_pitcher = pitcher_dict[pitcher_id]

    T_matchUp = (T_pitcher * p_weight + T_batter * (1 - p_weight))

    P_steady_matchUp = sim_steady_state(T_matchUp)

    if not _return_df:
        return P_steady_matchUp[0][-8:]
    else:
        P_steady_matchUp_df = convert_T_df(P_steady_matchUp)
        P_steady_matchUp_df = P_steady_matchUp_df.iloc[:12][P_steady_matchUp_df.columns[-8:]].copy()

        return P_steady_matchUp_df


def get_emp_matchUp_T(pitcher_id, batter_id, data_df) -> np.array:
    '''Retrieve empirical match-up transition matrix from pitcher and batter ID'''
    matchUp_df = data_df[(data_df['pitcher'] == pitcher_id) & (data_df['batter'] == batter_id)]
    if matchUp_df.empty:
        raise Exception("player IDs did not return any data")
    matchUp_df.reset_index(inplace=True, drop=True)
    T_list = []
    for count_from in COUNT_LIST:
        T_list_row = []
        for count_to in COUNT_LIST:
            T_list_row.append(get_count_prob(count_from, count_to, matchUp_df))

        for event in EVENT_LIST:
            T_list_row.append(get_count_prob(count_from, event, matchUp_df))
        T_list.append(T_list_row)

    T1_arr = np.array(T_list)

    B = np.zeros([8, 12])
    C = np.zeros([8, 8])
    np.fill_diagonal(C, 1)

    T = np.block([
        [T1_arr],
        [B, C]
    ])

    return T


def test_T_dict(test_df, batter_dict, pitcher_dict, p_weight=0.5, _return_arr=False, is_pitch_data=True) -> pd.DataFrame:
    pitcher_list = list(pitcher_dict.keys())
    batter_list = list(batter_dict.keys())

    data_quali_df = test_df[(test_df['pitcher'].isin(pitcher_list) & (test_df['batter'].isin(batter_list)))].copy()
    if is_pitch_data:
        pa_df = data_quali_df.groupby(['game_year', 'game_pk', 'at_bat_number', 'pitcher', 'batter']).agg({
            'home_team': 'first',
            'stand': 'first',
            'p_throws': 'first',
            'event_type': 'first',
            'des': 'first'}).reset_index()
        event_df = pd.get_dummies(pa_df['event_type'])
        event_df = event_df[EVENT_LIST].copy()

        pa_df = pd.concat([pa_df, event_df], axis=1).copy()
    else:
        pa_df = test_df.reset_index(drop=True).copy()
    matchup_to_predict = list(zip(pa_df['pitcher'], pa_df['batter']))
    matchup_sets_df = data_quali_df[['pitcher', 'batter']].drop_duplicates()
    matchup_sets_list = list(zip(matchup_sets_df['pitcher'], matchup_sets_df['batter']))

    P_matchup_dict = {}
    for pair in tqdm(matchup_sets_list):
        P_matchup_dict[pair] = sim_steady_state_matchUp(
            pair[0], pair[1], p_weight=p_weight, _return_df=False,
            batter_dict=batter_dict, pitcher_dict=pitcher_dict
        )

    P_predicted = np.array([P_matchup_dict[pair] for pair in matchup_to_predict])
    P_predicted = normalize_P(P_predicted)
    if _return_arr:
        return P_predicted
    else:
        P_predicted_df = pd.DataFrame(P_predicted, columns=PROBA_LIST)
        pa_df = pd.concat([pa_df, P_predicted_df], axis=1)
        return pa_df

def get_weighted_T(cur_year, T_seasons, weight_func) -> np.array:
    T_weighted_all = np.zeros([20, 20])
    total_weight = 0
    for T in T_seasons:
        day = (cur_year - T[0])*365 + 1
        weight = weight_func(day)
        T_weighted_all+=(T[1]*weight)
        total_weight+=weight
    T_weighted_all/=total_weight
    return T_weighted_all

def get_player_T_by_season(player_df=None, seasons=None) -> list:
    T_seasons = []
    for season in seasons:
        season_df = player_df[player_df['game_year'] == season].copy()
        season_df.reset_index(inplace=True, drop=True)
        T_list = []
        for count_from in COUNT_LIST:
            T_list_row = []
            for count_to in COUNT_LIST:
                T_list_row.append(get_count_prob(count_from, count_to, season_df))
            for event in EVENT_LIST:
                T_list_row.append(get_count_prob(count_from, event, season_df))
            T_list.append(T_list_row)

        T1_arr = np.array(T_list)
        B = np.zeros([8, 12])
        C = np.zeros([8, 8])
        np.fill_diagonal(C, 1)
        T = np.block([
            [T1_arr],
            [B, C]
        ])

        T_seasons.append((season, T))
    return T_seasons

def get_weighted_player_T_by_season(player_id, data_df, is_pitcher=False,
                                    weight_func=lambda x: np.log(np.exp(1 / (599 + x) / np.log(np.exp(1 / 600)))),
                                    cur_year=2018, annual_pa_threshold=70) -> np.array:
    if is_pitcher:
        player_df = data_df[data_df['pitcher'] == player_id].copy()
    else:
        player_df = data_df[data_df['batter'] == player_id].copy()
    pa_df = player_df.groupby(['game_year', 'game_pk', 'at_bat_number']).agg({'n_count': 'first'}).reset_index()
    pa_count_df = pa_df.groupby('game_year').agg({'n_count': 'sum'}).reset_index()
    quali_seasons_list = pa_count_df[pa_count_df['n_count'] >= annual_pa_threshold]['game_year'].to_list()

    T_seasons_list = get_player_T_by_season(player_df=player_df, seasons=quali_seasons_list)

    T_seasons_weighted = get_weighted_T(cur_year, T_seasons_list, weight_func)

    return T_seasons_weighted

def get_allplayers_T_weighted(player_id_list, data_df, is_pitcher=False,
                              weight_func=lambda x:np.log(np.exp(1/(599+x)/np.log(np.exp(1/600)))),
                              cur_year=2018, annual_pa_threshold=70):
    T_dict = {}
    for player_id in tqdm(player_id_list):
        T = get_weighted_player_T_by_season(player_id, data_df, is_pitcher, weight_func, cur_year, annual_pa_threshold)
        T_dict[player_id] = T

    return T_dict