# Evolved factor from Evolved_2_7 and Evolved_2_7
def get_factor(): return 'DELTA($close, 3) / (DELTA($close, 3) + 1e-08) * TS_RANK(TS_MEAN($open_interest, 7), 15)'