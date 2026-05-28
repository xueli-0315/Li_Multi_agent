# Evolved factor from Evolved_2_6 and Evolved_2_3
def get_factor(): return 'RANK(TS_MEAN($high - $low, 24)) - ZSCORE($volume)'