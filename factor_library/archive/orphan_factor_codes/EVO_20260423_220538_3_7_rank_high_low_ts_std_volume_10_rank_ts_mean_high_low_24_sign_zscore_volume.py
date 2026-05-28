# Evolved factor from Evolved_2_1 and Evolved_2_6
def get_factor(): return 'RANK(($high - $low) / TS_STD($volume, 10) * RANK(TS_MEAN($high - $low, 24))) * SIGN(ZSCORE($volume))'