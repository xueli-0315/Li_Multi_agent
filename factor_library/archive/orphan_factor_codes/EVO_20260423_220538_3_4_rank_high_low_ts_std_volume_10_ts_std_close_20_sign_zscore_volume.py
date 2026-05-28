# Evolved factor from Evolved_2_1 and Evolved_2_1
def get_factor(): return 'RANK(($high - $low) / TS_STD($volume, 10) * TS_STD($close, 20)) * SIGN(ZSCORE($volume))'