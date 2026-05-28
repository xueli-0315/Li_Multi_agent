# Evolved factor from Evolved_2_8 and Evolved_2_7
def get_factor(): return 'RANK(($high - $low) / TS_STD($volume, 10) * TS_STD($close, 20)) * SIGN($close - $open)'