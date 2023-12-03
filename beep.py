import winsound

def make_beep(beep_on):
    freq = 500
    duration = 100
    winsound.Beep(freq, duration)
    beep_on = 0
    return beep_on
