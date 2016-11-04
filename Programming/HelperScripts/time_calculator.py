import time


class TimeCalculator(object):
    def __init__(self):
        self.t = time.time()
        self.lastly = self.t

    def show(self, message):
        current_time = time.time()
        print 'Time for '+message+': ' + self.toString(current_time - self.lastly)
        print 'Total time: ' + self.toString(self.getTotalTime())
        self.lastly = current_time

    def getTotalTime(self):
        current_time = time.time()
        return current_time - self.t

    def toString(self, ti):
        if ti > 90 * 60:
            return "%.2fh" % (ti/(60*60))
        elif ti > 90:
            return "%.2fm" % (ti/60)
        else:
            return "%.2fs" % ti
