import time


class TimeKeeper:
    def __init__(self, arg):
        self.arg = arg

    def print_log(self, string, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            string = f"[ {localtime} ] {string}"
        print(string)
        if self.arg.print_log:
            with open(f"{self.arg.work_dir}/log.txt", "a") as f:
                print(string, file=f)

    def record_time(self):
        self.cur_time = time.time()

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
