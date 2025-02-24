from datetime import datetime
import matplotlib.pyplot as plt

class Logger():
    log_path = None

    def scatter(self, x, y, save_path, **fig_kwargs):
        fig, ax = plt.subplots(**fig_kwargs)
        
        ax.scatter(x, y)
        fig.savefig(save_path)

    def log(self, message):
        if self.log_path is None: return 

        with open(self.log_path, "a") as log:
            log.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    def clear_log(self):
        if self.log_path is None: return 

        with open(self.log_path, "w") as log:
            log.write("")