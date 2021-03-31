from matplotlib import use
use('Qt5Agg')
import matplotlib.pyplot as plt


class Plot():
    def __init__(self, name, color, position):
        self.name = name
        self.color = color
        plt.style.use('dark_background')
        #self.fig, self.axs = plt.subplots(1, 2, figsize=(8, 4), dpi=100, constrained_layout=True)
        self.fig, self.axs = plt.subplots(1, 2, constrained_layout=True)
        self.fig.canvas.set_window_title(f"{name}")
        self.axs[0].set_title(f"{name} - reward sum")
        self.axs[1].set_title(f"{name} - running reward")

        mgr = plt.get_current_fig_manager()
        screen_width, screen_height = self.get_screen_size(mgr)
        plot_width = screen_width / 2
        plot_height = screen_height / 3 - 50
        position_x, position_y = self.calculate_position(screen_width, screen_height, plot_width, plot_height, position)
        mgr.window.setGeometry(position_x, position_y, plot_width, plot_height)

        self.is_shown = False


    def get_screen_size(self, mgr):
        #Since the fig manager incorrectly recognizes the resolution of the 4K screen, you have to do this trick to get the "resolution"
        mgr.full_screen_toggle()
        screen_width = mgr.window.geometry().width()
        screen_height = mgr.window.geometry().height()
        mgr.full_screen_toggle()
        return screen_width, screen_height


    def calculate_position(self, screen_width, screen_height, plot_width, plot_height, position):
        positions_y = {
            'T': 20,
            'C': (screen_height - plot_height) / 2,
            'B': screen_height - plot_height - 40
        }

        positions_x = {
            'L': 0,
            'C': (screen_width - plot_width) / 2,
            'R': screen_width - plot_width
        }

        position_y = positions_y.get(position[0],"Invalid plot position")
        position_x = positions_x.get(position[1],"Invalid plot position")
        return position_x, position_y


    def update(self, v_x, v_y1, v_y2):
        if not self.is_shown:
            plt.show(block=False)
            self.is_shown = True

        for i,x in enumerate(v_x):
            y1 = v_y1[i]
            y2 = v_y2[i]
            bar_width = int(x - self.axs[1].collections[-1].get_offsets()[0][0]) if len(self.axs[1].collections) > 0 else x
            self.axs[0].bar(x, y1, bar_width, color=self.color)
            self.axs[1].scatter(x, y2, color=self.color, s=2)
        self.fig.canvas.draw()
        #plt.draw()