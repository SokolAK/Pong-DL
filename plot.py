import matplotlib.pyplot as plt


class Plot():
    def __init__(self):
        self.is_shown = False

    def prepare_plots(self, net_A, net_B):
        fig = None
        axs = None

        if net_A.is_active or net_B.is_active:
            plt.style.use('dark_background')
            fig, axs = plt.subplots(2, 2, constrained_layout=True)
            #fig.suptitle('Reward balance vs game no')
            axs[0,0].set_title('Network A - reward balance')
            axs[1,0].set_title('Network B - reward balance')
            axs[0,1].set_title('Network A - running reward')
            axs[1,1].set_title('Network B - running reward')
            #plt.show(block=False)

        return [fig, axs]


    def update_plots(self, figure, net_A, net_B):
        if not self.is_shown:
            plt.show(block=False)
            self.is_shown = True

        fig = figure[0]
        axs = figure[1]
        if fig is not None and axs is not None:
            if net_A.is_active:
                axs[0,0].bar(net_A.episode_no, net_A.reward_sum, 1, color='tab:blue')
                axs[0,1].scatter(net_A.episode_no, net_A.running_reward, color='tab:blue', s=2)
            if net_B.is_active:
                axs[1,0].bar(net_B.episode_no, net_B.reward_sum, 1, color='tab:orange')
                axs[1,1].scatter(net_B.episode_no, net_B.running_reward, color='tab:orange', s=2)
            plt.draw()