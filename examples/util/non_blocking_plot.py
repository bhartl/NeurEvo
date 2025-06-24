import matplotlib.pyplot as plt
import numpy as np
import time as t


def main():
    from matplotlib import pyplot as plt
    plt.axis([-50,50,0,10000])
    plt.ion()
    plt.show()

    x = np.arange(-50, 51)
    for pow in range(1,10):   # plot x^1, x^2, ..., x^4
        y = [Xi**pow for Xi in x]
        plt.plot(x, y)
        plt.draw()
        plt.pause(0.001)
        t.sleep(0.1)


class Plotter:
    def __init__(self):
        self.plt, self.fig, self.ax = None, None, None

    def render(self):
        if self.plt is None:
            import matplotlib.pyplot as plt
            self.plt = plt
            plt.ion()
            plt.show()

            self.fig, self.ax = plt.subplots(1, 1)

        # self.ax.clear()
        self.ax.scatter(*np.random.rand(2, 2))
        self.plt.draw()
        self.plt.pause(0.0001)


if __name__ == '__main__':
    print("All in a function.")
    main()

    print("\nAll in a class.")
    p = Plotter()
    for i in range(10):
        p.render()
        t.sleep(0.1)
