import numpy as np
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def plot_hexagonal_grid(size, spacing):
    # Create an empty black image
    img = np.ones((size, size, 3), np.uint8) * 255

    # Define the coordinates of the hexagonal grid
    hex_points = np.array([[0, spacing], [spacing/2, 0], [spacing*3/2, 0], [spacing*2, spacing],
                           [spacing*3/2, spacing*2], [spacing/2, spacing*2]], np.int32)

    # import matplotlib.pyplot as plt
    # plt.scatter(*hex_points.T)
    # plt.gca().set_aspect("equal")
    # plt.show()

    hex_points = hex_points.reshape((-1, 1, 2))

    # Draw the hexagonal grid
    for i in range(-int(3/2*size/spacing), int(size/spacing)):
        for j in range(0, int(size/spacing * 3 / 2)):
            center = (spacing*j*1.5, spacing*(j + i*2.))
            pts = np.asarray(hex_points + center, dtype=np.int32)
            img = cv2.polylines(img, [np.flip(pts, axis=0)], True, (0, 0, 0), 2)

    img = cv2.polylines(img, [hex_points + (int(size//2) + spacing, int(size//2))], True, (255, 0, 0), 2)

    # Show the image
    cv2.imshow("Hexagonal Grid", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


import matplotlib.pyplot as plt

class MyDataPlotter:
    def __init__(self):
        self.x = []
        self.y = []
        self.fig, self.ax = plt.subplots()

    def add_data(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def render(self):
        self.ax.clear()
        self.ax.scatter(self.x, self.y)
        self.fig.canvas.draw()


if __name__ == '__main__':
    import time
    plt.ion()
    plt.show()
    p = MyDataPlotter()
    for i in range(10):
        p.add_data(*np.random.rand(2))
        p.render()
        time.sleep(0.1)


