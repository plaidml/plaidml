from op import *

import matplotlib.pyplot as plt


def hypersphere_volume_exact(r):
    return (math.pi**2) * 0.5 * (r**4)


def hypersphere_area_exact(r):
    return 2 * (math.pi**2) * (r**3)


def hypersphere(X, Y, Z, W, vars):
    R = vars[0]  #radius
    p = vars[1]  #dim
    return edsl.pow(edsl.pow(X, p) + edsl.pow(Y, p) + edsl.pow(Z, p) + edsl.pow(W, p), (1 / p)) - R


def generate_plot_data(min_N, max_N, jump, R, p, check_integral, type):
    error_chart = np.array([])
    error_chart_N_by_delta = np.array([])

    for N in range(min_N, max_N, jump):
        print("evaluating for N: {}".format(N))
        minval = -1.25 * R
        maxval = 1.25 * R
        delta = (maxval - minval) / (N - 1)  # grid spacing
        eps = 1.0e-8
        error = check_integral(type, N, minval, maxval, eps, hypersphere, [R, p])
        error_chart = np.append(error_chart, math.log(error, 10))
        error_chart_N_by_delta = np.append(error_chart_N_by_delta, math.log((R / delta), 10))

    return [error_chart_N_by_delta, error_chart]


def get_line_y(x, y):
    den = x.dot(x) - x.mean() * x.sum()
    m = (x.dot(y) - y.mean() * x.sum()) / den
    b = (y.mean() * x.dot(x) - x.mean() * x.dot(y)) / den
    y_line = (m * x + b)
    res = y - y_line
    tot = y - y.mean()
    R_sq = 1 - res.dot(res) / tot.dot(tot)
    label = "f(x) = " + '%.2f' % m + "x +" + '%.2f' % b + "| R^2 = " + '%.2f' % R_sq
    return [y_line, label]


def main(
):  #TODO:work in progress to generate graphs will clean up after documentation is completed
    #TODO:incorrect results produced
    #TODO:write tests once 4Dintegral correctness proof is found
    R = 1.0  # radius
    p = 2

    min_N = 32
    max_N = 130
    interval = 32

    def check_integral(type, N, minval, maxval, eps, frep, vars):
        if type == 'volume':
            exact_value = hypersphere_volume_exact(R)
            result = integral_volume_4D(N, minval, maxval, eps, frep, vars)
        if type == 'surface_area':
            exact_value = hypersphere_area_exact(R)
            result = integral_surface_area_4D(N, minval, maxval, eps, frep, vars,
                                              integrand_empty_4D)
        print("Exact value: {}".format(exact_value))
        print("computed result using integral: {}".format(result))
        return (abs(result - exact_value) / exact_value) * 100

    fig = plt.figure()
    fig.suptitle("convergence study Genus 1 : hypersphere - 2 ", fontsize=14)
    ax1 = fig.add_subplot(111)
    ax1.set(xlabel='log(R/delta)', ylabel='log(error percentage)')

    x1, y1 = generate_plot_data(min_N, max_N, interval, R, p, check_integral, 'volume')
    x2, y2 = generate_plot_data(min_N, max_N, interval, R, p, check_integral, 'surface_area')

    ax1.scatter(x1, y1, label='Volume')
    ax1.scatter(x2, y2, label='SurfaceArea')

    y_line1, label1 = get_line_y(x1, y1)
    ax1.plot(x1, y_line1, label=label1)
    y_line2, label2 = get_line_y(x2, y2)
    ax1.plot(x2, y_line2, label=label2)

    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    main()
