from op import *

import matplotlib.pyplot as plt


def toroidal_shell_integral_moment_of_innertia_exact(R, r):
    return 2 * (math.pi**2) * r * R * ((2 * (R**2)) + (3 * (r**2)))


def torus_volume_exact(R, r):
    return 2 * (math.pi**2) * (r**2) * R


def torus_surface_area_exact(R, r):
    return 4 * (math.pi**2) * r * R


def torus(X, Y, Z, vars):
    R = vars[0]  # major radius
    r = vars[1]  # minor radius
    return sq(edsl.sqrt(sq(X) + sq(Y)) - R) + sq(Z) - sq(r)


def integrand_inertia(X, Y, Z):
    return sq(X) + sq(Y)


def generate_plot_data(min_N, max_N, jump, R, r, check_integral, type):
    error_chart = np.array([])
    error_chart_N_by_delta = np.array([])

    for N in range(min_N, max_N, jump):
        print("evaluating for N: {}".format(N))
        minval = -1.25 * R
        maxval = 1.25 * R
        delta = (maxval - minval) / (N - 1)  # grid spacing
        eps = 1.0e-8
        error = check_integral(type, N, minval, maxval, eps, torus, [R, r])
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
    R = 10.0  # major radius
    r = 2.0  # minor radius

    min_N = 32
    max_N = 130
    interval = 32

    def check_integral(type, N, minval, maxval, eps, frep, vars):
        if type == 'volume':
            exact_value = torus_volume_exact(R, r)
            result = integral_volume(N, minval, maxval, eps, frep, vars)
        if type == 'surface_area':
            exact_value = torus_surface_area_exact(R, r)
            result = integral_surface_area(N, minval, maxval, eps, frep, vars, integrand_empty)
        if type == 'inertia':
            exact_value = toroidal_shell_integral_moment_of_innertia_exact(R, r)
            result = integral_surface_area(N, minval, maxval, eps, frep, vars, integrand_inertia)
        print("Exact value: {}".format(exact_value))
        print("computed result using integral: {}".format(result))
        return (abs(result - exact_value) / exact_value) * 100

    fig = plt.figure()
    fig.suptitle("convergence study Genus 1 : Torus ", fontsize=14)
    ax1 = fig.add_subplot(111)
    ax1.set(xlabel='log(R/delta)', ylabel='log(error percentage)')

    x1, y1 = generate_plot_data(min_N, max_N, interval, R, r, check_integral, 'volume')
    x2, y2 = generate_plot_data(min_N, max_N, interval, R, r, check_integral, 'surface_area')
    x3, y3 = generate_plot_data(min_N, max_N, interval, R, r, check_integral, 'inertia')

    ax1.scatter(x1, y1, label='Volume')
    ax1.scatter(x2, y2, label='SurfaceArea')
    ax1.scatter(x3, y3, label='Inertia')

    y_line1, label1 = get_line_y(x1, y1)
    ax1.plot(x1, y_line1, label=label1)
    y_line2, label2 = get_line_y(x2, y2)
    ax1.plot(x2, y_line2, label=label2)
    y_line3, label3 = get_line_y(x3, y3)
    ax1.plot(x3, y_line3, label=label3)
    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    main()
