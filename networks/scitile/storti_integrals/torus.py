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


def generate_plot_data(n, jump, R, r, check_integral, type):
    error_chart = np.array([])
    error_chart_N_by_delta = np.array([])

    for N in range(32, n, jump):
        print("evaluating for N: {}".format(N))
        minval = -1.25 * R
        maxval = 1.25 * R
        delta = (maxval - minval) / (N - 1)  # grid spacing
        eps = 1.0e-8
        error = check_integral(type, N, minval, maxval, eps, torus, [R, r])
        error_chart = np.append(error_chart, math.log(error))
        error_chart_N_by_delta = np.append(error_chart_N_by_delta, math.log(R / delta))

    return [error_chart_N_by_delta, error_chart]


def main(
):  #TODO:work in progress to generate graphs will clean up after documentation is completed
    R = 10.0  # major radius
    r = 2.0  # minor radius

    max_N = 256
    interval = 32

    def get_line_y(x, y):
        den = x.dot(x) - x.mean() * x.sum()
        m = (x.dot(y) - y.mean() * x.sum()) / den
        b = (y.mean() * x.dot(x) - x.mean() * x.dot(y)) / den
        y_line = (m * x + b)
        label = "f(x) = " + '%.2f' % m + "x +" + str(b)
        return [y_line, label]

    def check_integral(type, N, minval, maxval, eps, frep, vars):
        if type == 'volume':
            exact_value = torus_volume_exact(R, r)
            result = integral_volume(N, minval, maxval, eps, frep, vars)
        if type == 'surface_area':
            exact_value = torus_volume_exact(R, r)
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

    data1 = generate_plot_data(max_N, interval, R, r, check_integral, 'volume')
    data2 = generate_plot_data(max_N, interval, R, r, check_integral, 'surface_area')
    data3 = generate_plot_data(max_N, interval, R, r, check_integral, 'inertia')

    ax1.scatter(data1[0], data1[1], label='Volume')
    ax1.scatter(data2[0], data2[1], label='SurfaceArea')
    ax1.scatter(data3[0], data3[1], label='Inertia')

    y_line1, label1 = get_line_y(data1[0], data1[1])
    ax1.plot(data1[0], y_line1)
    y_line2, label2 = get_line_y(data2[0], data2[1])
    ax1.plot(data2[0], y_line2)
    y_line3, label3 = get_line_y(data3[0], data3[1])
    ax1.plot(data3[0], y_line3)
    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    main()
