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


def main(
):  #TODO:work in progress to generate graphs will clean up after documentation is completed
    R = 10.0  # major radius
    r = 2.0  # minor radius
    #N = 128  # number of grid points
    error_chart = np.array([])
    error_chart_N_by_delta = np.array([])
    for N in range(32, 1024, 128):
        print("evaluating for N: {}".format(N))
        minval = -1.25 * R
        maxval = 1.25 * R
        delta = (maxval - minval) / (N - 1)  # grid spacing
        eps = 1.0e-8
        #compute exact value
        #exact_value = toroidal_shell_integral_moment_of_innertia_exact(R, r)
        exact_value = torus_volume_exact(R, r)
        print(exact_value)
        #run integral computation
        print("Exact value: {}".format(exact_value))
        #compare the result
        #result = toroidal_shell_moment_of_inertia(N, minval, maxval, eps, R, r)
        result = integral_volume(N, minval, maxval, eps, torus, [R, r])
        print("computed result using integral: {}".format(result))
        error = (abs(result - exact_value) / exact_value) * 100
        print("error: {} %".format(error))
        error_chart = np.append(error_chart, math.log(error))
        error_chart_N_by_delta = np.append(error_chart_N_by_delta, math.log(R / delta))

    fig = plt.figure()
    ax = plt.axes()
    ax.set(
        xlabel='log(R/delta)',
        ylabel='log(error percentage)',
        title=
        'Result of convergence study genus 1 grid-based evaluationof moment of innertia of toroidal shell'
    )
    ax.scatter(error_chart_N_by_delta, error_chart)
    plt.show()


if __name__ == '__main__':
    main()
