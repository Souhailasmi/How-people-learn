#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:42:29 2019

@author: jecker, tenderini, ronssin
"""
from __future__ import division
import numpy as np
from IPython.display import display, Latex
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)
from IPython.core.magic import register_cell_magic
from IPython.display import HTML
import ipywidgets as widgets
import random
from ipywidgets import interact_manual


@register_cell_magic
def bgc(color):
    script = (
        "var cell = this.closest('.jp-CodeCell');"
        "var editor = cell.querySelector('.jp-Editor');"
        "editor.style.background='{}';"
        "this.parentNode.removeChild(this)"
    ).format(color)

    display(HTML('<img src onerror="{}">'.format(script)))


###############################################################################

## PRINTS Equations, systems, matrix

def printMonomial(coeff, index=None, include_zeros=False):
    """Prints the monomial coeff*x_{index} in optimal way

    :param coeff: value of the coefficient
    :type coeff: float
    :param index: index of the monomial. If None, only the numerical value of the coefficient is displayed
    :type index: int or NoneType
    :param include_zeros: if True, monomials of type 0x_n are printed. Defaults to False
    :type include_zeros: bool
    :return: string representative of the monomial
    :rtype: str
    """

    if index is not None:
        coeff = abs(coeff)

    if coeff % 1:
        return str(round(coeff, 3)) + ('x_' + str(index) if index is not None else "")
    elif not coeff:
        if index is None:
            return str(0)
        else:
            return str(0) + 'x_' + str(index) if include_zeros else ""
    elif coeff == 1:
        return 'x_' + str(index) if index is not None else str(int(coeff))
    elif coeff == -1:
        return 'x_' + str(index) if index is not None else str(int(coeff))
    else:
        return str(int(coeff)) + ('x_' + str(index) if index is not None else "")


def printPlusMinus(coeff, include_zeros=False):
    """Prints a plus or minus sign, depending on the sign of the coefficient

    :param coeff: value of the coefficient
    :type coeff: float
    :param include_zeros: if True, 0-coefficients are assigned a "+" sign
    :type include_zeros: bool
    :return: "+" if the coefficient is positive, "-" if it is negative, "" if it is 0
    :rtype: str
    """
    if coeff > 0:
        return "+"
    elif coeff < 0:
        return "-"
    else:
        return "+" if include_zeros else ""


def strEq(n, coeff):
    """Method that provides the Latex string of a linear equation, given the number of unknowns and the values
    of the coefficients. If no coefficient value is provided, then a symbolic equation with `n` unknowns is plotted.
    In particular:

        * **SYMBOLIC EQUATION**: if the number of unknowns is either 1 or 2, then all the equation is
          displayed while, if the number of unknowns is higher than 2, only the first and last term of the equation
          are displayed
        * **NUMERICAL EQUATION**: whichever the number of unknowns, the whole equation is plotted. Numerical values
          of the coefficients are rounded to the third digit

    :param n: number of unknowns of the equation
    :type n: int
    :param coeff: coefficients of the linear equation. It must be [] if a symbolic equation is desired
    :type: list[float]
    :return: Latex string representing the equation
    :rtype: str
    """

    Eq = ''
    if not len(coeff):
        if n is 1:
            Eq = Eq + 'a_1x_1 = b'
        elif n is 2:
            Eq = Eq + 'a_1x_1 + a_2x_2 = b'
        else:
            Eq = Eq + 'a_1x_1 + \ldots + ' + 'a_' + str(n) + 'x_' + str(n) + '= b'
    else:
        all_zeros = len(set(coeff[:-1])) == 1 and not coeff[0]  # check if all lhs coefficients are 0
        start_put_sign = all_zeros
        if n is 1:
            Eq += "-" if coeff[0] < 0 else ""
            Eq += printMonomial(coeff[0], index=1, include_zeros=all_zeros) + "=" + printMonomial(coeff[-1])
        else:
            Eq += "-" if coeff[0] < 0 else ""
            Eq += printMonomial(coeff[0], index=1, include_zeros=all_zeros)
            start_put_sign = start_put_sign or coeff[0] is not 0
            for i in range(1, n):
                Eq += printPlusMinus(coeff[i], include_zeros=all_zeros) if start_put_sign \
                      else "-" if coeff[i] < 0 else ""
                Eq += printMonomial(coeff[i], index=i+1, include_zeros=all_zeros)
                start_put_sign = start_put_sign or coeff[i] is not 0
            Eq += "=" + printMonomial(coeff[-1])
    return Eq


def printEq(coeff, b, *args):
    """Method that prints the Latex string of a linear equation, given the values of the coefficients. If no coefficient
     value is provided, then a symbolic equation with `n` unknowns is plotted. In particular:

        * **SYMBOLIC EQUATION**: if the number of unknowns is either 1 or 2, then all the equation is
          displayed while, if the number of unknowns is higher than 2, only the first and last term of the equation
          are displayed
        * **NUMERICAL EQUATION**: whichever the number of unknowns, the whole equation is plotted. Numerical values
          of the coefficients are rounded to the third digit

    :param coeff: coefficients of the left-hand side of the linear equation
    :type: list[float]
    :param b: right-hand side coefficient of the linear equation
    :type b: float
    :param *args: optional; if passed, it contains the number of unknowns to be considered. If not passed, all the
        unknowns are considered, i.e. n equals the length of the coefficients list
    :type: *args: list
    """

    if len(args) == 1:
        n = args[0]
    else:
        n = len(coeff)
    coeff = coeff + b
    texEq = '$'
    texEq = texEq + strEq(n, coeff)
    texEq = texEq + '$'
    display(Latex(texEq))
    return


def printSyst(A, b, *args):
    """Method that prints a linear system of `n` unknowns and `m` equations. If `A` and `b` are empty, then a symbolic
    system is printed; otherwise a system containing the values of the coefficients stored in `A` and `b`, approximated
    up to their third digit is printed.

    :param A: left-hand side matrix. It must be [] if a symbolic system is desired
    :type: list[list[float]]
    :param b: right-hand side vector. It must be [] if a symbolic system is desired
    :type b: list[float]
    :param args: optional; if not empty, it is a list of two integers representing the number of equations of the
        linear system (i.e. `m`) and the number of unknowns of the system (i.e. `n`)
    :type: list
    """

    if (len(args) == 2) or (len(A) == len(b)):  # ensures that MatCoeff has proper dimensions
        if len(args) == 2:
            m = args[0]
            n = args[1]
        else:
            m = len(A)
            n = len(A[0])

        texSyst = '$\\begin{cases}'
        Eq_list = []
        if len(A) and len(b):
            if type(b[0]) is list:
                b = np.array(b).astype(float)
                A = np.concatenate((A, b), axis=1)
            else:
                A = [A[i] + [b[i]] for i in range(0, m)]  # becomes augmented matrix
            A = np.array(A)  # just in case it's not

        for i in range(m):
            if not len(A) or not len(b):
                Eq_i = ''
                if n is 1:
                    Eq_i = Eq_i + 'a_{' + str(i + 1) + '1}' + 'x_1 = b_' + str(i + 1)
                elif n is 2:
                    Eq_i = Eq_i + 'a_{' + str(i + 1) + '1}' + 'x_1 + ' + 'a_{' + str(i + 1) + '2}' + 'x_2 = b_' + str(
                        i + 1)
                else:
                    Eq_i = Eq_i + 'a_{' + str(i + 1) + '1}' + 'x_1 + \ldots +' + 'a_{' + str(i + 1) + str(
                        n) + '}' + 'x_' + str(n) + '= b_' + str(i + 1)
            else:
                Eq_i = strEq(n, A[i, :])  # attention A is (A|b)
            Eq_list.append(Eq_i)
            texSyst = texSyst + Eq_list[i] + '\\\\'
        texSyst = texSyst + '\\end{cases}$'
        display(Latex(texSyst))
    else:
        print("La matrice des coefficients n'a pas les bonnes dimensions")

    return


def texMatrix(*args):
    """Method which produces the Latex string corresponding to the input matrix.

    .. note:: if two inputs are passed, they represent A and b respectively; as a result the augmented matrix A|B is
      plotted. Otherwise, if the input is unique, just the matrix A is plotted

    :param args: input arguments; they could be either a matrix and a vector or a single matrix
    :type args: list[list] or list[numpy.ndarray]
    :return: Latex string representing the input matrix or the input matrix augmented by the input vector
    :rtype: str
    """

    if len(args) == 2:  # matrice augmentée
        if not type(args[0]) is np.ndarray:
            A = np.array(args[0]).astype(float)
        else:
            A = args[0].astype(float)
        m = A.shape[1]
        if not type(args[1]) is np.array:
            b = np.array(args[1]).astype(float)
        else:
            b = args[1].astype(float)

        A = np.concatenate((A, b), axis=1)
        texApre = '\\left[\\begin{array}{'
        texA = ''
        for i in np.asarray(A):
            texALigne = ''
            texALigne = texALigne + str(round(i[0], 4) if i[0] % 1 else int(i[0]))
            if texA == '':
                texApre = texApre + 'c'
            for j in i[1:m]:
                if texA == '':
                    texApre = texApre + 'c'
                texALigne = texALigne + ' & ' + str(round(j, 4) if j % 1 else int(j))
            if texA == '':
                texApre = texApre + '| c'
            for j in i[m:]:
                if texA == '':
                    texApre = texApre + 'c'
                texALigne = texALigne + ' & ' + str(round(j, 4) if j % 1 else int(j))
            texALigne = texALigne + ' \\\\'
            texA = texA + texALigne
        texA = texApre + '}  ' + texA[:-2] + ' \\end{array}\\right]'
    elif len(args) == 1:  # matrice des coefficients
        if not type(args[0]) is np.ndarray:
            A = np.array(args[0]).astype(float)
        else:
            A = args[0].astype(float)
        texApre = '\\left[\\begin{array}{'
        texA = ''
        for i in np.asarray(A):
            texALigne = ''
            texALigne = texALigne + str(round(i[0], 4) if i[0] % 1 else int(i[0]))
            if texA == '':
                texApre = texApre + 'c'
            for j in i[1:]:
                if texA == '':
                    texApre = texApre + 'c'
                texALigne = texALigne + ' & ' + str(round(j, 4) if j % 1 else int(j))
            texALigne = texALigne + ' \\\\'
            texA = texA + texALigne
        texA = texApre + '}  ' + texA[:-2] + ' \\end{array}\\right]'
    else:
        print("Ce n'est pas une matrice des coefficients ni une matrice augmentée")
        texA = ''
    return texA


def printA(*args):
    """Method which prints the input matrix.

    .. note:: if two inputs are passed, they represent A and b respectively; as a result the augmented matrix A|B is
      plotted. Otherwise, if the input is unique, just the matrix A is plotted

    :param args: input arguments; they could be either a matrix and a vector or a single matrix
    :type args: list
    """

    texA = '$' + texMatrix(*args) + '$'
    display(Latex(texA))
    return


def printEquMatrices(*args):
    """Method which prints the list of input matrices.

    .. note:: if two inputs are passed, they represent the list of coefficient matrices A and the list of rhs b
      respectively; as a result the augmented matrices A|B are plotted. Otherwise, if the input is unique, just the
      matrices A are plotted

    :param args: input arguments; they could be either a list of matrices and a list of vectors or
        a single list of matrices
    :type args: list
    """

    # list of matrices is M=[M1, M2, ..., Mn] where Mi=(Mi|b)
    if len(args) == 2:
        listOfMatrices = args[0]
        listOfRhS = args[1]
        texEqu = '$' + texMatrix(listOfMatrices[0], listOfRhS[0])
        for i in range(1, len(listOfMatrices)):
            texEqu = texEqu + '\\quad \\sim \\quad' + texMatrix(listOfMatrices[i], listOfRhS[i])
        texEqu = texEqu + '$'
        display(Latex(texEqu))
    else:
        listOfMatrices = args[0]
        texEqu = '$' + texMatrix(listOfMatrices[0])
        for i in range(1, len(listOfMatrices)):
            texEqu = texEqu + '\\quad \\sim \\quad' + texMatrix(listOfMatrices[i])
        texEqu = texEqu + '$'
        display(Latex(texEqu))
    return


# %% Functions to enter something

def EnterInt(n=None):
    """Function to allow the user to enter a non-negative integer

    :param n: first integer, passed to the function. If null or negative or None, an integer is requested to the user.
       Defaults to None
    :type n: int or NoneType
    :return: positive integer
    :rtype: int
    """

    while type(n) is not int or (type(n) is int and n <= 0):
        try:
            n = int(n)
            if n <= 0:
                print("Le nombre ne peut pas être négatif o zero!")
                print("Entrez à nouveau: ")
                n = input()
        except:
            if n is not None:
                print("Ce n'est pas un entier!")
                print("Entrez à nouveau:")
                n = input()
            else:
                print("Entrez un entier positif")
                n = input()
    return n


def EnterListReal(n):
    """Function which allows the user to enter a list of `n` real numbers

    :param n: number of real numbers in the desired list
    :type n: int
    :return: list of `n` real numbers
    :rtype: list[float]
    """

    if n < 0:
        print(f"Impossible de générer une liste de {n} nombres réels")
    elif n == 0:
        return []
    else:
        print(f"Entrez une liste de {n} nombres réels")
        coeff = None
        while type(coeff) is not list:
            try:
                coeff = input()
                coeff = [float(eval(x)) for x in coeff.split(',')]
                if len(coeff) != n:
                    print("Vous n'avez pas entré le bon nombre de réels!")
                    print("Entrez à nouveau : ")
                    coeff = input()
            except:
                print("Ce n'est pas le bon format!")
                print("Entrez à nouveau")
                coeff = input()
        return coeff


def SolOfEq(sol, coeff, i):
    """Method that verifies if `sol` is a solution to the linear equation `i`with coefficients `coeff`

    :param sol: candidate solution vector
    :type sol: list
    :param coeff: coefficients of the linear equation
    :type coeff: list
    :param i: index of the equation
    :type i: int
    :return: True if `sol` is a solution, False otherwise
    :rtype: bool
    """

    try:
        assert len(sol) == len(coeff)-1
    except AssertionError:
        print(f"La suite entrée n'est pas une solution de l'équation {i}; Les dimensions ne correspondent pas")
        return False

    A = np.array(coeff[:-1])
    isSol = abs(np.dot(A, sol) - coeff[-1]) < 1e-8
    if isSol:
        print(f"La suite entrée est une solution de l'équation {i}")
    else:
        print(f"La suite entrée n'est pas une solution de l'équation {i}")
    return isSol


def SolOfSyst(solution, A, b):
    """Method that verifies if `solution` is a solution to the linear system with left-hand side matrix `A` and
    right-hand side vector `b`

    :param solution: candidate solution vector
    :type solution: list
    :param A: left-hand side matrix of the linear system
    :type A: list[list[float]] or numpy.ndarray
    :param b: right-hand side vector of the linear system
    :type b: list[float] or numpy.ndarray
    :return: True if `sol` is a solution, False otherwise
    :rtype: bool
    """

    try:
        assert len(solution) == (len(A[0]) if type(A) is list else A.shape[1])
    except AssertionError:
        print(f"La suite entrée n'est pas une solution du système; Les dimensions ne correspondent pas")
        return False

    A = [A[i] + [b[i]] for i in range(0, len(A))]
    A = np.array(A)
    isSol = [SolOfEq(solution, A[i, :], i+1) for i in range(len(A))]
    if all(isSol):
        print("C'est une solution du système")
        return True
    else:
        print("Ce n'est pas une solution du système")
        return False


# PLOTS WITH PLOTLY #

def drawLine(p, d):
    """Method which allows to plot lines, points and arrows in the 2D-place or 3D-space, using plotly library

    :param p: point
    :type p: list[list[float]]
    :param d: direction vector. If made of all zeros, just the reference point is plotted; if different from 0 a line
      passing through `p` and with direction `d` is plotted
    :type d: list[list[float]]
    :return: generated plot
    """

    blue = 'rgb(51, 214, 255)'
    colors = [blue]
    colorscale = [[0.0, colors[0]],
                  [0.1, colors[0]],
                  [0.2, colors[0]],
                  [0.3, colors[0]],
                  [0.4, colors[0]],
                  [0.5, colors[0]],
                  [0.6, colors[0]],
                  [0.7, colors[0]],
                  [0.8, colors[0]],
                  [0.9, colors[0]],
                  [1.0, colors[0]]]
    vec = 0.9 * np.array(d)
    if len(p) == 2:
        data = []
        t = np.linspace(-5, 5, 51)
        s = np.linspace(0, 1, 10)
        if all(dd == [0] for dd in d):
            vector = go.Scatter(x=p[0] + s*0, y=p[1] + s*0, marker=dict(symbol=6, size=12, color=colors[0]),
                                name ='Point')
        else:
            trace = go.Scatter(x=p[0] + t * d[0], y=p[1] + t * d[1], name='Droite')
            peak = go.Scatter(x=d[0], y=d[1], marker=dict(symbol=6, size=12, color=colors[0]), showlegend=False)
            vector = go.Scatter(x=p[0] + s * d[0], y=p[1] + s * d[1], mode='lines',
                                line=dict(width=5, color=colors[0]), name='Vecteur directeur')
        zero = go.Scatter(x=t*0, y=t*0, name='Origine', marker=dict(symbol=6, size=12, color=colors[0]),
                          showlegend=False)

        data.append(vector)
        data.append(zero)
        if not all(dd == [0] for dd in d):
            data.append(trace)
            data.append(peak)

        fig = go.FigureWidget(data=data)
        plotly.offline.iplot(fig)
    elif len(p) == 3:
        data = [
            {
                'type': 'cone',
                'x': [1], 'y': vec[1], 'z': vec[2],
                'u': d[0], 'v': d[1], 'w': d[2],
                "sizemode": "absolute",
                'colorscale': colorscale,
                'sizeref': 1,
                "showscale": False,
                'hoverinfo': 'none'
            }
        ]
        t = np.linspace(-5, 5, 51)
        s = np.linspace(0, 1, 10)
        zero = go.Scatter3d(x=t*0, y=t*0, z=t*0, name='Origine', marker=dict(size=3), showlegend=False)
        if all(dd == [0] for dd in d):
            vector = go.Scatter3d(x=p[0] + s*0, y=p[1] + s*0, z=p[2] + s*0, marker=dict(size=5),
                                  name='Point')
        else:
            trace = go.Scatter3d(x=p[0] + t * d[0], y=p[1] + t * d[1], z=p[2] + t * d[2], mode='lines', name='Droite')
            vector = go.Scatter3d(x=p[0] + s * d[0], y=p[1] + s * d[1], z=p[2] + s * d[2], mode='lines',
                                  line=dict(width=5,color=colors[0], dash='solid'), name='Vecteur directeur',
                                  hoverinfo='none')
        data.append(zero)
        data.append(vector)
        if not all(dd == [0] for dd in d):
            data.append(trace)
        layout = {
            'scene': {
                'camera': {
                    'eye': {'x': -0.76, 'y': 1.8, 'z': 0.92}
                }
            }
        }
        fig = go.FigureWidget(data=data, layout=layout)
        plotly.offline.iplot(fig)
    return fig


def Plot2DSys(xL, xR, p, A, b):
    """Function for the graphical visualization of a 2D system of equations, plotting the straight lines characterizing
    the different equations appearing in the system

    :param xL: left limit of the plot in both coordinates
    :type xL: int or float
    :param xR: right limit of the plot in both coordinates
    :type xR: int or float
    :param p: number of points used to draw the straight lines
    :type p: int
    :param A: matrix of the linear system
    :type A: list[list[float]] or numpy.ndarray
    :param b: right-hand side vector of the linear system
    :type b: list[float] or numpy.ndarray
    """

    A = [A[i] + [b[i]] for i in range(0, len(A))]
    A = np.array(A)
    t = np.linspace(xL, xR, p)
    data = []
    for i in range(1, len(A) + 1):
        if (abs(A[i - 1, 1])) > abs(A[i - 1, 0]):
            # p0=[0,A[i-1,2]/A[i-1,1]]
            # p1=[1,(A[i-1,2]-A[i-1,0])/A[i-1,1]]
            trace = go.Scatter(x=t, y=(A[i-1, 2] - A[i-1, 0] * t) / A[i-1, 1], name='Droite %d' % i)
        else:
            trace = go.Scatter(x=(A[i-1, 2] - A[i-1, 1] * t) / A[i-1, 0], y=t, name='Droite %d' % i)
        data.append(trace)

    fig = go.Figure(data=data)
    plotly.offline.iplot(fig)
    return


def Plot3DSys(xL, xR, p, A, b):
    """Function for the graphical visualization of a 3D system of equations, plotting the straight lines characterizing
       the different equations appearing in the system

       :param xL: left limit of the plot in all coordinates
       :type xL: int or float
       :param xR: right limit of the plot in all coordinates
       :type xR: int or float
       :param p: number of points used to draw the straight lines
       :type p: int
       :param A: matrix of the linear system
       :type A: list[list[float]] or numpy.ndarray
       :param b: right-hand side vector of the linear system
       :type b: list[float] or numpy.ndarray
       """
    A = [A[i] + [b[i]] for i in range(0, len(A))]
    A = np.array(A)
    gr = 'rgb(102,255,102)'
    org = 'rgb(255,117,26)'
    # red = 'rgb(255,0,0)'
    blue = 'rgb(51, 214, 255)'
    colors = [blue, gr, org]
    s = np.linspace(xL, xR, p)
    t = np.linspace(xL, xR, p)
    tGrid, sGrid = np.meshgrid(s, t)
    data = []
    for i in range(len(A)):
        colorscale = [[0.0, colors[i]],
                      [0.1, colors[i]],
                      [0.2, colors[i]],
                      [0.3, colors[i]],
                      [0.4, colors[i]],
                      [0.5, colors[i]],
                      [0.6, colors[i]],
                      [0.7, colors[i]],
                      [0.8, colors[i]],
                      [0.9, colors[i]],
                      [1.0, colors[i]]]
        j = i + 1
        if (abs(A[i, 2])) > abs(A[i, 1]):  # z en fonction de x,y
            x = sGrid
            y = tGrid
            surface = go.Surface(x=x, y=y, z=(A[i, 3] - A[i, 0] * x - A[i, 1] * y) / A[i, 2],
                                 showscale=False, showlegend=True, colorscale=colorscale, opacity=1, name='Plan %d' % j)
        elif A[i, 2] == 0 and A[i, 1] == 0:  # x =b
            y = sGrid
            z = tGrid
            surface = go.Surface(x=A[i, 3] - A[i, 1] * y, y=y, z=z,
                                 showscale=False, showlegend=True, colorscale=colorscale, opacity=1, name='Plan %d' % j)
        else:  # y en fonction de x,z
            x = sGrid
            z = tGrid
            surface = go.Surface(x=x, y=(A[i, 3] - A[i, 0] * x - A[i, 2] * z) / A[i, 1], z=z,
                                 showscale=False, showlegend=True, colorscale=colorscale, opacity=1, name='Plan %d' % j)

        data.append(surface)
        layout = go.Layout(
            showlegend=True,  # not there WHY???? --> LEGEND NOT YET IMPLEMENTED FOR SURFACE OBJECTS!!
            legend=dict(orientation="h"),
            autosize=True,
            width=800,
            height=800,
            scene=go.layout.Scene(
                xaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                )
            )
        )
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)
    return


def isDiag(M):
    """Method which checks if a matrix is diagonal

    :param M: input matrix
    :type M: list[list[float]] or numpy.ndarray
    :return: True if M is diagonal else False
    :rtype: bool
    """
    if not type(M) is np.ndarray:
        M = np.array(M)

    i, j = M.shape
    try:
        assert i == j
    except AssertionError:
        print("A non-squared matrix cannot be diagonal!")
        return False

    test = M.reshape(-1)[:-1].reshape(i - 1, j + 1)
    return ~np.any(test[:, 1:])


def isSym(M):
    """Method which checks if a matrix is symmetric

    :param M: input matrix
    :type M: list[list[float]] or numpy.ndarray
    :return: True if M is symmetric else False
    :rtype: bool
    """

    if not type(M) is np.ndarray:
        M = np.array(M)

    i, j = M.shape
    try:
        assert i == j
    except AssertionError:
        print("A non-squared matrix cannot be symmetric!")
        return False

    return ~np.any(M - np.transpose(M))


# ECHELONNAGE #

def echZero(indice, M):
    """Method which sets to zero the entries of matrix M that correspond to a True value in the boolean vector indice

    :param indice: vector of booleans; if an element is True, it means that the element with the corresponding index in
       matrix M must be set to 0
    :type indice: list[bool]
    :param M: matrix to be processed
    :type: numpy.ndarray
    :return: processed matrix M, where the given entries have been properly set to 0
    :rtype: numpy.ndarray
    """

    Mat = M[not indice, :].ravel()
    Mat = np.concatenate([Mat, M[indice, :].ravel()])
    Mat = Mat.reshape(len(M), len(M[0, :]))
    return Mat


def Eij(M, i, j):
    """Method to swap line `i` and line `j` of matrix `M`

    :param M: matrix to be processed
    :type M: numpy.ndarray
    :param i: first line index
    :type i: int
    :param j: second line index
    :type j: int
    :return: processed matrix, with line `i` and `j` having been swapped
    :rtype: numpy.ndarray
    """

    M = np.array(M)
    M[[i, j], :] = M[[j, i], :]
    return M


def Ealpha(M, i, alpha):
    """Method to multiply line `i` of matrix `M` by the scalar coefficient `alpha`

    :param M: matrix to be processed
    :type M: numpy.ndarray
    :param i: reference line index
    :type i: int
    :param alpha: scalar coefficient
    :type alpha: float
    :return: processed matrix, with line `i` multiplied by the scalar `alpha`
    :rtype: numpy.ndarray
    """

    M = np.array(M)
    M[i, :] = alpha * M[i, :]
    return M


def Eijalpha(M, i, j, alpha):
    """Method to add to line `i` of matrix `M` line `j` of the same matrix, multiplied by the scalar coefficient `alpha`

    :param M: matrix to be processed
    :type M: numpy.ndarray
    :param i: line to be modified
    :type i: int
    :param j: line whose multiple has tobe added to line `i`
    :type j: int
    :param alpha: scalar coefficient
    :type alpha: float
    :return: processed matrix, with line `i` being summed up with line `j` multiplied by `alpha`
    :rtype: numpy.ndarray
    """

    M = np.array(M)
    M[i, :] = M[i, :] + alpha * M[j, :]
    return M


def echelonMat(ech, *args):
    """Method to perform Gauss elimination on either the matrix of the coefficients (if `len(args)==1`) or on the
    augmented matrix (if `len(args)==2`); the elimination can be either in standard form (if `ech=='E` or in reduced
    form (if `ech=='ER'`).

    :param ech:
    :type ech:
    :param args:
    :type args:
    :return:
    :rtype:
    """
    if len(args) == 2:  # matrice augmentée
        A = np.array(args[0]).astype(float)
        m = A.shape[0]
        n = A.shape[1]
        b = args[1]
        if type(b[0]) == list:
            b = np.array(b).astype(float)
            A = np.concatenate((A, b), axis=1)
        else:
            b = [b[i] for i in range(m)]
            A = [A[i] + [b[i]] for i in range(0, m)]
    else:  # matrice coeff
        A = np.array(args[0]).astype(float)
        m = A.shape[0]
        n = A.shape[1]

    if ech == 'E':  # Echelonnée
        Mat = np.array(A)
        Mat = Mat.astype(float)  # in case the array in int instead of float.
        numPivot = 0
        for i in range(len(Mat)):
            j = i
            while all(abs(Mat[j:, i]) < 1e-15) and j != len(Mat[0, :]) - 1:  # if column (or rest of) is 0, take next
                j += 1
            if j == len(Mat[0, :]) - 1:
                if len(Mat[0, :]) > j:
                    Mat[i + 1:len(Mat), :] = 0
                break
            if abs(Mat[i, j]) < 1e-15:
                Mat[i, j] = 0
                zero = abs(Mat[i:, j]) < 1e-15
                M = echZero(zero, Mat[i:, :])
                Mat[i:, :] = M
            Mat = Ealpha(Mat, i, 1 / Mat[i, j])  # usually Mat[i,j]!=0
            for k in range(i + 1, len(A)):
                Mat = Eijalpha(Mat, k, i, -Mat[k, j])
                # Mat[k,:]=[0 if abs(Mat[k,l])<1e-15 else Mat[k,l] for l in range(len(MatCoeff[0,:]))]
            numPivot += 1
            Mat[abs(Mat) < 1e-15] = 0

        print("La matrice est sous la forme échelonnée")
        if len(args) == 2:
            printEquMatrices([A[:, :n], Mat[:, :n]], [A[:, n:], Mat[:, n:]])
        else:
            printEquMatrices([A, Mat])

    elif ech == 'ER':  # Echelonnée réduite
        Mat = np.array(A)
        Mat = Mat.astype(float)  # in case the array in int instead of float.
        numPivot = 0
        for i in range(len(Mat)):
            j = i
            while all(abs(Mat[j:, i]) < 1e-15) and j != len(
                    Mat[0, :]) - 1:  # if column (or rest of) is zero, take next column
                j += 1
            if j == len(Mat[0, :]) - 1:
                # ADD ZERO LINES BELOW!!!!!!
                if len(Mat[0, :]) > j:
                    Mat[i + 1:len(Mat), :] = 0
                break
            if abs(Mat[i, j]) < 1e-15:
                Mat[i, j] = 0
                zero = abs(Mat[i:, j]) < 1e-15
                M = echZero(zero, Mat[i:, :])
                Mat[i:, :] = M
            Mat = Ealpha(Mat, i, 1 / Mat[i, j])  # normalement Mat[i,j]!=0
            for k in range(i + 1, len(A)):
                Mat = Eijalpha(Mat, k, i, -Mat[k, j])
                # Mat[k,:]=[0 if abs(Mat[k,l])<1e-15 else Mat[k,l] for l in range(len(MatCoeff[0,:]))]
            numPivot += 1
            Mat[abs(Mat) < 1e-15] = 0

        print("La matrice est sous la forme échelonnée")
        if len(args) == 2:
            printEquMatrices([A[:, :n], Mat[:, :n]], [A[:, n:], Mat[:, n:]])
        else:
            printEquMatrices([np.asmatrix(A), np.asmatrix(Mat)])

        Mat = np.array(Mat)
        i = (len(Mat) - 1)
        while i >= 1:
            while all(
                    abs(Mat[i, :len(Mat[0]) - 1]) < 1e-15) and i != 0:  # if ligne (or rest of) is zero, take next ligne
                i -= 1
            # we have a lign with one non-nul element
            j = i  # we can start at pos ij at least the pivot is there
            if abs(Mat[i, j]) < 1e-15:  # if element Aij=0 take next one --> find pivot
                j += 1
            # Aij!=0 and Aij==1 if echelonMat worked
            for k in range(i):  # put zeros above pivot (which is 1 now)
                Mat = Eijalpha(Mat, k, i, -Mat[k, j])
            i -= 1

        print("La matrice est sous la forme échelonnée réduite")
        if len(args) == 2:
            printEquMatrices([A[:, :n], Mat[:, :n]], [A[:, n:], Mat[:, n:]])
        else:
            printEquMatrices([A, Mat])

    else:
        print(f"Méthode d'échelonnage non reconnue {ech}. Méthodes disponibles: 'E' (pour la forme échelonnée standard)"
              f", 'ER' (pour la forme échelonnée réduite))")

    return np.asmatrix(Mat)


def randomA():
    """Method which generates a random matrix with rows and columns within 1 and 10 and integer entries between -100
    and 100

    :return: generated random matrix
    :rtype: numpy.ndarray
    """
    n = random.randint(1, 10)
    m = random.randint(1, 10)
    A = [[random.randint(-100, 100) for i in range(n)] for j in range(m)]
    printA(A)
    return np.array(A)


def dimensionA(A):
    """Method which allows the user to enter the matrix dimensions and verifies whether they are correct or not

    :param A: reference matrix
    :type A: numpy.ndarray
    """
    m = widgets.IntText(
        value=1,
        step=1,
        description='m:',
        disabled=False
    )
    n = widgets.IntText(
        value=1,
        step=1,
        description='n:',
        disabled=False
    )

    display(m)
    display(n)

    def f():
        if m.value == A.shape[0] and n.value == A.shape[1]:
            print('Correcte!')
        else:
            print('Incorrecte, entrez de nouvelles valeurs')

    interact_manual(f)
    return


def manualEch(*args):
    """Method which allows the user to perform the Gauss elimination method on the given input matrix, eventually
    extended by the right-hand side vector.

    :param args:
    :type args:
    :return:
    :rtype:
    """

    if len(args) == 2:  # matrice augmentée
        A = np.array(args[0]).astype(float)
        m = A.shape[0]
        b = args[1]
        if type(b[0]) is list:
            b = np.array(b).astype(float)
            A = np.concatenate((A, b), axis=1)
        else:
            b = [b[i] for i in range(m)]
            A = [A[i] + [b[i]] for i in range(m)]
    else:
        A = np.array(args[0]).astype(float)
        m = A.shape[0]

    j = widgets.BoundedIntText(
        value=1,
        min=1,
        max=m,
        step=1,
        description='Ligne j:',
        disabled=False
    )
    i = widgets.BoundedIntText(
        value=1,
        min=1,
        max=m,
        step=1,
        description='Ligne i:',
        disabled=False
    )

    r = widgets.RadioButtons(
        options=['Eij', 'Ei(alpha)', 'Eij(alpha)', 'Revert'],
        description='Opération:',
        disabled=False
    )

    alpha = widgets.Text(
        value='1',
        description='Coeff. alpha:',
        disabled=False
    )
    print("Régler les paramètres et évaluer la cellule suivante")
    print("Répéter cela jusqu'à obtenir une forme échelonnée réduite")
    display(r)
    display(i)
    display(j)
    display(alpha)
    return i, j, r, alpha


def echelonnage(i, j, r, alpha, A, m, *args):
    """Method which performs the Gauss elimination method step described by `r.value` with parameters `ì`, `j` and
    `alpha` on matrix `A`

    :param i: first reference line
    :type i: ipywidgets.Text
    :param j: second reference line
    :type j: ipywidgets.Text
    :param r: RadioButton describing the elementary matricial operation to be performed
    :type r: ipywidgets.radioButton
    :param alpha: scalar coefficient
    :type alpha: ipywidgets.Text
    :param A: starting matrix
    :type A: numpy.ndarray
    :param m: starting augmented matrix
    :type m: numpy.ndarray
    :param args: either the list of matrices or both the list of matrices and rhs having bee built during the
       application of the methos
    :type args: list[numpy.ndarray] or tuple(list[numpy.ndarray], list[numpy.ndarray])
    :return: processed matrix
    :rtype: numpy.ndarray
    """
    m = np.array(m).astype(float)
    if alpha.value == 0 and r.value in {'Ei(alpha)', 'Eij(alpha)'}:
        print('Le coefficient alpha doit être non-nul!')

    if r.value == 'Eij':
        m = Eij(m, i.value-1, j.value-1)
    if r.value == 'Ei(alpha)':
        m = Ealpha(m, i.value-1, eval(alpha.value))
    if r.value == 'Eij(alpha)':
        m = Eijalpha(m, i.value-1, j.value-1, eval(alpha.value))

    if len(args) == 2:
        A = np.asmatrix(A)
        MatriceList = args[0]
        RhSList = args[1]
        if r.value != 'Revert':
            MatriceList.append(m[:, :A.shape[1]])
            RhSList.append(m[:, A.shape[1]:])
        else:
            if len(MatriceList) > 1 and len(RhSList) > 1:
                MatriceList.pop()
                RhSList.pop()
                mat = MatriceList[-1]
                rhs = RhSList[-1]
                m = np.concatenate((mat,rhs), axis=1)
            else:
                print("Impossible de revenir sur l'opération!")
        printEquMatrices(MatriceList, RhSList)
    elif len(args) == 1:
        MatriceList = args[0]
        if r.value != 'Revert':
            MatriceList.append(m)
        else:
            if len(MatriceList) > 1:
                MatriceList.pop()
                m = MatriceList[-1]
            else:
                print("Impossible de revenir sur l'opération!")
        printEquMatrices(MatriceList)
    else:
        print("La liste des matrices ou des matrices et des vecteurs connus doit être donnée en entrée de la fonction!")
        raise ValueError
    return m


def manualOp(*args):
    """Method which allows the user to perform elementary operations on the given input matrix, eventually extended by
    the right-hand side vector.

    :param args:
    :type args:
    :return:
    :rtype:
    """

    if len(args) == 2:  # matrice augmentée
        A = np.array(args[0]).astype(float)
        M = A.shape[0]
        b = args[1]
        if type(b[0]) is list:
            b = np.array(b).astype(float)
            A = np.concatenate((A, b), axis=1)
        else:
            b = [b[i] for i in range(M)]
            A = [A[i] + [b[i]] for i in range(M)]
    else:
        A = np.array(args[0]).astype(float)
        M = A.shape[0]
    A = np.array(A)  # just in case it's not

    i = widgets.BoundedIntText(
        value=1,
        min=1,
        max=M,
        step=1,
        description='Ligne i:',
        disabled=False
    )

    j = widgets.BoundedIntText(
        value=1,
        min=1,
        max=M,
        step=1,
        description='Ligne j:',
        disabled=False
    )

    r = widgets.RadioButtons(
        options=['Eij', 'Ei(alpha)', 'Eij(alpha)'],
        description='Opération:',
        disabled=False
    )

    alpha = widgets.Text(
        value='1',
        description='Coeff. alpha:',
        disabled=False
    )

    print("Régler les paramètres et cliquer sur RUN INTERACT pour effectuer votre opération")

    def f(r, i, j, alpha):
        m = A
        MatriceList = [A[:, :len(A[0])-1]]
        RhSList = [A[:, len(A[0])-1:]]
        if alpha == 0 and r != 'Eij':
            print('Le coefficient alpha doit être non-nul!')
        if r == 'Eij':
            m = Eij(m, i-1, j-1)
        if r == 'Ei(alpha)':
            m = Ealpha(m, i-1, eval(alpha))
        if r == 'Eij(alpha)':
            m = Eijalpha(m, i-1, j-1, eval(alpha))
        MatriceList.append(m[:, :len(A[0])-1])
        RhSList.append(m[:, len(A[0])-1:])
        printEquMatricesAug(MatriceList, RhSList)
        return

    interact_manual(f, r=r, i=i, j=j, alpha=alpha)
    return


######################################## OBSOLETE ####################################################################
def printEquMatricesAug(listOfMatrices, listOfRhS):  # list of matrices is M=[M1, M2, ..., Mn] where Mi=(Mi|b)
    texEqu = '$' + texMatrix(listOfMatrices[0], listOfRhS[0])
    for i in range(1, len(listOfMatrices)):
        texEqu = texEqu + '\\quad \\sim \\quad' + texMatrix(listOfMatrices[i], listOfRhS[i])
    texEqu = texEqu + '$'
    display(Latex(texEqu))


def echelonMatCoeff(A):  # take echelonMAt but without b.
    b = [0 for _ in range(len(A))]
    Mat = [A[i] + [b[i]] for i in range(len(A))]
    Mat = np.array(Mat)
    Mat = Mat.astype(float)  # in case the array in int instead of float.
    numPivot = 0
    for i in range(len(Mat)):
        j = i
        while all(abs(Mat[j:, i]) < 1e-15) and j != len(
                Mat[0, :]) - 1:  # if column (or rest of) is zero, take next column
            j += 1
        if j == len(Mat[0, :]) - 1:
            # ADD ZERO LINES BELOW!!!!!!
            if len(Mat[0, :]) > j:
                Mat[i + 1:len(Mat), :] = 0
            print("La matrice est sous la forme échelonnée")
            printEquMatrices(np.asmatrix(A), np.asmatrix(Mat[:, :len(A[0])]))
            break
        if abs(Mat[i, j]) < 1e-15:
            Mat[i, j] = 0
            zero = abs(Mat[i:, j]) < 1e-15
            M = echZero(zero, Mat[i:, :])
            Mat[i:, :] = M
        Mat = Ealpha(Mat, i, 1 / Mat[i, j])  # normalement Mat[i,j]!=0
        for k in range(i + 1, len(A)):
            Mat = Eijalpha(Mat, k, i, -Mat[k, j])
            # Mat[k,:]=[0 if abs(Mat[k,l])<1e-15 else Mat[k,l] for l in range(len(MatCoeff[0,:]))]
        numPivot += 1
        Mat[abs(Mat) < 1e-15] = 0
        # printA(np.asmatrix(Mat[:, :len(A[0])]))
    return np.asmatrix(Mat)


def echelonRedMat(A, b):
    Mat = echelonMat('ER', A, b)
    Mat = np.array(Mat)
    MatAugm = np.concatenate((A, b), axis=1)
    # MatAugm = [A[i]+[b[i]] for i in range(0,len(A))]
    i = (len(Mat) - 1)
    while i >= 1:
        while all(abs(Mat[i, :len(Mat[0]) - 1]) < 1e-15) and i != 0:  # if ligne (or rest of) is zero, take next ligne
            i -= 1
        # we have a lign with one non-nul element
        j = i  # we can start at pos ij at least the pivot is there
        if abs(Mat[i, j]) < 1e-15:  # if element Aij=0 take next one --> find pivot
            j += 1
        # Aij!=0 and Aij==1 if echelonMat worked
        for k in range(i):  # put zeros above pivot (which is 1 now)
            Mat = Eijalpha(Mat, k, i, -Mat[k, j])
        i -= 1
        printA(Mat)
    print("La matrice est sous la forme échelonnée réduite")
    printEquMatrices(MatAugm, Mat)
    return np.asmatrix(Mat)


def printEquMatricesOLD(listOfMatrices):  # list of matrices is M=[M1, M2, ..., Mn]
    texEqu = '$' + texMatrix(listOfMatrices[0])
    for i in range(1, len(listOfMatrices)):
        texEqu = texEqu + '\\quad \\sim \\quad' + texMatrix(listOfMatrices[i])
    texEqu = texEqu + '$'
    display(Latex(texEqu))
    return


def texMatrixAug(A, b):  # return tex expression of one matrix (A|b) where b can also be a matrix
    m = len(A[0])
    A = np.concatenate((A, b), axis=1)
    texApre = '\\left(\\begin{array}{'
    texA = ''
    for i in np.asarray(A):
        texALigne = ''
        texALigne = texALigne + str(round(i[0], 3) if i[0] % 1 else int(i[0]))
        if texA == '':
            texApre = texApre + 'c'
        for j in i[1:m]:
            if texA == '':
                texApre = texApre + 'c'
            texALigne = texALigne + ' & ' + str(round(j, 3) if j % 1 else int(j))
        if texA == '':
            texApre = texApre + '| c'
        for j in i[m:]:
            if texA == '':
                texApre = texApre + 'c'
            texALigne = texALigne + ' & ' + str(round(j, 3) if j % 1 else int(j))
        texALigne = texALigne + ' \\\\'
        texA = texA + texALigne
    texA = texApre + '}  ' + texA[:-2] + ' \\end{array}\\right)'
    return texA


def printAAug(A, b):  # Print matrix (A|b)
    texA = '$' + texMatrixAug(A, b) + '$'
    display(Latex(texA))
    return


def printEquMatricesOLD(*args):  # M=[M1, M2, ..., Mn] n>1 VERIFIED OK
    texEqu = '$' + texMatrix(args[0])
    for i in range(1, len(args)):
        texEqu = texEqu + '\\quad \\sim \\quad' + texMatrix(args[i])
    texEqu = texEqu + '$'
    display(Latex(texEqu))
    return
