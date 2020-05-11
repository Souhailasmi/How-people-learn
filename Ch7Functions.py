#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""



# Import the necessaries libraries

# Set notebook mode to work in offline
from __future__ import division
import numpy as np
from IPython.display import display, Latex, display_latex
import plotly
import plotly.graph_objs as go
import matplotlib.pylab as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
import plotly.offline as pyo
import plotly.graph_objs as go


plotly.offline.init_notebook_mode(connected=True)
from IPython.core.magic import register_cell_magic
from IPython.display import HTML
import ipywidgets as widgets
import random
from ipywidgets import interact_manual, Layout

import sympy as sp


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

## Chapter 7.1 

def red_matrix(A, i, j):
    """ Return reduced matrix (without row i and col j)"""
    row = [0, 1, 2]
    col = [0, 1, 2]
    row.remove(i-1)
    col.remove(j-1)
    return A[row, col]


def pl_mi(i,j, first=False):
    """ Return '+', '-' depending on row and col index"""
    if (-1)**(i+j)>0:
        if first:
            return ""
        else:
            return "+"
    else:
        return "-"
    
def brackets(expr):
    """Takes a sympy expression, determine if it needs parenthesis and returns a string containing latex of expr
    with or without the parenthesis."""
    expr_latex = sp.latex(expr)
    if '+' in expr_latex or '-' in expr_latex:
        return "(" + expr_latex + ")"
    else:
        return expr_latex

    
def Determinant_3x3(A, step_by_step=True ,row=True, n=1):
    """
    Step by step computation of the determinant of a 3x3 sympy matrix strating with given row/col number
    :param A: 3 by 3 sympy matrix 
    :param step_by_step: Boolean, True: print step by step derivation of det, False: print only determinant 
    :param row: True to compute determinant from row n, False to compute determinant from col n
    :param n: row or col number to compute the determinant from (int between 1 and 3)
    :return: display step by step solution for 
    """
    
    if A.shape!=(3,3):
        raise ValueError('Dimension of matrix A should be 3x3. The input A must be a sp.Matrix of shape (3,3).')
    if n<1 or n>3 or not isinstance(n, int):
        raise ValueError('n should be an integer between 1 and 3.')
    
    # Construct string for determinant of matrix A
    detA_s = sp.latex(A).replace('[','|').replace(']','|')
    
    # To print all the steps
    if step_by_step:

        # If we compute the determinant with row n 
        if row:
            # Matrix with row i and col j removed (red_matrix(A, i, j))
            A1 = red_matrix(A, n, 1)
            A2 = red_matrix(A, n, 2)
            A3 = red_matrix(A, n, 3)
            detA1_s = sp.latex(A1).replace('[','|').replace(']','|')

            detA2_s = sp.latex(A2).replace('[','|').replace(']','|')
            detA3_s = sp.latex(A3).replace('[','|').replace(']','|')

            line1 = "$" + detA_s + ' = ' + pl_mi(n,1, True) + sp.latex(A[n-1, 0])  + detA1_s + pl_mi(n,2) + \
                    sp.latex(A[n-1, 1]) + detA2_s + pl_mi(n,3) + sp.latex(A[n-1, 2]) + detA3_s + '$'

            line2 = '$' + detA_s + ' = ' + pl_mi(n,1, True) + sp.latex(A[n-1, 0]) + "\cdot (" + sp.latex(sp.det(A1)) \
                    +")" + pl_mi(n,2) + sp.latex(A[n-1, 1]) + "\cdot (" +  sp.latex(sp.det(A2)) + ")"+ \
                    pl_mi(n,3) + sp.latex(A[n-1, 2]) + "\cdot (" + sp.latex(sp.det(A3)) + ')$'
            line3 = '$' + detA_s + ' = ' + sp.latex(sp.simplify(sp.det(A))) + '$'

        # If we compute the determinant with col n 
        else:
            # Matrix with row i and col j removed (red_matrix(A, i, j))
            A1 = red_matrix(A, 1, n)
            A2 = red_matrix(A, 2, n)
            A3 = red_matrix(A, 3, n)
            detA1_s = sp.latex(A1).replace('[','|').replace(']','|')
            detA2_s = sp.latex(A2).replace('[','|').replace(']','|')
            detA3_s = sp.latex(A3).replace('[','|').replace(']','|')

            line1 = "$" + detA_s + ' = ' + pl_mi(n,1, True) + brackets(A[0, n-1])  + detA1_s + pl_mi(n,2) + \
                    brackets(A[1, n-1]) + detA2_s + pl_mi(n,3) + brackets(A[2, n-1]) + detA3_s + '$'

            line2 = '$' + detA_s + ' = ' + pl_mi(n,1, True) + brackets(A[0, n-1]) + "\cdot (" + sp.latex(sp.det(A1))\
                    +")" + pl_mi(n,2) + brackets(A[1, n-1]) + "\cdot (" +  sp.latex(sp.det(A2)) + ")"+ \
                    pl_mi(n,3) + brackets(A[2, n-1]) + "\cdot (" + sp.latex(sp.det(A3)) + ')$'

            line3 = '$' + detA_s + ' = ' + sp.latex(sp.simplify(sp.det(A))) + '$'

        # Display step by step computation of determinant
        display(Latex(line1))
        display(Latex(line2))
        display(Latex(line3))
    # Only print the determinant without any step
    else:
        display(Latex("$" + detA_s + "=" + sp.latex(sp.det(A)) + "$"))

def Sarrus_3x3(A):
    if A.shape!=(3,3):
        raise ValueError('Dimension of matrix A should be 3x3. The input A must be a sp.Matrix of shape (3,3).')
        
    # Construct string for determinant of matrix A
    detA_s = sp.latex(A).replace('[','|').replace(']','|')
    
    A11 = sp.latex(A[0,0]); A12 = sp.latex(A[0,1]); A13 = sp.latex(A[0,2])
    A21 = sp.latex(A[1,0]); A22 = sp.latex(A[1,1]); A23 = sp.latex(A[1,2])
    A31 = sp.latex(A[2,0]); A32 = sp.latex(A[2,1]); A33 = sp.latex(A[2,2])
    
    x1 = '(' + A11 + '\cdot ' + A22 + '\cdot ' + A33 + ')'
    x2 = '(' + A12 + '\cdot ' + A23 + '\cdot ' + A31 + ')'
    x3 = '(' + A13 + '\cdot ' + A21 + '\cdot ' + A32 + ')'
    
    y1 = '(' + A13 + '\cdot ' + A22 + '\cdot ' + A31 + ')'
    y2 = '(' + A11 + '\cdot ' + A23 + '\cdot ' + A32 + ')'
    y3 = '(' + A12 + '\cdot ' + A21 + '\cdot ' + A33 + ')'
    
    prod1 = A[0,0] * A[1,1] * A[2,2]; 
    prod2 = A[0,1] * A[1,2] * A[2,0];
    prod3 = A[0,2] * A[1,0] * A[2,1];
    
    prod4 = A[0,2] * A[1,1] * A[2,0]; 
    prod5 = A[0,0] * A[1,2] * A[2,1]; 
    prod6 = A[0,1] * A[1,0] * A[2,2]; 
    
    display(Latex('$' + detA_s + ' = ' + x1 + "+" + x2 + "+" + x3 + "-" + y1 + "-" + y2 + "-" + y3 + '$'  ))
    display(Latex('$' + detA_s + ' = ' + sp.latex(sp.simplify(prod1)) + " + " + sp.latex(sp.simplify(prod2)) + " + " \
                  + sp.latex(sp.simplify(prod3)) + " - " + sp.latex(sp.simplify(prod4)) + " - "\
                  + sp.latex(sp.simplify(prod5)) + " - " + sp.latex(sp.simplify(prod6)) + '$'))
    display(Latex('$' + detA_s + ' = ' + sp.latex(sp.simplify(prod1+prod2+prod3)) + ' - ' + sp.latex(sp.simplify(prod4+prod5+prod6)) + '$'))
    
    display(Latex('$' + detA_s + ' = ' + sp.latex(sp.simplify(sp.det(A))) + '$'))
            

## 7.4
def question7_4_solution(reponse):
    if np.abs(reponse - 2/7) > 0.01:
        display("La solution est donnée ci desous.")
        A = sp.Matrix([[3,-2,7], [1,-1,4], [2,5,-6]])
        B = sp.Matrix([[4, -3, 0],[9, -4, -1],[-7, 1, 1]])
        Determinant_3x3(A, step_by_step = True)
        Determinant_3x3(B, step_by_step = True)
        display(Latex("$$ \det C = \det B \det A^{-1} = \det B \left( \\frac{1}{\det A} \\right) = \\frac{\det B}{\det A} = \\frac{-6}{-21} = \\frac{2}{7} $$"))
    else:
        display("Bravo! Vous avez trouvé la réponse")
        

## 7.5
def plotDeterminant3D(A):
    """
    This function is used to plot a 3D representation of a determinant as the volume
    of a parallelopiped. This gets the vertices and sends it to another function below to plot.
    """
    # Will only execute if it is 3x3
    if (np.shape(A) != (3,3)):
        print('Le matrice A doit être 3x3.')
        return
    
    # This creates the parallelopiped coordonates
    cube = np.array([[-1, -1, -1],
                  [1, -1, -1 ],
                  [1, 1, -1],
                  [-1, 1, -1],
                  [-1, -1, 1],
                  [1, -1, 1 ],
                  [1, 1, 1],
                  [-1, 1, 1]])
    vertices = np.zeros((8,3))
    for i in range(8):
        vertices[i,:] = np.dot(cube[i,:], A)
    
    data_x = vertices[:,0]
    data_y = vertices[:,1]
    data_z = vertices[:,2]
    
    vol = np.abs(np.linalg.det(A))
    vol = np.round(vol, decimals = 3) 
    plot3d(data_x, data_y, data_z, vol) 
    
    
    
def plot3d(data_x, data_y, data_z, vol):  
    """
    This function plots an interactive 3D plot of the determinant as a volume of a 
    parallelopiped
    """
    fig = go.Figure(
    data = [
        go.Mesh3d(
            x = data_x,
            y = data_y,
            z = data_z,
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], # These are needed, numbers from documentation
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            colorscale=[[0, 'darkblue'],
                    [0.5, 'lightskyblue'],
                    [1, 'darkblue']],
            intensity = np.linspace(0, 1, 8, endpoint=True),
            showscale=False,
            opacity = 0.6
        )
    ],
    layout = go.Layout(
        title = "Le volume est: " + str(vol),
        autosize = True
        )
    )

    # This prints it
    pyo.iplot(fig, filename='Determinant-Volume')

def plotDeterminant2D(A):
    # See; https://stackoverflow.com/questions/44881885/python-draw-parallelepiped
    """
    This function creates a 2D plot of the area of a determinant for a 2x2 matrix. 
    """
    
    # Will only execute if it is 2x2
    if (np.shape(A) != (2,2)):
        print('Le matrice A doit être 2x2.')
        return
    
    # Define vertices for a cube to multiply with input matrix A to get parallelopiped
    rect = np.array([[1, -1],
                     [1, 1],
                     [-1, 1],
                     [-1, -1]])
    
    vertices = np.zeros((5,2))
    
    for i in range(4): vertices[i,:] = np.dot(rect[i,:], A)
    vertices[4,0] = vertices[0,0]
    vertices[4,1] = vertices[0,1]
    
        # Create figure / grid to plot
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
  
    # Plot vertices
    ax.plot(vertices[:,0], vertices[:,1])
    vol = np.abs(np.linalg.det(A)) # absolute value of the determinant
    vol = np.round(vol, decimals = 3)
    print("L'aire est:", vol)

    coll = PolyCollection([vertices])
    ax.add_collection(coll)
    ax.autoscale_view()
    plt.grid()
    plt.show()

    