import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y):
    """
    Grafica los puntos X (compuesto por x1 y x2) en una figura. Se grafica los
    datos positivos (1) con triángulos amarillos, y los negativos (0) con
    círculos verdes.
    
    Argumentos
    ----------
        X - Matriz (2,n) que contiene cada una de las n instancias como columnas.
            Solo se grafica los dos primeros atributos (dos primeras filas)
        y - Vector (1,n) que contiene las clases de las instancias
    
    """
    pos = np.where(y.flatten()==1)[0]
    neg = np.where(y.flatten()==0)[0]
    Xpos = X[:,pos]; Xneg = X[:,neg];
    plt.figure(figsize=(6,6))
    plt.plot(Xpos[0,:], Xpos[1,:],'y^',label='Clase 1')
    plt.plot(Xneg[0,:], Xneg[1,:],'go',label='Clase 0')
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    plt.legend(); plt.axis('scaled')


def  normalizar(X):
    """
    Normaliza cada atributo de X usando la media y la desviación estándar
    
    Argumentos
    ----------
        X - Matriz (d,n) que contiene todos cada una de las n instancias como columnas. Se considera
            que cada instancia tiene d atributos.
    
    Retorna
    -------
        Xnorm - Matriz (d,n) que contiene cada atributo normalizado.
          mu  - Vector de tamaño (d,1) que contiene las medias de cada atributo 
        sigma - Vector de tamaño (d,1) que contiene las desviaciones estándar de cada atributo
        
    """
    Xnorm = X.copy()
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, keepdims=True)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma


def generacion_bases(X, grado=6):
    """
    Genera bases polinomiales x1, x2, x1^2, x1*x2, x2^2, ... hasta un cierto grado

    Argumentos
    ----------
          X - Matriz de tamaño (2,n) donde n es el número de instancias
      grado - grado de las bases polinomiales

    Retorna
    -------
       Xout - Matriz de tamaño (2+m, n) donde se añade m filas según el grado

    """
    X1 = X[0,:]; X2 = X[1,:]
    res = []
    for i in range(1, grado + 1):
        for j in range(i + 1):
            res.append((X1 ** (i - j)) * (X2 ** j))
    return np.array(res)


def plot_frontera(w0, w):
    """
    Grafica la frontera de decisión definida por w y w0

    Argumentos
    ----------
        w0 - Bias del modelo
         w - Vector (d,1) que contiene los parámetros del modelo (w1, w2, ... wd)

    """
    # Rango de las celdas
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    # Celdas
    z = np.zeros((u.size, v.size))
    # Evaluación de cada una de las celdas
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            base = generacion_bases(np.array([[ui], [vj]]))
            z[i,j] = np.dot(w.T, base) + w0
            
    z = z.T
    # Gráfico de z = 0
    plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
    plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)