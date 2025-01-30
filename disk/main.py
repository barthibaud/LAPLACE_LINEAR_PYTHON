import numpy as np
import math
import matplotlib.pyplot as plt
import time

##############################################################################
#           Fonctions de base pour la BEM (déjà décrites précédemment)
##############################################################################

def G_2D(x, y):
    """
    Calcule la fonction de Green pour l'équation de Laplace en 2D.
    G(x,y) = -1/(2π) * ln(||x - y||).
    x, y : tableaux numpy de taille (2,) représentant des points en 2D
    """
    r = np.linalg.norm(x - y)
    return -1.0/(2.0*np.pi) * np.log(r + 1e-15)  # petit epsilon pour éviter log(0)

def dGdn_2D(x, y, ny):
    """
    Calcule la dérivée normale de la fonction de Green par rapport à la variable 'y'.
    (∂/∂n_y) G(x,y) = ∇_y G(x,y) · ny.

    x, y : tableaux numpy (2,)
    ny   : tableau numpy (2,) la normale sortante au point y

    ∇_y G(x,y) = 1/(2π) * ((x - y)/||x - y||^2).
    => dGdn_2D = [ (x - y)·ny ] / [2π * ||x - y||^2 ].
    """
    r_vec = x - y
    r = np.linalg.norm(r_vec)
    dot = r_vec.dot(ny)
    return -1.0/(2.0*np.pi) * (dot / (r**2 + 1e-30))

def gauss_legendre_1D(n):
    """
    Retourne les (points, poids) de Gauss-Legendre sur l'intervalle [-1,1].
    n = ordre (2,3,4,...)
    """
    if n == 2:
        x = np.array([-0.57735026919, 0.57735026919])
        w = np.array([1.0, 1.0])
    elif n == 3:
        x = np.array([-0.77459666924, 0.0, 0.77459666924])
        w = np.array([0.55555555556, 0.88888888889, 0.55555555556])
    elif n == 4:
        x = np.array([-0.86113631159, -0.33998104358, 
                       0.33998104358,  0.86113631159])
        w = np.array([0.34785484514, 0.65214515486, 
                      0.65214515486, 0.34785484514])
    elif n == 8:
        x = np.array([
            -0.9602898565,
            -0.7966664774,
            -0.5255324099,
            -0.1834346425,
             0.1834346425,
             0.5255324099,
             0.7966664774,
             0.9602898565
        ])
        w = np.array([
            0.1012285363,
            0.2223810345,
            0.3137066459,
            0.3626837834,
            0.3626837834,
            0.3137066459,
            0.2223810345,
            0.1012285363
        ])
    else:
        raise ValueError("Implémentez d'autres ordres si besoin.")

    return x, w

def build_circle_geometry(N, radius):
    """
    Construit un maillage de N points sur le cercle de rayon 'radius'.
    Retourne:
      coords : tableau (N,2) contenant les coordonnées (x_j,y_j) du bord
      normals: tableau (N,2) contenant la normale sortante en (x_j,y_j)
      ds     : la longueur d'arc (constante) d'un segment (élément)
    """
    coords = np.zeros((N, 2))
    normals = np.zeros((N, 2))

    dtheta = 2.0*np.pi/N
    for j in range(N):
        theta = dtheta*j+0.5*dtheta
        xj = radius * np.cos(theta)
        yj = radius * np.sin(theta)
        coords[j,:] = [xj, yj]
        # Normale sortante (pour un cercle centré à l'origine)
        normals[j,:] = [np.cos(theta), np.sin(theta)]

    # Longueur d’arc approximative par élément
    ds = radius * dtheta

    return coords, normals, ds

def build_matrices_gauss(coords, normals, gauss_order=8):
    """
    Construit les matrices G et H pour la formulation BEM, en utilisant
    une quadrature de Gauss-Legendre sur chaque segment du maillage.

    - coords   : (N,2) points du bord, indexés par k
    - normals  : (N,2) normales sortantes en ces points (utile si on interpole la normale)
    - gauss_order : nombre de points de Gauss-Legendre par segment

    Retourne:
       G, H (matrices NxN).

    Remarque : On suppose que coords[k+1] (mod N) relie le segment k.
               On paramètre s(t) = x_k + t*(x_{k+1}-x_k), t in [0,1].
               On calcule la normale s(t)/||s(t)|| pour un cercle
               (ou un interpolé plus précis si la géométrie n'est pas un cercle).
    """
    from math import sqrt

    N = coords.shape[0]
    G = np.zeros((N, N))
    H = np.zeros((N, N))

    # Récupération des points et poids Gauss-Legendre sur [-1,1]
    xi_gauss, w_gauss = gauss_legendre_1D(gauss_order)

    for k in range(N):
        # Segment reliant coords[k] à coords[k+1 (mod N)]
        kp1 = (k+1) % N
        xA = coords[k]
        xB = coords[kp1]

        # Longueur du segment
        vecAB = (xB - xA)
        Lk = np.linalg.norm(vecAB)

        # Pour chaque collocation j (chaque ligne)
        for j in range(N):
            x_j = coords[j]
            x_jp1 = coords[(j+1) % N]
            x_m = 0.5*(x_j + x_jp1)  # milieu du segment

            # --- G_{j,k} et H_{j,k} ---
            # Intégration de G_2D(x_j, s(t)) ou dGdn_2D(x_j, s(t)) sur t in [0,1]

            # Cas diagonal: on force H_{j,j} = 0 => plus tard on ajoutera +0.5*phi_j dans le système.
            if j == k:
                # S'il s'agit du même index (strictement) => on met 0.
                # (NB: il y aurait la question de "j est dans le segment k?" 
                #  Dans un BEM plus fin, on regarde la near-singularity quand x_j est "dans" le segment k.
                #  Pour la démo, on fait simple.)
                G[j,k] = 0.0
                H[j,k] = 0.0
                continue

            # Sinon on fait la quadrature sur le segment
            sum_g = 0.0
            sum_h = 0.0

            for ig in range(gauss_order):
                xi = xi_gauss[ig]   # point de référence dans [-1,1]
                w  = w_gauss[ig]    # poids associé

                # Transformation : t in [0,1], t = (xi+1)/2
                t = 0.5*(xi + 1.0)

                # Point d'intégration s(t)
                s_t = xA + t*vecAB   # vecteur 2D
                # Normale sortante. Pour un cercle centré à l'origine,
                # on peut prendre n(t) = s_t / ||s_t||. 
                # (Pour une géométrie plus générale, on interpolerait la normale.)
                n_seg = [-vecAB[1], vecAB[0]]  # normal au segment
                r_s = np.linalg.norm(s_t)
                if r_s < 1e-14:
                    # Evite la division par zéro
                    n_t = normals[k]  # fallback
                else:
                    n_t = s_t / r_s

                # Valeurs de la fonction de Green et sa dérivée normale
                # g_val = G_2D(x_j, s_t)
                # dg_val = dGdn_2D(x_j, s_t, -n_t)
                g_val = G_2D(x_m, s_t)
                dg_val = dGdn_2D(x_m, s_t, n_seg)

                # On accumule
                sum_g += g_val * w
                sum_h += dg_val * w

            # Facteur du changement de variable:
            #  - On est passé de t in [0,1] => xi in [-1,1], Jacobien = (Lk * 0.5).
            #  - En paramètre direct on a s(t) = xA + t*(xB - xA), dΓ = Lk dt.
            #  - Avec la transfo Gauss, dt = (xi+1)/2 => J = 0.5. 
            # => Au total, dΓ = Lk dt => Lk * (0.5) quand on multiplie par w.
            # => sum = sum * (Lk/2).
            G[j,k] = sum_g * (0.5 * Lk)
            H[j,k] = sum_h * (0.5 * Lk)

    return G, H

def solve_laplace_dirichlet_bem(N):
    """
    Résout le problème de Laplace dans le disque unité avec condition
    Dirichlet phi(x,y)=x sur le bord. Utilise la formulation BEM directe.

    Retourne:
      q : dérivée normale (flux) en chaque point du bord (taille N)
      coords : coordonnées (N,2) des points du bord
      normals: normales (N,2) associées
      ds     : longueur d'arc élémentaire
      phi_bd : la condition de Dirichlet imposée (phi_j = x_j)
    """
    # 1) Géométrie
    rad = 1.0

    coords, normals, ds = build_circle_geometry(N, rad)

    # 2) Matrices G et H
    start = time.process_time()
    G, H = build_matrices_gauss(coords, normals, gauss_order=8)
    end = time.process_time()
    print(f"{N} elem build time:", end - start)

    # 3) Condition Dirichlet: phi_j = x_j
    phi_bd = coords[:,0]  # x_j

    # 4) Assemblage du système:
    #    0.5*phi_j = sum_k( H[j,k]*phi_k ) - sum_k( G[j,k]*q_k )
    # => G q = H phi - 0.5 phi
    RHS = np.dot(H, phi_bd) - 0.5*phi_bd

    # 5) On résout pour q
    start = time.process_time()
    Ginv = np.linalg.inv(G)

    q = np.matmul(Ginv, RHS)
    end = time.process_time()
    print(f"{N} elem solving time:", end - start)

    return q, coords, normals, ds, phi_bd

##############################################################################
#       Nouvelle fonction : reconstruction de la solution à l'intérieur
##############################################################################

def compute_interior_solution(coords, normals, ds, phi_bd, q_bd, Nx=50, Ny=50):
    """
    Calcule la solution phi(x,y) à l'intérieur du disque en un maillage (Nx x Ny).
    On effectue la représentation intégrale :
      phi(x) = ∫ dGdn_2D(x,s)*phi(s) dΓ(s) - ∫ G_2D(x,s)*q(s) dΓ(s)

    coords : (N,2) points du bord
    normals: (N,2) normales sortantes
    ds     : longueur d'arc (constante) des éléments
    phi_bd : valeur de phi sur le bord (Dirichlet)
    q_bd   : valeur de la dérivée normale sur le bord
    Nx,Ny  : taille du maillage dans le carré [-1,1]x[-1,1]

    Retourne:
       X, Y        : grilles 2D (shape Nx x Ny) des coordonnées
       phi_interior: solution reconstruite (même shape), np.nan en dehors du disque
    """
    # Maillage régulier dans le carré [-1,1] x [-1,1]
    x_vals = np.linspace(-1.0, 1.0, Nx)
    y_vals = np.linspace(-1.0, 1.0, Ny)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')

    N = coords.shape[0]
    phi_interior = np.full_like(X, np.nan, dtype=float)  # on met NaN par défaut

    for i in range(Nx):
        for j in range(Ny):
            xx = X[i,j]
            yy = Y[i,j]
            # On ne calcule que si (xx,yy) est à l'intérieur du cercle unité
            if xx*xx + yy*yy <= 1.0:
                x_pt = np.array([xx, yy])

                # Représentation intégrale
                s1 = 0.0  # correspond à l'intégrale de dGdn_2D * phi_bd
                s2 = 0.0  # correspond à l'intégrale de G_2D * q_bd
                for k in range(N):
                    x_k = coords[k,:]
                    n_k = normals[k,:]
                    # intégration élément constant
                    g_val = G_2D(x_pt, x_k)
                    dgdn_val = dGdn_2D(x_pt, x_k, n_k)

                    s1 += dgdn_val * phi_bd[k]
                    s2 += g_val    * q_bd[k]

                phi_interior[i,j] = (s1 - s2) * ds

    return X, Y, phi_interior

def dphi_error_on_bc(q, coords, N):
    """
    Calcul de l'erreur de la dérivée normale sur le bord.
    q : dérivée normale numérique (solution BEM)
    coords : coordonnées des points du bord
    N : nombre de points du bord
    """
    linf = 0.0
    l2 = 0.0
    for j in range(N):
        x_j = coords[j,:]
        x_jp1 = coords[(j+1) % N,:]
        x_m = 0.5*(x_j + x_jp1)
        vecAB = (x_jp1 - x_j)
        lj = np.linalg.norm(vecAB)
        n_x_m = [-vecAB[1]/lj, vecAB[0]/lj]  # normale au segment dirigée vers l'intérieur
        q_num = q[j]
        dphi_exact = [1.0,0.0]
        q_exact = (dphi_exact[0]*n_x_m[0]+dphi_exact[1]*n_x_m[1])
        linf = np.max([linf,(q_num - q_exact)])
        l2 += (q_num - q_exact)**2

    return linf,np.sqrt(l2)/np.linalg.norm(dphi_exact)/N


##############################################################################
#        Exemple d'utilisation: plot solution num. vs solution exacte
##############################################################################

if __name__ == "__main__":
    # 1) On résout le problème BEM sur le disque

    errs = []
    times = []
    fig, axes = plt.subplots(4, 3, figsize=(18,24))
    i = 0
    for N in [10, 50, 100, 500]:
        start = time.process_time()
        q_bd, coords, normals, ds, phi_bd = solve_laplace_dirichlet_bem(N)
        end = time.process_time()
        times.append(end - start)
        linf, l2 = dphi_error_on_bc(q_bd, coords, N)
        print(f"{N} elem erreur L1 sur le bord: {l2}")
        errs.append(l2)

        # calcul de l'érreur sur le bord

        # 2) On reconstruit la solution à l'intérieur
        Nx, Ny = 100, 100  # taille du maillage pour la visualisation
        start = time.process_time()
        X, Y, phi_num = compute_interior_solution(coords, normals, ds, phi_bd, q_bd, Nx, Ny)
        end = time.process_time()
        print(f"{N} elem reconstruction time:", end - start)

        # 3) Solution exacte: phi(x,y) = x
        #    On la stocke dans un tableau de même dimension, en mettant NaN en dehors
        phi_exact = np.full_like(phi_num, np.nan)
        mask_in = (X**2 + Y**2 <= 1.0)
        phi_exact[mask_in] = X[mask_in]  # car phi_exact(x,y) = x

        # 4) Erreur
        error_map = np.abs(phi_num - phi_exact)/np.abs(phi_exact)

        # 5) Affichage côte à côte


        # (a) Solution numérique
        pcm1 = axes[i,0].pcolormesh(X, Y, phi_num, shading='auto', cmap='jet', vmin=-1, vmax=1)
        #pcm1 = axes[0].pcolormesh(X, Y, phi_num, shading='auto', cmap='jet')
        axes[i,0].set_title("Solution numérique (BEM)")
        axes[i,0].set_aspect('equal')
        plt.colorbar(pcm1, ax=axes[0])

        # (b) Solution exacte
        pcm2 = axes[i,1].pcolormesh(X, Y, phi_exact, shading='auto', cmap='jet')
        axes[i,1].set_title("Solution exacte (x)")
        axes[i,1].set_aspect('equal')
        plt.colorbar(pcm2, ax=axes[1])

        # (c) Carte d'erreur
        pcm3 = axes[i,2].pcolormesh(X, Y, error_map*100, shading='auto', cmap='jet',vmin=0, vmax=10)
        axes[i,2].set_title("Erreur (num - exact)")
        axes[i,2].set_aspect('equal')
        plt.colorbar(pcm3, ax=axes[2])

        i += 1

    #plt.tight_layout()
    plt.savefig(f"laplace_bem_solutions.png")

    fig, ax = plt.subplots(1, 2, figsize=(18,6))
    ax[0].plot([10, 50, 100, 500], errs, label="L1 dphi_dn", marker='x')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("Erreur L1")
    ax[0].legend()

    ax[1].plot([10, 50, 100, 500], times, label="temps de calcul")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("Temps de calcul")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f"laplace_bem_errors.png")