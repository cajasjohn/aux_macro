###############################################
### FUNCIONES AUXILIARES PARA MACROECONOMIA ###
###############################################

####################
### Preliminares ###
####################
from __future__ import annotations
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from itertools import cycle, combinations
from contextlib import contextmanager
from matplotlib.collections import LineCollection
#from typing import Callable, Dict, Iterable, Tuple, Union, Optional
from scipy.integrate import solve_ivp


#####################################################################
### Función para obtener derivadas en ambos lados de una ecuación ###
#####################################################################
def diff_eq(eq, var, n=1):
    if isinstance(eq, sp.Equality):
        d_izq = sp.simplify(sp.diff(eq.lhs, var, n))
        d_der = sp.simplify(sp.diff(eq.rhs, var, n))
        return sp.Eq(d_izq, d_der)
    else:
        return sp.simplify(sp.diff(eq, var, n))


##############################################################################
### Función para aplicar log-diferenciación en ambos lados de una ecuación ###
##############################################################################
def diff_ln_eq(eq, var, n=1):
    if isinstance(eq, sp.Equality):
        d_ln_izq = sp.simplify(sp.diff(sp.log(eq.lhs),var,n))
        d_ln_der = sp.simplify(sp.diff(sp.log(eq.rhs),var,n))   
        return sp.Eq(d_ln_izq, d_ln_der)
    else:
        return sp.simplify(sp.diff(sp.log(eq),var,n))


#################################################
### Función para pasar de "puntos" a "gorros" ###
#################################################
def dot_to_hat(expr, var_list, t, target_idx=0, do_simplify=True):
    # Construir funciones "gorro" \hat{v}(t)
    hat = {v: sp.Function(rf'\hat{{{v.name}}}') for v in var_list}

    # ----- CASO 1: expr es ecuación -> comportamiento original -----
    if isinstance(expr, sp.Equality):
        sistema = [expr]
        variables = [hat[var_list[target_idx]](t)]  # incógnita principal: \hat{var_target}(t)
        for j in var_list:
            sistema.append(sp.Eq(hat[j](t), sp.diff(j, t)/j))  # \hat{j} = \dot{j}/j
            variables.append(sp.diff(j, t))                    # y las \dot{j} como incógnitas auxiliares

        sol = sp.solve(sistema, variables, dict=True)
        if not sol:
            raise ValueError("No se pudo resolver el sistema para la ecuación dada.")
        rhs = sol[0][variables[0]]
        return sp.Eq(variables[0], sp.simplify(rhs) if do_simplify else rhs)

    # ----- CASO 2: expr es término -> sustitución directa -----
    # Reemplaza d v/dt -> \hat{v}(t) * v
    repl = {sp.diff(v, t): hat[v](t) * v for v in var_list}

    out = expr
    # Solo aplicamos .subs si el objeto lo soporta (por seguridad si pasan números)
    if hasattr(out, "subs"):
        out = out.subs(repl)

    # Intentar cancelar factores v/v y simplificar
    try:
        out = sp.together(out)
        out = sp.cancel(out)
    except Exception:
        pass

    return sp.simplify(out) if do_simplify else out


#################################################
### Función para pasar de "gorros" a "puntos" ###
#################################################
def hat_to_dot(expr,var_list,t,target_idx=0):
    sistema=[expr]
    variables=[
        sp.diff(var_list[0],t)
    ]    
    for j in var_list:
        sistema.append(sp.Eq(sp.Function(rf'\hat{{{j.name}}}')(t),
                             sp.diff(j,t)/j))
        variables.append(sp.Function(rf'\hat{{{j.name}}}')(t))
    solucion=sp.solve(sistema,variables,dict=True)[0]
    return sp.Eq(variables[0],sp.simplify(solucion[variables[0]]))




#ChatGPT
######################################################################################
### Función que resuelve sistema de ecuaciones y arroja soluciones como igualdades ###
######################################################################################
def solve_as_equalities(eqs, vars=None, *, simplify=True, vector=False):
    """
    Resuelve un sistema y devuelve las soluciones como igualdades (Eq).
    
    Parámetros
    ----------
    eqs : Eq o iterable de Eq/expresiones
        Si se pasa una expresión no-Eq, se interpreta como == 0.
    vars : iterable de símbolos/variables (opcional)
        Orden deseado de incógnitas. Si es None, SymPy elige (orden no garantizado).
    simplify : bool (default True)
        Aplica sp.simplify al lado derecho de cada igualdad.
    vector : bool (default False)
        Si True, devuelve Eq(Matrix(vars), Matrix(valores)) por solución.
    
    Retorna
    -------
    list
        Lista de soluciones. Cada solución es:
          - una lista de Eq(...) (si vector=False), o
          - una Eq(Matrix(...), Matrix(...)) (si vector=True).
        Si no hay solución, retorna [].
    """
    # Normaliza a lista
    if not isinstance(eqs, (list, tuple, set)):
        eqs = [eqs]

    # Asegura que todas sean igualdades
    eqs_norm = [e if isinstance(e, sp.Equality) else sp.Eq(e, 0) for e in eqs]

    # Resuelve (dict=True para trabajar por nombre de variable)
    sol_list = (sp.solve(eqs_norm, vars, dict=True) if vars is not None
                else sp.solve(eqs_norm, dict=True))

    if not sol_list:
        return []

    def _vals_in_order(sol, keys):
        vals = []
        kept = []
        for v in keys:
            if v in sol:         # sólo las que están determinadas en esta rama
                vals.append(sp.simplify(sol[v]) if simplify else sol[v])
                kept.append(v)
        return kept, vals

    out = []
    for sol in sol_list:
        keys = list(vars) if vars is not None else list(sol.keys())
        kept, vals = _vals_in_order(sol, keys)

        if vector:
            out.append(sp.Eq(sp.Matrix(kept), sp.Matrix(vals)))
        else:
            out.append([sp.Eq(v, val) for v, val in zip(kept, vals)])
    return out    


#ChatGPT
####################################################################
### Función para gráficar funciones de producción Y=f(K,L) en 3D ###
####################################################################
def plot_prod_func_3d(eq, L_sym, K_sym, Y_sym=None, params=None,
                         L=(0.1, 100.0), K=(0.1, 100.0), n=80,
                         width=1100, height=750,
                         aspectmode='cube',          # 'cube' | 'data' | 'manual'
                         camera_eye=(1.6, 1.6, 1.2), # posición de la cámara
                         xlim=None, ylim=None, zlim=None,
                         colorscale='Magma',
                         show=True):
    """
    Grafica en 3D una ecuación SymPy (sp.Eq) que contenga L y K.
    L_sym y K_sym pueden ser símbolos (L, K) o funciones aplicadas (L(t), K(t)).
    Si Y_sym no se pasa, intento inferir la variable dependiente.
    params: dict con valores de parámetros para sustituir antes de evaluar.
    """

    if not isinstance(eq, sp.Equality):
        raise TypeError("eq debe ser una igualdad de SymPy (sp.Eq).")

    # Sustituciones de parámetros
    subs = params or {}
    eq_eval = sp.Eq(sp.simplify(eq.lhs.subs(subs)), sp.simplify(eq.rhs.subs(subs)))

    # Elegir la variable objetivo (eje z)
    target = Y_sym
    if target is None:
        # Si uno de los lados es "la salida" distinta de L y K, úsalo
        if (eq_eval.lhs != L_sym and eq_eval.lhs != K_sym) and eq_eval.lhs.has(L_sym, K_sym):
            target = eq_eval.lhs
        elif (eq_eval.rhs != L_sym and eq_eval.rhs != K_sym) and eq_eval.rhs.has(L_sym, K_sym):
            target = eq_eval.rhs
        else:
            # Último recurso: primer símbolo distinto de L y K
            cand = [s for s in eq_eval.atoms(sp.Symbol) if s not in {L_sym, K_sym}]
            if not cand:
                raise ValueError("No pude inferir la variable dependiente; pasa Y_sym=...")
            target = sorted(cand, key=lambda s: s.sort_key())[0]

    # Aislar Z(L,K)
    if eq_eval.lhs == target:
        Z_expr = sp.simplify(eq_eval.rhs)
    elif eq_eval.rhs == target:
        Z_expr = sp.simplify(eq_eval.lhs)
    else:
        sols = sp.solve(eq_eval, target, dict=False)
        if not sols:
            raise ValueError(f"No se pudo aislar {target} en términos de {L_sym} y {K_sym}.")
        Z_expr = sp.simplify(sols[0])

    # Sustituir L(t), K(t) por símbolos mudos para lambdify
    _L, _K = sp.symbols('_L _K', real=True)
    Z_num = sp.simplify(Z_expr.subs({L_sym: _L, K_sym: _K}))
    Z_fun = sp.lambdify((_L, _K), Z_num, modules='numpy')

    # Malla
    Lv = np.linspace(*L, n)
    Kv = np.linspace(*K, n)
    Lm, Km = np.meshgrid(Lv, Kv, indexing='ij')

    # Superficie Z(L,K)
    Z = Z_fun(Lm, Km)

    # Superficie con contornos proyectados al “piso”
    surf = go.Surface(
        x=Lm, y=Km, z=Z, colorscale=colorscale, showscale=True,
        contours=dict(z=dict(show=True, usecolormap=True,
                             highlightcolor="white", project_z=True))
    )
    fig = go.Figure(data=[surf])

    # Títulos de ejes (muestran L(t), K(t) si los pasas como funciones)
    #xlab = sp.latex(L_sym) if isinstance(L_sym, sp.Basic) else str(L_sym)
    #ylab = sp.latex(K_sym) if isinstance(K_sym, sp.Basic) else str(K_sym)
    #zlab = sp.latex(target) if isinstance(target, sp.Basic) else str(target)

    # Layout y límites
    #scene_kwargs = dict(xaxis_title=xlab, yaxis_title=ylab, zaxis_title=zlab, aspectmode=aspectmode)
    scene_kwargs = dict(xaxis_title="L", yaxis_title="K", zaxis_title="Y", aspectmode=aspectmode)

    if xlim is not None: scene_kwargs['xaxis'] = dict(range=list(xlim))
    if ylim is not None: scene_kwargs['yaxis'] = dict(range=list(ylim))
    if zlim is not None: scene_kwargs['zaxis'] = dict(range=list(zlim))

    fig.update_layout(width=width, height=height, margin=dict(l=0, r=0, t=40, b=0), scene=scene_kwargs)

    # Cámara
    if camera_eye is not None:
        fig.update_layout(scene_camera=dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2])))

    if show:
        fig.show()
    # return fig


#ChatGPT
########################################
### Función para graficar isocuantas ###
########################################
def plot_isoquants_from_eq(eq, L_sym, K_sym, Y_sym, *,
                           params=None,
                           Y_values=(1.0,),
                           L_range=(0.0, 100.0),
                           K_range=(0.0, 100.0),
                           n=500,
                           xlim=None, ylim=None,
                           loglog=False,
                           show=True,
                           eps=1e-8):
    """
    Dibuja isocuantas K(L;Y) a partir de una sp.Eq con L, K y Y.
    - Maneja Leontief (Min/Piecewise) con “L” y codo.
    - Caso general vía contorno G(L,K;Y)=0.
    - Leyenda consistente con color por curva.
    """

    if not isinstance(eq, sp.Equality):
        raise TypeError("eq debe ser una igualdad de SymPy (sp.Eq).")

    # Sustitución de parámetros
    subs = params or {}
    eq_eval = sp.Eq(sp.simplify(eq.lhs.subs(subs)),
                    sp.simplify(eq.rhs.subs(subs)))

    # Reemplazo de L(t),K(t),Y(t) por símbolos mudos
    _L, _K, _Y = sp.symbols('_L _K _Y', real=True)
    repl = {L_sym: _L, K_sym: _K, Y_sym: _Y}
    lhs = sp.simplify(eq_eval.lhs.xreplace(repl))
    rhs = sp.simplify(eq_eval.rhs.xreplace(repl))

    # Residual G(L,K;Y)=0 (para contornos en el caso general)
    G = sp.simplify(lhs - rhs)
    G_fun = sp.lambdify((_L, _K, _Y), G, modules='numpy')

    # Detectar Leontief: Y = Min(f(L), g(K)) (o al revés)
    def leontief_parts():
        side = None
        if eq_eval.lhs == Y_sym and isinstance(eq_eval.rhs, sp.Min):
            side = eq_eval.rhs
        elif eq_eval.rhs == Y_sym and isinstance(eq_eval.lhs, sp.Min):
            side = eq_eval.lhs
        if side is None:
            return None, None
        a, b = side.xreplace(repl).args
        cond_a = a.has(_L) and (not a.has(_K)) and (not a.has(_Y))
        cond_b = b.has(_K) and (not a.has(_L)) and (not a.has(_Y))
        if cond_a and cond_b:
            return a, b
        cond_a2 = a.has(_K) and (not a.has(_L)) and (not a.has(_Y))
        cond_b2 = b.has(_L) and (not b.has(_K)) and (not b.has(_Y))
        if cond_a2 and cond_b2:
            return b, a
        return None, None

    fL, gK = leontief_parts()

    # Figura
    fig, ax = plt.subplots()
    # try:
    #     ax.set_xlabel(sp.latex(L_sym)); ax.set_ylabel(sp.latex(K_sym))
    # except Exception:
    #     ax.set_xlabel(str(L_sym)); ax.set_ylabel(str(K_sym))
    ax.set_xlabel("L"); ax.set_ylabel("K")

    # Malla/rangos
    Lmin, Lmax = L_range
    Kmin, Kmax = K_range
    Lv = np.linspace(Lmin, Lmax, n)
    Kv = np.linspace(Kmin, Kmax, n)
    Lm, Km = np.meshgrid(Lv, Kv, indexing='ij')

    # Normalizar Y_values
    if not hasattr(Y_values, '__iter__') or isinstance(Y_values, (str, bytes)):
        Y_values = (float(Y_values),)

    # Ciclo de colores estable
    rc_cycle = plt.rcParams.get('axes.prop_cycle', None)
    color_list = (rc_cycle.by_key().get('color', ['C0'])) if rc_cycle else ['C0']

    legend_handles, legend_labels = [], []

    for i, yv in enumerate(Y_values):
        color = color_list[i % len(color_list)]

        if fL is not None:
            # ===== Leontief: codo (Ls, Ks), con rayos a la derecha y hacia arriba =====
            try:
                L_star = sp.solve(sp.Eq(fL, yv), _L)[0]
                K_star = sp.solve(sp.Eq(gK, yv), _K)[0]
            except Exception:
                continue
            Ls = float(sp.N(L_star)); Ks = float(sp.N(K_star))

            # Horizontal: desde max(Ls, Lmin) hasta Lmax a altura Ks (si Ks está en rango)
            if (Kmin <= Ks <= Kmax):
                x0 = max(Lmin, Ls)
                if x0 <= Lmax:
                    ax.plot([x0, Lmax], [Ks, Ks], color=color)
                    legend_handles.append(plt.Line2D([0], [0], color=color))
                    legend_labels.append(fr"$Y={yv}$")

            # Vertical (corrección): desde max(Ks, Kmin) hasta Kmax en L=Ls
            if (Lmin <= Ls <= Lmax):
                y0 = max(Ks, Kmin, eps if loglog else Kmin)
                y1 = Kmax
                if y1 > y0:
                    ax.plot([Ls, Ls], [y0, y1], color=color)

        else:
            # ===== General: contorno de G(L,K;Y)=0 =====
            Z = np.array(G_fun(Lm, Km, yv), dtype=float)
            try:
                cs = ax.contour(Lm, Km, Z, levels=[0.0], colors=[color])
                if cs.allsegs and cs.allsegs[0]:
                    legend_handles.append(plt.Line2D([0], [0], color=color))
                    legend_labels.append(fr"$Y={yv}$")
            except Exception:
                pass

    # Escalas y límites
    if loglog:
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(*(xlim if xlim is not None else (Lmin, Lmax)))
    ax.set_ylim(*(ylim if ylim is not None else (Kmin, Kmax)))
    ax.grid(True, which="both")

    # Leyenda
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='best')

    if show:
        plt.show()
    # return ax



#ChatGPT
########################################
### Función que simplifica potencias ###
########################################
def simplify_powers_func(eq_or_expr, t, func_list):
    """
    Simplifica potencias/productos sustituyendo temporalmente variables/funciones por símbolos positivos.
    Acepta en func_list: K(t), L(t), A(t) (AppliedUndef), K, L, A (Symbol), o K, L, A como FunctionClass.

    Parámetros
    ----------
    eq_or_expr : sp.Eq | sp.Expr
        Ecuación o expresión a simplificar.
    t          : sp.Symbol
        Variable temporal (para aplicar FunctionClass como f(t)).
    func_list  : iterable
        Elementos Sympy: FunctionClass (K), AppliedUndef (K(t)) o Symbol (K).

    Retorna
    -------
    sp.Eq | sp.Expr (misma clase que la entrada)
    """
    if not func_list:
        raise ValueError("func_list no puede estar vacío.")

    normalized = []
    for f in func_list:
        # f es una "clase" de función: K -> K(t)
        if isinstance(f, sp.FunctionClass):
            f = f(t)
        # f es una función aplicada: K(t)
        elif getattr(f, "is_Function", False):
            # asegúrate de que dependa de t (si no, aplícala)
            if hasattr(f, "args") and (t not in f.args):
                f = f.func(t)
        # f es símbolo puro: K
        elif isinstance(f, sp.Symbol):
            pass
        else:
            raise TypeError("Cada elemento de func_list debe ser FunctionClass, AppliedUndef o Symbol.")

        normalized.append(f)

    # Quitar duplicados por objeto (no por nombre), preservando orden
    seen = set()
    targets = []
    for f in normalized:
        if f not in seen:
            targets.append(f)
            seen.add(f)

    # Construir sustituciones -> positivos, con nombres únicos
    subs_fwd, subs_bwd = {}, {}
    used_names = set()

    def _base_name(obj):
        # Nombre base distinto para símbolo vs función aplicada
        if isinstance(obj, sp.Symbol):
            return obj.name
        elif getattr(obj, "is_Function", False):
            return obj.func.__name__
        else:
            return str(obj)

    for i, f in enumerate(targets):
        base = _base_name(f)
        # Para distinguir K(t) de K, añade sufijo cuando sea función
        suf = "_t" if getattr(f, "is_Function", False) else ""
        pos_name = f"{base}pos{suf}"
        # Garantiza unicidad si ya existe
        k = 1
        while pos_name in used_names:
            k += 1
            pos_name = f"{base}pos{suf}_{k}"
        used_names.add(pos_name)

        vpos = sp.Symbol(pos_name, positive=True)
        subs_fwd[f] = vpos
        subs_bwd[vpos] = f

    def _simplify(expr):
        tmp = expr.subs(subs_fwd)
        # Fuerza reglas de potencias/prod (útil para (a*b)^p, etc.)
        tmp = sp.powsimp(tmp, force=True)
        # Cancela factores racionales
        tmp = sp.cancel(tmp)
        # Limpieza final
        tmp = sp.simplify(tmp)
        # Revertir a las variables originales
        return tmp.subs(subs_bwd)

    if isinstance(eq_or_expr, sp.Equality):
        return sp.Eq(_simplify(eq_or_expr.lhs), _simplify(eq_or_expr.rhs))
    else:
        return _simplify(eq_or_expr)



#ChatGPT
###############################################
### Función para graficar varias ecuaciones ###
###############################################
# ===== COMPAT: Shim de compatibilidad SymPy (AppliedUndef/UndefinedFunction/FunctionClass) =====
try:
    from sympy.core.function import AppliedUndef as _AppliedUndef, UndefinedFunction as _UndefinedFunction, FunctionClass as _FunctionClass
except Exception:
    # Algunas versiones no exportan AppliedUndef en sympy; obtenemos clases por introspección
    try:
        from sympy.core.function import FunctionClass as _FunctionClass
    except Exception:
        _FunctionClass = type(sp.Function('f'))  # clase de función indefinida
    # Construye un ejemplo aplicado para capturar su clase
    _AppliedUndef = type(sp.Function('f')(sp.Symbol('x')))
    _UndefinedFunction = getattr(sp, 'UndefinedFunction', _FunctionClass)


# ===== Helpers compactos =====
def _broadcast(lst, m):
    if lst is None: return None
    if len(lst) == m: return list(lst)
    if len(lst) == 1 and m > 1: return [lst[0]]*m
    raise ValueError("Longitud inválida: se esperaba 1 o m elementos.")

def _hide_canvas_ui(fig):
    try:
        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False
    except Exception:
        pass

def _in_notebook_like_env():
    try:
        from IPython import get_ipython
        shell = str(type(get_ipython())).lower()
        return ("zmq" in shell) or ("terminalinteractive" in shell)
    except Exception:
        return False

def _is_widget_or_inline_backend():
    try:
        b = str(plt.get_backend()).lower()
        return ("inline" in b) or ("ipympl" in b) or ("widget" in b)
    except Exception:
        return False

@contextmanager
def _autoclose_fig(fig, enable=True):
    try:
        yield
    finally:
        if enable:
            try: plt.close(fig)
            except Exception: pass

def _normalize_eq(e):
    if isinstance(e, sp.Equality): return e
    if isinstance(e, sp.Expr):     return sp.Eq(e, 0)
    raise TypeError("Cada elemento de `eqs` debe ser sp.Eq o sp.Expr.")

def _subs(obj, params):
    return obj.subs(params) if (params and hasattr(obj, "subs")) else obj

def _roots_1d(obj, sym):
    """Raíces numéricas finitas de Eq/expr respecto a sym; número -> constante."""
    vals = []
    try:
        if isinstance(obj, sp.Equality):
            roots = sp.solve(obj, sym)
            seq = roots if isinstance(roots, (list, tuple)) else [roots]
            for r in seq:
                try:
                    v = float(sp.N(r)); 
                    if np.isfinite(v): vals.append(v)
                except Exception: pass
        else:
            expr = sp.sympify(obj)
            if expr.has(sym):
                roots = sp.solve(sp.Eq(expr, 0), sym)
                seq = roots if isinstance(roots, (list, tuple)) else [roots]
                for r in seq:
                    try:
                        v = float(sp.N(r)); 
                        if np.isfinite(v): vals.append(v)
                    except Exception: pass
            else:
                v = float(sp.N(expr))
                if np.isfinite(v): vals.append(v)
    except Exception:
        pass
    return vals

# ===== NUEVO: validador para claves de sliders (símbolo, función indefinida o aplicada) =====
def _is_slider_key(obj):
    return (
        isinstance(obj, sp.Symbol)              # símbolo
        or isinstance(obj, _UndefinedFunction)  # p.ej. sp.Function('a')
        or isinstance(obj, _FunctionClass)      # seguridad extra
        or isinstance(obj, _AppliedUndef)       # p.ej. a(t0), a(x)
    )

# ===== NUEVO: _make_pre_funcs con proxies para funciones aplicadas/indefinidas =====
def _make_pre_funcs(eqs_norm, x_sym, y_sym, colors, labels, linestyles,  # << añadido linestyles
                    per_eq_params, shared_params, slider_syms):
    """
    Prepara funciones lambdify y metadatos por solución.
    Soporta parámetros:
      - sp.Symbol
      - sp.UndefinedFunction (sp.Function('a')) no aplicada
      - sp.AppliedUndef (a(t0), a(x), ...)
    Mediante símbolos proxy internos para lambdify.
    """
    m = len(eqs_norm)
    per_keys_by_i = [set((per_eq_params[i] or {}).keys()) if (per_eq_params and i < len(per_eq_params) and per_eq_params[i]) else set()
                     for i in range(m)]
    shared_keys = set((shared_params or {}).keys())
    pre = []

    # --- detecta candidatos parametrizables robustamente (compat) ---
    def _collect_param_candidates(expr):
        syms = set(expr.free_symbols)  # sp.Symbol

        # UndefinedFunction “desnudas” (a, b, ...)
        try:
            ufun = set(expr.atoms(_UndefinedFunction))
        except Exception:
            # Fallback: recorre el árbol y toma clases de función no aplicadas
            ufun = {node for node in sp.preorder_traversal(expr) if isinstance(node, _FunctionClass)}

        # Funciones aplicadas (a(t0), a(x), ...)
        try:
            appl = set(expr.atoms(_AppliedUndef))
        except Exception:
            # Fallback: nodos con is_Function True que no son clases (llamadas reales)
            appl = {
                node for node in sp.preorder_traversal(expr)
                if getattr(node, 'is_Function', False) and not isinstance(node, _FunctionClass)
            }

        return syms.union(ufun).union(appl)

    _proxy_counter = 0
    def _make_proxy_for(obj):
        nonlocal _proxy_counter
        _proxy_counter += 1
        name = f"__par{_proxy_counter}__{sp.srepr(obj)}"  # nombre único y rastreable
        return sp.Symbol(name)

    for i, eq in enumerate(eqs_norm):
        try:
            sols = sp.solve(eq, y_sym, dict=False)
        except Exception as e:
            print(f"[Aviso] No se pudo resolver la ecuación {i+1}: {e}")
            pre.append([])
            continue

        funcs_i = []
        for j, s in enumerate(sols or []):
            cand = _collect_param_candidates(s)
            per_keys_i = per_keys_by_i[i]

            # parámetros “controlables”
            param_objs = [p for p in cand if (p in slider_syms) or (p in per_keys_i) or (p in shared_keys)]

            # Aviso: si AppliedUndef depende de x_sym, se trata como escalar
            for p in param_objs:
                if isinstance(p, _AppliedUndef):
                    try:
                        if any(arg.has(x_sym) for arg in p.args):
                            print(f"[Aviso] '{p}' depende de {x_sym}. Se tratará como escalar (valor fijo) para deslizadores.")
                    except Exception:
                        pass

            # Mapa original -> proxy (símbolos quedan igual)
            repl_to_proxy = {}
            order_originals = []
            for p in sorted(param_objs, key=lambda z: getattr(z, "name", str(z))):
                if isinstance(p, sp.Symbol):
                    repl_to_proxy[p] = p
                else:
                    repl_to_proxy[p] = _make_proxy_for(p)
                order_originals.append(p)

            # Reemplaza en la solución y lambdify
            s_sub = s.xreplace(repl_to_proxy)
            order_proxies = [repl_to_proxy[p] for p in order_originals]
            try:
                f = sp.lambdify((x_sym, *order_proxies), s_sub, "numpy")
            except Exception as e:
                print(f"[Aviso] Lambdify falló (ec {i+1}, rama {j+1}): {e}")
                continue

            label = (labels[i] if labels else f"Ecuación {i+1}") + (f" (rama {j+1})" if len(sols) > 1 else "")
            per_i = per_eq_params[i] if (per_eq_params and i < len(per_eq_params) and per_eq_params[i]) else {}

            funcs_i.append((
                f,                   # función lambdify(x, *proxies)
                order_proxies,       # proxies usados en lambdify (info interna)
                order_originals,     # objetos originales a buscar en params_now/per/shared
                label, colors[i], linestyles[i],  # << guardamos estilo por ecuación
                per_i
            ))
        pre.append(funcs_i)
    return pre

# === Ajustes internos para permitir “capa congelada” ===
def _render_curves(ax, pre_funcs, x_sym, x_range, n, *, params_now, shared_params,
                   clear=True, alpha=1.0, linestyle=None, include_labels=True):  # << linestyle opcional (override)
    if clear:
        ax.clear()
    x_vals = np.linspace(x_range[0], x_range[1], n)
    for funcs_i in pre_funcs:
        for item in funcs_i:
            # Recibimos (f, order_proxies, order_originals, label, color, style_item, per_i)
            f, order_proxies, order_originals, label, color, style_item, per_i = item
            try:
                # sliders -> per_eq -> shared -> 0
                args = []
                for p_orig in order_originals:
                    if p_orig in params_now:                           args.append(float(params_now[p_orig]))
                    elif per_i and p_orig in per_i:                    args.append(float(per_i[p_orig]))
                    elif shared_params and p_orig in shared_params:    args.append(float(shared_params[p_orig]))
                    else:                                              args.append(0.0)
                with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                    y_vals = f(x_vals, *args)
                y_arr = np.array(np.real(y_vals), dtype=float)
                if y_arr.ndim == 0: y_arr = np.full_like(x_vals, y_arr, dtype=float)
                mask = np.isfinite(y_arr)
                if np.any(mask):
                    use_style = linestyle if (linestyle is not None) else style_item  # prioridad a override global
                    ax.plot(x_vals[mask], y_arr[mask], linewidth=2, color=color,
                            linestyle=use_style, alpha=alpha,
                            label=(label if include_labels else None))
            except Exception as e:
                print(f"[Aviso] Error evaluando curva: {e}")

def _plot_guides_both(ax, x_sym, y_sym,
                      vertical_eq, vertical_params, vertical_labels, vline_color, vline_style,
                      horizontal_eq, horizontal_params, horizontal_labels, hline_color, hline_style,
                      params_now, shared_params, *,
                      alpha=1.0, include_labels=True):
    def _as_list(obj): return obj if isinstance(obj, (list, tuple)) else [obj]
    # Vertical
    if vertical_eq is not None:
        vlist = _as_list(vertical_eq)
        vline_color_list = _as_list(vline_color)*len(vlist)
        vline_style_list = _as_list(vline_style)*len(vlist)
        vparams = vertical_params if isinstance(vertical_params, list) else [vertical_params]*len(vlist)
        if isinstance(vertical_params, list) and len(vertical_params) != len(vlist): vparams = [None]*len(vlist)
        vlabels = vertical_labels if isinstance(vertical_labels, list) else [vertical_labels]*len(vlist)
        if isinstance(vertical_labels, list) and len(vertical_labels) != len(vlist): vlabels = [None]*len(vlist)
        for raw, prm, lab, vcol, vsty in zip(vlist, vparams, vlabels, vline_color_list, vline_style_list):
            obj = _subs(_subs(_subs(raw, params_now), prm), shared_params)
            vals = _roots_1d(obj, x_sym)
            for k, v in enumerate(vals):
                labk = None
                if include_labels:
                    labk = None if lab is None else (lab if len(vals)==1 else f"{lab} (r{k+1})")
                ax.axvline(v, color=vcol, linestyle=vsty, linewidth=1.5, alpha=alpha, label=labk)
    # Horizontal
    if horizontal_eq is not None:
        hlist = _as_list(horizontal_eq)
        hline_color_list = _as_list(hline_color)*len(hlist)
        hline_style_list = _as_list(hline_style)*len(hlist)
        hparams = horizontal_params if isinstance(horizontal_params, list) else [horizontal_params]*len(hlist)
        if isinstance(horizontal_params, list) and len(horizontal_params) != len(hlist): hparams = [None]*len(hlist)
        hlabels = horizontal_labels if isinstance(horizontal_labels, list) else [horizontal_labels]*len(hlist)
        if isinstance(horizontal_labels, list) and len(horizontal_labels) != len(hlist): hlabels = [None]*len(hlist)
        for raw, prm, lab, hcol, hsty in zip(hlist, hparams, hlabels, hline_color_list, hline_style_list):
            obj = _subs(_subs(_subs(raw, params_now), prm), shared_params)
            vals = _roots_1d(obj, y_sym)
            for k, v in enumerate(vals):
                labk = None
                if include_labels:
                    labk = None if lab is None else (lab if len(vals)==1 else f"{lab} (r{k+1})")
                ax.axhline(v, color=hcol, linestyle=hsty, linewidth=1.5, alpha=alpha, label=labk)

def _apply_legend(ax, show_legend):
    """Leyenda con TODO (curvas+guías) y sin duplicados."""
    if not show_legend:
        return
    handles, labels = ax.get_legend_handles_labels()
    new_h, new_l, seen = [], [], set()
    for h, l in zip(handles, labels):
        if not l or l in seen: 
            continue
        seen.add(l); new_h.append(h); new_l.append(l)
    if new_h:
        ax.legend(new_h, new_l, loc="upper left", bbox_to_anchor=(1.02, 1),
                  borderaxespad=0.0, frameon=True)

# ====== FUNCIÓN PRINCIPAL (misma firma + xlim/ylim y NUEVO linestyles) ======
def plot_from_eq_list(eqs, x_sym, y_sym, 
                      x_range=(0.1, 5), n=400,
                      shared_params=None,
                      per_eq_params=None,
                      show_legend=False,
                      colors=None,
                      labels=None,
                      linestyles=None,          # << NUEVO
                      title="",
                      tam_fig=(7.5, 4.8),
                      # NUEVO: límites de ejes (xlim por defecto = x_range)
                      xlim=None,
                      ylim=None,
                      # --- VERTICALES (opcional) ---
                      vertical_eq=None,            # expr/Eq/número o lista
                      vertical_params=None,        # dict o lista[dict]
                      vertical_labels=None,        # str o lista[str]
                      vline_color="gray",
                      vline_style="--",
                      # --- HORIZONTALES (opcional) ---
                      horizontal_eq=None,          # expr/Eq/número o lista
                      horizontal_params=None,      # dict o lista[dict]
                      horizontal_labels=None,      # str o lista[str]
                      hline_color="gray",
                      hline_style="--",
                      # --- Ejes y título ---
                      label_x="",
                      label_y="",
                      # --- Interactividad con múltiples sliders ---
                      interactive=False,
                      sliders=None,                # {sp.Symbol / sp.Function / AppliedUndef: {...}}
                      slider_use_latex=True,       # usa etiquetas LaTeX si el frontend lo soporta
                      # --- NUEVO: congelar la gráfica inicial en modo interactivo ---
                      freeze_initial=False         # <<<<<< NUEVO (default False: no cambia comportamiento)
                      ):
    """Grafica ecuaciones (sp.Eq o sp.Expr) resolviendo para y(x). xlim = x_range por defecto."""
    # Validaciones y normalizaciones
    if not isinstance(eqs, (list, tuple)) or len(eqs) == 0:
        raise ValueError("`eqs` debe ser una lista no vacía de sp.Eq o sp.Expr.")
    m = len(eqs)
    if per_eq_params is not None and len(per_eq_params) != m:
        raise ValueError("`per_eq_params` debe tener la misma longitud que `eqs`.")
    labels_b = _broadcast(labels, m) if labels is not None else None
    colors_b = _broadcast(colors, m) if colors is not None else None
    if colors_b is None:
        default_colors = [d['color'] for d in plt.rcParams['axes.prop_cycle']]
        color_cycle = cycle(default_colors)
        colors_b = [next(color_cycle) for _ in range(m)]
    linestyles_b = _broadcast(linestyles, m) if linestyles is not None else ["-"]*m  # << NUEVO

    eqs_norm = [_normalize_eq(e) for e in eqs]

    # xlim consistente con x_range si no se pasó explícito
    if xlim is None: xlim = x_range

    # === Preparación común: pre-funciones ===
    shared_params = dict(shared_params) if shared_params else {}
    per_eq_params = list(per_eq_params) if per_eq_params else [None]*m
    slider_syms = set(sliders.keys()) if (interactive and isinstance(sliders, dict)) else set()
    pre_funcs = _make_pre_funcs(eqs_norm, x_sym, y_sym, colors_b, labels_b, linestyles_b,  # << pasa linestyles
                                per_eq_params, shared_params, slider_syms)

    # === Modo estático ===
    if not interactive:
        fig, ax = plt.subplots(figsize=tam_fig)
        _hide_canvas_ui(fig)
        enable_autoclose = not (_in_notebook_like_env() or _is_widget_or_inline_backend())
        with _autoclose_fig(fig, enable=enable_autoclose):
            _render_curves(ax, pre_funcs, x_sym, x_range, n,
                           params_now={}, shared_params=shared_params, clear=True, linestyle=None)  # << usa estilos por ecuación
            _plot_guides_both(ax, x_sym, y_sym,
                              vertical_eq, vertical_params, vertical_labels, vline_color, vline_style,
                              horizontal_eq, horizontal_params, horizontal_labels, hline_color, hline_style,
                              params_now={}, shared_params=shared_params, alpha=1.0)
            _apply_legend(ax, show_legend)
            if xlim is not None: ax.set_xlim(xlim)
            if ylim is not None: ax.set_ylim(ylim)
            ax.set_xlabel(label_x); ax.set_ylabel(label_y); ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.tight_layout()
            try:
                plt.show(block=not (_in_notebook_like_env() or _is_widget_or_inline_backend()))
            except TypeError:
                plt.show()
            try: fig.canvas.draw_idle(); fig.canvas.flush_events()
            except Exception: pass
        return

    # === Modo interactivo ===
    if not isinstance(sliders, dict) or len(sliders) == 0:
        raise ValueError("Con `interactive=True` pasa `sliders` como dict no vacío: {sp.Symbol o sp.Function/aplicadas: {...}}")

    # Normalizar sliders (con validador compatible)
    slider_specs = {}
    for sym, spec in sliders.items():
        if not _is_slider_key(sym):
            raise TypeError("Las claves de `sliders` deben ser símbolos SymPy, funciones indefinidas (sp.Function('a')) o funciones aplicadas (p.ej. a(t0)).")
        if "min" not in spec or "max" not in spec:
            raise ValueError(f"Faltan 'min'/'max' para el slider de {sym}.")
        mn, mx = spec["min"], spec["max"]
        slider_specs[sym] = {
            "min": mn, "max": mx,
            "step": spec.get("step", (mx - mn)/100.0 if mx > mn else 0.1),
            "init": spec.get("init", (mn + mx)/2.0),
            "desc": spec.get("desc", str(sym)),
            "desc_latex": spec.get("desc_latex", None),
        }

    # --- normalizador de LaTeX para etiquetas ---
    def _latexify(s):
        """Devuelve una cadena con EXACTAMENTE un par de $...$ para MathJax."""
        if s is None: return None
        t = str(s).strip()
        if t.startswith("$$") and t.endswith("$$"):
            t = t[2:-2].strip()
        if not (t.startswith("$") and t.endswith("$")):
            t = f"${t}$"
        return t

    try:
        import ipywidgets as widgets
        from IPython.display import display

        @contextmanager
        def _no_inline_show():
            was = plt.isinteractive(); plt.ioff()
            try: yield
            finally:
                if was: plt.ion()

        with _no_inline_show():
            fig, ax = plt.subplots(figsize=tam_fig)
            _hide_canvas_ui(fig)
            base_params = dict(shared_params)
            slider_rows, slider_widgets = [], []
            for sym, spec in slider_specs.items():
                base_params[sym] = float(spec["init"])
                sld = widgets.FloatSlider(min=spec["min"], max=spec["max"], step=spec["step"],
                                          value=spec["init"], continuous_update=True,
                                          description='', readout=True,
                                          style={'description_width': '0px'})
                if slider_use_latex:
                    raw_ltx = spec.get("desc_latex", None)
                    latex_text = _latexify(raw_ltx if raw_ltx is not None else sp.latex(sym))
                    if hasattr(widgets, "Latex"):
                        lbl = widgets.Latex(value=latex_text)
                    elif hasattr(widgets, "HTMLMath"):
                        lbl = widgets.HTMLMath(value=latex_text)
                    else:
                        lbl = widgets.Label(value=latex_text)
                    row = widgets.HBox([lbl, sld])
                else:
                    sld.description = spec["desc"]
                    row = widgets.HBox([sld])

                slider_rows.append(row); slider_widgets.append((sym, sld))

            # --- Primer render (y, si procede, quedará “congelado”) ---
            _render_curves(ax, pre_funcs, x_sym, x_range, n,
                           params_now=base_params, shared_params=shared_params, clear=True, linestyle=None)  # << estilos por ecuación
            _plot_guides_both(ax, x_sym, y_sym,
                              vertical_eq, vertical_params, vertical_labels, vline_color, vline_style,
                              horizontal_eq, horizontal_params, horizontal_labels, hline_color, hline_style,
                              params_now=base_params, shared_params=shared_params, alpha=1.0)
            _apply_legend(ax, show_legend)
            if xlim is not None: ax.set_xlim(xlim)
            if ylim is not None: ax.set_ylim(ylim)
            ax.set_xlabel(label_x); ax.set_ylabel(label_y); ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.tight_layout()

            # Guardamos la configuración inicial para la capa “congelada”
            frozen_params = dict(base_params)

        def _on_change(change):
            if change["name"] != "value": return
            params_now = dict(base_params)
            for sym, sld in slider_widgets:
                params_now[sym] = float(sld.value)

            ax = fig.axes[0]

            if freeze_initial:
                # 1) capa congelada, tenue y sin etiquetas (override global :)
                _render_curves(ax, pre_funcs, x_sym, x_range, n,
                               params_now=frozen_params, shared_params=shared_params,
                               clear=True, alpha=0.45, linestyle=":", include_labels=False)
                _plot_guides_both(ax, x_sym, y_sym,
                                  vertical_eq, vertical_params, vertical_labels, vline_color, vline_style,
                                  horizontal_eq, horizontal_params, horizontal_labels, hline_color, hline_style,
                                  params_now=frozen_params, shared_params=shared_params,
                                  alpha=0.45, include_labels=False)
                # 2) configuración actual encima (usa estilos por ecuación)
                _render_curves(ax, pre_funcs, x_sym, x_range, n,
                               params_now=params_now, shared_params=shared_params,
                               clear=False, alpha=1.0, linestyle=None, include_labels=True)
                _plot_guides_both(ax, x_sym, y_sym,
                                  vertical_eq, vertical_params, vertical_labels, vline_color, vline_style,
                                  horizontal_eq, horizontal_params, horizontal_labels, hline_color, hline_style,
                                  params_now=params_now, shared_params=shared_params,
                                  alpha=1.0, include_labels=True)
            else:
                _render_curves(ax, pre_funcs, x_sym, x_range, n,
                               params_now=params_now, shared_params=shared_params, clear=True, linestyle=None)
                _plot_guides_both(ax, x_sym, y_sym,
                                  vertical_eq, vertical_params, vertical_labels, vline_color, vline_style,
                                  horizontal_eq, horizontal_params, horizontal_labels, hline_color, hline_style,
                                  params_now=params_now, shared_params=shared_params, alpha=1.0)

            _apply_legend(ax, show_legend)
            if xlim is not None: ax.set_xlim(xlim)
            if ylim is not None: ax.set_ylim(ylim)
            ax.set_xlabel(label_x); ax.set_ylabel(label_y); ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.6)
            try: fig.canvas.draw_idle()
            except Exception: pass

        for _, sld in slider_widgets:
            sld.observe(_on_change, names="value")

        # Layout de sliders
        if len(slider_rows) <= 3:
            sliders_box = widgets.HBox(slider_rows)
        else:
            sliders_box = widgets.VBox(
                [widgets.HBox(slider_rows[i:i+3]) for i in range(0, len(slider_rows), 3)]
            )

        backend = str(plt.get_backend()).lower()
        if ("ipympl" in backend or "widget" in backend) and getattr(fig, "canvas", None) is not None:
            display(widgets.VBox([sliders_box, fig.canvas]))
        else:
            display(sliders_box); display(fig)

    except Exception as e:
        print("[Aviso] Interactividad requiere ipywidgets en entorno Jupyter. "
              "Se mostrará una gráfica estática. Detalle:", e)
        fig, ax = plt.subplots(figsize=tam_fig); _hide_canvas_ui(fig)
        enable_autoclose = not (_in_notebook_like_env() or _is_widget_or_inline_backend())
        with _autoclose_fig(fig, enable=enable_autoclose):
            _render_curves(ax, pre_funcs, x_sym, x_range, n,
                           params_now={}, shared_params=shared_params, clear=True, linestyle=None)  # << estilos por ecuación
            _plot_guides_both(ax, x_sym, y_sym,
                              vertical_eq, vertical_params, vertical_labels, vline_color, vline_style,
                              horizontal_eq, horizontal_params, horizontal_labels, hline_color, hline_style,
                              params_now={}, shared_params=shared_params, alpha=1.0)
            _apply_legend(ax, show_legend)
            if xlim is not None: ax.set_xlim(xlim)
            if ylim is not None: ax.set_ylim(ylim)
            ax.set_xlabel(label_x); ax.set_ylabel(label_y); ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.tight_layout()
            try:
                plt.show(block=not (_in_notebook_like_env() or _is_widget_or_inline_backend()))
            except TypeError:
                plt.show()
            try: fig.canvas.draw_idle(); fig.canvas.flush_events()
            except Exception: pass





#ChatGPT
###################################################################
### Función para graficar un campo de direcciones (slope field) ###
###################################################################
# ============================================
#  Slope field completo y autocontenido (1D)
#  - Helpers renombrados con sufijo _v02
# ============================================
# ===== COMPAT: Shim SymPy (AppliedUndef/UndefinedFunction/FunctionClass) =====
try:
    from sympy.core.function import AppliedUndef as _AppliedUndef, UndefinedFunction as _UndefinedFunction, FunctionClass as _FunctionClass
except Exception:
    try:
        from sympy.core.function import FunctionClass as _FunctionClass
    except Exception:
        _FunctionClass = type(sp.Function('f'))
    _AppliedUndef = type(sp.Function('f')(sp.Symbol('x')))
    _UndefinedFunction = getattr(sp, 'UndefinedFunction', _FunctionClass)

# ---------------------- Helpers internos (v02) ----------------------
@contextmanager
def _autoclose_fig_v02(fig, enable=True):
    try:
        yield
    finally:
        pass

def _is_slider_key_v02(obj):
    if isinstance(obj, sp.Symbol): return True
    if isinstance(obj, _AppliedUndef): return True
    try:
        return isinstance(obj, _FunctionClass)
    except Exception:
        return False

def _normalize_ode_v02(ode, t_sym, y_sym):
    if isinstance(ode, sp.Equality):
        lhs, rhs = ode.lhs, ode.rhs
        if lhs.has(sp.Derivative(y_sym, t_sym)): return rhs
        if rhs.has(sp.Derivative(y_sym, t_sym)): return lhs
        return lhs - rhs
    elif isinstance(ode, sp.Expr):
        return ode
    raise TypeError("`ode` debe ser sp.Eq o sp.Expr representando y' = f(t,y).")

def _lambdify_rhs_v02(rhs_expr, t_sym, y_sym, slider_syms, shared_params):
    def _collect_param_candidates(expr):
        syms = set(expr.free_symbols)
        try:
            ufun = set(expr.atoms(_UndefinedFunction))
        except Exception:
            ufun = {node for node in sp.preorder_traversal(expr) if isinstance(node, _FunctionClass)}
        try:
            appl = set(expr.atoms(_AppliedUndef))
        except Exception:
            appl = {
                node for node in sp.preorder_traversal(expr)
                if getattr(node, 'is_Function', False) and not isinstance(node, _FunctionClass)
            }
        return syms.union(ufun).union(appl)

    cand = _collect_param_candidates(rhs_expr)
    cand = {c for c in cand if c not in {t_sym, y_sym}}

    shared_keys = set((shared_params or {}).keys())
    param_objs = [p for p in cand if (p in slider_syms) or (p in shared_keys)]

    for p in param_objs:
        if isinstance(p, _AppliedUndef):
            try:
                if any(arg.has(t_sym) for arg in p.args):
                    print(f"[Aviso] '{p}' depende de {t_sym}. Se tratará como escalar para sliders.")
            except Exception:
                pass

    _proxy_counter = 0
    def _make_proxy_for(obj):
        nonlocal _proxy_counter
        _proxy_counter += 1
        return sp.Symbol(f"__par{_proxy_counter}__{sp.srepr(obj)}")

    repl_to_proxy = {}
    order_originals = []
    for p in sorted(param_objs, key=lambda z: getattr(z, "name", str(z))):
        if isinstance(p, sp.Symbol):
            repl_to_proxy[p] = p
        else:
            repl_to_proxy[p] = _make_proxy_for(p)
        order_originals.append(p)

    rhs_sub = rhs_expr.xreplace(repl_to_proxy)
    order_proxies = [repl_to_proxy[p] for p in order_originals]

    f = sp.lambdify((t_sym, y_sym, *order_proxies), rhs_sub, "numpy")
    shared = dict(shared_params) if shared_params else {}
    return f, order_originals, shared

def _evaluate_f_v02(rhs_fun, params_order, params_now, shared_params, T, Y):
    args = []
    for p in params_order:
        if p in params_now: args.append(float(params_now[p]))
        elif shared_params and p in shared_params: args.append(float(shared_params[p]))
        else: args.append(0.0)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        S = rhs_fun(T, Y, *args)
    return np.array(np.real(S), dtype=float)

def _make_segments_v02(T, Y, S, seg_len=0.08, cap=10.0):
    S = np.clip(S, -cap, cap)
    denom = np.sqrt(1.0 + S*S)
    dx = (seg_len/2.0) / denom
    dy = S * dx
    x0 = T - dx; x1 = T + dx
    y0 = Y - dy; y1 = Y + dy
    mask = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(y0) & np.isfinite(y1)
    segs = np.stack([np.stack([np.stack([x0[mask], y0[mask]], axis=-1),
                               np.stack([x1[mask], y1[mask]], axis=-1)], axis=-2)], axis=0)
    return segs.reshape(-1, 2, 2), mask

def _color_values_from_slope_v02(S, color_norm=3.0):
    return np.tanh(np.abs(S)/float(color_norm))

def _integrate_solution_v02(rhs_fun, params_order, params_now, shared_params, t_span, y0, n=400):
    try:
        from math import isfinite
        from scipy.integrate import solve_ivp
    except Exception:
        print("[Aviso] Para integrar soluciones necesitas SciPy (solve_ivp).")
        return None, None

    def rhs(t, y):
        args = []
        for p in params_order:
            if p in params_now: args.append(float(params_now[p]))
            elif shared_params and p in shared_params: args.append(float(shared_params[p]))
            else: args.append(0.0)
        try:
            val = rhs_fun(t, float(y[0]), *args)
            val = float(np.real(val))
            if not isfinite(val): return 0.0
            return val
        except Exception:
            return 0.0

    sol = solve_ivp(rhs, t_span=t_span, y0=[y0], dense_output=True, max_step=(t_span[1]-t_span[0])/n)
    if not sol.success:
        return None, None
    ts = np.linspace(t_span[0], t_span[1], n)
    ys = sol.sol(ts)[0]
    return ts, ys

def _roots_yprime0_at_t0_v02(rhs_fun, params_order, params_now, shared_params, y_range, samples=1200, tol=1e-10, itmax=60):
    y_min, y_max = float(y_range[0]), float(y_range[1])
    ys = np.linspace(y_min, y_max, int(samples))
    args = []
    for p in params_order:
        if p in params_now: args.append(float(params_now[p]))
        elif shared_params and p in shared_params: args.append(float(shared_params[p]))
        else: args.append(0.0)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        vals = rhs_fun(0.0, ys, *args)
    vals = np.array(np.real(vals), dtype=float)
    mask = np.isfinite(vals)
    ys = ys[mask]; vals = vals[mask]
    roots = []
    if len(ys) < 2: return roots
    s = np.sign(vals)
    for i in range(len(ys)-1):
        a, b = ys[i], ys[i+1]
        fa, fb = vals[i], vals[i+1]
        if not (np.isfinite(fa) and np.isfinite(fb)): continue
        if abs(fa) < tol: roots.append(a); continue
        if abs(fb) < tol: roots.append(b); continue
        if s[i] == 0 or s[i+1] == 0 or s[i]*s[i+1] < 0:
            lo, hi = a, b
            flo, fhi = fa, fb
            if flo * fhi > 0:
                m = (fhi - flo) / (hi - lo) if hi != lo else 0.0
                if m != 0:
                    x0 = lo - flo/m
                    if y_min <= x0 <= y_max: roots.append(x0)
                continue
            for _ in range(itmax):
                mid = 0.5*(lo+hi)
                fm = rhs_fun(0.0, float(mid), *args)
                fm = float(np.real(fm))
                if not np.isfinite(fm): break
                if abs(fm) < tol or abs(hi-lo) < tol:
                    roots.append(mid)
                    break
                if flo*fm <= 0:
                    hi, fhi = mid, fm
                else:
                    lo, flo = mid, fm
    roots = sorted([r for r in roots if y_min <= r <= y_max])
    dedup = []
    for r in roots:
        if not dedup or abs(r - dedup[-1]) > 1e-6:
            dedup.append(r)
    return dedup

# ========================== FUNCIÓN PRINCIPAL ==========================
def plot_slope_field(ode, t_sym, y_sym,
                     t_range=(0.0, 5.0), y_range=(-2.0, 2.0),
                     n_t=25, n_y=25,
                     shared_params=None,
                     show_legend=False,
                     # Apariencia del campo
                     field_color="C0",
                     cmap=None,
                     color_norm=3.0,
                     seg_len=0.08,
                     # Flechas (quiver) en vez de segmentos
                     arrows=False,
                     arrow_width=0.0025,
                     arrow_headwidth=3.5,
                     arrow_headlength=5.5,
                     # Ejes y rotulación
                     title="",
                     label_t="t", label_y="y",
                     tam_fig=(7.5, 4.8),
                     xlim=None, ylim=None,
                     # Curvas solución opcionales
                     solutions=None,
                     equilibrium_curve=False,
                     eq_color="red",
                     eq_lw=2.0,
                     # Marcar valor inicial
                     mark_initial=True,
                     init_marker='o',
                     init_size=36,
                     init_edgecolor='white',
                     init_zorder=6,
                     # Interactividad
                     interactive=False,
                     sliders=None,             # {sp.Symbol / sp.Function / a(t): {...}}
                     slider_use_latex=True,
                     solution_slider=None,
                     # Reporte y'(0)=0
                     report_eq_at_t0=True,
                     eq_digits=4,
                     # Soluciones desde y'(0)=0
                     eq_solution=True,
                     eq_solution_all=True,
                     eq_solution_color="tab:red",
                     eq_solution_lw=2.0):

    rhs_expr = _normalize_ode_v02(ode, t_sym, y_sym)
    slider_syms = set(sliders.keys()) if (interactive and isinstance(sliders, dict)) else set()
    rhs_fun, params_order, shared_params = _lambdify_rhs_v02(rhs_expr, t_sym, y_sym, slider_syms, shared_params)

    t0, t1 = float(t_range[0]), float(t_range[1])
    y0_lim, y1_lim = float(y_range[0]), float(y_range[1])
    if xlim is None: xlim = (t0, t1)
    if ylim is None: ylim = (y0_lim, y1_lim)

    tt = np.linspace(t0, t1, int(n_t))
    yy = np.linspace(y0_lim, y1_lim, int(n_y))
    T, Y = np.meshgrid(tt, yy)

    def _draw_field(ax, S):
        if arrows:
            denom = np.sqrt(1.0 + S*S)
            U = (seg_len) / denom
            V = S * U
            mask = np.isfinite(U) & np.isfinite(V)
            Tm, Ym, Um, Vm = T[mask], Y[mask], U[mask], V[mask]
            if cmap is None:
                q = ax.quiver(Tm, Ym, Um, Vm,
                              angles="xy", scale_units="xy", scale=1,
                              width=arrow_width, headwidth=arrow_headwidth, headlength=arrow_headlength,
                              color=field_color, pivot="tail")
            else:
                cvals = _color_values_from_slope_v02(S[mask], color_norm=color_norm)
                q = ax.quiver(Tm, Ym, Um, Vm, cvals,
                              angles="xy", scale_units="xy", scale=1,
                              width=arrow_width, headwidth=arrow_headwidth, headlength=arrow_headlength,
                              cmap=cmap, pivot="tail")
            return q, mask, (Tm, Ym)
        else:
            segs, mask = _make_segments_v02(T, Y, S, seg_len=seg_len)
            if cmap is None:
                lc = LineCollection(segs, colors=field_color, linewidths=1.5)
                ax.add_collection(lc)
                return lc, mask, None
            else:
                cvals = _color_values_from_slope_v02(S[mask], color_norm=color_norm)
                lc = LineCollection(segs, cmap=cmap, linewidths=1.5)
                lc.set_array(cvals)
                ax.add_collection(lc)
                return lc, mask, None

    # ============================ Modo estático ============================
    if not interactive:
        fig, ax = plt.subplots(figsize=tam_fig)
        try:
            fig.canvas.header_visible = False
            fig.canvas.toolbar_visible = False
        except Exception:
            pass

        with _autoclose_fig_v02(fig, enable=True):
            roots = _roots_yprime0_at_t0_v02(rhs_fun, params_order, {}, shared_params, y_range) if report_eq_at_t0 else []
            if report_eq_at_t0:
                txt = ", ".join(f"{val:.{eq_digits}f}" for val in roots) if roots else "— (sin raíces en el rango)"
                print(f"f(0) tal que f'(0)=0 en [{y_range[0]}, {y_range[1]}]: {txt}")

            S = _evaluate_f_v02(rhs_fun, params_order, params_now={}, shared_params=shared_params, T=T, Y=Y)
            artist, mask0, offsets0 = _draw_field(ax, S)

            if equilibrium_curve:
                try:
                    Sm = np.ma.masked_invalid(S)
                    ax.contour(T, Y, Sm, levels=[0], colors=[eq_color], linewidths=[eq_lw], zorder=4)
                except Exception:
                    pass

            if eq_solution and roots:
                t_start = 0.0 if xlim[0] <= 0.0 <= xlim[1] else xlim[0]
                to_plot = roots if eq_solution_all else [roots[0]]
                for y0eq in to_plot:
                    ts, ys = _integrate_solution_v02(rhs_fun, params_order, {}, shared_params, (t_start, t1), float(y0eq), n=400)
                    if ts is not None:
                        ax.plot(ts, ys, color=eq_solution_color, lw=eq_solution_lw,
                                label=f"y0@eq={y0eq:.{eq_digits}f}")
                        if mark_initial:
                            ax.scatter([t_start], [float(y0eq)], s=init_size, c=[eq_solution_color],
                                       marker=init_marker, edgecolors=init_edgecolor, zorder=init_zorder)

            if solutions:
                for sol in solutions:
                    try:
                        tspan = tuple(sol.get("t_span", (t0, t1)))
                        color_sol = sol.get("color", "k")
                        lw_sol = sol.get("lw", 2.0)
                        ts, ys = _integrate_solution_v02(rhs_fun, params_order, {}, shared_params, tspan, sol["y0"], n=sol.get("n", 400))
                        if ts is not None:
                            ax.plot(ts, ys, label=sol.get("label", None), color=color_sol, lw=lw_sol)
                            if mark_initial:
                                ax.scatter([tspan[0]], [sol["y0"]],
                                           s=init_size, c=[color_sol], marker=init_marker,
                                           edgecolors=init_edgecolor, zorder=init_zorder)
                    except Exception:
                        pass

            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(label_t); ax.set_ylabel(label_y); ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.6)
            if show_legend:
                try: ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
                except Exception: pass
            fig.tight_layout()
            plt.show()
        return

    # ============================ Modo interactivo ============================
    if not isinstance(sliders, dict) or len(sliders) == 0:
        raise ValueError("Con `interactive=True` pasa `sliders` como dict no vacío: {sp.Symbol / sp.Function / a(t): {...}}")

    slider_specs = {}
    for sym, spec in sliders.items():
        if not _is_slider_key_v02(sym):
            raise TypeError("Las claves de `sliders` deben ser sp.Symbol, sp.Function('a') o funciones aplicadas (p.ej. a(t0), a(t)).")
        if "min" not in spec or "max" not in spec:
            raise ValueError(f"Faltan 'min'/'max' para el slider de {sym}.")
        mn, mx = spec["min"], spec["max"]
        slider_specs[sym] = {
            "min": mn, "max": mx,
            "step": spec.get("step", (mx - mn)/100.0 if mx > mn else 0.1),
            "init": spec.get("init", (mn + mx)/2.0),
            "desc": spec.get("desc", str(sym)),
            "desc_latex": spec.get("desc_latex", None),
        }

    sol_slider_cfg = None
    if isinstance(solution_slider, dict) and "y0" in solution_slider and isinstance(solution_slider["y0"], dict):
        y0_spec = solution_slider["y0"]
        if "min" in y0_spec and "max" in y0_spec:
            ymn, ymx = y0_spec["min"], y0_spec["max"]
            sol_slider_cfg = {
                "y0": {
                    "min": ymn, "max": ymx,
                    "step": y0_spec.get("step", (ymx-ymn)/100.0 if ymx>ymn else 0.1),
                    "init": y0_spec.get("init", (ymn+ymx)/2.0),
                    "desc_latex": y0_spec.get("desc_latex", r"$y_0$")
                },
                "t_start": float(solution_slider.get("t_start", t0)),
                "t_end": float(solution_slider.get("t_end", t1)),
                "n": int(solution_slider.get("n", 400)),
                "label": solution_slider.get("label", "Solución"),
                "color": solution_slider.get("color", "k"),
                "lw": float(solution_slider.get("lw", 2.0))
            }

    def _latexify_v02(s):
        if s is None: return None
        t = str(s).strip()
        if t.startswith("$$") and t.endswith("$$"):
            t = t[2:-2].strip()
        if not (t.startswith("$") and t.endswith("$")):
            t = f"${t}$"
        return t

    try:
        import ipywidgets as widgets
        from IPython.display import display

        @contextmanager
        def _no_inline_show_v02():
            was = plt.isinteractive(); plt.ioff()
            try: yield
            finally:
                if was: plt.ion()

        with _no_inline_show_v02():
            fig, ax = plt.subplots(figsize=tam_fig)
            try:
                fig.canvas.header_visible = False
                fig.canvas.toolbar_visible = False
            except Exception:
                pass

            base_params = dict(shared_params) if shared_params else {}
            slider_rows, slider_widgets = [], []
            for sym, spec in slider_specs.items():
                base_params[sym] = float(spec["init"])
                sld = widgets.FloatSlider(min=spec["min"], max=spec["max"], step=spec["step"],
                                          value=spec["init"], continuous_update=True,
                                          description='', readout=True,
                                          style={'description_width': '0px'})
                if slider_use_latex:
                    raw_ltx = spec.get("desc_latex", None)
                    latex_text = _latexify_v02(raw_ltx if raw_ltx is not None else sp.latex(sym))
                    lbl = (widgets.Latex(value=latex_text)
                           if hasattr(widgets, "Latex") else
                           widgets.HTMLMath(value=latex_text) if hasattr(widgets, "HTMLMath")
                           else widgets.Label(value=latex_text))
                    row = widgets.HBox([lbl, sld])
                else:
                    sld.description = spec["desc"]
                    row = widgets.HBox([sld])
                slider_rows.append(row); slider_widgets.append((sym, sld))

            roots0 = _roots_yprime0_at_t0_v02(rhs_fun, params_order, base_params, shared_params, y_range) if report_eq_at_t0 else []

            def _fmt_roots_v02(lst):
                return "— (sin raíces en el rango)" if not lst else ", ".join(f"{val:.{eq_digits}f}" for val in lst)
            eq_label_widget = None
            if report_eq_at_t0:
                eq_label_widget = widgets.HTML(
                    value=f"<b>f(0) tal que f'(0)=0:</b> {_fmt_roots_v02(roots0)}  "
                          f"<span style='color:#888'>(rango y: [{y_range[0]}, {y_range[1]}])</span>"
                )
                slider_rows = [eq_label_widget] + slider_rows

            eq_lines, eq_markers = [], []

            if eq_solution and roots0:
                t_start = 0.0 if xlim[0] <= 0.0 <= xlim[1] else xlim[0]
                to_plot = roots0 if eq_solution_all else [roots0[0]]
                for y0eq in to_plot:
                    ts, ys = _integrate_solution_v02(rhs_fun, params_order, base_params, shared_params, (t_start, t1), float(y0eq), n=400)
                    if ts is not None:
                        ln, = ax.plot(ts, ys, color=eq_solution_color, lw=eq_solution_lw,
                                      label=f"y0@eq={y0eq:.{eq_digits}f}")
                        eq_lines.append(ln)
                        if mark_initial:
                            mk = ax.scatter([t_start], [float(y0eq)], s=init_size, c=[eq_solution_color],
                                            marker=init_marker, edgecolors=init_edgecolor, zorder=init_zorder)
                            eq_markers.append(mk)

            S0 = _evaluate_f_v02(rhs_fun, params_order, params_now=base_params, shared_params=shared_params, T=T, Y=Y)
            artist, mask0, offsets0 = _draw_field(ax, S0)

            if equilibrium_curve:
                try:
                    Sm0 = np.ma.masked_invalid(S0)
                    ax.contour(T, Y, Sm0, levels=[0], colors=[eq_color], linewidths=[eq_lw], zorder=4)
                except Exception:
                    pass

            lines_fixed = []
            if solutions:
                for sol in solutions:
                    try:
                        tspan = tuple(sol.get("t_span", (t0, t1)))
                        color_sol = sol.get("color", "k")
                        lw_sol = sol.get("lw", 2.0)
                        ts, ys = _integrate_solution_v02(rhs_fun, params_order, base_params, shared_params, tspan, sol["y0"], n=sol.get("n", 400))
                        if ts is not None:
                            ln, = ax.plot(ts, ys, label=sol.get("label", None), color=color_sol, lw=lw_sol)
                            lines_fixed.append(ln)
                            if mark_initial:
                                ax.scatter([tspan[0]], [sol["y0"]],
                                           s=init_size, c=[color_sol], marker=init_marker,
                                           edgecolors=init_edgecolor, zorder=init_zorder)
                    except Exception:
                        pass

            line_dyn = None
            dyn_marker = None
            y0_widget = None
            if isinstance(sol_slider_cfg, dict):
                y0s = sol_slider_cfg["y0"]
                y0_widget = widgets.FloatSlider(min=y0s["min"], max=y0s["max"], step=y0s["step"],
                                                value=y0s["init"], continuous_update=True,
                                                description='', readout=True,
                                                style={'description_width': '0px'})
                if slider_use_latex:
                    lbl_y0 = (widgets.Latex(value=_latexify_v02(y0s["desc_latex"]))
                              if hasattr(widgets, "Latex") else widgets.Label(value="y0"))
                    slider_rows.append(widgets.HBox([lbl_y0, y0_widget]))
                else:
                    y0_widget.description = "y0"
                    slider_rows.append(widgets.HBox([y0_widget]))

                ts, ys = _integrate_solution_v02(rhs_fun, params_order, base_params, shared_params,
                                                 (sol_slider_cfg["t_start"], sol_slider_cfg["t_end"]),
                                                 float(y0_widget.value), n=sol_slider_cfg["n"])
                if ts is not None:
                    line_dyn, = ax.plot(ts, ys, label=sol_slider_cfg["label"],
                                        color=sol_slider_cfg["color"], lw=sol_slider_cfg["lw"])
                    if mark_initial:
                        dyn_marker = ax.scatter([sol_slider_cfg["t_start"]], [float(y0_widget.value)],
                                                s=init_size, c=[sol_slider_cfg["color"]], marker=init_marker,
                                                edgecolors=init_edgecolor, zorder=init_zorder)

            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(label_t); ax.set_ylabel(label_y); ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.6)
            if show_legend:
                try: ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
                except Exception: pass
            fig.tight_layout()

        def _on_change_v02(change):
            if change["name"] != "value": return
            nonlocal artist, eq_lines, eq_markers, lines_fixed, line_dyn, dyn_marker

            params_now = dict(base_params)
            for sym, sld in slider_widgets:
                params_now[sym] = float(sld.value)

            roots_now = _roots_yprime0_at_t0_v02(rhs_fun, params_order, params_now, shared_params, y_range) if report_eq_at_t0 else []

            for ln in eq_lines:
                try: ln.remove()
                except Exception: pass
            eq_lines = []
            for mk in eq_markers:
                try: mk.remove()
                except Exception: pass
            eq_markers = []

            if eq_solution and roots_now:
                t_start = 0.0 if xlim[0] <= 0.0 <= xlim[1] else xlim[0]
                to_plot = roots_now if eq_solution_all else [roots_now[0]]
                for y0eq in to_plot:
                    ts, ys = _integrate_solution_v02(rhs_fun, params_order, params_now, shared_params, (t_start, t1), float(y0eq), n=400)
                    if ts is not None:
                        ln, = ax.plot(ts, ys, color=eq_solution_color, lw=eq_solution_lw,
                                      label=f"y0@eq={y0eq:.{eq_digits}f}")
                        eq_lines.append(ln)
                        if mark_initial:
                            mk = ax.scatter([t_start], [float(y0eq)], s=init_size, c=[eq_solution_color],
                                            marker=init_marker, edgecolors=init_edgecolor, zorder=init_zorder)
                            eq_markers.append(mk)

            if report_eq_at_t0 and (eq_label_widget is not None):
                txt = "— (sin raíces en el rango)" if not roots_now else ", ".join(f"{val:.{eq_digits}f}" for val in roots_now)
                eq_label_widget.value = (
                    f"<b>f(0) tal que f'(0)=0:</b> {txt}  "
                    f"<span style='color:#888'>(rango y: [{y_range[0]}, {y_range[1]}])</span>"
                )

            S = _evaluate_f_v02(rhs_fun, params_order, params_now=params_now, shared_params=shared_params, T=T, Y=Y)
            if arrows:
                denom = np.sqrt(1.0 + S*S)
                U = (seg_len) / denom; V = S * U
                U0 = U[mask0]; V0 = V[mask0]
                bad = ~(np.isfinite(U0) & np.isfinite(V0))
                U0 = U0.copy(); V0 = V0.copy(); U0[bad] = 0.0; V0[bad] = 0.0
                if artist is not None:
                    try:
                        if cmap is None:
                            artist.set_UVC(U0, V0)
                        else:
                            cvals = _color_values_from_slope_v02(S[mask0], color_norm=color_norm)
                            artist.set_UVC(U0, V0, cvals)
                    except Exception:
                        artist = None
                if artist is None:
                    artist, _, _ = _draw_field(ax, S)
            else:
                segs, mask = _make_segments_v02(T, Y, S, seg_len=seg_len)
                if (artist is not None) and hasattr(artist, "set_segments"):
                    artist.set_segments(segs)
                    if cmap is not None:
                        artist.set_array(_color_values_from_slope_v02(S[mask], color_norm=color_norm))
                else:
                    try:
                        if artist is not None: artist.remove()
                    except Exception:
                        pass
                    artist, _, _ = _draw_field(ax, S)

            if solutions and lines_fixed:
                for ln, sol in zip(lines_fixed, solutions):
                    try:
                        tspan = tuple(sol.get("t_span", (t0, t1)))
                        ts, ys = _integrate_solution_v02(rhs_fun, params_order, params_now, shared_params, tspan, sol["y0"], n=sol.get("n", 400))
                        if ts is not None:
                            ln.set_data(ts, ys)
                    except Exception:
                        pass

            if (y0_widget is not None) and (line_dyn is not None):
                y0_now = float(y0_widget.value)
                ts, ys = _integrate_solution_v02(rhs_fun, params_order, params_now, shared_params,
                                                 (sol_slider_cfg["t_start"], sol_slider_cfg["t_end"]),
                                                 y0_now, n=sol_slider_cfg["n"])
                if ts is not None: line_dyn.set_data(ts, ys)
                if mark_initial and (dyn_marker is not None):
                    dyn_marker.set_offsets(np.array([[sol_slider_cfg["t_start"], y0_now]]))

            try: fig.canvas.draw_idle()
            except Exception: pass

        for _, sld in slider_widgets:
            sld.observe(_on_change_v02, names="value")
        if 'y0_widget' in locals() and (y0_widget is not None):
            y0_widget.observe(_on_change_v02, names="value")

        sliders_box = widgets.HBox(slider_rows) if len(slider_rows) <= 3 else widgets.VBox(
            [widgets.HBox(slider_rows[i:i+3]) for i in range(0, len(slider_rows), 3)]
        )

        backend = str(plt.get_backend()).lower()
        if ("ipympl" in backend or "widget" in backend) and getattr(fig, "canvas", None) is not None:
            display(widgets.VBox([sliders_box, fig.canvas]))
        else:
            display(sliders_box); display(fig)

    except Exception as e:
        print("[Aviso] Interactividad requiere ipywidgets (y SciPy para integrar). "
              "Se mostrará una gráfica estática. Detalle:", e)
        fig, ax = plt.subplots(figsize=tam_fig)
        try:
            fig.canvas.header_visible = False
            fig.canvas.toolbar_visible = False
        except Exception:
            pass
        with _autoclose_fig_v02(fig, enable=True):
            roots = _roots_yprime0_at_t0_v02(rhs_fun, params_order, {}, shared_params, y_range) if report_eq_at_t0 else []
            if report_eq_at_t0:
                txt = ", ".join(f"{val:.{eq_digits}f}" for val in roots) if roots else "— (sin raíces en el rango)"
                print(f"f(0) tal que f'(0)=0 en [{y_range[0]}, {y_range[1]}]: {txt}")

            S = _evaluate_f_v02(rhs_fun, params_order, params_now={}, shared_params=shared_params, T=T, Y=Y)
            artist, _, _ = _draw_field(ax, S)

            if equilibrium_curve:
                try:
                    Sm = np.ma.masked_invalid(S)
                    ax.contour(T, Y, Sm, levels=[0], colors=[eq_color], linewidths=[eq_lw], zorder=4)
                except Exception:
                    pass

            if eq_solution and roots:
                t_start = 0.0 if xlim[0] <= 0.0 <= xlim[1] else xlim[0]
                to_plot = roots if eq_solution_all else [roots[0]]
                for y0eq in to_plot:
                    ts, ys = _integrate_solution_v02(rhs_fun, params_order, {}, shared_params, (t_start, t1), float(y0eq), n=400)
                    if ts is not None:
                        ax.plot(ts, ys, color=eq_solution_color, lw=eq_solution_lw,
                                label=f"y0@eq={y0eq:.{eq_digits}f}")
                        if mark_initial:
                            ax.scatter([t_start], [float(y0eq)], s=init_size, c=[eq_solution_color],
                                       marker=init_marker, edgecolors=init_edgecolor, zorder=init_zorder)

            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(label_t); ax.set_ylabel(label_y); ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.6)
            if show_legend:
                try: ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
                except Exception: pass
            fig.tight_layout()
            plt.show()




#####################################################
### Función que cambia de cuadrante a una gráfica ###
#####################################################
def change_quad(
     eq: sp.Eq,
     *,
     x_old: sp.Symbol,
     y_old: sp.Symbol,
     X: sp.Symbol = None,
     Y: sp.Symbol = None,
     quad="TR",
     include_axes: bool = True,
     x_bounds=None,   # (xmin, xmax) o None
     y_bounds=None    # (ymin, ymax) o None       
):
    
    sgn = {"TR":(+1,+1), "TL":(-1,+1), "BR":(+1,-1), "BL":(-1,-1)}[quad]

    if not isinstance(eq, sp.Equality):
        raise TypeError("`eq` debe ser sp.Eq")

    if X is None: X = x_old
    if Y is None: Y = y_old

    # 1) Reflejo de variables al nuevo sistema (X,Y)
    eq_ref = sp.Eq(
        sp.sympify(eq.lhs).subs({x_old: sgn[0]*X, y_old: sgn[1]*Y}),
        sp.sympify(eq.rhs).subs({x_old: sgn[0]*X, y_old: sgn[1]*Y}),
    )

    # 2) Resolver para Y (para que tu trazador pueda lambdificar y(x)=...)
    try:
        sols = sp.solve(eq_ref, Y, dict=False)
    except Exception:
        sols = []

    if not sols:
        # Si no se pudo resolver en Y, no forzamos nada (no se graficará con tu función)
        return []
    
    cond=sp.And(sgn[0]*X>=0,sgn[1]*Y>=0)
    
    return sp.Eq(eq_ref.lhs,
                 sp.Piecewise((eq_ref.rhs,cond),
                              (sp.nan, True))
                 )

###############################################################
### Función para dibujar campo-pendiente y soluciones en 2D ###
###############################################################
import sympy as sp

# ---- Compat sliders (símbolos / funciones indefinidas / aplicadas) ----
try:
    from sympy.core.function import AppliedUndef as _AppliedUndef, UndefinedFunction as _UndefinedFunction, FunctionClass as _FunctionClass
except Exception:
    try:
        from sympy.core.function import FunctionClass as _FunctionClass
    except Exception:
        _FunctionClass = type(sp.Function('f'))
    _AppliedUndef = type(sp.Function('f')(sp.Symbol('x')))
    _UndefinedFunction = getattr(sp, 'UndefinedFunction', _FunctionClass)


def _latexify(s):
    if s is None: return None
    t = str(s).strip()
    if t.startswith("$$") and t.endswith("$$"): t = t[2:-2].strip()
    if not (t.startswith("$") and t.endswith("$")): t = f"${t}$"
    return t


def plot_slope_field_and_solutions2D(
    eqs, states, indep,
    params=None,
    xlim=(-3,3), ylim=(-3,3), grid_n=25,
    init_points=None, t0=0.0, T=20.0, max_step=None,
    normalize_field=True,
    fig_size=(7.5,6), arrow_scale=25, arrow_width=0.002, arrow_alpha=0.6,
    equal_aspect=False,
    interactive=False,
    sliders=None,            # {clave: {"min":..,"max":..,"step":..,"init":..,"desc":..,"desc_latex":..}}
    slider_use_latex=True,
    freeze_initial=False,

    # --------- PARÁMETROS DE ESTILO ---------
    sol_color="tab:blue",               # color único (default)
    sol_colors=None,                    # lista/tupla/str con colores por solución
    initial_color = "black",
    title="",
    xlabel="x", ylabel="y",
    legend_outside=True,                # leyenda fuera, a la derecha
    show_equilibria=True,               # resaltar puntos de equilibrio
    equilibrium_style=None,             # dict matplotlib para ax.scatter
    eq_solver_opts=None,                # opciones del buscador de raíces

    # --------- LÍMITES ---------
    # "static" (igual que antes) | "solutions" (ajusta a las trayectorias)
    auto_limits="static",

    # --------- DIRECCIÓN TEMPORAL ---------
    # "both" (como antes), "forward" (solo t>t0), "backward" (solo t<t0)
    time_direction="forward",

    # --------- Longitud fija de flechas ---------
    # None => comportamiento original; float => longitud FIX en pantalla (compensando anisotropía)
    arrow_length_fixed=None,

    # ================= NULLCLINES (numéricas) =================
    show_nullclines=False,              # dibujar líneas ẋ=0 y ẏ=0 (vía contornos numéricos)
    nullcline_f_style=None,             # estilo para ẋ=0 (dict matplotlib)
    nullcline_g_style=None,             # estilo para ẏ=0 (dict matplotlib)
    #nullcline_labels=(r"$\\dot{x}=0$", r"$\\dot{y}=0$"), # etiquetas en leyenda (None para omitir)
    nullcline_labels=None,

    # ======= NULLCLINES exactas (opción alternativa, vía SymPy.solve) =======
    show_nullclines_exact=False,        # dibujar nullclines resolviendo f=0, g=0 simbólicamente
    nullcline_exact_target='auto',      # 'y'|'x'|'auto' (resolver y(x), x(y) o ambos si es posible)
    nullcline_exact_nsamples=800,       # número de muestras para graficar cada rama
    nullcline_exact_f_style=None,       # estilo para ẋ=0 exacta (dict matplotlib)
    nullcline_exact_g_style=None,       # estilo para ẏ=0 exacta (dict matplotlib)
    #nullcline_exact_labels=(r"$\\dot{x}=0$ (exact)", r"$\\dot{y}=0$ (exact)")
    nullcline_exact_labels=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import root

    # ---- Compat sliders (símbolos / funciones indefinidas / aplicadas) ----
    try:
        from sympy.core.function import AppliedUndef as _AppliedUndef, UndefinedFunction as _UndefinedFunction, FunctionClass as _FunctionClass
    except Exception:
        try:
            from sympy.core.function import FunctionClass as _FunctionClass
        except Exception:
            _FunctionClass = type(sp.Function('f'))
        _AppliedUndef = type(sp.Function('f')(sp.Symbol('x')))
        _UndefinedFunction = getattr(sp, 'UndefinedFunction', _FunctionClass)

    def _latexify(s):
        if s is None: return None
        t = str(s).strip()
        if t.startswith("$$") and t.endswith("$$"): t = t[2:-2].strip()
        if not (t.startswith("$") and t.endswith("$")): t = f"${t}$"
        return t

    if params is None: params = {}
    if init_points is None: init_points = []
    if len(eqs)!=2 or len(states)!=2:
        raise ValueError("Se esperan exactamente dos ecuaciones y dos estados.")
    if interactive and (not isinstance(sliders, dict) or not sliders):
        raise ValueError("Con interactive=True, pasa `sliders` como dict no vacío.")
    if auto_limits not in ("static", "solutions"):
        raise ValueError("auto_limits debe ser 'static' o 'solutions'.")
    if time_direction not in ("both", "forward", "backward"):
        raise ValueError("time_direction debe ser 'both', 'forward' o 'backward'.")

    # defaults
    if equilibrium_style is None:
        equilibrium_style = dict(marker="o", s=80, color="crimson", edgecolors="k", linewidths=0.6, zorder=5)
    if eq_solver_opts is None:
        eq_solver_opts = dict(tol=1e-9, maxiter=200)

    # estilos nullclines por defecto (contorno numérico)
    if nullcline_f_style is None:
        nullcline_f_style = dict(colors=['tab:red'], linewidths=2.0, linestyles='--', zorder=3.5)
    if nullcline_g_style is None:
        nullcline_g_style = dict(colors=['tab:green'], linewidths=2.0, linestyles='--', zorder=3.4)

    # estilos nullclines exactas por defecto
    if nullcline_exact_f_style is None:
        nullcline_exact_f_style = dict(lw=2.2, ls='-', color='tab:red', zorder=3.6)
    if nullcline_exact_g_style is None:
        nullcline_exact_g_style = dict(lw=2.2, ls='-', color='tab:green', zorder=3.55)

    # --- Normaliza sliders ---
    def _is_slider_key(obj):
        return (isinstance(obj, sp.Symbol)
                or isinstance(obj, _UndefinedFunction)
                or isinstance(obj, _AppliedUndef))

    slider_specs = {}
    if interactive:
        for key, spec in sliders.items():
            if not _is_slider_key(key):
                raise TypeError("Claves de sliders: sp.Symbol, sp.Function('a') o aplicadas (p.ej. a(t0)).")
            if "min" not in spec or "max" not in spec:
                raise ValueError(f"Faltan 'min'/'max' en slider de {key}.")
            mn, mx = float(spec["min"]), float(spec["max"])
            slider_specs[key] = {
                "min": mn, "max": mx,
                "step": float(spec.get("step", (mx-mn)/100.0 if mx>mn else 0.1)),
                "init": float(spec.get("init", (mn+mx)/2.0)),
                "desc": spec.get("desc", str(key)),
                "desc_latex": spec.get("desc_latex", None),
            }

    # --- Simbólico -> numérico ---
    f_expr = eqs[0].rhs
    g_expr = eqs[1].rhs
    x_s, y_s = sp.symbols('x_s y_s', real=True)
    f_xy = sp.simplify(f_expr.subs({states[0]: x_s, states[1]: y_s}))
    g_xy = sp.simplify(g_expr.subs({states[0]: x_s, states[1]: y_s}))

    all_param_objs = set(params.keys()) | set(slider_specs.keys())
    def _mkname(obj, k): return f"__par{k}__{sp.srepr(obj)}"
    repl_to_proxy, order_originals = {}, []
    for i, p in enumerate(sorted(all_param_objs, key=lambda z: getattr(z, "name", str(z))), start=1):
        repl_to_proxy[p] = p if isinstance(p, sp.Symbol) else sp.Symbol(_mkname(p, i))
        order_originals.append(p)
    order_proxies = [repl_to_proxy[p] for p in order_originals]

    f_sub, g_sub = f_xy.xreplace(repl_to_proxy), g_xy.xreplace(repl_to_proxy)
    f_num = sp.lambdify((indep, x_s, y_s, *order_proxies), f_sub, "numpy")
    g_num = sp.lambdify((indep, x_s, y_s, *order_proxies), g_sub, "numpy")

    xmin, xmax = xlim; ymin, ymax = ylim
    if max_step is None:
        max_step = (abs(T)/500.0) if T!=0 else 0.05

    def _par_vector(pdict):
        vals = []
        for p in order_originals:
            if p in pdict: vals.append(float(pdict[p]))
            elif p in params: vals.append(float(params[p]))
            else: vals.append(0.0)
        return vals

    # --- Colores por solución ---
    def _normalize_sol_colors(spec, m):
        if m == 0: return None
        if spec is None: return None
        if isinstance(spec, (list, tuple)):
            if len(spec) == m:
                return list(spec)
            if len(spec) == 1 and m > 1:
                return [spec[0]] * m
            raise ValueError("`sol_colors` debe tener longitud 1 o igual al número de puntos iniciales.")
        return [spec] * m

    sol_colors_b = _normalize_sol_colors(sol_colors, len(init_points))

    # ---- buscador de equilibrios (f=0, g=0) ----
    def _find_equilibria(par_vals, xlo, xhi, ylo, yhi, seeds_per_dim=6, merge_tol=1e-4):
        eqs_found = []
        sx = np.linspace(xlo, xhi, seeds_per_dim)
        sy = np.linspace(ylo, yhi, seeds_per_dim)
        for x0 in sx:
            for y0 in sy:
                try:
                    sol = root(lambda z: [f_num(t0, z[0], z[1], *par_vals),
                                          g_num(t0, z[0], z[1], *par_vals)],
                               x0=np.array([x0, y0], dtype=float),
                               tol=eq_solver_opts.get("tol", 1e-9),
                               options={"maxiter": eq_solver_opts.get("maxiter", 200)})
                    if sol.success:
                        xz, yz = float(sol.x[0]), float(sol.x[1])
                        if not (xlo<=xz<=xhi and ylo<=yz<=yhi): 
                            continue
                        keep = True
                        for (xa, ya) in eqs_found:
                            if (xa-xz)**2 + (ya-yz)**2 <= merge_tol**2:
                                keep = False; break
                        if keep:
                            eqs_found.append((xz, yz))
                except Exception:
                    pass
        return eqs_found

    def _draw(ax, par_vals, alpha_scale=1.0, include_labels=True, clear=True):
        if clear: ax.cla()

        # --- Ecuación diferencial para integrar trayectorias
        def ode_sys(t, z):
            xval, yval = z
            return [float(f_num(t, xval, yval, *par_vals)),
                    float(g_num(t, xval, yval, *par_vals))]

        # --- Integramos primero para poder autoajustar
        all_pts = []
        T_abs = abs(T)
        for idx, (x0, y0) in enumerate(init_points):
            color_i = (sol_colors_b[idx] if sol_colors_b is not None else sol_color)
            try:
                if time_direction in ("forward", "both"):
                    sol_fwd = solve_ivp(ode_sys, (t0, t0 + T_abs), [x0, y0],
                                        max_step=max_step, rtol=1e-6, atol=1e-9)
                    if sol_fwd.y.size:
                        ax.plot(sol_fwd.y[0], sol_fwd.y[1], lw=2, color=color_i,
                                label=(f"({x0:.2f},{y0:.2f}) →" if include_labels else None),
                                alpha=alpha_scale)
                        all_pts.append(sol_fwd.y.T)
                if time_direction in ("backward", "both"):
                    sol_bwd = solve_ivp(ode_sys, (t0, t0 - T_abs), [x0, y0],
                                        max_step=max_step, rtol=1e-6, atol=1e-9)
                    if sol_bwd.y.size:
                        ax.plot(sol_bwd.y[0], sol_bwd.y[1], lw=2, color=color_i, alpha=alpha_scale)
                        all_pts.append(sol_bwd.y.T)
            except Exception:
                pass
            ax.plot([x0], [y0], 'o', ms=5, color=initial_color, alpha=alpha_scale)

        # --- Límites (estáticos o por soluciones)
        if auto_limits == "solutions" and all_pts:
            P = np.vstack(all_pts)
            xmin2, ymin2 = np.nanmin(P, axis=0)
            xmax2, ymax2 = np.nanmax(P, axis=0)
            def pad(a,b):
                span = max(1e-9, b - a); d = 0.05*span
                return a - d, b + d
            xmin_, xmax_ = pad(xmin2, xmax2)
            ymin_, ymax_ = pad(ymin2, ymax2)
        else:
            xmin_, xmax_ = xmin, xmax
            ymin_, ymax_ = ymin, ymax

        # --- Campo vectorial (con opción de longitud fija visual) ---
        X = np.linspace(xmin_, xmax_, grid_n)
        Y = np.linspace(ymin_, ymax_, grid_n)
        XX, YY = np.meshgrid(X, Y)

        U = f_num(t0, XX, YY, *par_vals)
        V = g_num(t0, XX, YY, *par_vals)
        U = np.where(np.isfinite(U), U, 0.0)
        V = np.where(np.isfinite(V), V, 0.0)

        if arrow_length_fixed is None:
            # --- Comportamiento original (opcionalmente normalizado)
            if normalize_field:
                spd = np.hypot(U, V)
                U = np.divide(U, spd, out=np.zeros_like(U), where=spd>0)
                V = np.divide(V, spd, out=np.zeros_like(V), where=spd>0)
            ax.quiver(
                XX, YY, U, V,
                angles='xy', scale_units='xy', scale=arrow_scale,
                width=arrow_width, alpha=arrow_alpha * alpha_scale, color='k'
            )
        else:
            # --- Longitud fija VISUAL compensando anisotropía del eje
            spd = np.hypot(U, V)
            U = np.divide(U, spd, out=np.zeros_like(U), where=spd > 0)
            V = np.divide(V, spd, out=np.zeros_like(V), where=spd > 0)
            p0 = ax.transData.transform((0.0, 0.0))
            px = ax.transData.transform((1.0, 0.0))
            py = ax.transData.transform((0.0, 1.0))
            sx = abs(px[0] - p0[0]) or 1.0
            sy = abs(py[1] - p0[1]) or 1.0
            spd_pix = np.hypot(sx * U, sy * V)
            U = np.divide(U, spd_pix, out=np.zeros_like(U), where=spd_pix > 0)
            V = np.divide(V, spd_pix, out=np.zeros_like(V), where=spd_pix > 0)
            Lpx = float(arrow_length_fixed) * sx
            U = (Lpx / sx) * U
            V = (Lpx / sy) * V
            ax.quiver(
                XX, YY, U, V,
                angles='xy', scale_units='xy', scale=1.0,
                width=arrow_width, alpha=arrow_alpha * alpha_scale, color='k'
            )

        # ================= DIBUJO DE NULLCLINES (numéricas) =================
        if show_nullclines:
            try:
                cf = ax.contour(XX, YY, U, levels=[0.0], **nullcline_f_style)
                cg = ax.contour(XX, YY, V, levels=[0.0], **nullcline_g_style)
                if nullcline_labels is not None:
                    if len(cf.collections) > 0 and nullcline_labels[0]:
                        cf.collections[0].set_label(nullcline_labels[0])
                    if len(cg.collections) > 0 and nullcline_labels[1]:
                        cg.collections[0].set_label(nullcline_labels[1])
            except Exception:
                pass

        # ============== DIBUJO DE NULLCLINES EXACTAS (SymPy.solve) ==============
        if show_nullclines_exact:
            try:
                # Preparar expresiones f(x,y)=0, g(x,y)=0 con parámetros numéricos y t=t0
                # Usamos f_sub/g_sub (ya con proxies de parámetros) y sustituimos valores
                subs_par = {sym: val for sym, val in zip(order_proxies, par_vals)}
                # si el sistema dependiera de t, fijamos t=t0 para nullclines
                subs_par[indep] = float(t0)
                f0 = sp.simplify(f_sub.subs(subs_par))
                g0 = sp.simplify(g_sub.subs(subs_par))

                def _plot_exact(expr, which):
                    curves_plotted = 0
                    if which in ('y', 'auto'):
                        try:
                            sols_y = sp.solve(sp.Eq(expr, 0), y_s, domain=sp.S.Reals)
                            if not isinstance(sols_y, (list, tuple)):
                                sols_y = [sols_y]
                            xs = np.linspace(xmin_, xmax_, int(nullcline_exact_nsamples))
                            for sol in sols_y:
                                fy = sp.lambdify(x_s, sol, 'numpy')
                                yy = fy(xs)
                                mask = np.isfinite(yy) & (yy >= ymin_) & (yy <= ymax_)
                                if np.any(mask):
                                    idx = np.where(mask)[0]
                                    # detectar rupturas y segmentar
                                    gaps = np.where(np.diff(idx) > 1)[0]
                                    start = 0
                                    for gk in list(gaps) + [len(idx)-1]:
                                        j0 = idx[start]
                                        j1 = idx[gk]
                                        ax.plot(xs[j0:j1+1], yy[j0:j1+1], **(nullcline_exact_f_style if expr is f0 else nullcline_exact_g_style))
                                        start = gk+1
                                    curves_plotted += 1
                        except Exception:
                            pass
                    if which in ('x', 'auto'):
                        try:
                            sols_x = sp.solve(sp.Eq(expr, 0), x_s, domain=sp.S.Reals)
                            if not isinstance(sols_x, (list, tuple)):
                                sols_x = [sols_x]
                            ys = np.linspace(ymin_, ymax_, int(nullcline_exact_nsamples))
                            for sol in sols_x:
                                fx = sp.lambdify(y_s, sol, 'numpy')
                                xx = fx(ys)
                                mask = np.isfinite(xx) & (xx >= xmin_) & (xx <= xmax_)
                                if np.any(mask):
                                    idx = np.where(mask)[0]
                                    gaps = np.where(np.diff(idx) > 1)[0]
                                    start = 0
                                    for gk in list(gaps) + [len(idx)-1]:
                                        j0 = idx[start]
                                        j1 = idx[gk]
                                        ax.plot(xx[j0:j1+1], ys[j0:j1+1], **(nullcline_exact_f_style if expr is f0 else nullcline_exact_g_style))
                                        start = gk+1
                                    curves_plotted += 1
                        except Exception:
                            pass
                    return curves_plotted

                cf_n = _plot_exact(f0, nullcline_exact_target)
                cg_n = _plot_exact(g0, nullcline_exact_target)

                if nullcline_exact_labels is not None:
                    if cf_n > 0 and nullcline_exact_labels[0]:
                        ax.plot([], [], **nullcline_exact_f_style, label=nullcline_exact_labels[0])
                    if cg_n > 0 and nullcline_exact_labels[1]:
                        ax.plot([], [], **nullcline_exact_g_style, label=nullcline_exact_labels[1])
            except Exception:
                pass

        # --- Equilibrios
        if show_equilibria:
            pts = _find_equilibria(par_vals, xmin_, xmax_, ymin_, ymax_)
            if pts:
                ex, ey = zip(*pts)
                ax.scatter(ex, ey, **equilibrium_style)

        # --- Estética
        ax.set_xlim(xmin_, xmax_); ax.set_ylim(ymin_, ymax_)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if include_labels and init_points:
            h, l = ax.get_legend_handles_labels()
            if l:
                seen=set(); hh=[]; ll=[]
                for hi, li in zip(h,l):
                    if li and li not in seen:
                        seen.add(li); hh.append(hi); ll.append(li)
                if legend_outside:
                    try: ax.figure.subplots_adjust(right=0.80)
                    except Exception: pass
                    ax.legend(hh, ll, frameon=False, fontsize=9,
                              loc='center left', bbox_to_anchor=(1.02, 0.5))
                else:
                    ax.legend(hh, ll, frameon=False, fontsize=9, loc='best')
        if equal_aspect: ax.set_aspect('equal', adjustable='box')
        else: ax.set_aspect('auto'); ax.margins(x=0.02, y=0.02)

    # ---------- ESTÁTICO ----------
    if not interactive:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        try:
            if hasattr(fig.canvas, "header_visible"):
                fig.canvas.header_visible = False
        except Exception:
            pass
        _draw(ax, _par_vector(params))
        plt.show()
        return

    # ---------- INTERACTIVO ----------
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        base = dict(params)
        for key, spec in slider_specs.items():
            base[key] = spec["init"]

        with plt.ioff():
            fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        try:
            if hasattr(fig.canvas, "header_visible"):
                fig.canvas.header_visible = False
        except Exception:
            pass

        frozen_vals = _par_vector(base)
        _draw(ax, frozen_vals, alpha_scale=1.0, include_labels=True, clear=True)

        rows, slider_widgets = [], []
        for key, spec in slider_specs.items():
            if slider_use_latex:
                txt = spec["desc_latex"] if spec["desc_latex"] is not None else sp.latex(key)
                if hasattr(widgets, "HTMLMath"):
                    lbl = widgets.HTMLMath(value=_latexify(txt))
                elif hasattr(widgets, "Latex"):
                    lbl = widgets.Latex(value=_latexify(txt))
                elif hasattr(widgets, "HTML"):
                    lbl = widgets.HTML(value=_latexify(txt))
                else:
                    lbl = widgets.Label(value=str(key))
            else:
                lbl = widgets.Label(value=spec.get("desc", str(key)))
            lbl.layout = widgets.Layout(width='80px')

            sld = widgets.FloatSlider(
                min=spec["min"], max=spec["max"], step=spec["step"],
                value=spec["init"], continuous_update=True, readout=True,
                readout_format=".4f",
                description='', style={'description_width':'0px'},
                layout=widgets.Layout(width='240px')
            )

            row = widgets.HBox([lbl, sld])
            row.layout = widgets.Layout(width='360px', align_items='center', margin='0 12px 10px 0')
            rows.append(row); slider_widgets.append((key, sld))

        sliders_box = widgets.Box(
            rows,
            layout=widgets.Layout(display='flex', flex_flow='row wrap',
                                  justify_content='flex-start', align_items='center', width='100%')
        )

        backend = str(plt.get_backend()).lower()
        use_ipympl = ("ipympl" in backend) or ("widget" in backend and getattr(fig, "canvas", None) is not None)
        out_fig = widgets.Output()

        def _on_change(_):
            cur = dict(params)
            for k, w in slider_widgets:
                cur[k] = float(w.value)
            cur_vals = _par_vector(cur)

            if freeze_initial:
                _draw(ax, frozen_vals, alpha_scale=0.45, include_labels=False, clear=True)
                _draw(ax, cur_vals, alpha_scale=1.0, include_labels=True, clear=False)
            else:
                _draw(ax, cur_vals, alpha_scale=1.0, include_labels=True, clear=True)

            if use_ipympl:
                try: fig.canvas.draw_idle()
                except Exception: pass
            else:
                with out_fig:
                    clear_output(wait=True)
                    display(fig)

        for _, w in slider_widgets:
            w.observe(_on_change, names="value")

        if not use_ipympl:
            with out_fig:
                clear_output(wait=True)
                display(fig)
            display(widgets.VBox([sliders_box, out_fig], layout=widgets.Layout(width='100%')))
        else:
            display(widgets.VBox([sliders_box, fig.canvas], layout=widgets.Layout(width='100%')))

    except Exception as e:
        print("[Aviso] Interactividad requiere ipywidgets en entorno Jupyter. "
              "Se mostrará una gráfica estática. Detalle:", e)
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        _draw(ax, _par_vector(params))
        plt.show()



#########################################################################
### Función que grafica series temporales de soluciones (x(t) y y(t)) ###
#########################################################################
def plot_time_series_solutions2D(
    eqs, states, indep,
    params=None,
    init_points=None,         # lista de tuplas [(x0,y0), ...]
    solution_names=None,      # lista de nombres (mismo largo que init_points)

    # Filtrar soluciones a graficar
    show_only=None,           # None | int (índice) | str (nombre) -> filtra la solución a graficar
    solution_picker=False,    # si True (y interactive), agrega Dropdown para elegir solución

    t0=0.0, T=20.0, T_back=0.0,
    max_step=None,
    grid_n_t=400,
    fig_size=(8,5),
    interactive=False,
    sliders=None,             # {clave: {"min":..,"max":..,"step":..,"init":..,"desc":..,"desc_latex":..}}
    slider_use_latex=True,
    freeze_initial=False,

    # --- estética y controles ---
    title="Series temporales de las soluciones",
    xlabel="t", ylabel="x(t)",
    ylabel_right="y(t)",
    legend_position="outside_bottom",   # "outside_bottom" | "outside_right" | "inside"
    show_equilibria=True,
    equilibrium_line_style_x=None,
    equilibrium_line_style_y=None,
    color_x="tab:blue",
    color_y="tab:orange",
    init_point_style=None,
    mark_initial_points=True,
    eq_solver_opts=None,
    dual_yaxis=False,
    var_to_plot="both",       # "both" | "x" | "y" | estado sympy (x(t)/y(t)) | texto ("x(t)","$x$",...)

    # --- límites para “zoom” ---
    xlim=None,
    ylim_left=None,
    ylim_right=None
):
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import root

    # ---- compat (símbolos / funciones indefinidas / aplicadas) ----
    try:
        from sympy.core.function import AppliedUndef as _AppliedUndef, UndefinedFunction as _UndefinedFunction, FunctionClass as _FunctionClass
    except Exception:
        try:
            from sympy.core.function import FunctionClass as _FunctionClass
        except Exception:
            _FunctionClass = type(sp.Function('f'))
        _AppliedUndef = type(sp.Function('f')(sp.Symbol('x')))
        _UndefinedFunction = getattr(sp, 'UndefinedFunction', _FunctionClass)

    def _is_slider_key(obj):
        return (isinstance(obj, sp.Symbol)
                or isinstance(obj, _UndefinedFunction)
                or isinstance(obj, _AppliedUndef))

    def _latexify(s):
        if s is None: return None
        t = str(s).strip()
        if t.startswith("$$") and t.endswith("$$"): t = t[2:-2].strip()
        if not (t.startswith("$") and t.endswith("$")): t = f"${t}$"
        return t

    # ----- defaults -----
    if params is None: params = {}
    if init_points is None or len(init_points)==0:
        init_points = [(0.0, 0.0)]
    if max_step is None:
        max_step = (abs(T)+abs(T_back))/500.0 if (T or T_back) else 0.05
    if eq_solver_opts is None:
        eq_solver_opts = dict(tol=1e-9, maxiter=200)
    if equilibrium_line_style_x is None:
        equilibrium_line_style_x = dict(color=color_x, linestyle="--", linewidth=1.2, alpha=0.9)
    if equilibrium_line_style_y is None:
        equilibrium_line_style_y = dict(color=color_y, linestyle="-.", linewidth=1.2, alpha=0.9)
    if init_point_style is None:
        init_point_style = dict(marker="o", s=50, edgecolors="k", linewidths=0.6)

    # --- helper: normaliza qué variable(s) graficar ---
    def _norm_var_to_plot(v, states):
        """
        Acepta:
        - "both", "x", "y"
        - el propio estado sympy: x(t) o y(t)
        - el nombre como string: "x", "y", "x(t)", "y(t)", e incluso $...$
        Devuelve: (plot_x: bool, plot_y: bool)
        """
        def _clean(s):
            s = str(s).strip()
            if s.startswith("$$") and s.endswith("$$"): s = s[2:-2].strip()
            if s.startswith("$") and s.endswith("$"): s = s[1:-1].strip()
            return s

        if v is None or str(v).lower() == "both":
            return True, True

        def _keys(st):
            K = {st, str(st), _clean(st)}
            func = getattr(st, "func", None)
            if func is not None:
                K.add(func)
                name = getattr(func, "__name__", None)
                if name:
                    K.add(name); K.add(_clean(name))
            name2 = getattr(st, "name", None)
            if name2:
                K.add(name2); K.add(_clean(name2))
            return K

        keys_x = _keys(states[0])
        keys_y = _keys(states[1])

        targets = v if isinstance(v, (list, tuple, set)) else [v]
        want_x = want_y = False
        for t in targets:
            t_clean = _clean(t)
            if (t in keys_x) or (t_clean in keys_x):
                want_x = True
            if (t in keys_y) or (t_clean in keys_y):
                want_y = True

        s = _clean(v).lower()
        if not (want_x or want_y):
            if s in {"x", "x(t)"}: want_x = True
            if s in {"y", "y(t)"}: want_y = True

        if not (want_x or want_y):
            raise ValueError("`var_to_plot` no coincide con ninguno de los estados provistos en `states`.")
        return want_x, want_y

    # --- qué variable(es) graficar ---
    plot_x, plot_y = _norm_var_to_plot(var_to_plot, states)

    # --- nombres de soluciones ---
    if solution_names is not None:
        if len(solution_names) != len(init_points or []):
            raise ValueError("`solution_names` debe tener el mismo tamaño que `init_points`.")
        solution_names = [str(n) for n in solution_names]

    # ----- sliders (normalización) -----
    slider_specs = {}
    if interactive:
        if not isinstance(sliders, dict) or not sliders:
            raise ValueError("Con interactive=True, pasa `sliders` como dict no vacío.")
        for key, spec in sliders.items():
            if not _is_slider_key(key):
                raise TypeError("Claves de sliders: sp.Symbol, sp.Function('a') o aplicadas (p.ej. a(t0)).")
            if "min" not in spec or "max" not in spec:
                raise ValueError(f"Faltan 'min'/'max' en slider de {key}.")
            mn, mx = float(spec["min"]), float(spec["max"])
            slider_specs[key] = {
                "min": mn, "max": mx,
                "step": float(spec.get("step", (mx-mn)/100.0 if mx>mn else 0.1)),
                "init": float(spec.get("init", (mn+mx)/2.0)),
                "desc": spec.get("desc", str(key)),
                "desc_latex": spec.get("desc_latex", None),
            }

    # ----- simbólico -> numérico -----
    f_expr = eqs[0].rhs
    g_expr = eqs[1].rhs
    x, y = states
    t = indep

    x_s, y_s = sp.symbols('x_s y_s', real=True)
    f_xy = sp.simplify(f_expr.subs({x: x_s, y: y_s}))
    g_xy = sp.simplify(g_expr.subs({x: x_s, y: y_s}))

    all_param_objs = set(params.keys()) | set(slider_specs.keys())
    def _mkname(obj, k): return f"__par{k}__{sp.srepr(obj)}"
    repl_to_proxy, order_originals = {}, []
    for i, p in enumerate(sorted(all_param_objs, key=lambda z: getattr(z, "name", str(z))), start=1):
        repl_to_proxy[p] = p if isinstance(p, sp.Symbol) else sp.Symbol(_mkname(p, i))
        order_originals.append(p)
    order_proxies = [repl_to_proxy[p] for p in order_originals]

    f_sub, g_sub = f_xy.xreplace(repl_to_proxy), g_xy.xreplace(repl_to_proxy)
    f_num = sp.lambdify((t, x_s, y_s, *order_proxies), f_sub, "numpy")
    g_num = sp.lambdify((t, x_s, y_s, *order_proxies), g_sub, "numpy")

    # ----- utilidades numéricas -----
    def _par_vector(pdict):
        vals = []
        for p in order_originals:
            if p in pdict: vals.append(float(pdict[p]))
            elif p in params: vals.append(float(params[p]))
            else: vals.append(0.0)
        return vals

    def _solve(par_vals, x0, y0, t_span):
        def ode_sys(t_, z):
            return [float(f_num(t_, z[0], z[1], *par_vals)),
                    float(g_num(t_, z[0], z[1], *par_vals))]
        return solve_ivp(ode_sys, t_span, [x0, y0],
                         max_step=max_step, rtol=1e-6, atol=1e-9, dense_output=True)

    def _find_equilibrium(par_vals, seeds=9):
        xs = np.linspace(-1, 1, seeds)
        ys = np.linspace(-1, 1, seeds)
        x_centers = [x0 for (x0, _) in init_points] + [0.0]
        y_centers = [y0 for (_, y0) in init_points] + [0.0]
        tried = set()
        for xc in x_centers:
            for yc in y_centers:
                for dx in xs:
                    for dy in ys:
                        guess = (float(xc+dx), float(yc+dy))
                        if guess in tried:
                            continue
                        tried.add(guess)
                        try:
                            sol = root(lambda z: [f_num(t0, z[0], z[1], *par_vals),
                                                  g_num(t0, z[0], z[1], *par_vals)],
                                       x0=np.array(guess, dtype=float),
                                       tol=eq_solver_opts.get("tol", 1e-9),
                                       options={"maxiter": eq_solver_opts.get("maxiter", 200)})
                            if sol.success and np.isfinite(sol.x).all():
                                return float(sol.x[0]), float(sol.x[1])
                        except Exception:
                            pass
        return None

    # --- helper: aplica selección show_only ---
    def _select_points(points, names, select):
        if select is None:
            return points, names
        if isinstance(select, int):
            if select < 0 or select >= len(points):
                raise IndexError("`show_only` fuera de rango.")
            sel_pts = [points[select]]
            sel_nms = None if names is None else [names[select]]
            return sel_pts, sel_nms
        if isinstance(select, str):
            if names is None:
                raise ValueError("`show_only` es str, pero `solution_names` no fue provisto.")
            try:
                idx = names.index(select)
            except ValueError:
                raise ValueError(f"`show_only='{select}'` no coincide con `solution_names`.")
            return [points[idx]], [names[idx]]
        raise TypeError("`show_only` debe ser None, int o str.")

    # --- contenedor eje derecho para recrearlo en cada redraw ---
    ax2_holder = {"ax2": None}

    def _draw(ax, par_vals, active_points, active_names,
              alpha_scale=1.0, include_labels=True, clear=True):
        # Limpia eje izquierdo
        if clear:
            ax.cla()
        # Elimina eje derecho previo si existe (arregla bug de acumulación)
        if ax2_holder["ax2"] is not None:
            try: ax2_holder["ax2"].remove()
            except Exception: pass
            ax2_holder["ax2"] = None

        # ¿necesitamos eje derecho?
        need_ax2 = dual_yaxis and plot_y
        ax2 = ax.twinx() if need_ax2 else None
        if need_ax2:
            ax2_holder["ax2"] = ax2
            ax2.set_ylabel(ylabel_right)

        # malla temporal común
        t_start = t0 - max(T_back, 0.0)
        t_end   = t0 + max(T, 0.0)
        tt = np.linspace(t_start, t_end, grid_n_t)

        # etiqueta simbólica de los estados para la leyenda
        state_labels = [str(states[0]), str(states[1])]

        # resolver y trazar
        for i, (x0, y0) in enumerate(active_points):
            sol_b = _solve(par_vals, x0, y0, (t0, t0 - T_back)) if T_back > 0 else None
            sol_f = _solve(par_vals, x0, y0, (t0, t0 + T))

            xt_list, yt_list = [], []
            if sol_b is not None and sol_b.sol is not None:
                z = sol_b.sol(tt[tt <= t0]);  xt_list.append(z[0]); yt_list.append(z[1])
            elif sol_b is not None and len(sol_b.t) > 1:
                mask = sol_b.t <= t0
                xt_list.append(np.interp(tt[tt <= t0], sol_b.t[mask], sol_b.y[0][mask]))
                yt_list.append(np.interp(tt[tt <= t0], sol_b.t[mask], sol_b.y[1][mask]))

            if sol_f is not None and sol_f.sol is not None:
                z = sol_f.sol(tt[tt >= t0]);  xt_list.append(z[0]); yt_list.append(z[1])
            elif sol_f is not None and len(sol_f.t) > 1:
                mask = sol_f.t >= t0
                xt_list.append(np.interp(tt[tt >= t0], sol_f.t[mask], sol_f.y[0][mask]))
                yt_list.append(np.interp(tt[tt >= t0], sol_f.t[mask], sol_f.y[1][mask]))

            if not xt_list:
                continue

            xt = np.concatenate(xt_list) if len(xt_list) > 1 else xt_list[0]
            yt = np.concatenate(yt_list) if len(yt_list) > 1 else yt_list[0]
            tt_left, tt_right = tt[tt <= t0], tt[tt >= t0]
            tt_plot = (np.concatenate([tt_left, tt_right])
                       if (len(xt_list) == 2)
                       else (tt_left if len(xt_list)==1 and T_back>0 else tt_right))

            # etiquetas (si hay nombres de solución, úsalos + nombre del estado)
            if active_names is not None:
                base = active_names[i]
                lbl_x = f"{base}: {state_labels[0]}"
                lbl_y = f"{base}: {state_labels[1]}"
            else:
                # sin nombres, etiqueta solo una vez por estado para no saturar
                lbl_x = state_labels[0] if i == 0 else None
                lbl_y = state_labels[1] if i == 0 else None

            # x(t) en eje izquierdo si corresponde
            if plot_x:
                ax.plot(tt_plot, xt, lw=2, color=color_x, alpha=alpha_scale, label=lbl_x)

            # y(t) en eje derecho si corresponde (o izquierdo si no hay ax2)
            if plot_y:
                target_ax = ax2 if (ax2 is not None) else ax
                target_ax.plot(tt_plot, yt, lw=2, color=color_y, alpha=alpha_scale, label=lbl_y)

            # puntos iniciales
            if mark_initial_points:
                if plot_x:
                    ax.scatter([t0], [x0], **init_point_style, color=color_x, zorder=5, alpha=alpha_scale)
                if plot_y:
                    (ax2 if (ax2 is not None) else ax).scatter([t0], [y0], **init_point_style,
                                                               color=color_y, zorder=5, alpha=alpha_scale)

        # equilibrios (líneas horizontales)
        if show_equilibria:
            par_eq = _par_vector(params) if not isinstance(par_vals, (list, tuple, np.ndarray)) else par_vals
            eq = _find_equilibrium(par_eq)
            if eq is not None:
                xeq, yeq = eq
                if plot_x:
                    ax.axhline(y=xeq, **equilibrium_line_style_x)
                if plot_y:
                    (ax2 if (ax2 is not None) else ax).axhline(y=yeq, **equilibrium_line_style_y)

        # límites/zoom
        if xlim is not None: ax.set_xlim(*xlim)
        else: ax.set_xlim(t_start, t_end)
        if ylim_left is not None: ax.set_ylim(*ylim_left)
        if plot_y and (ax2 is not None) and (ylim_right is not None):
            ax2.set_ylim(*ylim_right)

        # estética general
        ax.set_xlabel(xlabel)
        if not plot_y or (ax2 is None):
            if plot_x:
                ax.set_ylabel(ylabel)
            elif plot_y:
                ax.set_ylabel(ylabel_right)
        else:
            ax.set_ylabel(ylabel)
            ax2.set_ylabel(ylabel_right)

        ax.set_title(title)
        ax.grid(True, alpha=0.25)

        # leyenda
        def _dedup(handles, labels):
            seen=set(); hh=[]; ll=[]
            for h, lab in zip(handles, labels):
                if lab and lab not in seen:
                    seen.add(lab); hh.append(h); ll.append(lab)
            return hh, ll

        if legend_position:
            if (plot_y and ax2 is not None):
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                h, l = _dedup(h1+h2, l1+l2)
            else:
                h, l = _dedup(*ax.get_legend_handles_labels())

            if l:
                if legend_position == "outside_right":
                    try: ax.figure.subplots_adjust(right=0.80)
                    except Exception: pass
                    ax.legend(h, l, frameon=False, fontsize=9,
                              loc='center left', bbox_to_anchor=(1.02, 0.5))
                elif legend_position == "outside_bottom":
                    try: ax.figure.subplots_adjust(bottom=0.20)
                    except Exception: pass
                    ncol = max(2, min(4, len(l)))
                    ax.legend(h, l, frameon=False, fontsize=9,
                              loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=ncol)
                elif legend_position == "inside":
                    ax.legend(h, l, frameon=False, fontsize=9, loc='best')

    # ---------- SELECCIÓN INICIAL (no interactivo) ----------
    active_points, active_names = _select_points(init_points, solution_names, show_only)

    # ---------- ESTÁTICO ----------
    if not interactive:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        try:
            if hasattr(fig.canvas, "header_visible"):
                fig.canvas.header_visible = False
        except Exception:
            pass
        _draw(ax, _par_vector(params), active_points, active_names)
        plt.show()
        return

    # ---------- INTERACTIVO (con sliders y selector opcional) ----------
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        base = dict(params)
        for key, spec in slider_specs.items():
            base[key] = spec["init"]

        with plt.ioff():
            fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        try:
            if hasattr(fig.canvas, "header_visible"):
                fig.canvas.header_visible = False
        except Exception:
            pass

        # estado mutable para show_only en modo interactivo
        current_points, current_names = _select_points(init_points, solution_names, show_only)
        frozen_vals = _par_vector(base)
        _draw(ax, frozen_vals, current_points, current_names, alpha_scale=1.0, include_labels=True, clear=True)

        # ---- sliders (responsivos con flex-wrap) ----
        rows, slider_widgets = [], []
        for key, spec in slider_specs.items():
            if slider_use_latex:
                txt = spec["desc_latex"] if spec["desc_latex"] is not None else sp.latex(key)
                if hasattr(widgets, "HTMLMath"):
                    lbl = widgets.HTMLMath(value=_latexify(txt))
                elif hasattr(widgets, "Latex"):
                    lbl = widgets.Latex(value=_latexify(txt))
                elif hasattr(widgets, "HTML"):
                    lbl = widgets.HTML(value=_latexify(txt))
                else:
                    lbl = widgets.Label(value=str(key))
            else:
                lbl = widgets.Label(value=spec.get("desc", str(key)))
            lbl.layout = widgets.Layout(width='80px')

            sld = widgets.FloatSlider(
                min=spec["min"], max=spec["max"], step=spec["step"],
                value=spec["init"], continuous_update=True, readout=True,
                readout_format=".4f",   # más decimales en el readout
                description='', style={'description_width':'0px'},
                layout=widgets.Layout(width='240px')
            )

            row = widgets.HBox([lbl, sld])
            row.layout = widgets.Layout(width='360px', align_items='center', margin='0 12px 10px 0')
            rows.append(row); slider_widgets.append((key, sld))

        sliders_box = widgets.Box(
            rows,
            layout=widgets.Layout(display='flex', flex_flow='row wrap',
                                  justify_content='flex-start', align_items='center', width='100%')
        )

        # ---- selector de solución (Dropdown) opcional ----
        if solution_picker and len(init_points) > 1:
            if solution_names is None:
                opts = [(f"Solución {i+1}", i) for i in range(len(init_points))]
            else:
                opts = [(name, i) for i, name in enumerate(solution_names)]
            dd_solution = widgets.Dropdown(options=[("Todas", -1)] + opts,
                                           value=(-1 if show_only is None else
                                                  (show_only if isinstance(show_only,int) else
                                                   (solution_names.index(show_only) if solution_names else -1))),
                                           description="Ver:",
                                           layout=widgets.Layout(width="240px"))
        else:
            dd_solution = None

        backend = str(plt.get_backend()).lower()
        use_ipympl = ("ipympl" in backend) or ("widget" in backend and getattr(fig, "canvas", None) is not None)
        out_fig = widgets.Output()

        def _refresh():
            cur = dict(params)
            for k, w in slider_widgets:
                cur[k] = float(w.value)
            cur_vals = _par_vector(cur)

            _draw(ax, cur_vals, current_points, current_names,
                  alpha_scale=1.0, include_labels=True, clear=True)

            if use_ipympl:
                try: fig.canvas.draw_idle()
                except Exception: pass
            else:
                with out_fig:
                    clear_output(wait=True)
                    display(fig)

        def _on_slider_change(_):
            if freeze_initial:
                pass
            _refresh()

        def _on_solution_change(change):
            nonlocal current_points, current_names
            if change["name"] == "value":
                val = change["new"]
                if val == -1:
                    current_points, current_names = init_points, solution_names
                else:
                    current_points, current_names = _select_points(init_points, solution_names, int(val))
                _refresh()

        for _, w in slider_widgets:
            w.observe(_on_slider_change, names="value")
        if dd_solution is not None:
            dd_solution.observe(_on_solution_change, names="value")

        # Primer pintado inline si aplica
        if not use_ipympl:
            with out_fig:
                clear_output(wait=True)
                display(fig)
            if dd_solution is not None:
                display(widgets.VBox([sliders_box, dd_solution, out_fig], layout=widgets.Layout(width='100%')))
            else:
                display(widgets.VBox([sliders_box, out_fig], layout=widgets.Layout(width='100%')))
        else:
            if dd_solution is not None:
                display(widgets.VBox([sliders_box, dd_solution, fig.canvas], layout=widgets.Layout(width='100%')))
            else:
                display(widgets.VBox([sliders_box, fig.canvas], layout=widgets.Layout(width='100%')))

    except Exception as e:
        print("[Aviso] Interactividad requiere ipywidgets en entorno Jupyter. "
              "Se mostrará una gráfica estática. Detalle:", e)
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        _draw(ax, _par_vector(params), * _select_points(init_points, solution_names, show_only))
        plt.show()


#########################################################################
### Función que grafica series temporales de soluciones (x,y,z) en 3D ###
#########################################################################
def plot_time_series_solutions3D(
    eqs, states, indep,
    params=None,
    init_points=None,         # lista de tuplas [(x0,y0,z0), ...]
    solution_names=None,      # lista de nombres (mismo largo que init_points)

    # Filtrar soluciones a graficar
    show_only=None,           # None | int (índice) | str (nombre)
    solution_picker=False,    # si True (y interactive), agrega Dropdown para elegir solución

    t0=0.0, T=20.0, T_back=0.0,
    max_step=None,
    grid_n_t=400,
    fig_size=(9,5.5),
    interactive=False,
    sliders=None,             # {clave: {"min":..,"max":..,"step":..,"init":..,"desc":..,"desc_latex":..}}
    slider_use_latex=True,
    freeze_initial=False,

    # --- estética y controles ---
    title="Series temporales de las soluciones (3D)",
    xlabel="t",
    ylabel="x(t)",
    ylabel_right="y(t)",
    ylabel_right2="z(t)",
    legend_position="outside_bottom",   # "outside_bottom" | "outside_right" | "inside"
    show_equilibria=True,
    equilibrium_line_style_x=None,
    equilibrium_line_style_y=None,
    equilibrium_line_style_z=None,
    color_x="tab:blue",
    color_y="tab:orange",
    color_z="tab:green",
    init_point_style=None,
    mark_initial_points=True,
    eq_solver_opts=None,
    triple_yaxis=True,         # si True, usa tres ejes Y; si False, todo en el eje izquierdo
    var_to_plot="all",         # "all" | "x" | "y" | "z" | colecc. | estados sympy | textos

    # --- límites para “zoom” ---
    xlim=None,
    ylim_left=None,           # para x
    ylim_right=None,          # para y
    ylim_right2=None          # para z
):
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import root
    from matplotlib.transforms import Affine2D

    # ---- compat (símbolos / funciones indefinidas / aplicadas) ----
    try:
        from sympy.core.function import AppliedUndef as _AppliedUndef, UndefinedFunction as _UndefinedFunction, FunctionClass as _FunctionClass
    except Exception:
        try:
            from sympy.core.function import FunctionClass as _FunctionClass
        except Exception:
            _FunctionClass = type(sp.Function('f'))
        _AppliedUndef = type(sp.Function('f')(sp.Symbol('x')))
        _UndefinedFunction = getattr(sp, 'UndefinedFunction', _FunctionClass)

    def _is_slider_key(obj):
        return (isinstance(obj, sp.Symbol)
                or isinstance(obj, _UndefinedFunction)
                or isinstance(obj, _AppliedUndef))

    def _latexify(s):
        if s is None: return None
        t = str(s).strip()
        if t.startswith("$$") and t.endswith("$$"): t = t[2:-2].strip()
        if not (t.startswith("$") and t.endswith("$")): t = f"${t}$"
        return t

    # ----- defaults -----
    if params is None: params = {}
    if init_points is None or len(init_points)==0:
        init_points = [(0.0, 0.0, 0.0)]
    if max_step is None:
        max_step = (abs(T)+abs(T_back))/500.0 if (T or T_back) else 0.05
    if eq_solver_opts is None:
        eq_solver_opts = dict(tol=1e-9, maxiter=200)
    if equilibrium_line_style_x is None:
        equilibrium_line_style_x = dict(color=color_x, linestyle="--", linewidth=1.2, alpha=0.9)
    if equilibrium_line_style_y is None:
        equilibrium_line_style_y = dict(color=color_y, linestyle="-.", linewidth=1.2, alpha=0.9)
    if equilibrium_line_style_z is None:
        equilibrium_line_style_z = dict(color=color_z, linestyle=":", linewidth=1.2, alpha=0.9)
    if init_point_style is None:
        init_point_style = dict(marker="o", s=50, edgecolors="k", linewidths=0.6)

    # --- helper: normaliza qué variable(s) graficar ---
    def _norm_var_to_plot(v, states):
        """
        Acepta:
        - "all", "x", "y", "z"
        - el/los propios estados sympy: x(t), y(t), z(t)
        - nombres como string: "x", "y", "z", "x(t)", ...
        Devuelve: (plot_x, plot_y, plot_z)
        """
        def _clean(s):
            s = str(s).strip()
            if s.startswith("$$") and s.endswith("$$"): s = s[2:-2].strip()
            if s.startswith("$") and s.endswith("$"): s = s[1:-1].strip()
            return s

        if v is None or str(v).lower() in {"all", "both"}:
            return True, True, True

        def _keys(st):
            K = {st, str(st), _clean(st)}
            func = getattr(st, "func", None)
            if func is not None:
                K.add(func)
                name = getattr(func, "__name__", None)
                if name:
                    K.add(name); K.add(_clean(name))
            name2 = getattr(st, "name", None)
            if name2:
                K.add(name2); K.add(_clean(name2))
            return K

        keys = [_keys(s) for s in states]
        targets = v if isinstance(v, (list, tuple, set)) else [v]
        want = [False, False, False]
        for t in targets:
            t_clean = _clean(t)
            for i in range(3):
                if (t in keys[i]) or (t_clean in keys[i]):
                    want[i] = True

        s = _clean(v).lower()
        if not any(want):
            if s in {"x","x(t)"}: want[0]=True
            if s in {"y","y(t)"}: want[1]=True
            if s in {"z","z(t)"}: want[2]=True

        if not any(want):
            raise ValueError("`var_to_plot` no coincide con ninguno de los estados provistos en `states`.")
        return tuple(want)

    # --- qué variable(es) graficar ---
    plot_x, plot_y, plot_z = _norm_var_to_plot(var_to_plot, states)

    # --- nombres de soluciones ---
    if solution_names is not None:
        if len(solution_names) != len(init_points or []):
            raise ValueError("`solution_names` debe tener el mismo tamaño que `init_points`.")
        solution_names = [str(n) for n in solution_names]

    # ----- sliders (normalización) -----
    slider_specs = {}
    if interactive:
        if not isinstance(sliders, dict) or not sliders:
            raise ValueError("Con interactive=True, pasa `sliders` como dict no vacío.")
        for key, spec in sliders.items():
            if not _is_slider_key(key):
                raise TypeError("Claves de sliders: sp.Symbol, sp.Function('a') o aplicadas (p.ej. a(t0)).")
            if "min" not in spec or "max" not in spec:
                raise ValueError(f"Faltan 'min'/'max' en slider de {key}.")
            mn, mx = float(spec["min"]), float(spec["max"])
            slider_specs[key] = {
                "min": mn, "max": mx,
                "step": float(spec.get("step", (mx-mn)/100.0 if mx>mn else 0.1)),
                "init": float(spec.get("init", (mn+mx)/2.0)),
                "desc": spec.get("desc", str(key)),
                "desc_latex": spec.get("desc_latex", None),
            }

    # ----- simbólico -> numérico -----
    f_expr = eqs[0].rhs
    g_expr = eqs[1].rhs
    h_expr = eqs[2].rhs
    x, y, z = states
    t = indep

    x_s, y_s, z_s = sp.symbols('x_s y_s z_s', real=True)
    f_xyz = sp.simplify(f_expr.subs({x: x_s, y: y_s, z: z_s}))
    g_xyz = sp.simplify(g_expr.subs({x: x_s, y: y_s, z: z_s}))
    h_xyz = sp.simplify(h_expr.subs({x: x_s, y: y_s, z: z_s}))

    all_param_objs = set(params.keys()) | set(slider_specs.keys())
    def _mkname(obj, k): return f"__par{k}__{sp.srepr(obj)}"
    repl_to_proxy, order_originals = {}, []
    for i, p in enumerate(sorted(all_param_objs, key=lambda z_: getattr(z_, "name", str(z_))), start=1):
        repl_to_proxy[p] = p if isinstance(p, sp.Symbol) else sp.Symbol(_mkname(p, i))
        order_originals.append(p)
    order_proxies = [repl_to_proxy[p] for p in order_originals]

    f_sub, g_sub, h_sub = (f_xyz.xreplace(repl_to_proxy),
                           g_xyz.xreplace(repl_to_proxy),
                           h_xyz.xreplace(repl_to_proxy))
    f_num = sp.lambdify((t, x_s, y_s, z_s, *order_proxies), f_sub, "numpy")
    g_num = sp.lambdify((t, x_s, y_s, z_s, *order_proxies), g_sub, "numpy")
    h_num = sp.lambdify((t, x_s, y_s, z_s, *order_proxies), h_sub, "numpy")

    # ----- utilidades numéricas -----
    def _par_vector(pdict):
        vals = []
        for p in order_originals:
            if p in pdict: vals.append(float(pdict[p]))
            elif p in params: vals.append(float(params[p]))
            else: vals.append(0.0)
        return vals

    def _solve(par_vals, x0, y0, z0, t_span):
        def ode_sys(t_, w):
            return [float(f_num(t_, w[0], w[1], w[2], *par_vals)),
                    float(g_num(t_, w[0], w[1], w[2], *par_vals)),
                    float(h_num(t_, w[0], w[1], w[2], *par_vals))]
        return solve_ivp(ode_sys, t_span, [x0, y0, z0],
                         max_step=max_step, rtol=1e-6, atol=1e-9, dense_output=True)

    def _find_equilibrium(par_vals, seeds=7):
        xs = np.linspace(-1, 1, seeds)
        ys = np.linspace(-1, 1, seeds)
        zs = np.linspace(-1, 1, seeds)
        x_centers = [p[0] for p in init_points] + [0.0]
        y_centers = [p[1] for p in init_points] + [0.0]
        z_centers = [p[2] for p in init_points] + [0.0]
        tried = set()
        for xc in x_centers:
            for yc in y_centers:
                for zc in z_centers:
                    for dx in xs:
                        for dy in ys:
                            for dz in zs:
                                guess = (float(xc+dx), float(yc+dy), float(zc+dz))
                                if guess in tried: continue
                                tried.add(guess)
                                try:
                                    sol = root(lambda w: [f_num(t0, w[0], w[1], w[2], *par_vals),
                                                          g_num(t0, w[0], w[1], w[2], *par_vals),
                                                          h_num(t0, w[0], w[1], w[2], *par_vals)],
                                               x0=np.array(guess, dtype=float),
                                               tol=eq_solver_opts.get("tol", 1e-9),
                                               options={"maxiter": eq_solver_opts.get("maxiter", 200)})
                                    if sol.success and np.isfinite(sol.x).all():
                                        return float(sol.x[0]), float(sol.x[1]), float(sol.x[2])
                                except Exception:
                                    pass
        return None

    # --- helper: aplica selección show_only ---
    def _select_points(points, names, select):
        if select is None:
            return points, names
        if isinstance(select, int):
            if select < 0 or select >= len(points):
                raise IndexError("`show_only` fuera de rango.")
            sel_pts = [points[select]]
            sel_nms = None if names is None else [names[select]]
            return sel_pts, sel_nms
        if isinstance(select, str):
            if names is None:
                raise ValueError("`show_only` es str, pero `solution_names` no fue provisto.")
            try:
                idx = names.index(select)
            except ValueError:
                raise ValueError(f"`show_only='{select}'` no coincide con `solution_names`.")
            return [points[idx]], [names[idx]]
        raise TypeError("`show_only` debe ser None, int o str.")

    # --- contenedor ejes derechos para recrearlos en cada redraw ---
    ax2_holder = {"ax2": None, "ax3": None}

    def _make_third_axis(ax):
        # Crea un segundo eje derecho desplazado para z
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.12))
        # Evita que el borde se tape
        ax3.spines["right"].set_visible(True)
        return ax3

    def _draw(ax, par_vals, active_points, active_names,
              alpha_scale=1.0, include_labels=True, clear=True):
        # Limpia eje izquierdo
        if clear:
            ax.cla()
        # Elimina ejes derechos previos si existen (evita acumulación)
        for k in ("ax2","ax3"):
            if ax2_holder[k] is not None:
                try: ax2_holder[k].remove()
                except Exception: pass
                ax2_holder[k] = None

        # ¿necesitamos ejes derechos?
        need_ax2 = triple_yaxis and (plot_y or plot_z)
        ax2 = ax.twinx() if need_ax2 else None
        if need_ax2 and plot_y:
            ax2_holder["ax2"] = ax2
            ax2.set_ylabel(ylabel_right, color=color_y)
            ax2.tick_params(axis='y', labelcolor=color_y)

        ax3 = None
        if need_ax2 and plot_z:
            ax3 = _make_third_axis(ax)
            ax2_holder["ax3"] = ax3
            ax3.set_ylabel(ylabel_right2, color=color_z)
            ax3.tick_params(axis='y', labelcolor=color_z)

        # malla temporal común
        t_start = t0 - max(T_back, 0.0)
        t_end   = t0 + max(T, 0.0)
        tt = np.linspace(t_start, t_end, grid_n_t)

        # etiquetas de los estados para la leyenda
        state_labels = [str(states[0]), str(states[1]), str(states[2])]

        # resolver y trazar
        for i, (x0, y0, z0) in enumerate(active_points):
            sol_b = _solve(par_vals, x0, y0, z0, (t0, t0 - T_back)) if T_back > 0 else None
            sol_f = _solve(par_vals, x0, y0, z0, (t0, t0 + T))

            xt_list, yt_list, zt_list = [], [], []
            if sol_b is not None and sol_b.sol is not None:
                w = sol_b.sol(tt[tt <= t0]);  xt_list.append(w[0]); yt_list.append(w[1]); zt_list.append(w[2])
            elif sol_b is not None and len(sol_b.t) > 1:
                mask = sol_b.t <= t0
                tseg = tt[tt <= t0]
                xt_list.append(np.interp(tseg, sol_b.t[mask], sol_b.y[0][mask]))
                yt_list.append(np.interp(tseg, sol_b.t[mask], sol_b.y[1][mask]))
                zt_list.append(np.interp(tseg, sol_b.t[mask], sol_b.y[2][mask]))

            if sol_f is not None and sol_f.sol is not None:
                w = sol_f.sol(tt[tt >= t0]);  xt_list.append(w[0]); yt_list.append(w[1]); zt_list.append(w[2])
            elif sol_f is not None and len(sol_f.t) > 1:
                mask = sol_f.t >= t0
                tseg = tt[tt >= t0]
                xt_list.append(np.interp(tseg, sol_f.t[mask], sol_f.y[0][mask]))
                yt_list.append(np.interp(tseg, sol_f.t[mask], sol_f.y[1][mask]))
                zt_list.append(np.interp(tseg, sol_f.t[mask], sol_f.y[2][mask]))

            if not xt_list:
                continue

            xt = np.concatenate(xt_list) if len(xt_list) > 1 else xt_list[0]
            yt = np.concatenate(yt_list) if len(yt_list) > 1 else yt_list[0]
            zt = np.concatenate(zt_list) if len(zt_list) > 1 else zt_list[0]
            tt_left, tt_right = tt[tt <= t0], tt[tt >= t0]
            tt_plot = (np.concatenate([tt_left, tt_right])
                       if (len(xt_list) == 2)
                       else (tt_left if len(xt_list)==1 and T_back>0 else tt_right))

            # etiquetas (si hay nombres de solución, úsalos + nombre del estado)
            if active_names is not None:
                base = active_names[i]
                lbl_x = f"{base}: {state_labels[0]}"
                lbl_y = f"{base}: {state_labels[1]}"
                lbl_z = f"{base}: {state_labels[2]}"
            else:
                lbl_x = state_labels[0] if i == 0 else None
                lbl_y = state_labels[1] if i == 0 else None
                lbl_z = state_labels[2] if i == 0 else None

            # x(t) en eje izquierdo
            if plot_x:
                ax.plot(tt_plot, xt, lw=2, color=color_x, alpha=alpha_scale, label=lbl_x)

            # y(t) en eje derecho principal (o izquierdo si no hay ejes extra)
            if plot_y:
                target_ax = ax2 if (ax2 is not None) else ax
                target_ax.plot(tt_plot, yt, lw=2, color=color_y, alpha=alpha_scale, label=lbl_y)

            # z(t) en segundo eje derecho desplazado (o izquierdo si no hay ejes extra)
            if plot_z:
                target_ax = ax3 if (ax3 is not None) else (ax2 if (ax2 is not None) else ax)
                target_ax.plot(tt_plot, zt, lw=2, color=color_z, alpha=alpha_scale, label=lbl_z)

            # puntos iniciales
            if mark_initial_points:
                if plot_x:
                    ax.scatter([t0], [x0], **init_point_style, color=color_x, zorder=5, alpha=alpha_scale)
                if plot_y:
                    (ax2 if (ax2 is not None) else ax).scatter([t0], [y0], **init_point_style,
                                                               color=color_y, zorder=5, alpha=alpha_scale)
                if plot_z:
                    (ax3 if (ax3 is not None) else (ax2 if (ax2 is not None) else ax)).scatter(
                        [t0], [z0], **init_point_style, color=color_z, zorder=5, alpha=alpha_scale
                    )

        # equilibrios (líneas horizontales)
        if show_equilibria:
            par_eq = _par_vector(params) if not isinstance(par_vals, (list, tuple, np.ndarray)) else par_vals
            eq = _find_equilibrium(par_eq)
            if eq is not None:
                xeq, yeq, zeq = eq
                if plot_x:
                    ax.axhline(y=xeq, **equilibrium_line_style_x)
                if plot_y:
                    (ax2 if (ax2 is not None) else ax).axhline(y=yeq, **equilibrium_line_style_y)
                if plot_z:
                    (ax3 if (ax3 is not None) else (ax2 if (ax2 is not None) else ax)).axhline(
                        y=zeq, **equilibrium_line_style_z
                    )

        # límites/zoom
        if xlim is not None: ax.set_xlim(*xlim)
        else:
            ax.set_xlim(t_start, t_end)
        if ylim_left is not None: ax.set_ylim(*ylim_left)
        if plot_y and (ax2 is not None) and (ylim_right is not None):
            ax2.set_ylim(*ylim_right)
        if plot_z and (ax3 is not None) and (ylim_right2 is not None):
            ax3.set_ylim(*ylim_right2)

        # estética general
        ax.set_xlabel(xlabel)
        if not triple_yaxis:
            # todo en el izquierdo
            if plot_x and not (plot_y or plot_z):
                ax.set_ylabel(ylabel)
            elif plot_y and not (plot_x or plot_z):
                ax.set_ylabel(ylabel_right)
            elif plot_z and not (plot_x or plot_y):
                ax.set_ylabel(ylabel_right2)
            else:
                ax.set_ylabel("valor")
        else:
            # múltiples ejes con colores
            ax.set_ylabel(ylabel, color=color_x)
            ax.tick_params(axis='y', labelcolor=color_x)
            if ax2 is None and (plot_y or plot_z):
                # caso triple_yaxis=False o sin ax2
                pass

        ax.set_title(title)
        ax.grid(True, alpha=0.25)

        # leyenda (unificada deduplicada)
        def _dedup(handles, labels):
            seen=set(); hh=[]; ll=[]
            for h, lab in zip(handles, labels):
                if lab and lab not in seen:
                    seen.add(lab); hh.append(h); ll.append(lab)
            return hh, ll

        if legend_position:
            h_all, l_all = ax.get_legend_handles_labels()
            if ax2_holder["ax2"] is not None:
                h2, l2 = ax2_holder["ax2"].get_legend_handles_labels()
                h_all += h2; l_all += l2
            if ax2_holder["ax3"] is not None:
                h3, l3 = ax2_holder["ax3"].get_legend_handles_labels()
                h_all += h3; l_all += l3
            h, l = _dedup(h_all, l_all)
            if l:
                if legend_position == "outside_right":
                    try: ax.figure.subplots_adjust(right=0.82)
                    except Exception: pass
                    ax.legend(h, l, frameon=False, fontsize=9,
                              loc='center left', bbox_to_anchor=(1.02, 0.5))
                elif legend_position == "outside_bottom":
                    try: ax.figure.subplots_adjust(bottom=0.22)
                    except Exception: pass
                    ncol = max(2, min(4, len(l)))
                    ax.legend(h, l, frameon=False, fontsize=9,
                              loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=ncol)
                elif legend_position == "inside":
                    ax.legend(h, l, frameon=False, fontsize=9, loc='best')

    # ---------- SELECCIÓN INICIAL (no interactivo) ----------
    active_points, active_names = _select_points(init_points, solution_names, show_only)

    # ---------- ESTÁTICO ----------
    if not interactive:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        try:
            if hasattr(fig.canvas, "header_visible"):
                fig.canvas.header_visible = False
        except Exception:
            pass
        _draw(ax, _par_vector(params), active_points, active_names)
        plt.show()
        return

    # ---------- INTERACTIVO (con sliders y selector opcional) ----------
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        base = dict(params)
        for key, spec in slider_specs.items():
            base[key] = spec["init"]

        with plt.ioff():
            fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        try:
            if hasattr(fig.canvas, "header_visible"):
                fig.canvas.header_visible = False
        except Exception:
            pass

        # estado mutable para show_only en modo interactivo
        current_points, current_names = _select_points(init_points, solution_names, show_only)
        frozen_vals = _par_vector(base)
        _draw(ax, frozen_vals, current_points, current_names, alpha_scale=1.0, include_labels=True, clear=True)

        # ---- sliders (responsivos con flex-wrap) ----
        rows, slider_widgets = [], []
        for key, spec in slider_specs.items():
            if slider_use_latex:
                txt = spec["desc_latex"] if spec["desc_latex"] is not None else sp.latex(key)
                if hasattr(widgets, "HTMLMath"):
                    lbl = widgets.HTMLMath(value=_latexify(txt))
                elif hasattr(widgets, "Latex"):
                    lbl = widgets.Latex(value=_latexify(txt))
                elif hasattr(widgets, "HTML"):
                    lbl = widgets.HTML(value=_latexify(txt))
                else:
                    lbl = widgets.Label(value=str(key))
            else:
                lbl = widgets.Label(value=spec.get("desc", str(key)))
            lbl.layout = widgets.Layout(width='80px')

            sld = widgets.FloatSlider(
                min=spec["min"], max=spec["max"], step=spec["step"],
                value=spec["init"], continuous_update=True, readout=True,
                readout_format=".4f",
                description='', style={'description_width':'0px'},
                layout=widgets.Layout(width='240px')
            )

            row = widgets.HBox([lbl, sld])
            row.layout = widgets.Layout(width='360px', align_items='center', margin='0 12px 10px 0')
            rows.append(row); slider_widgets.append((key, sld))

        sliders_box = widgets.Box(
            rows,
            layout=widgets.Layout(display='flex', flex_flow='row wrap',
                                  justify_content='flex-start', align_items='center', width='100%')
        )

        # ---- selector de solución (Dropdown) opcional ----
        if solution_picker and len(init_points) > 1:
            if solution_names is None:
                opts = [(f"Solución {i+1}", i) for i in range(len(init_points))]
            else:
                opts = [(name, i) for i, name in enumerate(solution_names)]
            dd_solution = widgets.Dropdown(options=[("Todas", -1)] + opts,
                                           value=(-1 if show_only is None else
                                                  (show_only if isinstance(show_only,int) else
                                                   (solution_names.index(show_only) if solution_names else -1)) ),
                                           description="Ver:",
                                           layout=widgets.Layout(width="240px"))
        else:
            dd_solution = None

        backend = str(plt.get_backend()).lower()
        use_ipympl = ("ipympl" in backend) or ("widget" in backend and getattr(fig, "canvas", None) is not None)
        out_fig = widgets.Output()

        def _refresh():
            cur = dict(params)
            for k, w in slider_widgets:
                cur[k] = float(w.value)
            cur_vals = _par_vector(cur)

            _draw(ax, cur_vals, current_points, current_names,
                  alpha_scale=1.0, include_labels=True, clear=True)

            if use_ipympl:
                try: fig.canvas.draw_idle()
                except Exception: pass
            else:
                with out_fig:
                    clear_output(wait=True)
                    display(fig)

        def _on_slider_change(_):
            if freeze_initial:
                pass
            _refresh()

        def _on_solution_change(change):
            nonlocal current_points, current_names
            if change["name"] == "value":
                val = change["new"]
                if val == -1:
                    current_points, current_names = init_points, solution_names
                else:
                    current_points, current_names = _select_points(init_points, solution_names, int(val))
                _refresh()

        for _, w in slider_widgets:
            w.observe(_on_slider_change, names="value")
        if dd_solution is not None:
            dd_solution.observe(_on_solution_change, names="value")

        # Primer pintado inline si aplica
        if not use_ipympl:
            with out_fig:
                clear_output(wait=True)
                display(fig)
            if dd_solution is not None:
                display(widgets.VBox([sliders_box, dd_solution, out_fig], layout=widgets.Layout(width='100%')))
            else:
                display(widgets.VBox([sliders_box, out_fig], layout=widgets.Layout(width='100%')))
        else:
            if dd_solution is not None:
                display(widgets.VBox([sliders_box, dd_solution, fig.canvas], layout=widgets.Layout(width='100%')))
            else:
                display(widgets.VBox([sliders_box, fig.canvas], layout=widgets.Layout(width='100%')))

    except Exception as e:
        print("[Aviso] Interactividad requiere ipywidgets en entorno Jupyter. "
              "Se mostrará una gráfica estática. Detalle:", e)
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        _draw(ax, _par_vector(params), * _select_points(init_points, solution_names, show_only))
        plt.show()



#########################################################
### Campo-pendiente y soluciones en 3D (x,y,z vs t)   ###
#########################################################
def plot_slope_field_and_solutions3D(
    eqs, states, indep,
    params=None,
    xlim=(-3,3), ylim=(-3,3), zlim=(-3,3), grid_n=9,
    init_points=None,              # [(x0,y0,z0), ...]
    t0=0.0, T=20.0, max_step=None,
    normalize_field=True,
    fig_size=(8,7),

    # --- Campo vectorial ---
    arrow_alpha=0.6,
    show_field=False,
    # modo de longitud de flecha: "relative" (fracción de la diagonal) o "data" (unidades de datos)
    arrow_length_mode="relative",
    arrow_length_rel=0.06,     # si mode="relative"
    arrow_length=0.10,         # si mode="data"

    # --- Estética general ---
    equal_aspect=True,
    interactive=False,
    sliders=None,
    slider_use_latex=True,
    freeze_initial=False,

    sol_color="tab:blue",          # color único (default, se mantiene)
    sol_colors=None,               # <<< NUEVO: colores por trayectoria
    initial_color="black",
    title="",
    xlabel="x", ylabel="y", zlabel="z",
    legend_outside=True,
    show_equilibria=False,
    equilibrium_style=None,
    eq_solver_opts=None,

    axes_only=False,               # sin grid/panes, sólo ejes
    auto_limits="solutions",       # "static" | "solutions"
    fill_figure=False,             # ocupa todo el canvas

    time_direction="forward",      # "forward" | "backward" | "both"

    tick_labelsize=8,
    axis_labelsize=11,
    title_fontsize=12,
    
    # --- NUEVO: escalas de unidad por eje ---
    xunit_scale=1.0,
    yunit_scale=1.0,
    zunit_scale=1.0
    
):
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import root
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # ===== COMPAT: Shim SymPy
    try:
        from sympy.core.function import AppliedUndef as _AppliedUndef, UndefinedFunction as _UndefinedFunction, FunctionClass as _FunctionClass
    except Exception:
        try:
            from sympy.core.function import FunctionClass as _FunctionClass
        except Exception:
            _FunctionClass = type(sp.Function('f'))
        _AppliedUndef = type(sp.Function('f')(sp.Symbol('x')))
        _UndefinedFunction = getattr(sp, 'UndefinedFunction', _FunctionClass)

    def _latexify(s):
        if s is None: return ""
        t = str(s).strip()
        if t.startswith("$$") and t.endswith("$$"): return t
        if t.startswith("$") and t.endswith("$"): return t
        return f"${t}$"

    # ---- Defaults/validaciones
    if params is None: params = {}
    if init_points is None: init_points = []
    if len(eqs)!=3 or len(states)!=3:
        raise ValueError("Se esperan exactamente tres ecuaciones y tres estados.")
    if interactive and (not isinstance(sliders, dict) or not sliders):
        raise ValueError("Con interactive=True, pasa `sliders` como dict no vacío.")
    if auto_limits not in ("static","solutions"):
        raise ValueError("auto_limits debe ser 'static' o 'solutions'.")
    if time_direction not in ("forward","backward","both"):
        raise ValueError("time_direction: 'forward' | 'backward' | 'both'.")
    if arrow_length_mode not in ("relative","data"):
        raise ValueError("arrow_length_mode: 'relative' | 'data'.")

    if equilibrium_style is None:
        equilibrium_style = dict(marker="o", s=50, color="crimson",
                                 edgecolors="k", linewidths=0.6, zorder=8, depthshade=True)
    if eq_solver_opts is None:
        eq_solver_opts = dict(tol=1e-9, maxiter=200)

    # ---- Sliders
    def _is_slider_key(obj):
        return (isinstance(obj, sp.Symbol)
                or isinstance(obj, _UndefinedFunction)
                or isinstance(obj, _AppliedUndef))

    slider_specs = {}
    if interactive:
        for key, spec in sliders.items():
            if not _is_slider_key(key):
                raise TypeError("Claves de sliders: sp.Symbol, sp.Function('a') o aplicadas (p.ej. a(t0)).")
            if "min" not in spec or "max" not in spec:
                raise ValueError(f"Faltan 'min'/'max' en slider de {key}.")
            mn, mx = float(spec["min"]), float(spec["max"])
            slider_specs[key] = {
                "min": mn, "max": mx,
                "step": float(spec.get("step", (mx-mn)/100.0 if mx>mn else 0.1)),
                "init": float(spec.get("init", (mn+mx)/2.0)),
                "desc": spec.get("desc", str(key)),
                "desc_latex": spec.get("desc_latex", None),
            }

    # ---- Simbólico -> numérico
    f_expr = eqs[0].rhs; g_expr = eqs[1].rhs; h_expr = eqs[2].rhs
    x_s, y_s, z_s = sp.symbols('x_s y_s z_s', real=True)

    f_xyz = sp.simplify(f_expr.subs({states[0]: x_s, states[1]: y_s, states[2]: z_s}))
    g_xyz = sp.simplify(g_expr.subs({states[0]: x_s, states[1]: y_s, states[2]: z_s}))
    h_xyz = sp.simplify(h_expr.subs({states[0]: x_s, states[1]: y_s, states[2]: z_s}))

    all_param_objs = set(params.keys()) | set(slider_specs.keys())
    def _mkname(obj, k): return f"__par{k}__{sp.srepr(obj)}"
    repl_to_proxy, order_originals = {}, []
    for i, p in enumerate(sorted(all_param_objs, key=lambda z: getattr(z, "name", str(z))), start=1):
        repl_to_proxy[p] = p if isinstance(p, sp.Symbol) else sp.Symbol(_mkname(p, i))
        order_originals.append(p)
    order_proxies = [repl_to_proxy[p] for p in order_originals]

    f_sub, g_sub, h_sub = (f_xyz.xreplace(repl_to_proxy),
                           g_xyz.xreplace(repl_to_proxy),
                           h_xyz.xreplace(repl_to_proxy))

    f_num = sp.lambdify((indep, x_s, y_s, z_s, *order_proxies), f_sub, "numpy")
    g_num = sp.lambdify((indep, x_s, y_s, z_s, *order_proxies), g_sub, "numpy")
    h_num = sp.lambdify((indep, x_s, y_s, z_s, *order_proxies), h_sub, "numpy")

    xmin, xmax = xlim; ymin, ymax = ylim; zmin, zmax = zlim
    if max_step is None:
        max_step = (abs(T)/500.0) if T!=0 else 0.05

    def _par_vector(pdict):
        vals = []
        for p in order_originals:
            if p in pdict: vals.append(float(pdict[p]))
            elif p in params: vals.append(float(params[p]))
            else: vals.append(0.0)
        return vals

    # --- NUEVO: normalizador de colores por solución
    def _normalize_sol_colors(spec, m):
        if m == 0: return None
        if spec is None: return None
        if isinstance(spec, (list, tuple)):
            if len(spec) == m: return list(spec)
            if len(spec) == 1 and m > 1: return [spec[0]] * m
            raise ValueError("`sol_colors` debe tener longitud 1 o igual al número de puntos iniciales.")
        return [spec] * m  # string/escala → replica

    sol_colors_b = _normalize_sol_colors(sol_colors, len(init_points))

    def _find_equilibria(par_vals, seeds_per_dim=4, merge_tol=1e-3):
        pts = []
        sx = np.linspace(xmin, xmax, seeds_per_dim)
        sy = np.linspace(ymin, ymax, seeds_per_dim)
        sz = np.linspace(zmin, zmax, seeds_per_dim)
        for x0 in sx:
            for y0 in sy:
                for z0 in sz:
                    try:
                        sol = root(lambda w: [f_num(t0, w[0], w[1], w[2], *par_vals),
                                              g_num(t0, w[0], w[1], w[2], *par_vals),
                                              h_num(t0, w[0], w[1], w[2], *par_vals)],
                                   x0=np.array([x0, y0, z0], dtype=float),
                                   tol=eq_solver_opts.get("tol", 1e-9),
                                   options={"maxiter": eq_solver_opts.get("maxiter", 200)})
                        if sol.success:
                            xs, ys, zs = map(float, sol.x)
                            if not (xmin<=xs<=xmax and ymin<=ys<=ymax and zmin<=zs<=zmax):
                                continue
                            keep = True
                            for (xa, ya, za) in pts:
                                if (xa-xs)**2 + (ya-ys)**2 + (za-zs)**2 <= merge_tol**2:
                                    keep = False; break
                            if keep: pts.append((xs, ys, zs))
                    except Exception:
                        pass
        return pts

    # ---- Dibujo principal
    def _draw(ax, par_vals, alpha_scale=1.0, include_labels=True, clear=True):
        if clear: ax.cla()

        # Integración primero (para auto_limits)
        all_pts = []
        def ode_sys(t, z):
            xval, yval, zval = z
            return [float(f_num(t, xval, yval, zval, *par_vals)),
                    float(g_num(t, xval, yval, zval, *par_vals)),
                    float(h_num(t, xval, yval, zval, *par_vals))]
        for idx, (x0, y0, z0) in enumerate(init_points):
            color_i = (sol_colors_b[idx] if sol_colors_b is not None else sol_color)  # <<< NUEVO

            if time_direction in ("forward","both"):
                try:
                    sf = solve_ivp(ode_sys, (t0, t0+T), [x0, y0, z0],
                                   max_step=max_step, rtol=1e-6, atol=1e-9)
                    if sf.y.size:
                        ax.plot3D(sf.y[0], sf.y[1], sf.y[2],
                                  lw=2, color=color_i,     # <<< NUEVO
                                  label=(f"({x0:.2f},{y0:.2f},{z0:.2f}) →" if include_labels else None),
                                  alpha=alpha_scale)
                        all_pts.append(sf.y.T)
                except Exception:
                    pass

            if time_direction in ("backward","both"):
                try:
                    sb = solve_ivp(ode_sys, (t0, t0-T), [x0, y0, z0],
                                   max_step=max_step, rtol=1e-6, atol=1e-9)
                    if sb.y.size:
                        ax.plot3D(sb.y[0][::-1], sb.y[1][::-1], sb.y[2][::-1],
                                  lw=2, color=color_i,     # <<< NUEVO
                                  alpha=alpha_scale)
                        all_pts.append(sb.y.T)
                except Exception:
                    pass

            ax.scatter([x0], [y0], [z0], s=18, c=initial_color, alpha=alpha_scale)

        # Equilibrios
        if show_equilibria:
            pts = _find_equilibria(par_vals)
            if pts:
                ex, ey, ez = zip(*pts)
                ax.scatter(ex, ey, ez, **equilibrium_style)
                if auto_limits == "solutions":
                    all_pts.append(np.array(list(zip(ex,ey,ez))))

        # Límites actuales
        if auto_limits == "solutions" and all_pts:
            P = np.vstack(all_pts)
            xmin2, ymin2, zmin2 = np.nanmin(P, axis=0)
            xmax2, ymax2, zmax2 = np.nanmax(P, axis=0)
            def pad(a,b):
                span = max(1e-9, b - a); d = 0.05 * span
                return a - d, b + d
            xmin_, xmax_ = pad(xmin2, xmax2)
            ymin_, ymax_ = pad(ymin2, ymax2)
            zmin_, zmax_ = pad(zmin2, zmax2)
        else:
            xmin_, xmax_ = xmin, xmax
            ymin_, ymax_ = ymin, ymax
            zmin_, zmax_ = zmin, zmax

        # Campo vectorial: grid con los límites *actuales*
        if show_field:
            X = np.linspace(xmin_, xmax_, grid_n)
            Y = np.linspace(ymin_, ymax_, grid_n)
            Z = np.linspace(zmin_, zmax_, grid_n)
            XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing='xy')

            U = f_num(t0, XX, YY, ZZ, *par_vals)
            V = g_num(t0, XX, YY, ZZ, *par_vals)
            W = h_num(t0, XX, YY, ZZ, *par_vals)
            U = np.where(np.isfinite(U), U, 0.0)
            V = np.where(np.isfinite(V), V, 0.0)
            W = np.where(np.isfinite(W), W, 0.0)

            if normalize_field:
                m = np.sqrt(U*U + V*V + W*W)
                U = np.divide(U, m, out=np.zeros_like(U), where=m>0)
                V = np.divide(V, m, out=np.zeros_like(V), where=m>0)
                W = np.divide(W, m, out=np.zeros_like(W), where=m>0)

            # longitud de flecha adaptativa
            if arrow_length_mode == "relative":
                diag = np.sqrt((xmax_-xmin_)**2 + (ymax_-ymin_)**2 + (zmax_-zmin_)**2)
                _len = max(1e-12, float(arrow_length_rel) * float(diag))
            else:
                _len = float(arrow_length)

            # aplanar para quiver3D
            ax.quiver(XX.ravel(), YY.ravel(), ZZ.ravel(),
                      U.ravel(), V.ravel(), W.ravel(),
                      length=_len, normalize=False,
                      linewidths=0.3, alpha=arrow_alpha*alpha_scale, color='k')

        # Ejes y estilos
        ax.set_xlim(xmin_, xmax_); ax.set_ylim(ymin_, ymax_); ax.set_zlim(zmin_, zmax_)
        ax.set_xlabel(xlabel, fontsize=axis_labelsize)
        ax.set_ylabel(ylabel, fontsize=axis_labelsize)
        ax.set_zlabel(zlabel, fontsize=axis_labelsize)
        ax.set_title(title, fontsize=title_fontsize)

        if axes_only:
            ax.grid(False)
            try:
                ax.w_xaxis.set_pane_color((1,1,1,0))
                ax.w_yaxis.set_pane_color((1,1,1,0))
                ax.w_zaxis.set_pane_color((1,1,1,0))
            except Exception:
                try:
                    ax.xaxis.pane.set_visible(False)
                    ax.yaxis.pane.set_visible(False)
                    ax.zaxis.pane.set_visible(False)
                except Exception:
                    pass
            ax.set_facecolor((1,1,1,0))
        else:
            ax.grid(True)

        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        try: ax.zaxis.set_tick_params(labelsize=tick_labelsize)
        except Exception: pass

        if equal_aspect:
            try:
                ax.set_box_aspect((
                    xunit_scale*(xmax_-xmin_), 
                    yunit_scale*(ymax_-ymin_), 
                    zunit_scale*(zmax_-zmin_)
                    ))
            except Exception:
                pass

        if include_labels and init_points:
            h,l = ax.get_legend_handles_labels()
            if l:
                seen=set(); hh=[]; ll=[]
                for hi, li in zip(h,l):
                    if li and li not in seen:
                        seen.add(li); hh.append(hi); ll.append(li)
                if legend_outside:
                    try: ax.figure.subplots_adjust(right=0.80)
                    except Exception: pass
                    ax.legend(hh, ll, frameon=False, fontsize=9,
                              loc='center left', bbox_to_anchor=(1.02, 0.5))
                else:
                    ax.legend(hh, ll, frameon=False, fontsize=9, loc='upper left')

    # ---------- Creación de figura (sin auto-display) ----------
    def _make_fig_ax():
        import matplotlib.pyplot as plt
        plt.ioff()
        fig = plt.figure(figsize=fig_size, constrained_layout=not fill_figure)
        ax = fig.add_subplot(111, projection='3d')
        try:
            if hasattr(fig.canvas, "header_visible"):
                fig.canvas.header_visible = False
        except Exception:
            pass
        if fill_figure:
            try: fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            except Exception: pass
        return fig, ax

    # ---------- Modo estático ----------
    if not interactive:
        from IPython.display import display
        fig, ax = _make_fig_ax()
        _draw(ax, _par_vector(params))
        display(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return

    # ---------- Modo interactivo ----------
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        base = dict(params)
        for key, spec in slider_specs.items():
            base[key] = spec["init"]

        fig, ax = _make_fig_ax()

        rows, slider_widgets = [], []
        for key, spec in slider_specs.items():
            if slider_use_latex:
                txt = spec["desc_latex"] if spec["desc_latex"] is not None else sp.latex(key)
                if hasattr(widgets, "HTMLMath"):
                    lbl = widgets.HTMLMath(value=_latexify(txt))
                elif hasattr(widgets, "Latex"):
                    lbl = widgets.Latex(value=_latexify(txt))
                elif hasattr(widgets, "HTML"):
                    lbl = widgets.HTML(value=_latexify(txt))
                else:
                    lbl = widgets.Label(value=str(key))
            else:
                lbl = widgets.Label(value=spec.get("desc", str(key)))
            lbl.layout = widgets.Layout(width='80px')

            sld = widgets.FloatSlider(
                min=spec["min"], max=spec["max"], step=spec["step"],
                value=spec["init"], continuous_update=True, readout=True,
                readout_format=".4f",
                description='', style={'description_width':'0px'},
                layout=widgets.Layout(width='240px')
            )
            row = widgets.HBox([lbl, sld])
            row.layout = widgets.Layout(width='360px', align_items='center', margin='0 12px 10px 0')
            rows.append(row); slider_widgets.append((key, sld))

        sliders_box = widgets.Box(
            rows,
            layout=widgets.Layout(display='flex', flex_flow='row wrap',
                                  justify_content='flex-start', align_items='center', width='100%')
        )

        backend = str(plt.get_backend()).lower()
        use_ipympl = ("ipympl" in backend) or ("widget" in backend and getattr(fig, "canvas", None) is not None)
        out_fig = widgets.Output()

        def _on_change(_):
            cur = dict(params)
            for k, w in slider_widgets:
                cur[k] = float(w.value)
            cur_vals = _par_vector(cur)

            if freeze_initial:
                _draw(ax, _par_vector(base), alpha_scale=0.45, include_labels=False, clear=True)
                _draw(ax, cur_vals,         alpha_scale=1.0,  include_labels=True,  clear=False)
            else:
                _draw(ax, cur_vals,         alpha_scale=1.0,  include_labels=True,  clear=True)

            if use_ipympl:
                try: fig.canvas.draw_idle()
                except Exception: pass
            else:
                with out_fig:
                    clear_output(wait=True)
                    display(fig)

        # render inicial y enganchar eventos
        _on_change(None)
        for _, w in slider_widgets: w.observe(_on_change, names="value")

        if not use_ipympl:
            display(widgets.VBox([sliders_box, out_fig], layout=widgets.Layout(width='100%')))
        else:
            display(widgets.VBox([sliders_box, fig.canvas], layout=widgets.Layout(width='100%')))

    except Exception as e:
        print("[Aviso] Interactividad requiere ipywidgets en entorno Jupyter. "
              "Se mostrará estática. Detalle:", e)
        from IPython.display import display
        fig, ax = _make_fig_ax()
        _draw(ax, _par_vector(params))
        display(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)



##############################################################
### Función para obtener menores principales de una matriz ###
##############################################################
def principal_minors(
    A, k,
    simplify=True,
    index_base=0,
    return_sum=False,   # si True, devuelve (dict_menores, suma)
    only_sum=False      # si True, devuelve solo la suma
):
    """
    Calcula todos los menores PRINCIPALES de orden k de la matriz cuadrada A.
    Un menor principal usa el mismo conjunto de índices para filas y columnas.

    Parámetros
    ----------
    A : sympy.Matrix
        Matriz cuadrada.
    k : int
        Orden del menor (1 <= k <= n).
    simplify : bool
        Si True, aplica sp.simplify a cada determinante y a la suma. Por defecto True.
    index_base : int
        Base para los índices reportados (0 o 1). Por defecto 0.
    return_sum : bool
        Si True, retorna (diccionario, suma_de_menores).
    only_sum : bool
        Si True, retorna solo la suma_de_menores.

    Retorna
    -------
    dict | sympy.Expr | (dict, sympy.Expr)
        Dependiendo de return_sum y only_sum.
    """
    if not isinstance(A, sp.MatrixBase):
        A = sp.Matrix(A)

    n, m = A.shape
    if n != m:
        raise ValueError("A debe ser cuadrada.")
    if not (1 <= k <= n):
        raise ValueError(f"k debe estar entre 1 y {n}.")

    menores = {}
    total = sp.Integer(0)

    for idx in combinations(range(n), k):
        sub = A.extract(idx, idx)
        det = sub.det()
        if simplify:
            det = sp.simplify(det)
        show_idx = tuple(i + index_base for i in idx)
        menores[show_idx] = det
        total += det

    if simplify:
        total = sp.simplify(total)

    if only_sum:
        return total
    if return_sum:
        return menores, total
    return menores

# (opcional) leading principal minors
def leading_principal_minors(A, simplify=True):
    if not isinstance(A, sp.MatrixBase):
        A = sp.Matrix(A)
    n, m = A.shape
    if n != m:
        raise ValueError("A debe ser cuadrada.")
    out = []
    for i in range(1, n+1):
        det = A[:i, :i].det()
        out.append(sp.simplify(det) if simplify else det)
    return out



#########################################################
### Plano de fase (2D) y soluciones de un sistema 3D  ###
### Basado en la API/opciones de plot_slope_field_and_ ###
### solutions3D, proyectando a 2 variables elegidas.  ###
#########################################################
def plot_2D_from3D(
    eqs, states, indep,
    params=None,
    # --- selección del plano ---
    plane=None,                  # None | (i,j) con i,j∈{0,1,2} | (sym_i, sym_j) | ("x","y")
    slice_value=0.0,             # valor fijo de la tercera variable para el campo vectorial (modo 'slice')
    slice_tol=1e-6,              # tolerancia para mostrar equilibrios cercanos al corte
    slice_auto_from=None,        # None | 'mean' | 'median' (si None, usa slice_value)

    # --- límites y grilla del plano ---
    xlim=(-3,3), ylim=(-3,3), grid_n=21,

    # --- condiciones iniciales (en 3D) ---
    init_points=None,            # [(x0,y0,z0), ...]
    t0=0.0, T=20.0, max_step=None,
    normalize_field=True,
    fig_size=(8,6),

    # --- Campo vectorial ---
    show_field=False,
    field_mode="slice",         # 'slice' (por defecto) | 'along_trajectory' (flechas tangentes a la proyección)
    field_on_traj_n=40,          # nº de flechas por trayectoria cuando field_mode='along_trajectory'
    field_time=None,             # si ecuaciones no autónomas y field_mode='slice': tiempo para evaluar el campo (None -> t0)
    arrow_alpha=0.6,
    # 'relative' (fracción de diagonal) | 'data' (unidades de datos del plano)
    arrow_length_mode="relative",
    arrow_length_rel=0.06,   # si mode='relative'
    arrow_length=0.10,       # si mode='data'

    # --- Estética general ---
    equal_aspect=True,
    # --- Control fino de proporciones ---
    xunit_scale=1.0,            # >1 ensancha X (duplica unidades horizontales si =2)
    yunit_scale=1.0,            # >1 estira Y (duplica unidades verticales si =2)
    data_aspect=None,           # None | float -> relación (unidad_y / unidad_x)
    box_aspect=None,            # None | float -> ancho/alto del recuadro del eje (w/h)
    interactive=False,
    sliders=None,
    slider_use_latex=True,
    freeze_initial=False,

    sol_color="tab:blue",
    initial_color="black",
    title="",
    xlabel=None, ylabel=None,
    legend_outside=True,
    legend_position="right",
    show_equilibria=False,
    equilibrium_style=None,
    eq_solver_opts=None,

    axes_only=False,
    auto_limits="solutions",    # 'static' | 'solutions'
    fill_figure=False,

    time_direction="forward",   # 'forward' | 'backward' | 'both'

    tick_labelsize=8,
    axis_labelsize=11,
    title_fontsize=12
):
    """
    Proyecta un sistema 3D (x,y,z) sobre un plano 2D entre dos estados elegidos
    y grafica campo vectorial (en ese corte) y soluciones proyectadas.

    Importante sobre el campo:
      - field_mode='slice' (por defecto): se evalúa el campo 2D fijando la tercera
        variable en `slice_value` (o en un valor agregado de las trayectorias si
        `slice_auto_from` ∈ {'mean','median'}). Esto representa la dinámica del
        subsistema con esa variable *congelada*, que en general NO coincide con
        la proyección del flujo 3D.
      - field_mode='along_trajectory': se dibujan flechas tangentes a lo largo de
        las trayectorias proyectadas, usando los verdaderos (dx/dt, dy/dt) con
        el z(t) correspondiente a cada punto (y el tiempo real si el sistema es
        no autónomo). Este modo "sigue" mejor órbitas cíclicas.
    """
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import root

    # ===== COMPAT: Shim SymPy (nombres únicos v02)
    try:
        from sympy.core.function import AppliedUndef as _AppliedUndef_v02, UndefinedFunction as _UndefinedFunction_v02, FunctionClass as _FunctionClass_v02
    except Exception:
        try:
            from sympy.core.function import FunctionClass as _FunctionClass_v02
        except Exception:
            _FunctionClass_v02 = type(sp.Function('f'))
        _AppliedUndef_v02 = type(sp.Function('f')(sp.Symbol('x')))
        _UndefinedFunction_v02 = getattr(sp, 'UndefinedFunction', _FunctionClass_v02)

    def _latexify_v02(s):
        if s is None: return ""
        t = str(s).strip()
        if t.startswith("$$") and t.endswith("$$"): return t
        if t.startswith("$") and t.endswith("$"): return t
        return f"${t}$"

    # ---- Defaults/validaciones
    if params is None: params = {}
    if init_points is None: init_points = []
    if len(eqs)!=3 or len(states)!=3:
        raise ValueError("Se esperan exactamente tres ecuaciones y tres estados.")
    if interactive and (not isinstance(sliders, dict) or not sliders):
        raise ValueError("Con interactive=True, pasa `sliders` como dict no vacío.")
    if auto_limits not in ("static","solutions"):
        raise ValueError("auto_limits debe ser 'static' o 'solutions'.")
    if time_direction not in ("forward","backward","both"):
        raise ValueError("time_direction: 'forward' | 'backward' | 'both'.")
    if arrow_length_mode not in ("relative","data"):
        raise ValueError("arrow_length_mode: 'relative' | 'data'.")
    if field_mode not in ("slice","along_trajectory"):
        raise ValueError("field_mode: 'slice' | 'along_trajectory'.")

    if equilibrium_style is None:
        equilibrium_style = dict(marker="o", s=50, color="crimson",
                                 edgecolors="k", linewidths=0.6, zorder=8)
    if eq_solver_opts is None:
        eq_solver_opts = dict(tol=1e-9, maxiter=200)

    # ---- Resolver `plane` a índices (i,j) en {0,1,2}
    def _resolve_plane_v02(plane, states):
        if plane is None:
            return (0,1)
        # índices
        if isinstance(plane, (tuple, list)) and len(plane)==2 and all(isinstance(k, int) for k in plane):
            i,j = plane
            if i==j or min(i,j)<0 or max(i,j)>2: raise ValueError("plane con índices inválidos.")
            return (i,j)
        # símbolos
        if isinstance(plane, (tuple, list)) and len(plane)==2 and all(isinstance(k, sp.Basic) for k in plane):
            i = states.index(plane[0]); j = states.index(plane[1])
            if i==j: raise ValueError("plane repetido.")
            return (i,j)
        # strings
        if isinstance(plane, (tuple, list)) and len(plane)==2 and all(isinstance(k, str) for k in plane):
            names = [str(s) for s in states]
            try:
                i = names.index(plane[0]); j = names.index(plane[1])
            except ValueError:
                raise ValueError("plane contiene nombres que no están en `states`.")
            if i==j: raise ValueError("plane repetido.")
            return (i,j)
        raise ValueError("plane debe ser None, (i,j), (sym_i,sym_j) o (name_i,name_j).")

    i_plane, j_plane = _resolve_plane_v02(plane, states)
    k_plane = [0,1,2][[0,1,2].index(i_plane) ^ [0,1,2].index(j_plane) ^ 3]  # índice restante

    # Símbolos auxiliares
    x_s, y_s, z_s = sp.symbols('x_s y_s z_s', real=True)

    # RHS
    f_expr = eqs[0].rhs; g_expr = eqs[1].rhs; h_expr = eqs[2].rhs

    # Sustituir estados por (x_s,y_s,z_s)
    subs_map = {states[0]: x_s, states[1]: y_s, states[2]: z_s}
    f_xyz = sp.simplify(f_expr.subs(subs_map))
    g_xyz = sp.simplify(g_expr.subs(subs_map))
    h_xyz = sp.simplify(h_expr.subs(subs_map))

    # Sliders (aceptan símbolos/funciones/funciones aplicadas)
    def _is_slider_key_v02(obj):
        return (isinstance(obj, sp.Symbol)
                or isinstance(obj, _UndefinedFunction_v02)
                or isinstance(obj, _AppliedUndef_v02))

    slider_specs = {}
    if interactive:
        for key, spec in sliders.items():
            if not _is_slider_key_v02(key):
                raise TypeError("Claves de sliders: sp.Symbol, sp.Function('a') o aplicadas (p.ej. a(t0)).")
            if "min" not in spec or "max" not in spec:
                raise ValueError(f"Faltan 'min'/'max' en slider de {key}.")
            mn, mx = float(spec["min"]), float(spec["max"])
            slider_specs[key] = {
                "min": mn, "max": mx,
                "step": float(spec.get("step", (mx-mn)/100.0 if mx>mn else 0.1)),
                "init": float(spec.get("init", (mn+mx)/2.0)),
                "desc": spec.get("desc", str(key)),
                "desc_latex": spec.get("desc_latex", None),
            }

    # Reemplazos para lambdify (para funciones aplicadas)
    all_param_objs = set(params.keys()) | set(slider_specs.keys())
    def _mkname_v02(obj, k): return f"__par{k}__{sp.srepr(obj)}"
    repl_to_proxy, order_originals = {}, []
    for i, p in enumerate(sorted(all_param_objs, key=lambda z: getattr(z, "name", str(z))), start=1):
        repl_to_proxy[p] = p if isinstance(p, sp.Symbol) else sp.Symbol(_mkname_v02(p, i))
        order_originals.append(p)
    order_proxies = [repl_to_proxy[p] for p in order_originals]

    f_sub, g_sub, h_sub = (f_xyz.xreplace(repl_to_proxy),
                           g_xyz.xreplace(repl_to_proxy),
                           h_xyz.xreplace(repl_to_proxy))

    f_num = sp.lambdify((indep, x_s, y_s, z_s, *order_proxies), f_sub, "numpy")
    g_num = sp.lambdify((indep, x_s, y_s, z_s, *order_proxies), g_sub, "numpy")
    h_num = sp.lambdify((indep, x_s, y_s, z_s, *order_proxies), h_sub, "numpy")

    if max_step is None:
        max_step = (abs(T)/500.0) if T!=0 else 0.05

    def _par_vector_v02(pdict):
        vals = []
        for p in order_originals:
            if p in pdict: vals.append(float(pdict[p]))
            elif p in params: vals.append(float(params[p]))
            else: vals.append(0.0)
        return vals

    # Equilibrios en 3D y filtrado por corte
    def _find_equilibria_v02(par_vals, seeds_per_dim=4, merge_tol=1e-3):
        xmin, xmax = xlim; ymin, ymax = ylim
        # Para semilla de la tercera variable usamos ±max(|slice_value|, span_promedio)
        span = 0.5*(abs(xmax-xmin) + abs(ymax-ymin))
        zmin = (slice_value if slice_value is not None else 0.0) - span
        zmax = (slice_value if slice_value is not None else 0.0) + span
        pts = []
        sx = np.linspace(xmin, xmax, seeds_per_dim)
        sy = np.linspace(ymin, ymax, seeds_per_dim)
        sz = np.linspace(zmin, zmax, seeds_per_dim)
        for x0 in sx:
            for y0 in sy:
                for z0 in sz:
                    try:
                        sol = root(lambda w: [f_num(t0, w[0], w[1], w[2], *par_vals),
                                              g_num(t0, w[0], w[1], w[2], *par_vals),
                                              h_num(t0, w[0], w[1], w[2], *par_vals)],
                                   x0=np.array([x0, y0, z0], dtype=float),
                                   tol=eq_solver_opts.get("tol", 1e-9),
                                   options={"maxiter": eq_solver_opts.get("maxiter", 200)})
                        if sol.success:
                            xs, ys, zs = map(float, sol.x)
                            keep = True
                            for (xa, ya, za) in pts:
                                if (xa-xs)**2 + (ya-ys)**2 + (za-zs)**2 <= merge_tol**2:
                                    keep = False; break
                            if keep: pts.append((xs, ys, zs))
                    except Exception:
                        pass
        return pts

    # ---------- Dibujo principal (2D) ----------
    def _draw_v02(ax, par_vals, alpha_scale=1.0, include_labels=True, clear=True):
        if clear: ax.cla()

        # Integrador 3D
        def ode_sys(t, z):
            xval, yval, zval = z
            return [float(f_num(t, xval, yval, zval, *par_vals)),
                    float(g_num(t, xval, yval, zval, *par_vals)),
                    float(h_num(t, xval, yval, zval, *par_vals))]

        # Proyección de soluciones
        all_pts = []       # para límites (2D)
        sols_3d = []       # para campo 'along_trajectory'
        for (x0, y0, z0) in init_points:
            if time_direction in ("forward","both"):
                try:
                    sf = solve_ivp(ode_sys, (t0, t0+T), [x0, y0, z0],
                                   max_step=max_step, rtol=1e-6, atol=1e-9)
                    if sf.y.size:
                        xi = sf.y[i_plane]; xj = sf.y[j_plane]
                        ax.plot(xi, xj, lw=2, color=sol_color,
                                label=(f"({x0:.2f},{y0:.2f},{z0:.2f}) →" if include_labels else None),
                                alpha=alpha_scale)
                        all_pts.append(np.vstack([xi,xj]).T)
                        sols_3d.append(dict(t=sf.t, X=sf.y[0], Y=sf.y[1], Z=sf.y[2]))
                except Exception:
                    pass
            if time_direction in ("backward","both"):
                try:
                    sb = solve_ivp(ode_sys, (t0, t0-T), [x0, y0, z0],
                                   max_step=max_step, rtol=1e-6, atol=1e-9)
                    if sb.y.size:
                        xi = sb.y[i_plane][::-1]; xj = sb.y[j_plane][::-1]
                        ax.plot(xi, xj, lw=2, color=sol_color, alpha=alpha_scale)
                        all_pts.append(np.vstack([xi,xj]).T)
                        # invertir para mantener t ascendente
                        sols_3d.append(dict(t=sb.t[::-1], X=sb.y[0][::-1], Y=sb.y[1][::-1], Z=sb.y[2][::-1]))
                except Exception:
                    pass
            ax.scatter([ (x0,y0,z0)[i_plane] ], [ (x0,y0,z0)[j_plane] ], s=18, c=initial_color, alpha=alpha_scale)

        # Valor automático del corte si se pidió
        slice_val_local = slice_value
        if slice_auto_from in ("mean","median") and sols_3d:
            z_all = np.concatenate([s['Z'] for s in sols_3d])
            if slice_auto_from == 'mean':
                slice_val_local = float(np.nanmean(z_all))
            else:
                slice_val_local = float(np.nanmedian(z_all))

        # Equilibrios proyectados (opcional)
        if show_equilibria:
            pts3 = _find_equilibria_v02(par_vals)
            if pts3:
                ex = []; ey = []
                for (a,b,c) in pts3:
                    third = [a,b,c][k_plane]
                    if slice_val_local is not None and abs(third - slice_val_local) <= slice_tol:
                        ex.append([a,b,c][i_plane]); ey.append([a,b,c][j_plane])
                if len(ex)>0:
                    ax.scatter(ex, ey, **equilibrium_style)
                    if auto_limits == "solutions":
                        all_pts.append(np.array(list(zip(ex,ey))))

        # Límites
        if auto_limits == "solutions" and all_pts:
            P = np.vstack(all_pts)
            xmin2, ymin2 = np.nanmin(P, axis=0)
            xmax2, ymax2 = np.nanmax(P, axis=0)
            def pad(a,b):
                span = max(1e-9, b-a); d = 0.05*span
                return a-d, b+d
            xmin_, xmax_ = pad(xmin2, xmax2)
            ymin_, ymax_ = pad(ymin2, ymax2)
        else:
            xmin_, xmax_ = xlim
            ymin_, ymax_ = ylim

        # Campo vectorial
        if show_field:
            if arrow_length_mode=='relative':
                diag = np.sqrt((xmax_-xmin_)**2 + (ymax_-ymin_)**2)
                _len = max(1e-12, float(arrow_length_rel) * float(diag))
            else:
                _len = float(arrow_length)

            if field_mode == 'slice':
                X = np.linspace(xmin_, xmax_, grid_n)
                Y = np.linspace(ymin_, ymax_, grid_n)
                XX, YY = np.meshgrid(X, Y, indexing='xy')
                # Preparar tripleta (x_i, x_j, x_k)
                coords = [None, None, None]
                coords[i_plane] = XX
                coords[j_plane] = YY
                use_t = t0 if field_time is None else float(field_time)
                coords[k_plane] = np.full_like(XX, float(slice_val_local))

                U3 = f_num(use_t, coords[0], coords[1], coords[2], *par_vals)
                V3 = g_num(use_t, coords[0], coords[1], coords[2], *par_vals)
                W3 = h_num(use_t, coords[0], coords[1], coords[2], *par_vals)
                U3 = np.where(np.isfinite(U3), U3, 0.0)
                V3 = np.where(np.isfinite(V3), V3, 0.0)
                W3 = np.where(np.isfinite(W3), W3, 0.0)

                comp = [U3, V3, W3]
                Ui = comp[i_plane]
                Uj = comp[j_plane]

                if normalize_field:
                    m = np.sqrt(Ui*Ui + Uj*Uj)
                    Ui = np.divide(Ui, m, out=np.zeros_like(Ui), where=m>0)
                    Uj = np.divide(Uj, m, out=np.zeros_like(Uj), where=m>0)

                ax.quiver(XX, YY, Ui, Uj, angles='xy', scale_units='xy', scale=1.0/_len,
                          width=0.0025, alpha=arrow_alpha, color='k')

            else:  # field_mode == 'along_trajectory'
                # Construir flechas tangentes usando z(t) real y, si aplica, t real
                Xi_list, Xj_list, Ui_list, Uj_list = [], [], [], []
                for s in sols_3d:
                    n = len(s['t'])
                    if n < 2: continue
                    # muestrear ~field_on_traj_n puntos
                    idx = np.linspace(0, n-1, num=min(field_on_traj_n, n), dtype=int)
                    tt = s['t'][idx]
                    Xv = [s['X'], s['Y'], s['Z']]
                    xi = Xv[i_plane][idx]; xj = Xv[j_plane][idx]
                    xk = Xv[k_plane][idx]
                    # evaluar derivadas reales en cada punto
                    Ui = f_num(tt, Xv[0][idx], Xv[1][idx], Xv[2][idx], *par_vals)
                    Uj_ = g_num(tt, Xv[0][idx], Xv[1][idx], Xv[2][idx], *par_vals)
                    Uk  = h_num(tt, Xv[0][idx], Xv[1][idx], Xv[2][idx], *par_vals)
                    comp = [Ui, Uj_, Uk]
                    Ui2 = comp[i_plane]
                    Uj2 = comp[j_plane]

                    Ui2 = np.where(np.isfinite(Ui2), Ui2, 0.0)
                    Uj2 = np.where(np.isfinite(Uj2), Uj2, 0.0)
                    if normalize_field:
                        m = np.sqrt(Ui2*Ui2 + Uj2*Uj2)
                        Ui2 = np.divide(Ui2, m, out=np.zeros_like(Ui2), where=m>0)
                        Uj2 = np.divide(Uj2, m, out=np.zeros_like(Uj2), where=m>0)

                    Xi_list.append(xi); Xj_list.append(xj)
                    Ui_list.append(Ui2); Uj_list.append(Uj2)

                if Xi_list:
                    Xi = np.concatenate(Xi_list)
                    Xj = np.concatenate(Xj_list)
                    Ui = np.concatenate(Ui_list)
                    Uj = np.concatenate(Uj_list)
                    ax.quiver(Xi, Xj, Ui, Uj, angles='xy', scale_units='xy', scale=1.0/_len,
                              width=0.0028, alpha=arrow_alpha, color='k')

        # Etiquetas y estilo
        ax.set_xlim(xmin_, xmax_); ax.set_ylim(ymin_, ymax_)
        xlabel_loc = str(states[i_plane]) if xlabel is None else xlabel
        ylabel_loc = str(states[j_plane]) if ylabel is None else ylabel
        ax.set_xlabel(xlabel_loc, fontsize=axis_labelsize)
        ax.set_ylabel(ylabel_loc, fontsize=axis_labelsize)
        ax.set_title(title, fontsize=title_fontsize)

        if axes_only:
            ax.grid(False)
            ax.set_facecolor((1,1,1,0))
        else:
            ax.grid(True)

        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

        # ---- Proporciones del eje ----
        try:
            # Prioridad: data_aspect explícito > (xunit_scale,yunit_scale) > equal_aspect
            if data_aspect is not None:
                # data_aspect = (unidad_y / unidad_x)
                ax.set_aspect(float(data_aspect), adjustable='box')
            elif (abs(float(xunit_scale) - 1.0) > 1e-12) or (abs(float(yunit_scale) - 1.0) > 1e-12):
                asp = float(yunit_scale) / float(xunit_scale)  # (px por unidad y) / (px por unidad x)
                ax.set_aspect(asp, adjustable='box')
            elif equal_aspect:
                ax.set_aspect('equal', adjustable='box')
            # Opcional: control del rectángulo (ancho/alto)
            if box_aspect is not None:
                try:
                    # Matplotlib interpreta box_aspect como alto/ancho; pedimos ancho/alto
                    ax.set_box_aspect(1.0/float(box_aspect))
                except Exception:
                    pass
        except Exception:
            pass

        if include_labels and init_points:
            h,l = ax.get_legend_handles_labels()
            if l:
                seen=set(); hh=[]; ll=[]
                for hi, li in zip(h,l):
                    if li and li not in seen:
                        seen.add(li); hh.append(hi); ll.append(li)
                # Limpia posibles leyendas previas de nivel figura (evita duplicados en modo interactivo)
                fig_loc = ax.figure
                for _leg in list(getattr(fig_loc, 'legends', [])):
                    try: _leg.remove()
                    except Exception: pass

                if legend_position == 'bottom':
                    # Leyenda centrada abajo a nivel de figura
                    ncol = max(1, min(4, len(ll)))
                    fig_loc.legend(hh, ll, loc='lower center', ncol=ncol,
                                   frameon=False, fontsize=9)
                elif legend_outside:
                    # Fuera a la derecha sin reservar margen fijo
                    ax.legend(hh, ll, frameon=False, fontsize=9,
                              loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
                else:
                    ax.legend(hh, ll, frameon=False, fontsize=9, loc='upper left')

    # ---------- Creación de figura (sin auto-display) ----------
    def _make_fig_ax_v02():
        plt.ioff()
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=not fill_figure)
        try:
            if hasattr(fig.canvas, "header_visible"):
                fig.canvas.header_visible = False
        except Exception:
            pass
        if fill_figure:
            try: fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            except Exception: pass
        return fig, ax

    # ---------- Modo estático ----------
    if not interactive:
        from IPython.display import display
        fig, ax = _make_fig_ax_v02()
        _draw_v02(ax, _par_vector_v02(params))
        display(fig)
        plt.close(fig)
        return

    # ---------- Modo interactivo con sliders ----------
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        base = dict(params)
        for key, spec in slider_specs.items():
            base[key] = spec["init"]

        fig, ax = _make_fig_ax_v02()

        rows, slider_widgets = [], []
        for key, spec in slider_specs.items():
            if slider_use_latex:
                txt = spec["desc_latex"] if spec["desc_latex"] is not None else sp.latex(key)
                if hasattr(widgets, "HTMLMath"):
                    lbl = widgets.HTMLMath(value=_latexify_v02(txt))
                elif hasattr(widgets, "Latex"):
                    lbl = widgets.Latex(value=_latexify_v02(txt))
                elif hasattr(widgets, "HTML"):
                    lbl = widgets.HTML(value=_latexify_v02(txt))
                else:
                    lbl = widgets.Label(value=str(key))
            else:
                lbl = widgets.Label(value=spec.get("desc", str(key)))
            lbl.layout = widgets.Layout(width='80px')

            sld = widgets.FloatSlider(
                min=spec["min"], max=spec["max"], step=spec["step"],
                value=spec["init"], continuous_update=True, readout=True,
                readout_format=".4f",
                description='', style={'description_width':'0px'},
                layout=widgets.Layout(width='240px')
            )
            row = widgets.HBox([lbl, sld])
            row.layout = widgets.Layout(width='360px', align_items='center', margin='0 12px 10px 0')
            rows.append(row); slider_widgets.append((key, sld))

        sliders_box = widgets.Box(
            rows,
            layout=widgets.Layout(display='flex', flex_flow='row wrap',
                                  justify_content='flex-start', align_items='center', width='100%')
        )

        backend = str(plt.get_backend()).lower()
        use_ipympl = ("ipympl" in backend) or ("widget" in backend and getattr(fig, "canvas", None) is not None)
        out_fig = widgets.Output()

        def _on_change_v02(_):
            cur = dict(params)
            for k, w in slider_widgets:
                cur[k] = float(w.value)
            cur_vals = _par_vector_v02(cur)

            if freeze_initial:
                _draw_v02(ax, _par_vector_v02(base), alpha_scale=0.45, include_labels=False, clear=True)
                _draw_v02(ax, cur_vals,         alpha_scale=1.0,  include_labels=True,  clear=False)
            else:
                _draw_v02(ax, cur_vals,         alpha_scale=1.0,  include_labels=True,  clear=True)

            if use_ipympl:
                try: fig.canvas.draw_idle()
                except Exception: pass
            else:
                with out_fig:
                    clear_output(wait=True)
                    display(fig)

        # render inicial y enganchar eventos
        _on_change_v02(None)
        for _, w in slider_widgets: w.observe(_on_change_v02, names="value")

        if not use_ipympl:
            display(widgets.VBox([sliders_box, out_fig], layout=widgets.Layout(width='100%')))
        else:
            display(widgets.VBox([sliders_box, fig.canvas], layout=widgets.Layout(width='100%')))

    except Exception as e:
        print("[Aviso] Interactividad requiere ipywidgets en entorno Jupyter. "
              "Se mostrará estática. Detalle:", e)
        from IPython.display import display
        fig, ax = _make_fig_ax_v02()
        _draw_v02(ax, _par_vector_v02(params))
        display(fig)
        plt.close(fig)

