from flask import Flask, render_template, request, send_file
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import io

app = Flask(__name__)


def domain_to_str(dom):
    """
    将 Sympy 的域（Interval、Union、FiniteSet 等）转换为更易读的字符串形式。
    """
    if isinstance(dom, sp.Interval):
        left_bracket = "(" if dom.left_open else "["
        right_bracket = ")" if dom.right_open else "]"
        left = "-∞" if dom.start == -sp.oo else str(dom.start)
        right = "∞" if dom.end == sp.oo else str(dom.end)
        return f"{left_bracket}{left}, {right}{right_bracket}"
    elif isinstance(dom, sp.Union):
        return " ∪ ".join(domain_to_str(a) for a in dom.args)
    elif isinstance(dom, sp.Intersection):
        return " ∩ ".join(domain_to_str(a) for a in dom.args)
    elif isinstance(dom, sp.FiniteSet):
        if len(dom) == 0:
            return "∅"
        else:
            return "{" + ", ".join(str(a) for a in sorted(dom)) + "}"
    else:
        return str(dom)


def analyze_function(func_str):
    # 定义符号变量
    x = sp.symbols('x')
    # 解析函数表达式
    try:
        expr = sp.sympify(func_str)
    except Exception as e:
        return {"error": f"解析表达式失败: {e}"}

    # 定义域（通过 Sympy 连续性检查）
    try:
        raw_domain = sp.calculus.util.continuous_domain(expr, x, sp.S.Reals)
    except Exception:
        raw_domain = sp.S.Reals  # 如果出错，默认全体实数

    if raw_domain == sp.S.Reals:
        domain = "(-∞, ∞)"
    else:
        domain = domain_to_str(raw_domain)

    # 求零点
    try:
        zeros = sp.solve(expr, x)
    except Exception:
        zeros = "无法求零点"

    # 求导
    derivative = sp.diff(expr, x)

    # 求驻点并判断极值
    critical_points = sp.solve(derivative, x)
    second_derivative = sp.diff(derivative, x)
    extrema = []
    for cp in critical_points:
        try:
            second_val = second_derivative.subs(x, cp)
            if second_val.is_real:
                if second_val > 0:
                    extrema.append(f"x = {cp} 为局部最小值")
                elif second_val < 0:
                    extrema.append(f"x = {cp} 为局部最大值")
                else:
                    extrema.append(f"x = {cp} 可能为拐点")
            else:
                extrema.append(f"x = {cp} 分析不充分")
        except Exception as e:
            extrema.append(f"x = {cp} 分析失败: {e}")

    # 关于单调性，在 [-10,10] 区间内采样判断
    xs = np.linspace(-10, 10, 400)
    f_deriv = sp.lambdify(x, derivative, 'numpy')
    try:
        deriv_vals = f_deriv(xs)
    except Exception:
        deriv_vals = np.zeros_like(xs)

    eps = 1e-7  # 容许的数值误差
    if np.all(deriv_vals >= -eps):
        monotonicity = "在区间 [-10,10] 内单调递增"
    elif np.all(deriv_vals <= eps):
        monotonicity = "在区间 [-10,10] 内单调递减"
    else:
        monotonicity = "在区间 [-10,10] 内存在单调性变化"

    # 下面将表达式转成 LaTeX，以便前端用 MathJax 渲染
    # zeros 如果是列表，则每个元素都转成 LaTeX；如果是字符串(例如 "无法求零点")，则直接保留
    if isinstance(zeros, list):
        zeros_latex = [sp.latex(z) for z in zeros]
    else:
        zeros_latex = zeros  # 保持字符串

    # extrema 里存放的是字符串描述（如 “x = 1 为局部最小值”），
    # 如果想要更精细的 LaTeX 化，可以自行拆分这里的逻辑。
    # 这里为了简单，先保持字符串，直接显示文本。

    result = {
        "expr": sp.latex(expr),
        "domain": domain,
        "zeros": zeros_latex,
        "derivative": sp.latex(derivative),
        "extrema": extrema if extrema else "无明显局部极值",
        "monotonicity": monotonicity
    }
    return result


def create_plot(expr):
    x = sp.symbols('x', real=True)
    f = sp.lambdify(x, expr, "numpy")
    xs = np.linspace(-10, 10, 400)
    try:
        ys = f(xs)
    except Exception:
        ys = np.zeros_like(xs)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, label=r"$y = " + sp.latex(expr) + "$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("函数图像")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        func_str = request.form["function"]
        result = analyze_function(func_str)
        # 保存图像到全局变量
        global plot_buf, expr_plot
        try:
            expr_plot = sp.sympify(func_str)
        except Exception:
            expr_plot = sp.sympify("0")
        plot_buf = create_plot(expr_plot)
    return render_template("index.html", result=result)


@app.route("/plot.png")
def plot_png():
    global plot_buf
    return send_file(plot_buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
