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
        left = "-∞" if dom.start.is_negative_infinity else str(dom.start)
        right = "∞" if dom.end.is_positive_infinity else str(dom.end)
        return f"{left_bracket}{left}, {right}{right_bracket}"
    elif isinstance(dom, sp.Union):
        # 对 Union 中的每个子集递归调用 domain_to_str，再用 ∪ 拼接
        return " ∪ ".join(domain_to_str(a) for a in dom.args)
    elif isinstance(dom, sp.Intersection):
        # 类似处理 Intersection
        return " ∩ ".join(domain_to_str(a) for a in dom.args)
    elif isinstance(dom, sp.FiniteSet):
        # 对有限点集做特殊处理
        if len(dom) == 0:
            return "∅"
        else:
            return "{" + ", ".join(str(a) for a in sorted(dom)) + "}"
    else:
        # 其他情况直接转成字符串
        return str(dom)


def analyze_function(func_str):
    x = sp.Symbol('x', real=True)
    # 1. 解析函数
    try:
        expr = sp.sympify(func_str)
    except Exception as e:
        return {"error": f"解析表达式失败: {e}"}

    # 2. 求定义域
    # continuous_domain 通常可视为函数在实数上的“连续”区间，也可视为定义域近似
    try:
        raw_domain = sp.calculus.util.continuous_domain(expr, x, sp.S.Reals)
        domain = domain_to_str(raw_domain)
    except Exception as e:
        domain = f"无法确定定义域：{e}"

    # 3. 求零点
    # solve 在部分情况下可能无法解析求解；此处做简单处理
    try:
        zeros = sp.solve(expr, x, dict=False)
    except Exception:
        zeros = []
    if not zeros:
        zeros_str = "无"
    else:
        # 转换成字符串，便于输出
        zeros_str = ", ".join(str(z) for z in zeros)

    # 4. 求导数
    derivative = sp.diff(expr, x)
    derivative_str = sp.pretty(derivative)

    # 5. 求驻点、极值
    critical_points = sp.solve(sp.Eq(derivative, 0), x, dict=False)
    extrema = []
    second_derivative = sp.diff(derivative, x)
    for cp in critical_points:
        # 二阶导数判断
        second_val = second_derivative.subs(x, cp)
        # 可能出现无法比较大小的情况，需要做一定判断
        if second_val.is_real:
            if second_val > 0:
                extrema.append(f"x = {cp} 为局部最小值")
            elif second_val < 0:
                extrema.append(f"x = {cp} 为局部最大值")
            else:
                extrema.append(f"x = {cp} 可能为拐点")
        else:
            extrema.append(f"x = {cp} 分析不充分")

    if not extrema:
        extrema_str = "无明显局部极值"
    else:
        # 多个极值点时，按列表输出
        extrema_str = "<br>".join(extrema)

    # 6. 单调性分析（此处只在 [-10, 10] 采样做示例）
    xs = np.linspace(-10, 10, 400)
    f_deriv = sp.lambdify(x, derivative, 'numpy')
    try:
        deriv_vals = f_deriv(xs)
    except Exception:
        deriv_vals = np.zeros_like(xs)
    if np.all(deriv_vals >= 0):
        monotonicity = "在区间 [-10,10] 内整体单调递增"
    elif np.all(deriv_vals <= 0):
        monotonicity = "在区间 [-10,10] 内整体单调递减"
    else:
        monotonicity = "在区间 [-10,10] 内存在单调性变化"

    # 7. 组织返回结果
    result = {
        "expr": sp.pretty(expr),
        "domain": domain,
        "zeros": zeros_str,
        "derivative": derivative_str,
        "extrema": extrema_str,
        "monotonicity": monotonicity
    }
    return result


def create_plot(expr):
    x = sp.symbols('x')
    f = sp.lambdify(x, expr, "numpy")
    xs = np.linspace(-10, 10, 400)
    ys = np.zeros_like(xs)

    for i, val in enumerate(xs):
        try:
            ys[i] = f(val)
        except Exception:
            ys[i] = np.nan  # 使用 np.nan 填充未定义的点

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, label=f"y = {sp.pretty(expr)}")
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
