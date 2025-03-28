from flask import Flask, render_template, request, send_file
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import io

app = Flask(__name__)

def analyze_function(func_str):
    # 定义符号变量
    x = sp.symbols('x')
    # 解析函数表达式
    try:
        expr = sp.sympify(func_str)
        if expr.is_constant():
            raise ValueError("表达式是一个常数，无法绘制图像")
    except Exception as e:
        return {"error": f"解析表达式失败: {e}"}

        # 自动求解定义域：利用 continuous_domain 求连续域（基本上可以视为定义域）
    try:
        domain_expr = sp.calculus.util.continuous_domain(expr, x, sp.S.Reals)
        domain = str(domain_expr)
    except Exception as e:
        domain = f"无法确定定义域：{e}"

    # 求零点（仅对多项式或可求解析解的函数）
    try:
        zeros = sp.solve(expr, x)
    except Exception:
        zeros = "无法求零点"

    # 求导
    derivative = sp.diff(expr, x)

    # 求驻点并判断极值
    critical_points = sp.solve(derivative, x)
    extrema = []
    # 计算二阶导数
    second_derivative = sp.diff(derivative, x)
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
    deriv_vals = f_deriv(xs)
    if np.all(deriv_vals >= -1e-6):
        monotonicity = "在区间 [-10,10] 内单调递增"
    elif np.all(deriv_vals <= 1e-6):
        monotonicity = "在区间 [-10,10] 内单调递减"
    else:
        monotonicity = "在区间 [-10,10] 内存在单调性变化"

    result = {
        "expr": sp.pretty(expr),
        "domain": domain,
        "zeros": zeros,
        "derivative": sp.pretty(derivative),
        "extrema": extrema if extrema else "无明显局部极值",
        "monotonicity": monotonicity
    }
    return result

def create_plot(expr):
    x = sp.symbols('x')
    # 将 sympy 表达式转换为可计算函数
    f = sp.lambdify(x, expr, "numpy")
    xs = np.linspace(-10, 10, 400)
    try:
        ys = f(xs)
        # 如果 ys 是一个标量值，将其扩展为与 xs 形状相同的数组
        if ys.ndim == 0:
            ys = np.full_like(xs, ys)
    except Exception:
        ys = np.zeros_like(xs)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, label=f"y = {sp.pretty(expr)}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("函数图像")
    plt.legend()
    plt.grid(True)

    # 将图像保存到内存中
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
