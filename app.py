from flask import Flask, render_template, request, send_file
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import io

app = Flask(__name__)

class FunctionAnalyzer:
    @staticmethod
    def domain_to_str(dom):
        """
        将 Sympy 的 的（Interval、Union、FiniteSet 等）转换为更易读的字符串形式。
        """
        if isinstance(dom, sp.Interval):
            left_bracket = "(" if dom.left_open else "["
            right_bracket = ")" if dom.right_open else "]"
            left = "-∞" if dom.start == -sp.oo else str(dom.start)
            right = "∞" if dom.end == sp.oo else str(dom.end)
            return f"{left_bracket}{left}, {right}{right_bracket}"
        elif isinstance(dom, sp.Union):
            return " ∪ ".join(FunctionAnalyzer.domain_to_str(a) for a in dom.args)
        elif isinstance(dom, sp.Intersection):
            return " ∩ ".join(FunctionAnalyzer.domain_to_str(a) for a in dom.args)
        elif isinstance(dom, sp.FiniteSet):
            if len(dom) == 0:
                return "∅"
            else:
                return "{" + ", ".join(str(a) for a in sorted(dom)) + "}"
        else:
            return str(dom)

    @staticmethod
    def analyze_function(func_str):
        x = sp.Symbol('x', real=True)
        # 1. 解析函数
        try:
            expr = sp.sympify(func_str)
        except Exception as e:
            return {"error": f"解析表达式失败: {e}"}

        # 2. 求定义域
        try:
            raw_domain = sp.calculus.util.continuous_domain(expr, x, sp.S.Reals)
            if raw_domain == sp.S.Reals:
                domain = "(-∞, ∞)"
            else:
                domain = FunctionAnalyzer.domain_to_str(raw_domain)
        except Exception as e:
            domain = f"无法确定定义域：{e}"

        # 3. 求零点
        try:
            zeros = sp.solve(expr, x, dict=False)
        except Exception:
            zeros = []
        zeros_str = "无" if not zeros else ", ".join(str(z) for z in zeros)

        # 4. 求导数
        derivative = sp.diff(expr, x)
        derivative_str = sp.pretty(derivative)

        # 5. 求驻点、极值
        critical_points = sp.solve(sp.Eq(derivative, 0), x, dict=False)
        extrema = []
        second_derivative = sp.diff(derivative, x)
        for cp in critical_points:
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
        extrema_str = "无明显局部极值" if not extrema else "<br>".join(extrema)

        # 6. 单调性分析：在定义域内分别分析每个连续区间
        f_deriv = sp.lambdify(x, derivative, 'numpy')
        monotonicity_list = []

        # 提取连续区间列表
        if raw_domain == sp.S.Reals:
            domain_intervals = [sp.Interval(-sp.oo, sp.oo)]
        elif isinstance(raw_domain, sp.Interval):
            domain_intervals = [raw_domain]
        elif isinstance(raw_domain, sp.Union):
            domain_intervals = [i for i in raw_domain.args if isinstance(i, sp.Interval)]
        else:
            domain_intervals = []

        EPS = 1e-3  # 避免采样时取到端点处不适定值

        for interval in domain_intervals:
            # 对于无限端点，采样时取较大或较小的有限值；显示时依然使用无限端点
            if interval.start == -sp.oo:
                a = -100
            else:
                # 如果区间左端点为 0（且为开区间），则取 0+EPS
                a = float(interval.start) if float(interval.start) != 0 else 0 + EPS

            if interval.end == sp.oo:
                b = 100
            else:
                # 如果区间右端点为 0（且为开区间），则取 0-EPS
                b = float(interval.end) if float(interval.end) != 0 else 0 - EPS

            # 对于开区间端点为0的情况，确保采样区间内不包含 0
            if a == 0:
                a += EPS
            if b == 0:
                b -= EPS

            xs_dom = np.linspace(a, b, 400)
            try:
                deriv_vals_dom = f_deriv(xs_dom)
            except Exception:
                deriv_vals_dom = np.zeros_like(xs_dom)
            if np.all(deriv_vals_dom >= 0):
                monotonicity_list.append(f"在区间 {FunctionAnalyzer.domain_to_str(interval)} 内单调递增")
            elif np.all(deriv_vals_dom <= 0):
                monotonicity_list.append(f"在区间 {FunctionAnalyzer.domain_to_str(interval)} 内单调递减")
            else:
                monotonicity_list.append(f"在区间 {FunctionAnalyzer.domain_to_str(interval)} 内存在单调性变化")
        monotonicity = "；".join(monotonicity_list) if monotonicity_list else "无法分析单调性"

        result = {
            "expr": sp.pretty(expr),
            "domain": domain,
            "zeros": zeros_str,
            "derivative": derivative_str,
            "extrema": extrema_str,
            "monotonicity": monotonicity
        }
        return result

class PlotCreator:
    @staticmethod
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
        analyzer = FunctionAnalyzer()
        result = analyzer.analyze_function(func_str)
        plot_creator = PlotCreator()
        try:
            expr_plot = sp.sympify(func_str)
        except Exception:
            expr_plot = sp.sympify("0")
        global plot_buf
        plot_buf = plot_creator.create_plot(expr_plot)
    return render_template("index.html", result=result)

@app.route("/plot.png")
def plot_png():
    global plot_buf
    return send_file(plot_buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
