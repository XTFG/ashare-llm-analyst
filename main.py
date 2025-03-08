import base64
import os
from datetime import datetime
from io import BytesIO
from string import Template
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pytz
from matplotlib.axes import Axes
import numpy as np

import Ashare as as_api
import MyTT as mt
from llm import LLMAnalyzer


def generate_trading_signals(df):
    """生成交易信号和建议"""
    signals = []

    # MACD信号
    if df['MACD'].iloc[-1] > 0 >= df['MACD'].iloc[-2]:
        signals.append("MACD金叉形成，可能上涨")
    elif df['MACD'].iloc[-1] < 0 <= df['MACD'].iloc[-2]:
        signals.append("MACD死叉形成，可能下跌")

    # KDJ信号
    if df['K'].iloc[-1] < 20 and df['D'].iloc[-1] < 20:
        signals.append("KDJ超卖，可能反弹")
    elif df['K'].iloc[-1] > 80 and df['D'].iloc[-1] > 80:
        signals.append("KDJ超买，注意回调")

    # RSI信号
    if df['RSI'].iloc[-1] < 20:
        signals.append("RSI超卖，可能反弹")
    elif df['RSI'].iloc[-1] > 80:
        signals.append("RSI超买，注意回调")

    # BOLL带信号
    if df['close'].iloc[-1] > df['BOLL_UP'].iloc[-1]:
        signals.append("股价突破布林上轨，超买状态")
    elif df['close'].iloc[-1] < df['BOLL_LOW'].iloc[-1]:
        signals.append("股价跌破布林下轨，超卖状态")

    # DMI信号
    if df['PDI'].iloc[-1] > df['MDI'].iloc[-1] and df['PDI'].iloc[-2] <= df['MDI'].iloc[-2]:
        signals.append("DMI金叉，上升趋势形成")
    elif df['PDI'].iloc[-1] < df['MDI'].iloc[-1] and df['PDI'].iloc[-2] >= df['MDI'].iloc[-2]:
        signals.append("DMI死叉，下降趋势形成")

    # 成交量分析
    if df['VR'].iloc[-1] > 160:
        signals.append("VR大于160，市场活跃度高")
    elif df['VR'].iloc[-1] < 40:
        signals.append("VR小于40，市场活跃度低")

    # ROC动量分析
    if df['ROC'].iloc[-1] > df['MAROC'].iloc[-1] and df['ROC'].iloc[-2] <= df['MAROC'].iloc[-2]:
        signals.append("ROC上穿均线，上升动能增强")
    elif df['ROC'].iloc[-1] < df['MAROC'].iloc[-1] and df['ROC'].iloc[-2] >= df['MAROC'].iloc[-2]:
        signals.append("ROC下穿均线，上升动能减弱")

    return signals if signals else ["当前无明显交易信号"]


def plot_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return image_base64


def _get_value_class(value):
    """根据数值返回CSS类名"""
    try:
        if isinstance(value, str) and '%' in value:
            value = float(value.strip('%'))
        elif isinstance(value, str):
            return 'neutral'
        if value > 0:
            return 'positive'
        elif value < 0:
            return 'negative'
        return 'neutral'
    except (ValueError, TypeError) as e:
        print(f"无法解析数值 {value}，错误信息: {e}")
        return 'neutral'


def _generate_table_row(key, value):
    """生成表格行HTML，包含样式"""
    value_class = _get_value_class(value)
    return f'<tr><td>{key}</td><td class="{value_class}">{value}</td></tr>'


class StockAnalyzer:
    def __init__(self, _stock_info, count=120, llm_api_key=None, llm_base_url=None, llm_model=None):
        """
        初始化股票分析器

        Args:
            _stock_info: 股票信息字典
            count: 获取的数据条数
            llm_api_key: llm API密钥，默认从环境变量LLM_API_KEY获取
            llm_base_url: llm API基础URL，默认从环境变量LLM_BASE_URL获取
            llm_model: llm 模型名称，默认从环境变量LLM_MODEL获取
        """
        self.stock_codes = list(_stock_info.values())
        self.stock_names = _stock_info
        self.count = count
        self.data = {}
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 从环境变量获取API密钥和基础URL
        self.llm_api_key = llm_api_key or os.environ.get('LLM_API_KEY')
        self.llm_base_url = llm_base_url or os.environ.get('LLM_BASE_URL')
        self.llm_model = llm_model or os.environ.get('LLM_MODEL')

        # 初始化llm分析器
        self.llm = LLMAnalyzer(self.llm_api_key, self.llm_base_url, self.llm_model) if self.llm_api_key else None

    def get_stock_name(self, code):
        """根据股票代码获取股票名称"""
        return {v: k for k, v in self.stock_names.items()}.get(code, code)

    def fetch_data(self):
        """获取股票数据"""
        for code in self.stock_codes:
            stock_name = self.get_stock_name(code)
            try:
                df = as_api.get_price(code, count=self.count, frequency='1d')
                self.data[code] = df
            except Exception as e:
                print(f"获取股票 {stock_name} ({code}) 数据失败: {str(e)}")

    def calculate_indicators(self, code):
        """计算技术指标"""
        df = self.data[code].copy()
        close = np.array(df['close'])
        open_price = np.array(df['open'])
        high = np.array(df['high'])
        low = np.array(df['low'])
        volume = np.array(df['volume'])

        # 计算基础指标
        dif, dea, macd = mt.MACD(close)
        k, d, j = mt.KDJ(close, high, low)
        upper, mid, lower = mt.BOLL(close)
        rsi = mt.RSI(close, N=14)
        rsi = np.nan_to_num(rsi, nan=50)
        psy, psyma = mt.PSY(close)
        wr, wr1 = mt.WR(close, high, low)
        bias1, bias2, bias3 = mt.BIAS(close)
        cci = mt.CCI(close, high, low)

        # 计算均线
        ma5 = mt.MA(close, 5)
        ma10 = mt.MA(close, 10)
        ma20 = mt.MA(close, 20)
        ma60 = mt.MA(close, 60)

        # 计算ATR和EMV
        atr = mt.ATR(close, high, low)
        emv, maemv = mt.EMV(high, low, volume)

        # 新增指标计算
        dpo, madpo = mt.DPO(close)  # 区间振荡
        trix, trma = mt.TRIX(close)  # 三重指数平滑平均
        pdi, mdi, adx, adxr = mt.DMI(close, high, low)  # 动向指标
        vr = mt.VR(close, volume)  # 成交量比率
        ar, br = mt.BRAR(open_price, close, high, low)  # 人气意愿指标
        roc, maroc = mt.ROC(close)  # 变动率
        mtm, mtmma = mt.MTM(close)  # 动量指标
        dif_dma, difma_dma = mt.DMA(close)  # 平行线差指标

        df['MACD'] = macd
        df['DIF'] = dif
        df['DEA'] = dea
        df['K'] = k
        df['D'] = d
        df['J'] = j
        df['BOLL_UP'] = upper
        df['BOLL_MID'] = mid
        df['BOLL_LOW'] = lower
        df['RSI'] = rsi
        df['PSY'] = psy
        df['PSYMA'] = psyma
        df['WR'] = wr
        df['WR1'] = wr1
        df['BIAS1'] = bias1
        df['BIAS2'] = bias2
        df['BIAS3'] = bias3
        df['CCI'] = cci
        df['MA5'] = ma5
        df['MA10'] = ma10
        df['MA20'] = ma20
        df['MA60'] = ma60
        df['ATR'] = atr
        df['EMV'] = emv
        df['MAEMV'] = maemv
        df['DPO'] = dpo
        df['MADPO'] = madpo
        df['TRIX'] = trix
        df['TRMA'] = trma
        df['PDI'] = pdi
        df['MDI'] = mdi
        df['ADX'] = adx
        df['ADXR'] = adxr
        df['VR'] = vr
        df['AR'] = ar
        df['BR'] = br
        df['ROC'] = roc
        df['MAROC'] = maroc
        df['MTM'] = mtm
        df['MTMMA'] = mtmma
        df['DIF_DMA'] = dif_dma
        df['DIFMA_DMA'] = difma_dma

        return df

    def plot_analysis(self, code):
        """绘制技术分析图表"""

        def _style_axis(ax: Axes, title: str):
            """统一设置坐标轴样式"""
            ax.set_title(title, pad=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#666666')
            ax.spines['bottom'].set_color('#666666')
            ax.tick_params(colors='#666666')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='upper left', frameon=True, facecolor='white',
                      edgecolor='none', fontsize=10)
            ax.set_facecolor('#F8F9FA')

        stock_name = self.get_stock_name(code)
        df = self.calculate_indicators(code)

        font_path = './static/fonts/微软雅黑.ttf'
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"找不到字体文件: {font_path}")

        # 注册字体文件
        custom_font = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)

        # 设置字体
        plt.rcParams['font.sans-serif'] = [custom_font.get_name()]
        plt.rcParams['axes.unicode_minus'] = False

        # 设置全局样式
        plt.rcParams.update({
            'axes.facecolor': '#F8F9FA',
            'axes.edgecolor': '#666666',
            'grid.color': '#666666',
            'grid.linestyle': '--',
            'xtick.color': '#666666',
            'ytick.color': '#666666',
            'font.size': 10,
            'axes.unicode_minus': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'figure.titlesize': 16,
            'lines.linewidth': 1.5,
            'lines.markersize': 6
        })

        # 创建图表
        fig = plt.figure(figsize=(15, 32))
        fig.patch.set_facecolor('#F0F2F6')

        # 配色方案
        colors = {
            'primary': '#2E4053',
            'ma5': '#E74C3C',
            'ma10': '#3498DB',
            'ma20': '#2ECC71',
            'boll': ['#E74C3C', '#F4D03F', '#2ECC71'],
            'volume': ['#E74C3C', '#2ECC71']
        }

        # 主图：K线 + 均线 + BOLL
        ax1 = plt.subplot2grid((12, 1), (0, 0), rowspan=2)
        ax1.plot(df.index, df['close'], color=colors['primary'], label='收盘价', alpha=0.8, linewidth=2)
        ax1.plot(df.index, df['MA5'], color=colors['ma5'], label='MA5', alpha=0.7)
        ax1.plot(df.index, df['MA10'], color=colors['ma10'], label='MA10', alpha=0.7)
        ax1.plot(df.index, df['MA20'], color=colors['ma20'], label='MA20', alpha=0.7)
        ax1.plot(df.index, df['BOLL_UP'], color=colors['boll'][0], linestyle='--', label='BOLL上轨', alpha=0.7)
        ax1.plot(df.index, df['BOLL_MID'], color=colors['boll'][1], linestyle='--', label='BOLL中轨', alpha=0.7)
        ax1.plot(df.index, df['BOLL_LOW'], color=colors['boll'][2], linestyle='--', label='BOLL下轨', alpha=0.7)
        _style_axis(ax1, f'{stock_name} ({code}) 技术指标')

        # MACD
        ax2 = plt.subplot2grid((12, 1), (2, 0))
        ax2.plot(df.index, df['DIF'], color='#E74C3C', label='DIF(差离值)', alpha=0.8)
        ax2.plot(df.index, df['DEA'], color='#2ECC71', label='DEA(讯号线)', alpha=0.8)
        ax2.bar(df.index, df['MACD'], color=np.where(df['MACD'] > 0, '#E74C3C', '#2ECC71'),
                label='MACD(指数平滑异同移动平均线)', alpha=0.6)
        _style_axis(ax2, 'MACD (指数平滑异同移动平均线)')

        # KDJ
        ax3 = plt.subplot2grid((12, 1), (3, 0))
        ax3.plot(df.index, df['K'], color='#E74C3C', label='K(随机指标K值)', alpha=0.8)
        ax3.plot(df.index, df['D'], color='#2ECC71', label='D(随机指标D值)', alpha=0.8)
        ax3.plot(df.index, df['J'], color='#3498DB', label='J(随机指标J值)', alpha=0.8)
        _style_axis(ax3, 'KDJ(随机指标)')

        # RSI
        ax4 = plt.subplot2grid((12, 1), (4, 0))
        ax4.plot(df.index, df['RSI'], color='#8E44AD', label='RSI(相对强弱指标)', alpha=0.8)
        ax4.axhline(y=80, color='#E74C3C', linestyle='--', alpha=0.5)
        ax4.axhline(y=20, color='#2ECC71', linestyle='--', alpha=0.5)
        _style_axis(ax4, 'RSI (相对强弱指标)')

        # BIAS
        ax5 = plt.subplot2grid((12, 1), (5, 0))
        ax5.plot(df.index, df['BIAS1'], color='#E74C3C', label='BIAS1', alpha=0.8)
        ax5.plot(df.index, df['BIAS2'], color='#2ECC71', label='BIAS2', alpha=0.8)
        ax5.plot(df.index, df['BIAS3'], color='#3498DB', label='BIAS3', alpha=0.8)
        _style_axis(ax5, 'BIAS (乖离率)')

        # DMI
        ax6 = plt.subplot2grid((12, 1), (6, 0))
        ax6.plot(df.index, df['PDI'], color='#E74C3C', label='PDI(上升方向线)', alpha=0.8)
        ax6.plot(df.index, df['MDI'], color='#2ECC71', label='MDI(下降方向线)', alpha=0.8)
        ax6.plot(df.index, df['ADX'], color='#3498DB', label='ADX(趋向指标)', alpha=0.8)
        ax6.plot(df.index, df['ADXR'], color='#F4D03F', label='ADXR(平均方向指数)', alpha=0.8)
        _style_axis(ax6, 'DMI(动向指标)')

        # TRIX
        ax7 = plt.subplot2grid((12, 1), (7, 0))
        ax7.plot(df.index, df['TRIX'], color='#E74C3C', label='TRIX', alpha=0.8)
        ax7.plot(df.index, df['TRMA'], color='#2ECC71', label='TRMA', alpha=0.8)
        _style_axis(ax7, 'TRIX(三重指数平滑平均线)')

        # ROC
        ax8 = plt.subplot2grid((12, 1), (8, 0))
        ax8.plot(df.index, df['ROC'], color='#E74C3C', label='ROC(变动率)', alpha=0.8)
        ax8.plot(df.index, df['MAROC'], color='#2ECC71', label='MAROC(移动平均线)', alpha=0.8)
        _style_axis(ax8, 'ROC(变动率)')

        # VR和AR/BR
        ax9 = plt.subplot2grid((12, 1), (9, 0))
        ax9.plot(df.index, df['VR'], color='#E74C3C', label='VR(成交量比率)', alpha=0.8)
        ax9.plot(df.index, df['AR'], color='#2ECC71', label='AR(人气指标)', alpha=0.8)
        ax9.plot(df.index, df['BR'], color='#3498DB', label='BR(意愿指标)', alpha=0.8)
        _style_axis(ax9, '成交量指标')

        # MTM
        ax10 = plt.subplot2grid((12, 1), (10, 0))
        ax10.plot(df.index, df['MTM'], color='#E74C3C', label='MTM', alpha=0.8)
        ax10.plot(df.index, df['MTMMA'], color='#2ECC71', label='MTMMA', alpha=0.8)
        _style_axis(ax10, 'MTM(动量指标)')

        # DMA
        ax11 = plt.subplot2grid((12, 1), (11, 0))
        ax11.plot(df.index, df['DIF_DMA'], color='#E74C3C', label='DIF_DMA', alpha=0.8)
        ax11.plot(df.index, df['DIFMA_DMA'], color='#2ECC71', label='DIFMA_DMA', alpha=0.8)
        _style_axis(ax11, 'DMA(平行线差指标)')

        # 调整子图间距
        plt.subplots_adjust(hspace=0.4)

        return plot_to_base64(fig)

    def generate_analysis_data(self, code):
        """生成股票分析数据"""
        df = self.data[code]
        latest_df = self.calculate_indicators(code)

        analysis_data = {
            "基础数据": {
                "股票代码": code,
                "最新收盘价": f"{df['close'].iloc[-1]:.2f}",
                "涨跌幅": f"{((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100):.2f}%",
                "最高价": f"{df['high'].iloc[-1]:.2f}",
                "最低价": f"{df['low'].iloc[-1]:.2f}",
                "成交量": f"{int(df['volume'].iloc[-1]):,}",
            },
            "技术指标": {
                "MA指标": {
                    "MA5": f"{latest_df['MA5'].iloc[-1]:.2f}",
                    "MA10": f"{latest_df['MA10'].iloc[-1]:.2f}",
                    "MA20": f"{latest_df['MA20'].iloc[-1]:.2f}",
                    "MA60": f"{latest_df['MA60'].iloc[-1]:.2f}",
                },
                "趋势指标": {
                    "MACD (指数平滑异同移动平均线)": f"{latest_df['MACD'].iloc[-1]:.2f}",
                    "DIF (差离值)": f"{latest_df['DIF'].iloc[-1]:.2f}",
                    "DEA (讯号线)": f"{latest_df['DEA'].iloc[-1]:.2f}",
                    "TRIX (三重指数平滑平均线)": f"{latest_df['TRIX'].iloc[-1]:.2f}",
                    "PDI (上升方向线)": f"{latest_df['PDI'].iloc[-1]:.2f}",
                    "MDI (下降方向线)": f"{latest_df['MDI'].iloc[-1]:.2f}",
                    "ADX (趋向指标)": f"{latest_df['ADX'].iloc[-1]:.2f}",
                },
                "摆动指标": {
                    "RSI (相对强弱指标)": f"{latest_df['RSI'].iloc[-1]:.2f}",
                    "KDJ-K (随机指标K值)": f"{latest_df['K'].iloc[-1]:.2f}",
                    "KDJ-D (随机指标D值)": f"{latest_df['D'].iloc[-1]:.2f}",
                    "KDJ-J (随机指标J值)": f"{latest_df['J'].iloc[-1]:.2f}",
                    "BIAS (乖离率)": f"{latest_df['BIAS1'].iloc[-1]:.2f}",
                    "CCI (顺势指标)": f"{latest_df['CCI'].iloc[-1]:.2f}",
                },
                "成交量指标": {
                    "VR (成交量比率)": f"{latest_df['VR'].iloc[-1]:.2f}",
                    "AR (人气指标)": f"{latest_df['AR'].iloc[-1]:.2f}",
                    "BR (意愿指标)": f"{latest_df['BR'].iloc[-1]:.2f}",
                },
                "动量指标": {
                    "ROC (变动率)": f"{latest_df['ROC'].iloc[-1]:.2f}",
                    "MTM (动量指标)": f"{latest_df['MTM'].iloc[-1]:.2f}",
                    "DPO (区间振荡)": f"{latest_df['DPO'].iloc[-1]:.2f}",
                },
                "布林带": {
                    "BOLL上轨": f"{latest_df['BOLL_UP'].iloc[-1]:.2f}",
                    "BOLL中轨": f"{latest_df['BOLL_MID'].iloc[-1]:.2f}",
                    "BOLL下轨": f"{latest_df['BOLL_LOW'].iloc[-1]:.2f}",
                }
            },
            "技术分析建议": generate_trading_signals(latest_df)
        }

        """添加AI分析结果"""
        # 获取原有的分析数据
        if self.llm:
            try:
                api_result = self.llm.request_analysis(df, latest_df)
                if api_result:
                    analysis_data.update(api_result)
            except Exception as e:
                print(f"AI分析过程出错: {str(e)}")

        return analysis_data

    def _generate_ai_analysis_html(self, ai_analysis):
        """生成AI分析结果的HTML代码"""
        html = """
        <div class="ai-analysis-section">
            <h3>AI智能分析结果</h3>
            <div class="analysis-grid">
        """

        # 添加各个分析部分
        for section_name, content in ai_analysis.items():
            if section_name == "分析状态" and content == "分析失败":
                continue
            html += f"""
                <div class="analysis-card">
                    <h4>{section_name}</h4>
                    {self._format_analysis_content(content)}
                </div>
            """

        html += """
            </div>
        </div>
        """
        return html

    def _format_analysis_content(self, content):
        """格式化分析内容为HTML"""
        if isinstance(content, dict):
            html = "<table class='analysis-table'>"
            for key, value in content.items():
                html += f"<tr><td>{key}</td><td>{self._format_analysis_content(value)}</td></tr>"
            html += "</table>"
            return html
        elif isinstance(content, list):
            return "<ul>" + "".join(f"<li>{item}</li>" for item in content) + "</ul>"
        else:
            return str(content)

    def generate_html_report(self):
        """生成HTML格式的分析报告"""
        # 读取模板文件
        with open('static/templates/report_template.html', 'r', encoding='utf-8') as f:
            html_template = f.read()

        # 读取样式文件
        with open('static/css/report.css', 'r', encoding='utf-8') as f:
            css_content = f.read()

        tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.now(tz).strftime('%Y年%m月%d日 %H时%M分%S秒')

        stock_contents = []
        for code in self.stock_codes:
            if code in self.data:
                analysis_data = self.generate_analysis_data(code)
                chart_base64 = self.plot_analysis(code)
                stock_name = self.get_stock_name(code)

                # 生成基础数据部分的HTML
                basic_data_html = f"""
                <div class="indicator-section">
                    <h3>基础数据</h3>
                    <table class="data-table">
                        <tr>
                            <th>指标</th>
                            <th>数值</th>
                        </tr>
                        {''.join(_generate_table_row(k, v) for k, v in analysis_data['基础数据'].items())}
                    </table>
                </div>
                """

                # 生成技术指标部分的HTML
                indicator_sections = []
                for section_name, indicators in analysis_data['技术指标'].items():
                    indicator_html = f"""
                    <div class="indicator-section">
                        <h3>{section_name}</h3>
                        <table class="data-table">
                            <tr>
                                <th>指标</th>
                                <th>数值</th>
                            </tr>
                            {''.join(_generate_table_row(k, v) for k, v in indicators.items())}
                        </table>
                    </div>
                    """
                    indicator_sections.append(indicator_html)

                # 生成交易信号部分的HTML
                signals_html = f"""
                <div class="indicator-section">
                    <h3>交易信号</h3>
                    <ul class="signal-list">
                        {''.join(f'<li>{signal}</li>' for signal in analysis_data['技术分析建议'])}
                    </ul>
                </div>
                """

                # 生成AI分析结果的HTML
                ai_analysis_html = ""
                if "AI分析结果" in analysis_data:
                    sections = analysis_data["AI分析结果"]
                    for section_name, content in sections.items():
                        if section_name != "分析状态":
                            ai_analysis_html += f"""
                            <div class="indicator-section">
                                <h3>{section_name}</h3>
                                <div class="analysis-content">
                                    {content}
                                </div>
                            </div>
                            """

                # 组合单个股票的完整内容
                stock_content = f"""
                <div class="stock-container">
                    <h2>{stock_name} ({code}) 分析报告</h2>
                    
                    <div class="section-divider">
                        <h2>基础技术分析</h2>
                    </div>
                    
                    <div class="data-grid">
                        {basic_data_html}
                        {signals_html}
                    </div>
                    
                    <div class="section-divider">
                        <h2>技术指标详情</h2>
                    </div>
                    
                    {''.join(indicator_sections)}
                    
                    <div class="section-divider">
                        <h2>技术指标图表</h2>
                    </div>
                    
                    <div class="chart-container">
                        <img src="data:image/png;base64,{chart_base64}" 
                             alt="{stock_name} ({code})技术分析图表"
                             loading="lazy">
                    </div>
            
                    <div class="section-divider">
                        <h2>人工智能分析报告</h2>
                    </div>
                    {ai_analysis_html}
                </div>
                """
                stock_contents.append(stock_content)

        # 将CSS样式和内容插入到模板中
        template = Template(html_template)
        html_content = template.substitute(
            styles=css_content,
            generate_time=current_time,
            content='\n'.join(stock_contents)
        )
        return html_content

    def run_analysis(self, output_path='public/index.html'):
        """运行分析并生成报告"""
        self.fetch_data()
        html_report = self.generate_html_report()

        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 写入HTML报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)

        return output_path


if __name__ == "__main__":
    stock_info = {
        '跨境通': 'sz002640',
        '农业银行': 'sh601288'
    }
    analyzer = StockAnalyzer(stock_info)
    report_path = analyzer.run_analysis()
    print(f"分析报告已生成: {report_path}")
