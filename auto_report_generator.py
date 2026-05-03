import pandas as pd
import numpy as np
import smtplib
import schedule
import time
import os
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
#  CONFIG — change these for each client
# ═══════════════════════════════════════════════════════
CONFIG = {
    'company_name'   : 'ABC Retail Store',
    'report_email'   : 'manager@example.com',
    'kpi_thresholds' : {
        'daily_sales_min'    : 50000,   # Alert if below Rs. 50k
        'customer_count_min' : 30,      # Alert if below 30 customers
        'return_rate_max'    : 0.10,    # Alert if returns > 10%
    }
}

# ═══════════════════════════════════════════════════════
#  1. Generate / Load Business Data
# ═══════════════════════════════════════════════════════
def generate_sample_data(days=30):
    np.random.seed(int(datetime.now().strftime('%d')))
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]

    data = []
    for date in dates:
        weekend = 1.4 if date.weekday() >= 5 else 1.0
        data.append({
            'date'            : date.strftime('%Y-%m-%d'),
            'daily_sales'     : round(np.random.uniform(45000, 120000) * weekend),
            'customer_count'  : round(np.random.uniform(25, 80) * weekend),
            'avg_order_value' : round(np.random.uniform(800, 2500)),
            'returns'         : round(np.random.uniform(0, 8)),
            'top_product'     : np.random.choice(
                ['Electronics', 'Clothing', 'Food', 'Home'])
        })

    df = pd.DataFrame(data)
    df['return_rate'] = df['returns'] / df['customer_count']
    return df

# ═══════════════════════════════════════════════════════
#  2. KPI Analysis
# ═══════════════════════════════════════════════════════
def analyze_kpis(df):
    today    = df.iloc[0]
    last_7   = df.head(7)
    prev_7   = df.iloc[7:14]

    kpis = {
        'today_sales'      : today['daily_sales'],
        'today_customers'  : today['customer_count'],
        'today_aov'        : today['avg_order_value'],
        'today_returns'    : today['return_rate'],
        'week_sales'       : last_7['daily_sales'].sum(),
        'week_customers'   : last_7['customer_count'].sum(),
        'sales_growth'     : ((last_7['daily_sales'].sum() /
                               prev_7['daily_sales'].sum()) - 1) * 100,
        'best_day'         : last_7.loc[
                               last_7['daily_sales'].idxmax(), 'date'],
        'top_product'      : last_7['top_product'].mode()[0],
    }

    # Check alerts
    alerts = []
    t = CONFIG['kpi_thresholds']
    if kpis['today_sales'] < t['daily_sales_min']:
        alerts.append(
            f"LOW SALES: Today Rs.{kpis['today_sales']:,} "
            f"(below Rs.{t['daily_sales_min']:,})")
    if kpis['today_customers'] < t['customer_count_min']:
        alerts.append(
            f"LOW TRAFFIC: {kpis['today_customers']} customers "
            f"(below {t['customer_count_min']})")
    if kpis['today_returns'] > t['return_rate_max']:
        alerts.append(
            f"HIGH RETURNS: {kpis['today_returns']:.1%} "
            f"(above {t['return_rate_max']:.0%})")

    return kpis, alerts

# ═══════════════════════════════════════════════════════
#  3. Auto Report Generator
# ═══════════════════════════════════════════════════════
def generate_report(df, kpis, alerts):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#f8f9fa')
    fig.suptitle(
        f"{CONFIG['company_name']} — Daily Business Report\n"
        f"{datetime.now().strftime('%A, %d %B %Y')}",
        fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35)

    # KPI Summary Cards
    card_data = [
        ("Today's Sales",    f"Rs. {kpis['today_sales']:,}",    '#2ecc71'),
        ("Customers Today",  str(kpis['today_customers']),       '#3498db'),
        ("Weekly Growth",    f"{kpis['sales_growth']:+.1f}%",
         '#2ecc71' if kpis['sales_growth'] > 0 else '#e74c3c'),
    ]
    for i, (label, value, color) in enumerate(card_data):
        ax = fig.add_axes([0.05 + i * 0.32, 0.87, 0.27, 0.09])
        ax.set_facecolor(color)
        ax.text(0.5, 0.6, value, transform=ax.transAxes,
                ha='center', fontsize=15,
                fontweight='bold', color='white')
        ax.text(0.5, 0.15, label, transform=ax.transAxes,
                ha='center', fontsize=9, color='white')
        ax.set_xticks([]); ax.set_yticks([])

    # 30-day Sales Trend
    ax1 = fig.add_subplot(gs[0, :2])
    sales = df['daily_sales'].values[::-1]
    dates_label = df['date'].values[::-1]
    ax1.plot(range(len(sales)), sales / 1000,
             color='#3498db', linewidth=2)
    ax1.fill_between(range(len(sales)), sales / 1000,
                     alpha=0.15, color='#3498db')
    ax1.axhline(y=CONFIG['kpi_thresholds']['daily_sales_min'] / 1000,
                color='red', linestyle='--',
                alpha=0.6, label='Min Target')
    ax1.set_title('30-Day Sales Trend (Rs. 000s)',
                  fontweight='bold')
    ax1.set_ylabel('Sales (Rs. 000s)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, len(sales), 5))
    ax1.set_xticklabels(
        [dates_label[i][-5:] for i in range(0, len(sales), 5)],
        rotation=30, ha='right', fontsize=7)

    # Product Mix
    ax2 = fig.add_subplot(gs[0, 2])
    prod_counts = df['top_product'].value_counts()
    ax2.pie(prod_counts.values,
            labels=prod_counts.index,
            autopct='%1.0f%%',
            colors=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
            startangle=90,
            textprops={'fontsize': 8})
    ax2.set_title('Top Product Mix', fontweight='bold')

    # Weekly comparison
    ax3 = fig.add_subplot(gs[1, :2])
    weekly = df.head(14).groupby(
        df.head(14).index // 7)['daily_sales'].sum()
    bars = ax3.bar(['This Week', 'Last Week'],
                   weekly.values / 1000,
                   color=['#2ecc71', '#95a5a6'],
                   width=0.4, edgecolor='white')
    ax3.set_title('This Week vs Last Week (Rs. 000s)',
                  fontweight='bold')
    ax3.set_ylabel('Total Sales (Rs. 000s)')
    for bar, val in zip(bars, weekly.values):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 5,
                 f'Rs.{val/1000:.0f}K',
                 ha='center', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Alerts Panel
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor('#fff3cd' if alerts else '#d4edda')
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_title('Alerts', fontweight='bold')
    if alerts:
        alert_text = '\n\n'.join(alerts)
        ax4.text(0.5, 0.5, alert_text,
                 transform=ax4.transAxes,
                 ha='center', va='center',
                 fontsize=8, color='#856404',
                 wrap=True)
    else:
        ax4.text(0.5, 0.5, 'All KPIs Normal\nNo alerts today',
                 transform=ax4.transAxes,
                 ha='center', va='center',
                 fontsize=10, color='#155724',
                 fontweight='bold')

    filename = (f"report_{datetime.now().strftime('%Y%m%d')}.png")
    plt.savefig(filename, dpi=150,
                bbox_inches='tight',
                facecolor='#f8f9fa')
    plt.show()
    print(f"Report saved: {filename}")
    return filename

# ═══════════════════════════════════════════════════════
#  4. Email Alert System
# ═══════════════════════════════════════════════════════
def send_email_alert(kpis, alerts, report_file):
    """
    Sends automated email with report.
    Configure SMTP settings for real use.
    """
    alert_html = ""
    if alerts:
        items = "".join(f"<li>{a}</li>" for a in alerts)
        alert_html = f"""
        <div style='background:#fff3cd;padding:10px;border-radius:5px'>
          <b>Alerts:</b><ul>{items}</ul>
        </div>"""

    html = f"""
    <html><body>
    <h2>{CONFIG['company_name']} — Daily Report</h2>
    <p>Date: {datetime.now().strftime('%d %B %Y')}</p>
    <table border='1' cellpadding='8' style='border-collapse:collapse'>
      <tr bgcolor='#3498db'>
        <th style='color:white'>KPI</th>
        <th style='color:white'>Value</th>
      </tr>
      <tr><td>Today Sales</td>
          <td>Rs. {kpis['today_sales']:,}</td></tr>
      <tr><td>Customers</td>
          <td>{kpis['today_customers']}</td></tr>
      <tr><td>Avg Order Value</td>
          <td>Rs. {kpis['today_aov']:,}</td></tr>
      <tr><td>Weekly Growth</td>
          <td>{kpis['sales_growth']:+.1f}%</td></tr>
      <tr><td>Top Product</td>
          <td>{kpis['top_product']}</td></tr>
    </table>
    {alert_html}
    <p>Report attached. Generated automatically.</p>
    </body></html>
    """
    print(f"\nEmail would be sent to: {CONFIG['report_email']}")
    print("(Configure SMTP credentials for live email sending)")
    print("\nEMAIL PREVIEW:")
    print(f"  To      : {CONFIG['report_email']}")
    print(f"  Subject : Daily Report — "
          f"{CONFIG['company_name']} "
          f"{datetime.now().strftime('%d %b %Y')}")
    print(f"  Alerts  : {len(alerts)} active")

# ═══════════════════════════════════════════════════════
#  5. Scheduler — runs automatically every day
# ═══════════════════════════════════════════════════════
def daily_job():
    print(f"\n{'='*50}")
    print(f"Running daily report: "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    df            = generate_sample_data(days=30)
    kpis, alerts  = analyze_kpis(df)
    report_file   = generate_report(df, kpis, alerts)
    send_email_alert(kpis, alerts, report_file)
    print("Done!")

# ═══════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Business Automation System — Starting")
    print(f"Company : {CONFIG['company_name']}")
    print(f"Report  → {CONFIG['report_email']}")
    print("-" * 40)

    # Run once immediately
    daily_job()

    # Schedule daily at 8 AM
    schedule.every().day.at("08:00").do(daily_job)
    print("\nScheduler active — report runs daily at 08:00 AM")
    print("Press Ctrl+C to stop\n")

    # Uncomment to run as daemon:
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)
